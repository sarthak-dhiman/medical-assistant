import { useRef, useState, useEffect, useCallback } from 'react'
import Webcam from 'react-webcam'
import * as MPFaceMesh from '@mediapipe/face_mesh'
// Fix for Vite production build where FaceMesh might be nested or undefined
const FaceMeshConstructor = MPFaceMesh.FaceMesh || window.FaceMesh || MPFaceMesh?.default?.FaceMesh;
import axios from 'axios'
import ResultDisplay from './ResultDisplay'
import { Camera, RefreshCw, Image as ImageIcon, SwitchCamera, Bug, HelpCircle, Info, ChevronRight, X as CloseIcon, Sun } from 'lucide-react'

const API_URL = `http://${window.location.hostname}:8000`

const WebcamCapture = ({ mode, uploadedImage, isNerdMode, setIsNerdMode, setShowHelp, isAppReady = true }) => {
    const webcamRef = useRef(null)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [isProcessing, setIsProcessing] = useState(false)
    const [fps, setFps] = useState(0)
    const [facingMode, setFacingMode] = useState("user")
    const [isCalibrateEnabled, setIsCalibrateEnabled] = useState(false)
    const [isGPUFull, setIsGPUFull] = useState(false)
    const [showRecs, setShowRecs] = useState(false)
    const [isFaceMeshReady, setIsFaceMeshReady] = useState(false)
    const [isInfant, setIsInfant] = useState(false)

    // Refs for MediaPipe
    const faceMeshRef = useRef(null)
    const canvasRef = useRef(null)
    const latestLandmarksRef = useRef(null)

    // Stable refs so the capture interval is never recreated on prop/state changes
    const captureAndPredictRef = useRef(null)
    const uploadedImageRef = useRef(uploadedImage)
    const isAppReadyRef = useRef(isAppReady)

    // Initialize FaceMesh
    useEffect(() => {
        if (!FaceMeshConstructor) {
            console.error("FaceMesh constructor not found!");
            return;
        }
        const faceMesh = new FaceMeshConstructor({
            locateFile: (file) => {
                // Serve strictly from Vite's local static public directory
                // Avoids CDN race conditions (Assertion failed: undefined) and jsDelivr 302 redirects
                return `/mediapipe/face_mesh/${file}`;
            }
        });

        faceMesh.setOptions({
            maxNumFaces: 1,
            refineLandmarks: true,
            minDetectionConfidence: 0.5,
            minTrackingConfidence: 0.5
        });

        faceMesh.onResults((results) => {
            if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
                latestLandmarksRef.current = results.multiFaceLandmarks[0];
            } else {
                latestLandmarksRef.current = null;
            }
        });

        faceMeshRef.current = faceMesh;
        setIsFaceMeshReady(true);

        return () => {
            if (faceMeshRef.current) {
                faceMeshRef.current.close();
            }
        };
    }, []);

    // Local Cropping Helper (Edge AI)
    const performLocalCrop = useCallback((video, landmarks, featureType) => {
        if (!canvasRef.current || !landmarks) return null;

        const canvas = canvasRef.current;
        const ctx = canvas.getContext('2d');
        const vW = video.videoWidth;
        const vH = video.videoHeight;

        // Landmark indices for cropping
        let indices = [];
        if (featureType === 'EYE') indices = [33, 133, 159, 145]; // Left Eye
        else if (featureType === 'MOUTH') indices = [61, 291, 0, 17]; // Outer Lips

        // Calculate bbox of these landmarks
        let minX = 1, minY = 1, maxX = 0, maxY = 0;
        indices.forEach(idx => {
            const pt = landmarks[idx];
            minX = Math.min(minX, pt.x);
            minY = Math.min(minY, pt.y);
            maxX = Math.max(maxX, pt.x);
            maxY = Math.max(maxY, pt.y);
        });

        // Add 50% padding
        const width = (maxX - minX) * vW;
        const height = (maxY - minY) * vH;
        const centerX = ((minX + maxX) / 2) * vW;
        const centerY = ((minY + maxY) / 2) * vH;

        const side = Math.max(width, height) * 2.0;
        const x = centerX - side / 2;
        const y = centerY - side / 2;

        // Normalized bbox of the crop region (for nerd mode overlay)
        const cropBbox = [
            Math.max(0, x / vW),
            Math.max(0, y / vH),
            Math.min(1, (x + side) / vW),
            Math.min(1, (y + side) / vH)
        ];

        // Compute mouth open ratio (same formula as backend analyze_mouth)
        // Landmarks: 13 = top lip inner, 14 = bottom lip inner, 10 = forehead, 152 = chin
        let mouthOpenRatio = null;
        if (featureType === 'MOUTH' && landmarks[13] && landmarks[14] && landmarks[10] && landmarks[152]) {
            const topLip = landmarks[13];
            const botLip = landmarks[14];
            const faceH = Math.abs(landmarks[152].y - landmarks[10].y) * vH;
            const mouthDist = Math.sqrt(
                Math.pow((topLip.x - botLip.x) * vW, 2) +
                Math.pow((topLip.y - botLip.y) * vH, 2)
            );
            mouthOpenRatio = mouthDist / (faceH + 1e-6);
        }

        // Perform Crop (224x224 standard)
        canvas.width = 224;
        canvas.height = 224;
        ctx.drawImage(video, x, y, side, side, 0, 0, 224, 224);

        return { dataUrl: canvas.toDataURL('image/jpeg', 0.85), cropBbox, mouthOpenRatio };
    }, []);

    // Toggle Camera Callback
    const toggleCamera = useCallback(() => {
        setFacingMode(prev => prev === "user" ? "environment" : "user")
    }, [])

    // Cache to prevent re-sending same static image
    const lastRequestRef = useRef({ image: null, mode: null })

    // Ref to track the *current* active mode for avoiding stale responses
    const latestModeRef = useRef(mode)

    // Ref to track and cancel active polling intervals
    const activePollRef = useRef(null)

    // Ref to track processing state (avoids stale closure issues)
    const isProcessingRef = useRef(false)

    // Settings
    const CAPTURE_INTERVAL = 500 // ms (2 FPS for inference)

    const captureAndPredict = useCallback(async () => {
        // Use ref for immediate check (avoids stale closure)
        if (isProcessingRef.current) {
            return;
        }

        // Determine Source with validation
        let imageSrc = null;
        let isLocalPreprocessed = false; // Local variable — never use window globals for this
        let localCropBbox = null; // Normalized crop coordinates for nerd mode bbox overlay
        let localMouthOpenRatio = null; // Mouth open ratio from FaceMesh (for TEETH mode gate)
        try {
            if (uploadedImage) {
                // Validate uploaded image format
                if (typeof uploadedImage !== 'string' || uploadedImage.length === 0) {
                    console.error("Invalid uploaded image format");
                    setError("Invalid Image Format");
                    return;
                }
                // Check for oversized images (>50MB base64)
                if (uploadedImage.length > 50 * 1024 * 1024 * 1.37) {
                    console.error("Image too large");
                    setError("Image Too Large (Max 50MB)");
                    return;
                }
                imageSrc = uploadedImage;
            } else if (webcamRef.current) {
                // EDGE AI: If FaceMesh has landmarks, and we are in a facial mode, crop LOCALLY
                const landmarks = latestLandmarksRef.current;
                // JAUNDICE_EYE excluded: its model needs dual inputs (skin + sclera)
                // that only server-side SegFormer can provide
                const facialModes = ['ORAL_CANCER', 'TEETH', 'CATARACT'];

                if (landmarks && facialModes.includes(mode) && webcamRef.current.video) {
                    const featureType = (mode === 'ORAL_CANCER' || mode === 'TEETH') ? 'MOUTH' : 'EYE';
                    const cropResult = performLocalCrop(webcamRef.current.video, landmarks, featureType);
                    if (cropResult) {
                        imageSrc = cropResult.dataUrl;
                        localCropBbox = cropResult.cropBbox;
                        localMouthOpenRatio = cropResult.mouthOpenRatio;
                        isLocalPreprocessed = true;
                    }
                }

                if (!imageSrc) {
                    imageSrc = webcamRef.current.getScreenshot();
                    // isLocalPreprocessed stays false
                }

                if (!imageSrc) {
                    console.warn("Failed to capture screenshot from webcam");
                    return;
                }
            }
        } catch (e) {
            console.error("Error getting image source:", e);
            setError("Camera Error");
            return;
        }

        if (!imageSrc) {
            console.log("No image source available");
            return;
        }

        // Optimization: Skip if same image & mode (Only for uploaded images or stable states)
        if (uploadedImage &&
            lastRequestRef.current.image === imageSrc &&
            lastRequestRef.current.mode === mode &&
            (result || error)
        ) {
            return;
        }

        // Update Cache
        lastRequestRef.current = { image: imageSrc, mode: mode }

        isProcessingRef.current = true;
        setIsProcessing(true)
        setError(null)
        const startTime = Date.now()
        const isPreprocessed = isLocalPreprocessed;

        try {
            // 1. DESKTOP BRIDGE (Native ONNX)
            if (window.pywebview) {
                // pywebview exposes 'predict' method via 'window.pywebview.api'
                // The backend returns the final result directly (synchronous inference)
                const result = await window.pywebview.api.predict(imageSrc, mode, isNerdMode)

                // Validate result
                if (!result || typeof result !== 'object') {
                    throw new Error("Invalid result from pywebview");
                }

                setResult(result)
                isProcessingRef.current = false;
                setIsProcessing(false)
                setIsGPUFull(false) // Reset GPU error state on success

                // FPS Calc
                const duration = Date.now() - startTime
                setFps(Math.round(1000 / duration))
                return // Exit early
            }

            // 2. WEB API (Standard Flow)
            console.log("Sending Request - Nerd Mode:", isNerdMode, "Mode:", mode, "Preprocessed:", isPreprocessed);

            // Add timeout and better error handling
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

            const response = await axios.post(`${API_URL}/predict`, {
                image: imageSrc,
                mode: mode,
                debug: isNerdMode, // Enable Grad-CAM/Stats if Nerd Mode is ON
                is_upload: !!uploadedImage, // True when using uploaded image (enables foot/nail fallback)
                is_preprocessed: isPreprocessed, // Signal backend to skip landmarks
                calibrate: isCalibrateEnabled,
                is_infant: isInfant, // Demographic routing for Adult Jaundice -> Skin Disease
                crop_bbox: localCropBbox, // Normalized crop coords for nerd mode bbox (Edge AI path)
                mouth_open_ratio: localMouthOpenRatio // Mouth open ratio for TEETH mode gate (Edge AI path)
            }, {
                signal: controller.signal,
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            clearTimeout(timeoutId);

            // Validate response
            if (!response.data || !response.data.task_id) {
                throw new Error("Invalid response from server: missing task_id");
            }

            const { task_id } = response.data
            const requestMode = mode; // Capture mode at request time

            // 2. Poll for Result with improved error handling
            let pollAttempts = 0
            const MAX_POLL_ATTEMPTS = 600 // 120s timeout
            const POLL_INTERVAL = 200 // ms

            // Cancel any existing poll before starting new one
            if (activePollRef.current) {
                clearInterval(activePollRef.current);
            }

            const pollInterval = setInterval(async () => {
                pollAttempts++

                // Check if mode changed - cancel this poll
                if (requestMode !== latestModeRef.current) {
                    clearInterval(pollInterval);
                    activePollRef.current = null;
                    return;
                }

                try {
                    const res = await axios.get(`${API_URL}/result/${task_id}`, {
                        timeout: 5000 // 5s timeout for poll requests
                    })

                    // Double-check mode hasn't changed during the request
                    if (requestMode !== latestModeRef.current) {
                        clearInterval(pollInterval);
                        activePollRef.current = null;
                        return;
                    }

                    if (res.data.state === 'SUCCESS') {
                        // Handle GPU OOM
                        if (res.data.result && res.data.result.status === 'oom_error') {
                            setIsGPUFull(true);
                            setError("GPU Out of Memory");
                            setResult(null);
                            clearInterval(pollInterval);
                            activePollRef.current = null;
                            isProcessingRef.current = false;
                            setIsProcessing(false);
                            return;
                        }

                        // Handle general errors
                        if (res.data.result && res.data.result.status === 'error') {
                            setError(res.data.result.error || "Prediction error");
                            setResult(null);
                            clearInterval(pollInterval);
                            activePollRef.current = null;
                            isProcessingRef.current = false;
                            setIsProcessing(false);
                            return;
                        }

                        // Validate result data
                        if (!res.data.result || typeof res.data.result !== 'object') {
                            setError("Invalid result format");
                            setResult(null);
                            clearInterval(pollInterval);
                            activePollRef.current = null;
                            isProcessingRef.current = false;
                            setIsProcessing(false);
                            return;
                        }

                        setResult(res.data.result)
                        setIsGPUFull(false) // Reset GPU error on success
                        clearInterval(pollInterval)
                        activePollRef.current = null;
                        isProcessingRef.current = false;
                        setIsProcessing(false)

                        // FPS Calc
                        const duration = Date.now() - startTime
                        setFps(Math.round(1000 / duration))
                    } else if (res.data.state === 'FAILURE') {
                        clearInterval(pollInterval)
                        activePollRef.current = null;
                        isProcessingRef.current = false;
                        setIsProcessing(false)
                        setError(res.data.error || "Prediction Failed")
                        setResult(null)
                    } else if (pollAttempts > MAX_POLL_ATTEMPTS) { // Timeout
                        clearInterval(pollInterval)
                        activePollRef.current = null;
                        isProcessingRef.current = false;
                        setIsProcessing(false)
                        setError("Timeout (Backend Slow)")
                        setResult(null)
                    }
                } catch (e) {
                    clearInterval(pollInterval)
                    activePollRef.current = null;
                    isProcessingRef.current = false;
                    setIsProcessing(false)
                    if (e.code === 'ECONNABORTED' || e.message.includes('timeout')) {
                        setError("Network Timeout")
                    } else if (e.response) {
                        setError(`Server Error: ${e.response.status}`)
                    } else {
                        setError("Network Error")
                    }
                    setResult(null)
                }
            }, POLL_INTERVAL)

            // Store reference so we can cancel on mode change
            activePollRef.current = pollInterval;

        } catch (error) {
            console.error("Prediction Error:", error)
            isProcessingRef.current = false;
            setIsProcessing(false)
            if (error.name === 'AbortError' || error.code === 'ECONNABORTED') {
                setError("Request Timeout")
            } else if (error.response) {
                setError(`Server Error: ${error.response.status}`)
            } else if (error.request) {
                setError("Network Error - No Response")
            } else {
                setError("Request Failed")
            }
        }
    }, [mode, uploadedImage, isNerdMode, isCalibrateEnabled])

    // Keep refs in sync with the latest versions (no interval recreation needed)
    useEffect(() => { captureAndPredictRef.current = captureAndPredict; }, [captureAndPredict])
    useEffect(() => { uploadedImageRef.current = uploadedImage; }, [uploadedImage])
    useEffect(() => { isAppReadyRef.current = isAppReady; }, [isAppReady])

    // Auto-capture loop — stable interval created once, uses refs to stay current
    useEffect(() => {
        const interval = setInterval(async () => {
            // Don't fire requests until backend models are warm
            if (!isAppReadyRef.current) return;

            // Only proceed once the webcam video stream is actually playing
            const isUploaded = !!uploadedImageRef.current;
            const videoReady = !isUploaded &&
                webcamRef.current &&
                webcamRef.current.video &&
                webcamRef.current.video.readyState === 4;

            // Run MediaPipe inference in the background (webcam only)
            if (faceMeshRef.current && videoReady) {
                try {
                    await faceMeshRef.current.send({ image: webcamRef.current.video });
                } catch (e) {
                    console.error("MediaPipe send error:", e);
                }
            }

            // For webcam: skip until stream is ready to avoid null getScreenshot()
            if (!isUploaded && !videoReady) return;

            captureAndPredictRef.current?.()
        }, CAPTURE_INTERVAL)
        return () => clearInterval(interval)
    }, []) // eslint-disable-line react-hooks/exhaustive-deps

    // CRITICAL FIX: Reset ALL state when mode changes
    useEffect(() => {
        // Cancel any active polling from previous mode
        if (activePollRef.current) {
            clearInterval(activePollRef.current);
            activePollRef.current = null;
        }

        latestModeRef.current = mode;
        setResult(null);
        setError(null);
        setIsGPUFull(false);
        isProcessingRef.current = false;
        setIsProcessing(false);
        setShowRecs(false);
        setIsCalibrateEnabled(false);
        lastRequestRef.current = { image: null, mode: null };

        // Default to Infant=true for Jaundice modes
        if (mode === "JAUNDICE_BODY" || mode === "JAUNDICE_EYE") {
            setIsInfant(true);
        } else {
            setIsInfant(false);
        }
    }, [mode])

    // Status Helper
    const getStatusText = () => {
        if (error) return error.toUpperCase();
        if (uploadedImage && !result && isProcessing) return "LOADING MODEL...";
        if (isProcessing) return "PROCESSING";
        if (!result) return "WAITING...";
        return "IDLE";
    }

    // Dynamic Instructions
    const getInstructions = () => {
        switch (mode) {
            case "JAUNDICE_EYE":
                return isInfant ? "Come closer. Best for babies." : "Routing to Adult Skin Model.";
            case "JAUNDICE_BODY":
                return isInfant ? "Ensure good lighting. Show face or skin clearly. Best for babies." : "Routing to Adult Skin Model.";
            case "SKIN_DISEASE":
                return "Focus camera on affected area. Keep steady.";
            case "NAIL_DISEASE":
                return "Focus specifically on the affected nail.";
            case "CATARACT":
                return "Zoom in on the eye. Good lighting is crucial.";
            case "ORAL_CANCER":
                return "Open mouth wide. Focus on the lesion.";
            case "TEETH":
                return "Open mouth wide. Focus clearly on the teeth.";
            default:
                return "";
        }
    }

    return (
        <div className="relative rounded-3xl overflow-hidden shadow-2xl bg-black h-full w-full border-2 border-gray-800 group">
            {/* Video Feed or Static Image */}
            {uploadedImage ? (
                <img
                    src={uploadedImage}
                    alt="Uploaded"
                    className="w-full h-full object-contain bg-black"
                />
            ) : (
                <Webcam
                    key={facingMode}
                    audio={false}
                    ref={webcamRef}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{
                        width: 1280,
                        height: 720,
                        facingMode: facingMode
                    }}
                    onUserMediaError={(err) => {
                        console.error("Camera Error:", err);
                        setError("Camera Access Denied. (HTTPS required?)");
                    }}
                    className="w-full h-full object-cover" // Ensure it covers
                />
            )}

            {/* GPU Memory Modal */}
            {isGPUFull && (
                <div className="absolute inset-0 z-[100] flex items-center justify-center p-6 bg-black/80 backdrop-blur-md animate-in fade-in duration-300">
                    <div className="bg-gray-900 border-2 border-red-500/50 rounded-3xl p-8 max-w-md w-full shadow-[0_0_50px_rgba(239,68,68,0.3)] animate-in zoom-in-95 duration-300">
                        <div className="flex justify-center mb-6">
                            <div className="w-16 h-16 rounded-2xl bg-red-500/20 flex items-center justify-center text-red-500">
                                <Bug className="w-10 h-10" />
                            </div>
                        </div>
                        <h2 className="text-2xl font-black text-white text-center mb-2 tracking-tight">GPU MEMORY FULL</h2>
                        <p className="text-gray-400 text-center text-sm mb-6 leading-relaxed">
                            The AI engine has reached its hardware limit. This happens when too many apps are using the GPU or the resolution is too high.
                        </p>
                        <div className="space-y-3 mb-8">
                            <div className="flex items-center gap-3 text-xs bg-white/5 p-3 rounded-xl border border-white/5 text-gray-300 italic">
                                <span className="text-red-400 font-bold">1.</span> Close other browser tabs or software.
                            </div>
                            <div className="flex items-center gap-3 text-xs bg-white/5 p-3 rounded-xl border border-white/5 text-gray-300 italic">
                                <span className="text-red-400 font-bold">2.</span> Restart the Docker containers.
                            </div>
                        </div>
                        <button
                            onClick={() => {
                                setIsGPUFull(false);
                                setError(null);
                            }}
                            className="w-full py-4 bg-red-600 hover:bg-red-500 text-white font-black rounded-2xl transition-all active:scale-95 shadow-lg shadow-red-900/40 uppercase tracking-widest text-xs"
                        >
                            I Understand
                        </button>
                    </div>
                </div>
            )}

            {/* HTTP Warning (Mobile) */}
            {!window.isSecureContext && window.location.hostname !== "localhost" && (
                <div className="absolute top-1/2 left-1/2 transform -translate-x-1/2 -translate-y-1/2 bg-red-900/90 p-6 rounded-2xl border border-red-500 text-center z-50 max-w-sm">
                    <h3 className="font-bold text-red-200 text-lg mb-2">⚠️ Camera Blocked</h3>
                    <p className="text-red-100 text-sm mb-4">
                        Browsers block camera access on insecure connections (HTTP) for mobile devices.
                    </p>
                    <p className="text-white text-xs opacity-75">
                        <strong>Fix:</strong> Use 'chrome://flags' to allow this IP or use a localhost tunnel.
                    </p>
                </div>
            )}

            {/* Overlay UI */}
            <div className="absolute inset-0 pointer-events-none">
                {/* Result Overlay */}
                <ResultDisplay result={result} mode={mode} isNerdMode={isNerdMode} />

                {/* Unified Glass Header Bar (Opaque to hide masks behind UI) */}
                <div className="absolute top-0 left-0 right-0 h-14 bg-gray-950 border-b border-white/10 flex items-center justify-between px-4 z-20 pointer-events-auto">

                    {/* Left: Mode Title (Hierarchy Swapped) */}
                    <div className="flex flex-col leading-none">
                        <span className="text-sm font-black text-cyan-400 tracking-tight uppercase">Medical AI</span>
                        <span className="text-[10px] font-bold text-white/60 uppercase tracking-widest truncate max-w-[100px] sm:max-w-none mt-0.5">
                            {mode.replace('_', ' ')}
                        </span>
                    </div>

                    {/* Center: Live Status & Processing (Fixed Widths to prevent shift) */}
                    <div className="flex items-center gap-1.5 bg-black/40 p-1 rounded-2xl border border-white/5 shrink-0">
                        <div className={`flex items-center justify-center gap-1.5 px-2.5 py-1 rounded-xl transition-all min-w-[70px] ${uploadedImage ? 'bg-blue-600/40 text-blue-200' : 'bg-green-600/40 text-green-200'
                            }`}>
                            <div className={`w-1.5 h-1.5 rounded-full shrink-0 ${uploadedImage ? 'bg-blue-400' : 'bg-green-400 animate-pulse'}`} />
                            <span className="text-[10px] font-bold uppercase tracking-tight">{uploadedImage ? 'Static' : 'Live'}</span>
                        </div>

                        <div className="flex items-center justify-center gap-1.5 px-2.5 py-1 rounded-xl bg-cyan-600/20 text-cyan-200 min-w-[75px]">
                            <RefreshCw className={`w-3 h-3 shrink-0 ${isProcessing ? 'animate-spin' : ''} ${error ? 'text-red-400' : ''}`} />
                            <span className={`text-[10px] font-bold uppercase tracking-tight ${error ? 'text-red-400' : ''}`}>
                                {isProcessing ? 'Proc' : 'Ready'}
                            </span>
                        </div>
                    </div>

                    {/* Right: Tools Group */}
                    <div className="flex items-center gap-1">
                        {!uploadedImage && (
                            <button
                                onClick={toggleCamera}
                                className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-xl transition-all active:scale-95"
                                title="Switch Camera"
                            >
                                <SwitchCamera className="w-5 h-5" />
                            </button>
                        )}

                        {/* Calibrate Lighting Toggle */}
                        <button
                            onClick={() => setIsCalibrateEnabled(!isCalibrateEnabled)}
                            className={`p-2 rounded-xl transition-all duration-300 ${isCalibrateEnabled ? 'bg-orange-500/20 text-orange-400 border border-orange-500/50' : 'bg-black/20 text-white/70 hover:bg-black/40 border border-white/10'}`}
                            title="Calibrate Lighting (Color Constancy)"
                        >
                            <Sun size={18} className={isCalibrateEnabled ? 'animate-pulse' : ''} />
                        </button>

                        {/* Demographic Toggle (Only visible for Jaundice Modes) */}
                        {(mode === "JAUNDICE_BODY" || mode === "JAUNDICE_EYE") && (
                            <button
                                onClick={() => setIsInfant(!isInfant)}
                                className={`p-2 rounded-xl transition-all duration-300 font-bold text-[10px] uppercase tracking-wider flex items-center justify-center min-w-[60px] ${isInfant ? 'bg-pink-500/20 text-pink-400 border border-pink-500/50' : 'bg-indigo-500/20 text-indigo-400 border border-indigo-500/50'}`}
                                title={isInfant ? "Neonatal Mode (Babies)" : "Adult Mode"}
                            >
                                {isInfant ? "👶 Infant" : "👤 Adult"}
                            </button>
                        )}

                        <button
                            onClick={() => setIsNerdMode(!isNerdMode)}
                            className={`p-2 rounded-xl transition-all duration-300 ${isNerdMode ? 'bg-purple-500/20 text-purple-400 border border-purple-500/50' : 'bg-black/20 text-white/70 hover:bg-black/40 border border-white/10'}`}
                            title="Nerd Mode (Debug)"
                        >
                            <Bug size={18} className={isNerdMode ? 'animate-pulse' : ''} />
                        </button>

                        {setShowHelp && (
                            <button
                                onClick={() => setShowHelp(true)}
                                className="p-2 text-gray-400 hover:text-white hover:bg-white/10 rounded-xl transition-all active:scale-95"
                                title="Help"
                            >
                                <HelpCircle className="w-5 h-5" />
                            </button>
                        )}
                    </div>
                </div>

                {/* Instructions Removed (Moved to Sidebar) */}

                {/* Mode Indicator & Instructions */}
                <div className="absolute bottom-32 md:bottom-28 lg:bottom-6 left-4 right-4 lg:left-6 lg:right-auto transition-all duration-300 pointer-events-auto flex justify-center lg:block">
                    {result && (
                        <div className="bg-black/80 backdrop-blur-md rounded-2xl p-4 border border-white/10 w-full lg:w-auto lg:max-w-sm shadow-2xl animate-in fade-in slide-in slide-in-from-bottom-4 duration-500">
                            <p className="text-gray-500 text-[10px] font-black uppercase tracking-widest mb-2">AI Diagnosis</p>
                            <div className="flex items-center justify-between gap-4">
                                <div className="flex flex-col">
                                    <div className="flex items-center gap-2">
                                        <p className={`text-xl font-bold ${(result.label || '').includes('Jaundice') || (result.label || '').includes('Disease')
                                            ? 'text-red-400' : 'text-green-400'
                                            }`}>
                                            {(result.label || '').replace(/unknown_normal/gi, 'Normal') || 'Analyzing...'}
                                        </p>
                                        {result.triage && (
                                            <span className={`px-2 py-0.5 text-[10px] font-bold uppercase tracking-wider rounded-md border ${result.triage.color}`}>
                                                {result.triage.level}
                                            </span>
                                        )}
                                    </div>
                                    {result.triage && (
                                        <p className="text-xs text-gray-400 mt-1 italic leading-tight">
                                            {result.triage.message}
                                        </p>
                                    )}
                                </div>
                                <span className="text-sm font-black text-white/40 shrink-0">
                                    {result.confidence ? `${(result.confidence * 100).toFixed(1)}%` : ''}
                                </span>
                            </div>
                            {result.recommendations && (
                                <button
                                    onClick={() => setShowRecs(true)}
                                    className="mt-3 w-full flex items-center justify-center gap-1.5 text-xs font-bold text-cyan-400 hover:text-cyan-300 bg-cyan-400/10 hover:bg-cyan-400/20 py-2 px-3 rounded-lg transition-all"
                                >
                                    Learn More <ChevronRight className="w-4 h-4" />
                                </button>
                            )}
                        </div>
                    )}
                </div>
            </div>

            {/* Recommendations Modal */}
            {showRecs && result?.recommendations && (
                <div className="fixed inset-0 z-[100] flex items-center justify-center p-4 bg-black/80 backdrop-blur-xl animate-in fade-in duration-300">
                    <div className="bg-gray-900 border border-white/10 rounded-3xl w-full max-w-lg shadow-[0_0_50px_rgba(0,0,0,0.5)] overflow-hidden flex flex-col max-h-[85vh] animate-in zoom-in-95 duration-300">
                        {/* Modal Header */}
                        <div className="p-5 border-b border-white/10 flex items-center justify-between bg-gradient-to-r from-blue-600/10 to-transparent shrink-0">
                            <div className="flex items-center gap-3">
                                <div className="p-2.5 bg-blue-600/20 rounded-xl text-blue-400">
                                    <Info className="w-5 h-5" />
                                </div>
                                <div>
                                    <h2 className="text-xl font-black text-white tracking-tight">Condition Details</h2>
                                    <p className="text-blue-400 font-bold text-[10px] uppercase tracking-widest">{result.label}</p>
                                </div>
                            </div>
                            <button
                                onClick={() => setShowRecs(false)}
                                className="p-2 hover:bg-white/10 rounded-xl text-gray-400 hover:text-white transition-all"
                            >
                                <CloseIcon className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Modal Body */}
                        <div className="flex-1 overflow-y-auto p-5 space-y-4 scrollbar-none [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
                            {/* Description */}
                            <section>
                                <h3 className="text-[10px] font-black text-gray-500 uppercase tracking-widest mb-1.5 flex items-center gap-2">
                                    <div className="w-1 h-2.5 bg-blue-500 rounded-full" /> Overview
                                </h3>
                                <p className="text-gray-300 text-sm leading-relaxed">{result.recommendations.description}</p>
                            </section>

                            {/* Causes */}
                            {result.recommendations.causes && (
                                <section className="bg-white/5 rounded-xl p-3 border border-white/5">
                                    <h3 className="text-[10px] font-black text-blue-400 uppercase tracking-widest mb-1.5">Common Causes</h3>
                                    <p className="text-gray-300 text-xs leading-relaxed">{result.recommendations.causes}</p>
                                </section>
                            )}

                            {/* Care & Recommendations */}
                            <div className="grid grid-cols-2 gap-3">
                                <section className="bg-green-500/5 rounded-xl p-3 border border-green-500/10">
                                    <h3 className="text-[10px] font-black text-green-400 uppercase tracking-widest mb-1.5">Care Tips</h3>
                                    <p className="text-gray-300 text-[11px] leading-relaxed">{result.recommendations.care}</p>
                                </section>
                                <section className="bg-cyan-500/5 rounded-xl p-3 border border-cyan-500/10">
                                    <h3 className="text-[10px] font-black text-cyan-400 uppercase tracking-widest mb-1.5">AI Guidance</h3>
                                    <p className="text-gray-300 text-[11px] leading-relaxed">{result.recommendations.recommendations}</p>
                                </section>
                            </div>

                            {/* Disclaimer */}
                            <p className="text-[9px] text-gray-500 italic text-center pt-3 border-t border-white/5">
                                This information is for educational purposes only and not a substitute for professional medical advice.
                            </p>
                        </div>

                        {/* Close Button */}
                        <div className="p-4 bg-gray-950/50 border-t border-white/10 shrink-0">
                            <button
                                onClick={() => setShowRecs(false)}
                                className="w-full py-3 bg-white text-black font-black rounded-xl hover:bg-gray-200 transition-all active:scale-[0.98] uppercase tracking-widest text-[11px]"
                            >
                                Close Details
                            </button>
                        </div>
                    </div>
                </div>
            )}
            {/* Hidden canvas for Edge AI cropping */}
            <canvas ref={canvasRef} style={{ display: 'none' }} />
        </div>
    )
}

export default WebcamCapture
