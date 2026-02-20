import { useRef, useState, useEffect, useCallback } from 'react'
import Webcam from 'react-webcam'
import axios from 'axios'
import ResultDisplay from './ResultDisplay'
import { Camera, RefreshCw, Image as ImageIcon, SwitchCamera, Bug, HelpCircle, Info, ChevronRight, X as CloseIcon } from 'lucide-react'

const API_URL = `http://${window.location.hostname}:8000`

const WebcamCapture = ({ mode, uploadedImage, isNerdMode, setIsNerdMode, setShowHelp }) => {
    const webcamRef = useRef(null)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [isProcessing, setIsProcessing] = useState(false)
    const [fps, setFps] = useState(0)
    const [facingMode, setFacingMode] = useState("user")
    const [isGPUFull, setIsGPUFull] = useState(false)
    const [showRecs, setShowRecs] = useState(false)

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
                imageSrc = webcamRef.current.getScreenshot();
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
            console.log("Sending Request - Nerd Mode:", isNerdMode, "Mode:", mode);

            // Add timeout and better error handling
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 30000); // 30s timeout

            const response = await axios.post(`${API_URL}/predict`, {
                image: imageSrc,
                mode: mode,
                debug: isNerdMode, // Enable Grad-CAM/Stats if Nerd Mode is ON
                is_upload: !!uploadedImage // True when using uploaded image (enables foot/nail fallback)
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
    }, [mode, uploadedImage, isNerdMode])

    // Auto-capture loop
    useEffect(() => {
        const interval = setInterval(() => {
            captureAndPredict()
        }, CAPTURE_INTERVAL)
        return () => clearInterval(interval)
    }, [captureAndPredict])

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
        lastRequestRef.current = { image: null, mode: null };
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
                return "Come closer & remove glasses. Best for adults.";
            case "JAUNDICE_BODY":
                return "Ensure good lighting. Show face or skin clearly. Best for babies.";
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
                                <p className={`text-xl font-bold ${(result.label || '').includes('Jaundice') || (result.label || '').includes('Disease')
                                    ? 'text-red-400' : 'text-green-400'
                                    }`}>
                                    {(result.label || '').replace(/unknown_normal/gi, 'Normal') || 'Analyzing...'}
                                </p>
                                <span className="text-sm font-black text-white/40">
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
        </div>
    )
}

export default WebcamCapture
