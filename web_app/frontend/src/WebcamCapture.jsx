import { useRef, useState, useEffect, useCallback } from 'react'
import Webcam from 'react-webcam'
import axios from 'axios'
import ResultDisplay from './ResultDisplay'
import { Camera, RefreshCw, Image as ImageIcon, SwitchCamera, Bug, HelpCircle } from 'lucide-react'

const API_URL = "" // Relative path (proxied by Nginx)

const WebcamCapture = ({ mode, uploadedImage, isNerdMode, setIsNerdMode, setShowHelp }) => {
    const webcamRef = useRef(null)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [isProcessing, setIsProcessing] = useState(false)
    const [fps, setFps] = useState(0)
    const [facingMode, setFacingMode] = useState("user")
    const [isGPUFull, setIsGPUFull] = useState(false)

    // Ref to track the *current* active mode for avoiding stale responses
    const latestModeRef = useRef(mode)
    useEffect(() => { latestModeRef.current = mode }, [mode])

    // Toggle Camera Callback
    const toggleCamera = useCallback(() => {
        setFacingMode(prev => prev === "user" ? "environment" : "user")
    }, [])

    // Cache to prevent re-sending same static image
    const lastRequestRef = useRef({ image: null, mode: null })

    // Settings
    const CAPTURE_INTERVAL = 500 // ms (2 FPS for inference)

    const captureAndPredict = useCallback(async () => {
        if (isProcessing) return // Skip if busy

        // Determine Source
        let imageSrc = null;
        if (uploadedImage) {
            imageSrc = uploadedImage;
        } else if (webcamRef.current) {
            imageSrc = webcamRef.current.getScreenshot();
        }

        if (!imageSrc) return

        // Optimization: Skip if same image & mode (Only for uploaded images or stable states)
        if (uploadedImage &&
            lastRequestRef.current.image === imageSrc &&
            lastRequestRef.current.mode === mode &&
            (result || error) // If we already have a result OR error, don't re-fetch
        ) {
            return;
        }

        setIsProcessing(true)
        setError(null)
        const startTime = Date.now()

        // Update Cache
        lastRequestRef.current = { image: imageSrc, mode: mode }

        try {
            // 1. DESKTOP BRIDGE (Native ONNX)
            if (window.pywebview) {
                // pywebview exposes 'predict' method via 'window.pywebview.api'
                // The backend returns the final result directly (synchronous inference)
                const result = await window.pywebview.api.predict(imageSrc, mode, isNerdMode)
                setResult(result)
                setIsProcessing(false)

                // FPS Calc
                const duration = Date.now() - startTime
                setFps(Math.round(1000 / duration))
                return // Exit early
            }

            // 2. WEB API (Standard Flow)
            console.log("🤓 Sending Request - Nerd Mode:", isNerdMode, "Mode:", mode);
            const response = await axios.post(`${API_URL}/predict`, {
                image: imageSrc,
                mode: mode,
                debug: isNerdMode // Enable Grad-CAM/Stats if Nerd Mode is ON
            })

            const { task_id } = response.data

            // 2. Poll for Result
            let pollAttempts = 0
            const pollInterval = setInterval(async () => {
                pollAttempts++
                try {
                    const res = await axios.get(`${API_URL}/result/${task_id}`)
                    if (res.data.state === 'SUCCESS') {
                        // PREVENT STALE UPDATES:
                        if (mode !== latestModeRef.current) return;

                        if (res.data.result && res.data.result.status === 'oom_error') {
                            setIsGPUFull(true);
                            setError("GPU Out of Memory");
                            clearInterval(pollInterval);
                            setIsProcessing(false);
                            return;
                        }

                        if (res.data.result && res.data.result.status === 'error') {
                            setError(res.data.result.error);
                            setResult(null);
                            clearInterval(pollInterval);
                            setIsProcessing(false);
                            return;
                        }

                        setResult(res.data.result)
                        clearInterval(pollInterval)
                        setIsProcessing(false)

                        // FPS Calc
                        const duration = Date.now() - startTime
                        setFps(Math.round(1000 / duration))
                    } else if (res.data.state === 'FAILURE') {
                        clearInterval(pollInterval)
                        setIsProcessing(false)
                        setError("Prediction Failed")
                    } else if (pollAttempts > 600) { // Timeout after 120s
                        clearInterval(pollInterval)
                        setIsProcessing(false)
                        setError("Timeout (Backend Slow)")
                    }
                } catch (e) {
                    clearInterval(pollInterval)
                    setIsProcessing(false)
                    setError("Network Error")
                }
            }, 200)

        } catch (error) {
            console.error("Prediction Error:", error)
            setIsProcessing(false)
            setError("Server Error")
        }
    }, [mode, uploadedImage]) // FIX: Simplified dependencies to prevent stale closures

    // Auto-capture loop
    useEffect(() => {
        const interval = setInterval(() => {
            // Only capture if NOT switching (unless switching takes too long)
            captureAndPredict()
        }, CAPTURE_INTERVAL)
        return () => clearInterval(interval)
    }, [captureAndPredict])

    // CRITICAL FIX: Reset ALL state when mode changes
    useEffect(() => {
        setResult(null);
        setError(null);
        setIsGPUFull(false); // Reset OOM status on mode change
        setIsProcessing(false); // FIX: Reset processing state
        lastRequestRef.current = { image: null, mode: null }; // FIX: Clear cache

        // FIX: Force immediate capture after mode switch
        setTimeout(() => {
            if (!uploadedImage) {
                captureAndPredict();
            }
        }, 100);
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
                <div className="absolute bottom-6 left-6 transition-all duration-300">
                    {result && (
                        <div className="bg-black/80 backdrop-blur-md rounded-2xl p-4 border border-white/10 max-w-sm shadow-2xl animate-in fade-in slide-in-from-bottom-4 duration-500">
                            <p className="text-gray-400 text-[10px] font-black uppercase tracking-widest mb-1">AI Diagnosis</p>
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
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default WebcamCapture
