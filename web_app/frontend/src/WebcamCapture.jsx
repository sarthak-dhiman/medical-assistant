import { useRef, useState, useEffect, useCallback } from 'react'
import Webcam from 'react-webcam'
import axios from 'axios'
import ResultDisplay from './ResultDisplay'
import { Camera, RefreshCw, Image as ImageIcon, SwitchCamera } from 'lucide-react'

const API_URL = "" // Relative path (proxied by Nginx)

const WebcamCapture = ({ mode, uploadedImage, isNerdMode }) => {
    const webcamRef = useRef(null)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [isProcessing, setIsProcessing] = useState(false)
    const [fps, setFps] = useState(0)
    const [facingMode, setFacingMode] = useState("user")

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
            // 1. Send to Backend
            const response = await axios.post(`${API_URL}/predict`, {
                image: imageSrc,
                mode: mode
            })

            const { task_id } = response.data

            // 2. Poll for Result
            let pollAttempts = 0
            const pollInterval = setInterval(async () => {
                pollAttempts++
                try {
                    const res = await axios.get(`${API_URL}/result/${task_id}`)
                    if (res.data.state === 'SUCCESS') {
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
                    } else if (pollAttempts > 60) { // Timeout
                        clearInterval(pollInterval)
                        setIsProcessing(false)
                        setError("Timeout")
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
    }, [mode, isProcessing, uploadedImage, result, error])

    // Auto-capture loop
    useEffect(() => {
        const interval = setInterval(() => {
            captureAndPredict()
        }, CAPTURE_INTERVAL)
        return () => clearInterval(interval)
    }, [captureAndPredict])

    // Reset result when uploaded image changes (to force loading state)
    // Reset logic
    useEffect(() => {
        setResult(null);
        setError(null);
    }, [mode])

    useEffect(() => {
        if (uploadedImage) {
            setResult(null);
            // The loop will pick it up and running prediction
        }
    }, [uploadedImage])

    // Status Helper
    const getStatusText = () => {
        if (error) return error.toUpperCase();
        if (uploadedImage && !result && isProcessing) return "LOADING MODEL...";
        if (isProcessing) return "PROCESSING";
        if (!result) return "WAITING...";
        return "IDLE";
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

                {/* HUD Info */}
                <div className="absolute top-20 lg:top-4 right-4 flex gap-3 pointer-events-auto">
                    {/* Camera Switcher */}
                    {!uploadedImage && (
                        <button
                            onClick={toggleCamera}
                            className="bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full flex items-center gap-2 border border-white/10 hover:bg-white/10 transition-colors"
                        >
                            <SwitchCamera className="w-4 h-4 text-white" />
                            <span className="text-xs font-mono text-white hidden sm:inline">
                                {facingMode === 'user' ? 'FRONT' : 'BACK'}
                            </span>
                        </button>
                    )}

                    <div className="bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full flex items-center gap-2 border border-white/10">
                        {uploadedImage ? <ImageIcon className="w-4 h-4 text-blue-400" /> : <Camera className="w-4 h-4 text-green-400" />}
                        <span className={`text-xs font-mono ${uploadedImage ? 'text-blue-400' : 'text-green-400'}`}>
                            {uploadedImage ? 'STATIC' : 'LIVE'}
                        </span>
                    </div>
                    <div className="bg-black/60 backdrop-blur-md px-3 py-1.5 rounded-full flex items-center gap-2 border border-white/10">
                        <RefreshCw className={`w-4 h-4 text-cyan-400 ${isProcessing ? 'animate-spin' : ''} ${error ? 'text-red-500' : ''}`} />
                        <span className={`text-xs font-mono ${error ? 'text-red-500' : 'text-cyan-400'}`}>
                            {getStatusText()}
                        </span>
                    </div>
                </div>

                {/* Mode Indicator */}
                <div className="absolute bottom-6 left-6">
                    <h2 className="text-2xl font-bold text-white drop-shadow-lg">{mode.replace('_', ' ')}</h2>
                    {result && (
                        <div className="mt-2 bg-black/70 backdrop-blur-md rounded-xl p-3 border border-white/10 max-w-sm">
                            <p className="text-gray-300 text-sm">Latest Result:</p>
                            <p className={`text-lg font-bold ${(result.label || '').includes('Jaundice') || (result.label || '').includes('Disease')
                                ? 'text-red-400' : 'text-green-400'
                                }`}>
                                {result.label || 'Analyzing...'}
                                <span className="ml-2 text-sm opacity-75">
                                    {result.confidence ? `${(result.confidence * 100).toFixed(1)}%` : ''}
                                </span>
                            </p>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default WebcamCapture
