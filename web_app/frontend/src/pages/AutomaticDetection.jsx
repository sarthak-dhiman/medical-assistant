import { useState, useEffect, useRef, useCallback } from 'react'
import Webcam from 'react-webcam'
import { Activity, Bug, Scan, RefreshCw, AlertCircle } from 'lucide-react'
import axios from 'axios'
import ResultDisplay from '../ResultDisplay'
import LoadingOverlay from '../components/LoadingOverlay'

function AutomaticDetection() {
    const webcamRef = useRef(null)
    const [isScanning, setIsScanning] = useState(true)
    const [result, setResult] = useState(null)
    const [debugInfo, setDebugInfo] = useState(null)
    const [isNerdMode, setIsNerdMode] = useState(false)
    const [error, setError] = useState(null)
    const [detectedMode, setDetectedMode] = useState("INITIALIZING...")
    const [isAppReady, setIsAppReady] = useState(false);
    const API_BASE = `http://${window.location.hostname}:8000`;
    const lastRequestTime = useRef(0)
    const isProcessing = useRef(false)
    const [facingMode, setFacingMode] = useState("user")

    // Poll Backend Health until Models are Ready
    useEffect(() => {
        const HEALTH_URL = `${API_BASE}/health`; // Ensure full URL if not proxied
        const checkHealth = async () => {
            try {
                const res = await fetch(HEALTH_URL);
                const data = await res.json();
                if (data.models_ready) {
                    setIsAppReady(true);
                }
            } catch (e) {
                console.error("Backend offline:", e);
            }
        };

        if (!isAppReady) {
            checkHealth();
            const interval = setInterval(checkHealth, 3000);
            return () => clearInterval(interval);
        }
    }, [isAppReady]);


    // 2 FPS Sampling (500ms interval)
    const SAMPLE_INTERVAL = 500;

    const captureAndPredict = useCallback(async () => {
        if (!isAppReady || !isScanning || !webcamRef.current) return;

        const now = Date.now();
        if (now - lastRequestTime.current < SAMPLE_INTERVAL || isProcessing.current) return;

        const imageSrc = webcamRef.current.getScreenshot();
        if (!imageSrc) return;

        lastRequestTime.current = now;
        isProcessing.current = true;

        try {
            // Send to Diagnostic Gateway
            // Using axios for simpler async handling
            const response = await axios.post(`${API_BASE}/predict/auto`, {
                image: imageSrc.split(',')[1],
                mode: 'AUTO', // Ignored by backend but good for schema
                debug: isNerdMode
            });

            const taskId = response.data.task_id;

            // Poll for result (short poll)
            // For a real 2FPS stream, we might want a websocket or just fire-and-forget/poll-once
            // But here we'll do a quick poll
            pollResult(taskId);

        } catch (err) {
            console.error("Auto Prediction Failed", err);
            setError("Connection Lost");
            isProcessing.current = false;
        }
    }, [isScanning, isNerdMode, API_BASE]);

    const pollResult = async (taskId) => {
        try {
            let attempts = 0;
            const maxAttempts = 20; // 2 seconds total wait

            const check = async () => {
                if (attempts >= maxAttempts) {
                    isProcessing.current = false;
                    return;
                }
                attempts++;

                const res = await axios.get(`${API_BASE}/result/${taskId}`);
                if (res.data.state === 'SUCCESS') {
                    const data = res.data.result;
                    if (data.status === 'error') {
                        setError(data.error || "Detection Error");
                        setResult(null);
                    } else {
                        setResult(data);
                        setDetectedMode(data.mode); // Update the detected mode UI
                        if (data.debug_info) setDebugInfo(data.debug_info);
                        setError(null);
                        isProcessing.current = false;
                    }
                } else if (res.data.state === 'PENDING' || res.data.state === 'STARTED') {
                    setTimeout(check, 100); // 100ms retry
                } else {
                    setError("Detection Failed");
                    isProcessing.current = false;
                }
            };

            check();
        } catch (e) {
            console.error("Polling failed", e);
        }
    };

    useEffect(() => {
        const interval = setInterval(captureAndPredict, SAMPLE_INTERVAL);
        return () => clearInterval(interval);
    }, [captureAndPredict]);

    const videoConstraints = {
        width: 640,
        height: 480,
        facingMode: facingMode
    };


    return (
        <div className="w-full h-full bg-gray-950 flex flex-col relative overflow-hidden">

            {/* Block UI until models are warm */}
            {!isAppReady && <LoadingOverlay />}

            {/* HUD Overlay */}
            <div className="absolute inset-0 z-10 pointer-events-none flex flex-col p-4 md:p-6 justify-between">

                {/* Top HUD */}
                <div className="flex justify-between items-start">
                    <div className="flex flex-col gap-1">
                        <span className="text-[8px] md:text-[10px] font-black text-cyan-500 uppercase tracking-widest animate-pulse">
                            DIAGNOSTIC_GATEWAY_V1
                        </span>
                        <h2 className="text-xl md:text-2xl font-black text-white uppercase tracking-tighter flex items-center gap-3">
                            {detectedMode}
                            {isScanning && <div className="w-2 h-2 rounded-full bg-red-500 animate-ping"></div>}
                        </h2>
                    </div>

                    <div className="flex gap-2 pointer-events-auto">
                        <button
                            onClick={() => setFacingMode(prev => prev === "user" ? "environment" : "user")}
                            className="p-2 md:p-3 rounded-xl border transition-all bg-black/40 border-white/10 text-gray-400 hover:text-white hover:border-white/20"
                        >
                            <RefreshCw className="w-4 h-4 md:w-5 md:h-5" />
                        </button>
                        <button
                            onClick={() => setIsNerdMode(!isNerdMode)}
                            className={`p-2 md:p-3 rounded-xl border transition-all ${isNerdMode ? 'bg-purple-500/20 border-purple-500 text-purple-400' : 'bg-black/40 border-white/10 text-gray-400'}`}
                        >
                            <Bug className="w-4 h-4 md:w-5 md:h-5" />
                        </button>
                    </div>
                </div>

                {/* Center Reticle */}
                <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-48 h-48 md:w-64 md:h-64 border border-white/20 rounded-full flex items-center justify-center opacity-50">
                    <div className="w-44 h-44 md:w-60 md:h-60 border-t border-b border-cyan-500/50 rounded-full animate-spin-slow"></div>
                    <div className="absolute w-2 h-2 bg-cyan-500 rounded-full"></div>
                </div>

                {/* Bottom Results Panel */}
                <div className="bg-black/60 backdrop-blur-xl border border-white/10 p-4 md:p-6 rounded-3xl w-full max-w-2xl mx-auto mb-20 md:mb-10 transition-all pointer-events-auto">
                    {result ? (
                        <div className="flex items-center gap-4 md:gap-6">
                            <div className={`w-14 h-14 md:w-20 md:h-20 rounded-2xl flex items-center justify-center border-2 shadow-[0_0_20px_rgba(0,0,0,0.5)] ${result.label === 'Normal' ? 'border-green-500 bg-green-500/10' :
                                result.label.includes('No') ? 'border-gray-500 bg-gray-500/10' : 'border-red-500 bg-red-500/10'
                                }`}>
                                <Activity className={`w-6 h-6 md:w-8 md:h-8 ${result.label === 'Normal' ? 'text-green-500' :
                                    result.label.includes('No') ? 'text-gray-500' : 'text-red-500'
                                    }`} />
                            </div>

                            <div className="flex flex-col flex-1">
                                <span className="text-[10px] md:text-xs font-bold text-gray-500 uppercase tracking-widest mb-1">Diagnosis</span>
                                <h3 className="text-xl md:text-3xl font-black text-white leading-none tracking-tight">
                                    {result.label}
                                </h3>
                                <div className="flex items-center gap-2 mt-2">
                                    <div className="h-1.5 flex-1 bg-gray-800 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-500 ${result.label === 'Normal' ? 'bg-green-500' : 'bg-red-500'
                                                }`}
                                            style={{ width: `${(result.confidence || 0) * 100}%` }}
                                        ></div>
                                    </div>
                                    <span className="text-[10px] font-bold text-gray-400">
                                        {((result.confidence || 0) * 100).toFixed(1)}% CONFIDENCE
                                    </span>
                                </div>
                            </div>
                        </div>
                    ) : (
                        <div className="flex items-center justify-center gap-3 text-gray-500 h-16 md:h-20">
                            <Scan className="w-5 h-5 animate-pulse" />
                            <span className="text-xs font-bold uppercase tracking-widest">Scanning Subject...</span>
                        </div>
                    )}
                </div>
            </div>

            {/* Webcam Feed */}
            <Webcam
                ref={webcamRef}
                audio={false}
                screenshotFormat="image/jpeg"
                videoConstraints={videoConstraints}
                className="absolute inset-0 w-full h-full object-cover"
            />

            {/* Visualizer Overlay (Masks, BBoxes, Nerd Stats) */}
            <div className="absolute inset-0 pointer-events-none">
                <ResultDisplay result={result} mode={detectedMode} isNerdMode={isNerdMode} />
            </div>

        </div>
    )
}

export default AutomaticDetection
