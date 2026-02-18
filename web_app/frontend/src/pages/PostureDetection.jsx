import { useState, useEffect, useRef, useCallback } from 'react'
import Webcam from 'react-webcam'
import { Activity, Scan, Target, User, RefreshCw } from 'lucide-react'
import axios from 'axios'

function PostureDetection() {
    const webcamRef = useRef(null)
    const [isScanning, setIsScanning] = useState(true)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const lastRequestTime = useRef(0)
    const isProcessing = useRef(false)
    const [facingMode, setFacingMode] = useState("user")

    // Loading State
    const [isAppReady, setIsAppReady] = useState(false);
    const API_BASE = `http://${window.location.hostname}:8000`;

    // Balanced FPS Sampling (250ms interval ~= 4 FPS)
    const SAMPLE_INTERVAL = 250;

    // Check Backend Health
    useEffect(() => {
        const HEALTH_URL = `${API_BASE}/health`;
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


    const captureAndPredict = useCallback(async () => {
        if (!isScanning || !webcamRef.current) return;

        const now = Date.now();
        if (now - lastRequestTime.current < SAMPLE_INTERVAL || isProcessing.current) return;

        const imageSrc = webcamRef.current.getScreenshot();
        if (!imageSrc) return;

        lastRequestTime.current = now;
        isProcessing.current = true;

        try {
            // Send to Diagnostic Gateway with Mode=POSTURE
            // Use /predict (Manual Endpoint) because /predict/auto ignores mode!
            const response = await axios.post(`${API_BASE}/predict`, {
                image: imageSrc.split(',')[1],
                mode: 'POSTURE',
                debug: true // Enable debug to get the skeletal overlay
            });

            const taskId = response.data.task_id;
            pollResult(taskId);

        } catch (err) {
            console.error("Posture Prediction Failed", err);
            setError("Connection Lost");
            isProcessing.current = false;
        }
    }, [isScanning, API_BASE]);

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
                        setError(null);
                        isProcessing.current = false;
                    }
                } else if (res.data.state === 'PENDING' || res.data.state === 'STARTED') {
                    setTimeout(check, 100);
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

    const canvasRef = useRef(null);

    // MediaPipe Pose Connections (Simplified for key body parts)
    const POSE_CONNECTIONS = [
        [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], // Arms
        [11, 23], [12, 24], [23, 24], // Torso
        [23, 25], [24, 26], [25, 27], [26, 28], [27, 29], [28, 30], [29, 31], [30, 32] // Legs
    ];

    const drawSkeleton = useCallback((landmarks) => {
        const canvas = canvasRef.current;
        if (!canvas || !webcamRef.current) return;

        const video = webcamRef.current.video;
        if (video.readyState !== 4) return;

        const ctx = canvas.getContext('2d');
        const { videoWidth, videoHeight } = video;

        canvas.width = videoWidth;
        canvas.height = videoHeight;

        ctx.clearRect(0, 0, videoWidth, videoHeight);

        // Draw Connections
        ctx.strokeStyle = '#06b6d4'; // Cyan-500
        ctx.lineWidth = 3;

        POSE_CONNECTIONS.forEach(([startIdx, endIdx]) => {
            const start = landmarks[startIdx];
            const end = landmarks[endIdx];

            if (start && end && (start.visibility > 0.3 && end.visibility > 0.3)) {
                ctx.beginPath();
                ctx.moveTo(start.x * videoWidth, start.y * videoHeight);
                ctx.lineTo(end.x * videoWidth, end.y * videoHeight);
                ctx.stroke();
            }
        });

        // --- NECK / POSTURE ALIGNMENT ---
        // Calculate Neck Base (Midpoint of Shoulders 11 & 12)
        const leftShoulder = landmarks[11];
        const rightShoulder = landmarks[12];

        // Calculate Head Center (Midpoint of Ears 7 & 8, or Nose 0 if side profile)
        const leftEar = landmarks[7];
        const rightEar = landmarks[8];
        const nose = landmarks[0];

        if (leftShoulder && rightShoulder && leftShoulder.visibility > 0.3 && rightShoulder.visibility > 0.3) {
            const neckBaseX = (leftShoulder.x + rightShoulder.x) / 2;
            const neckBaseY = (leftShoulder.y + rightShoulder.y) / 2;

            // Determine Head Point
            let headX, headY, headVis = 0;

            if (leftEar && rightEar && leftEar.visibility > 0.3 && rightEar.visibility > 0.3) {
                // Front view: Midpoint of ears
                headX = (leftEar.x + rightEar.x) / 2;
                headY = (leftEar.y + rightEar.y) / 2;
                headVis = 1;
            } else if (leftEar && leftEar.visibility > 0.3) {
                // Side view (Left)
                headX = leftEar.x;
                headY = leftEar.y;
                headVis = 1;
            } else if (rightEar && rightEar.visibility > 0.3) {
                // Side view (Right)
                headX = rightEar.x;
                headY = rightEar.y;
                headVis = 1;
            } else if (nose && nose.visibility > 0.3) {
                // Fallback to nose
                headX = nose.x;
                headY = nose.y;
                headVis = 1;
            }

            if (headVis) {
                // Draw Neck Line (Yellow for visibility)
                ctx.strokeStyle = '#fbbf24'; // Amber-400
                ctx.lineWidth = 5;
                ctx.beginPath();
                ctx.moveTo(neckBaseX * videoWidth, neckBaseY * videoHeight);
                ctx.lineTo(headX * videoWidth, headY * videoHeight);
                ctx.stroke();

                // Draw Neck Base Point
                ctx.fillStyle = '#fbbf24';
                ctx.beginPath();
                ctx.arc(neckBaseX * videoWidth, neckBaseY * videoHeight, 6, 0, 2 * Math.PI);
                ctx.fill();

                // Draw Head Point
                ctx.fillStyle = '#fbbf24';
                ctx.beginPath();
                ctx.arc(headX * videoWidth, headY * videoHeight, 6, 0, 2 * Math.PI);
                ctx.fill();
            }
        }


        // Draw Landmarks
        ctx.fillStyle = '#ec4899'; // Pink-500
        landmarks.forEach((lm) => {
            if (lm.visibility > 0.3) {
                const x = lm.x * videoWidth;
                const y = lm.y * videoHeight;
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, 2 * Math.PI);
                ctx.fill();
            }
        });

    }, []);

    useEffect(() => {
        if (result?.landmarks) {
            requestAnimationFrame(() => drawSkeleton(result.landmarks));
        }
    }, [result, drawSkeleton]);

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

            {/* HUD Overlay */}
            <div className="absolute inset-0 z-10 pointer-events-none flex flex-col p-4 md:p-6 justify-between">

                {/* Top HUD */}
                <div className="flex justify-between items-start">
                    <div className="flex flex-col gap-1">
                        <span className="text-[8px] md:text-[10px] font-black text-cyan-500 uppercase tracking-widest animate-pulse">
                            KINETIC_ANALYSIS_V1
                        </span>
                        <h2 className="text-xl md:text-2xl font-black text-white uppercase tracking-tighter flex items-center gap-3">
                            POSTURE TRACKING
                            {isScanning && <div className="w-2 h-2 rounded-full bg-cyan-500 animate-ping"></div>}
                        </h2>
                    </div>

                    <div className="flex gap-2 pointer-events-auto">
                        <button
                            onClick={() => setFacingMode(prev => prev === "user" ? "environment" : "user")}
                            className="p-2 md:p-3 rounded-xl border transition-all bg-black/40 border-white/10 text-gray-400 hover:text-white hover:border-white/20"
                        >
                            <RefreshCw className="w-4 h-4 md:w-5 md:h-5" />
                        </button>
                    </div>
                </div>

                {/* Center Reticle (Standard) */}
                {!result?.debug_image && (
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full h-full flex items-center justify-center opacity-30">
                        <div className="border border-cyan-500/20 w-[80%] h-[80%] rounded-3xl"></div>
                        <div className="absolute w-full h-[1px] bg-cyan-500/20"></div>
                        <div className="absolute h-full w-[1px] bg-cyan-500/20"></div>
                    </div>
                )}

                {/* Bottom Results Panel */}
                <div className="bg-black/60 backdrop-blur-xl border border-white/10 p-4 md:p-6 rounded-3xl w-full max-w-2xl mx-auto mb-20 md:mb-10 transition-all pointer-events-auto">
                    {result ? (
                        <div className="flex items-center gap-4 md:gap-6">
                            <div className={`w-14 h-14 md:w-20 md:h-20 rounded-2xl flex items-center justify-center border-2 shadow-[0_0_20px_rgba(0,0,0,0.5)] ${result.label?.includes('Bad') ? 'border-red-500 bg-red-500/10 shadow-[0_0_20px_rgba(239,68,68,0.3)]' :
                                result.label?.includes('No') ? 'border-gray-500 bg-gray-500/10' :
                                    'border-cyan-500 bg-cyan-500/10 shadow-[0_0_20px_rgba(6,182,212,0.3)]'
                                }`}>
                                <User className={`w-6 h-6 md:w-8 md:h-8 ${result.label?.includes('Bad') ? 'text-red-500' :
                                    result.label?.includes('No') ? 'text-gray-500' :
                                        'text-cyan-500'}`} />
                            </div>

                            <div className="flex flex-col flex-1">
                                <span className="text-[10px] md:text-xs font-bold text-gray-500 uppercase tracking-widest mb-1">Status</span>
                                <h3 className="text-xl md:text-3xl font-black text-white leading-none tracking-tight">
                                    {result.label}
                                </h3>
                                <div className="flex items-center gap-2 mt-2">
                                    <div className="h-1.5 flex-1 bg-gray-800 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-500 ${result.label?.includes('Bad') ? 'bg-red-500' : 'bg-cyan-500'}`}
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
                            <Target className="w-5 h-5 animate-pulse" />
                            <span className="text-xs font-bold uppercase tracking-widest">Acquiring Target...</span>
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

            {/* Posture Overlay (Canvas) */}
            <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full object-cover z-20 opacity-80 pointer-events-none"
            />


        </div>
    )
}

export default PostureDetection
