import { useState, useEffect, useRef, useCallback } from 'react'
import Webcam from 'react-webcam'
import { Activity, Scan, Target, User, RefreshCw, AlertTriangle } from 'lucide-react'
import axios from 'axios'
import LoadingOverlay from '../components/LoadingOverlay'

// ─── Ideal posture shape (normalised, centred at 0,0) ───────────────────────
// Drawn relative to the detected person's shoulder/hip midpoints.
// Tailored for SEATED POSE (Legs removed, focuses on torso/head/arms alignment)
const IDEAL = {
    head: { x: 0.00, y: -1.65 }, // Head squarely over shoulders
    lShoulder: { x: -0.30, y: -1.00 },
    rShoulder: { x: 0.30, y: -1.00 },
    lElbow: { x: -0.35, y: -0.40 }, // Arms resting comfortably
    rElbow: { x: 0.35, y: -0.40 },
    lHip: { x: -0.20, y: 0.00 },
    rHip: { x: 0.20, y: 0.00 },
    neck: { x: 0.00, y: -1.00 },  // midpoint shoulders
}

const IDEAL_CONNECTIONS = [
    ['head', 'neck'],
    ['neck', 'lShoulder'], ['neck', 'rShoulder'],
    ['lShoulder', 'lElbow'], ['rShoulder', 'rElbow'],
    ['lShoulder', 'lHip'], ['rShoulder', 'rHip'],
    ['lHip', 'rHip'],
]

// Full MediaPipe 33-landmark connections
const POSE_CONNECTIONS = [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24],
    [23, 25], [24, 26], [25, 27], [26, 28],
    [27, 29], [28, 30], [29, 31], [30, 32],
]

function PostureDetection() {
    const webcamRef = useRef(null)
    const canvasRef = useRef(null)
    const [isScanning, setIsScanning] = useState(true)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const lastRequestTime = useRef(0)
    const isProcessing = useRef(false)
    const [facingMode, setFacingMode] = useState("user")
    const [isAppReady, setIsAppReady] = useState(false)
    const API_BASE = `http://${window.location.hostname}:8000`
    const SAMPLE_INTERVAL = 250

    // ── Health check ────────────────────────────────────────────────────────
    useEffect(() => {
        const checkHealth = async () => {
            try {
                const res = await fetch(`${API_BASE}/health`)
                const data = await res.json()
                if (data.models_ready) setIsAppReady(true)
            } catch (e) { console.error("Backend offline:", e) }
        }
        if (!isAppReady) {
            checkHealth()
            const id = setInterval(checkHealth, 3000)
            return () => clearInterval(id)
        }
    }, [isAppReady])

    // ── Capture & predict ────────────────────────────────────────────────────
    const captureAndPredict = useCallback(async () => {
        if (!isAppReady || !isScanning || !webcamRef.current) return
        const now = Date.now()
        if (now - lastRequestTime.current < SAMPLE_INTERVAL || isProcessing.current) return
        const imageSrc = webcamRef.current.getScreenshot()
        if (!imageSrc) return
        lastRequestTime.current = now
        isProcessing.current = true
        try {
            const response = await axios.post(`${API_BASE}/predict`, {
                image: imageSrc.split(',')[1],
                mode: 'POSTURE',
                debug: true,
            })
            pollResult(response.data.task_id)
        } catch (err) {
            console.error("Prediction failed", err)
            setError("Connection Lost")
            isProcessing.current = false
        }
    }, [isScanning, API_BASE])

    const pollResult = async (taskId) => {
        let attempts = 0
        const check = async () => {
            if (attempts++ >= 20) { isProcessing.current = false; return }
            try {
                const res = await axios.get(`${API_BASE}/result/${taskId}`)
                if (res.data.state === 'SUCCESS') {
                    const data = res.data.result
                    if (data.status === 'error') { setError(data.error || "Error"); setResult(null) }
                    else { setResult(data); setError(null) }
                    isProcessing.current = false
                } else if (['PENDING', 'STARTED'].includes(res.data.state)) {
                    setTimeout(check, 100)
                } else {
                    setError("Detection Failed")
                    isProcessing.current = false
                }
            } catch { isProcessing.current = false }
        }
        check()
    }

    // ── Draw skeleton + corrective overlay ──────────────────────────────────
    const drawCanvas = useCallback((landmarks, isBad) => {
        const canvas = canvasRef.current
        if (!canvas || !webcamRef.current) return
        const video = webcamRef.current.video
        if (!video || video.readyState !== 4) return

        const W = video.videoWidth
        const H = video.videoHeight
        canvas.width = W
        canvas.height = H
        const ctx = canvas.getContext('2d')
        ctx.clearRect(0, 0, W, H)

        // ── 1. Detected skeleton ──────────────────────────────────────────
        const color = isBad ? '#f87171' : '#06b6d4'  // red-400 or cyan-500

        ctx.strokeStyle = color
        ctx.lineWidth = 3
        POSE_CONNECTIONS.forEach(([a, b]) => {
            const A = landmarks[a], B = landmarks[b]
            if (A && B && A.visibility > 0.3 && B.visibility > 0.3) {
                ctx.beginPath()
                ctx.moveTo(A.x * W, A.y * H)
                ctx.lineTo(B.x * W, B.y * H)
                ctx.stroke()
            }
        })

        // Neck line (shoulder midpoint → head)
        const ls = landmarks[11], rs = landmarks[12]
        const nose = landmarks[0]
        if (ls && rs && ls.visibility > 0.3 && rs.visibility > 0.3 && nose) {
            const nx = (ls.x + rs.x) / 2 * W
            const ny = (ls.y + rs.y) / 2 * H
            ctx.strokeStyle = '#fbbf24'; ctx.lineWidth = 5
            ctx.beginPath(); ctx.moveTo(nx, ny)
            ctx.lineTo(nose.x * W, nose.y * H); ctx.stroke()
            ctx.fillStyle = '#fbbf24'
            ctx.beginPath(); ctx.arc(nx, ny, 6, 0, 2 * Math.PI); ctx.fill()
            ctx.beginPath(); ctx.arc(nose.x * W, nose.y * H, 6, 0, 2 * Math.PI); ctx.fill()
        }

        // Joints
        ctx.fillStyle = isBad ? '#fca5a5' : '#ec4899'
        landmarks.forEach(lm => {
            if (lm.visibility > 0.3) {
                ctx.beginPath()
                ctx.arc(lm.x * W, lm.y * H, 5, 0, 2 * Math.PI)
                ctx.fill()
            }
        })

        // ── 2. Corrective "ideal posture" ghost overlay ───────────────────
        if (!isBad) return

        const lHip = landmarks[23], rHip = landmarks[24]
        const lSh = landmarks[11], rSh = landmarks[12]
        if (!lHip || !rHip || !lSh || !rSh) return
        if (lHip.visibility < 0.2 || rHip.visibility < 0.2) return

        // Anchor = midpoint of hips (world coords)
        const anchorX = (lHip.x + rHip.x) / 2 * W
        const anchorY = (lHip.y + rHip.y) / 2 * H

        // Scale = torso height (shoulder mid → hip mid) in pixels
        const shMidY = (lSh.y + rSh.y) / 2 * H
        const torsoH = Math.abs(anchorY - shMidY)
        if (torsoH < 20) return   // person too small / partially visible

        // Project ideal points
        const proj = key => ({
            x: anchorX + IDEAL[key].x * torsoH,
            y: anchorY + IDEAL[key].y * torsoH,
        })

        // Ghost lines
        ctx.save()
        ctx.globalAlpha = 0.55
        ctx.strokeStyle = '#4ade80'   // green-400
        ctx.lineWidth = 4
        ctx.setLineDash([10, 6])
        ctx.shadowColor = '#4ade80'
        ctx.shadowBlur = 8

        IDEAL_CONNECTIONS.forEach(([a, b]) => {
            const A = proj(a), B = proj(b)
            ctx.beginPath(); ctx.moveTo(A.x, A.y); ctx.lineTo(B.x, B.y); ctx.stroke()
        })
        ctx.setLineDash([])

        // Ghost joints
        ctx.fillStyle = '#4ade80'
        Object.keys(IDEAL).forEach(key => {
            const p = proj(key)
            ctx.beginPath(); ctx.arc(p.x, p.y, 5, 0, 2 * Math.PI); ctx.fill()
        })

        // Head circle
        const headP = proj('head')
        ctx.strokeStyle = '#4ade80'; ctx.lineWidth = 3
        ctx.beginPath(); ctx.arc(headP.x, headP.y, torsoH * 0.15, 0, 2 * Math.PI); ctx.stroke()

        // Label
        ctx.globalAlpha = 0.85
        ctx.setLineDash([])
        ctx.shadowBlur = 0
        ctx.font = `bold ${Math.max(12, torsoH * 0.13)}px sans-serif`
        ctx.fillStyle = '#4ade80'
        ctx.textAlign = 'center'
        ctx.fillText('↑ Ideal Posture', anchorX, anchorY + torsoH * 1.45)

        ctx.restore()
    }, [])

    useEffect(() => {
        if (result?.landmarks) {
            const isBad = result.label?.toLowerCase().includes('bad')
            requestAnimationFrame(() => drawCanvas(result.landmarks, isBad))
        }
    }, [result, drawCanvas])

    useEffect(() => {
        const id = setInterval(captureAndPredict, SAMPLE_INTERVAL)
        return () => clearInterval(id)
    }, [captureAndPredict])

    const videoConstraints = { width: 640, height: 480, facingMode }
    const isBad = result?.label?.toLowerCase().includes('bad')

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
                            KINETIC_ANALYSIS_V2
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

                {/* Bad posture hint badge */}
                {isBad && (
                    <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-[140%] flex items-center gap-2
                                    bg-red-500/20 border border-red-500/40 backdrop-blur-sm px-4 py-2 rounded-2xl
                                    animate-pulse pointer-events-none">
                        <AlertTriangle className="w-4 h-4 text-red-400 shrink-0" />
                        <span className="text-xs font-bold text-red-300 uppercase tracking-widest">
                            Align with the green guide
                        </span>
                    </div>
                )}

                {/* Center Reticle (only when no landmarks yet) */}
                {!result?.landmarks && (
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
                            <div className={`w-14 h-14 md:w-20 md:h-20 rounded-2xl flex items-center justify-center border-2 shadow-[0_0_20px_rgba(0,0,0,0.5)] ${isBad
                                ? 'border-red-500 bg-red-500/10 shadow-[0_0_20px_rgba(239,68,68,0.3)]'
                                : result.label?.includes('No')
                                    ? 'border-gray-500 bg-gray-500/10'
                                    : 'border-cyan-500 bg-cyan-500/10 shadow-[0_0_20px_rgba(6,182,212,0.3)]'
                                }`}>
                                <User className={`w-6 h-6 md:w-8 md:h-8 ${isBad ? 'text-red-500'
                                    : result.label?.includes('No') ? 'text-gray-500'
                                        : 'text-cyan-500'}`}
                                />
                            </div>

                            <div className="flex flex-col flex-1">
                                <span className="text-[10px] md:text-xs font-bold text-gray-500 uppercase tracking-widest mb-1">Status</span>
                                <h3 className="text-xl md:text-3xl font-black text-white leading-none tracking-tight">
                                    {result.label}
                                </h3>
                                <div className="flex items-center gap-2 mt-2">
                                    <div className="h-1.5 flex-1 bg-gray-800 rounded-full overflow-hidden">
                                        <div
                                            className={`h-full rounded-full transition-all duration-500 ${isBad ? 'bg-red-500' : 'bg-cyan-500'}`}
                                            style={{ width: `${(result.confidence || 0) * 100}%` }}
                                        ></div>
                                    </div>
                                    <span className="text-[10px] font-bold text-gray-400">
                                        {((result.confidence || 0) * 100).toFixed(1)}% CONFIDENCE
                                    </span>
                                </div>
                                {/* Corrective tip when bad */}
                                {isBad && (
                                    <p className="text-[10px] text-green-400/80 mt-2 font-medium tracking-wide">
                                        🟢 Follow the green ghost skeleton — sit upright, head over shoulders
                                    </p>
                                )}
                            </div>
                        </div>
                    ) : error ? (
                        <div className="flex flex-col items-center justify-center gap-2 text-red-400 h-16 md:h-20 animate-in fade-in slide-in-from-bottom-2">
                            <div className="flex items-center gap-2">
                                <AlertTriangle className="w-5 h-5" />
                                <span className="text-xs font-bold uppercase tracking-widest">Detection Error</span>
                            </div>
                            <p className="text-[10px] text-gray-500 max-w-[200px] text-center line-clamp-1">{error}</p>
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

            {/* Skeleton + Overlay Canvas */}
            <canvas
                ref={canvasRef}
                className="absolute inset-0 w-full h-full object-cover z-[5] opacity-90 pointer-events-none"
            />
        </div>
    )
}

export default PostureDetection
