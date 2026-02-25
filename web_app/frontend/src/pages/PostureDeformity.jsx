import { useState, useEffect, useRef, useCallback } from 'react'
import Webcam from 'react-webcam'
import { Bone, ScanLine, AlertCircle, CheckCircle, AlertTriangle, XCircle, Activity, Lock, Unlock, Info, X, ChevronUp, ChevronDown } from 'lucide-react'
import axios from 'axios'
import LoadingOverlay from '../components/LoadingOverlay'

const API_BASE = `http://${window.location.hostname}:8000`

const POSE_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 7],
    [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10],
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
    [11, 23], [12, 24], [23, 24],
    [23, 25], [24, 26], [25, 27], [26, 28],
    [27, 29], [28, 30], [29, 31], [30, 32],
]

const SEVERITY_CONFIG = {
    Normal: { color: '#22d399', bg: 'rgba(34,211,153,0.12)', border: 'rgba(34,211,153,0.3)', icon: CheckCircle },
    Mild: { color: '#facc15', bg: 'rgba(250,204,21,0.12)', border: 'rgba(250,204,21,0.3)', icon: AlertCircle },
    Moderate: { color: '#fb923c', bg: 'rgba(251,146,60,0.12)', border: 'rgba(251,146,60,0.3)', icon: AlertTriangle },
    Severe: { color: '#f87171', bg: 'rgba(248,113,113,0.12)', border: 'rgba(248,113,113,0.3)', icon: XCircle },
    Unknown: { color: '#475569', bg: 'rgba(71,85,105,0.1)', border: 'rgba(71,85,105,0.2)', icon: ScanLine },
}

const SKELETON_NAMES = [
    'Forward Head Posture', 'Kyphosis (Hunched Back)', 'Lordosis (Swayback)',
    'Pelvic Tilt', 'Scoliosis Indicator', 'Knee Alignment',
]

const CONDITION_INFO = {
    'Forward Head Posture': {
        desc: "Occurs when your neck slants forward, placing the head in front of the shoulders instead of directly above them. This adds up to 10 lbs of extra weight on the cervical spine for every inch forward.",
        ideal: "Ear aligned vertically with the center of the shoulder.",
        risks: "Chronic neck pain, tension headaches, muscle stiffness, and long-term spinal degeneration."
    },
    'Kyphosis (Hunched Back)': {
        desc: "An exaggerated, forward rounding of the upper back (thoracic spine). Often caused by prolonged slouching over screens or poor desk ergonomics.",
        ideal: "A natural, slight outward curve (20-40 degrees) of the upper back.",
        risks: "Back and neck pain, stiffness, and in severe cases, breathing difficulties or compressed organs."
    },
    'Lordosis (Swayback)': {
        desc: "An excessive inward curve of the lower back (lumbar spine). It pushes the stomach forward and the buttocks out.",
        ideal: "A gentle inward curve of the lower back that balances the head and upper body.",
        risks: "Lower back pain, muscle spasms, and increased risk of slipped discs or sciatic nerve compression."
    },
    'Pelvic Tilt': {
        desc: "An asymmetrical drop of the pelvis on one side, meaning the left and right hip bones are not horizontally level.",
        ideal: "Both iliac crests (hip bones) are completely horizontal and level.",
        risks: "Leg length discrepancy compensation, uneven gait, lower back pain, and hip joint wear on one side."
    },
    'Scoliosis Indicator': {
        desc: "A lateral (sideways) deviation in the spine's alignment, detected here heavily through asymmetrical shoulder and hip tilts.",
        ideal: "Shoulders and hips are level, with the spine running straight down the center line of the back.",
        risks: "Uneven muscle strain, chronic back pain, and potential nerve compression. Severe cases require clinical bracing."
    },
    'Knee Alignment': {
        desc: "Assesses leg straightness. Genu Valgum ('knock-knees') occurs when knees touch but ankles do not. Genu Varum ('bow-legs') occurs when ankles touch but knees do not.",
        ideal: "When standing naturally, both the knees and the ankles should have a roughly equal, small gap between them.",
        risks: "Uneven cartilage wear, early onset osteoarthritis in the knees, and collateral ligament strain."
    }
}

// Single-row compact card — fills its flex-1 parent height
function ConditionCard({ condition, isLocked, onLearnMore }) {
    const sev = condition.severity || 'Unknown'
    const cfg = SEVERITY_CONFIG[sev] || SEVERITY_CONFIG.Unknown
    const Icon = cfg.icon
    return (
        <div className="w-full h-full rounded-xl px-3 flex items-center gap-2.5 transition-all duration-300"
            style={{ background: cfg.bg, border: `1px solid ${cfg.border}` }}>
            <Icon className="w-3.5 h-3.5 shrink-0" style={{ color: cfg.color }} />
            <div className="flex-1 min-w-0">
                <p className="text-white font-bold text-[11px] leading-tight truncate">{condition.condition}</p>
                <div className="flex items-center gap-2">
                    {condition.measured && (
                        <p className="text-[10px] font-mono truncate" style={{ color: cfg.color }}>{condition.measured}</p>
                    )}
                    {condition.ideal && (
                        <p className="text-[9px] font-mono truncate text-gray-500">
                            <span className="text-emerald-600">↔</span> {condition.ideal}
                        </p>
                    )}
                </div>
            </div>
            {isLocked && sev !== 'Normal' && sev !== 'Unknown' && (
                <button
                    onClick={() => onLearnMore(condition)}
                    className="p-1.5 rounded-lg transition-colors shrink-0 mr-1"
                    style={{ background: 'rgba(255,255,255,0.1)', color: 'white' }}
                    title="Learn More"
                >
                    <Info className="w-3.5 h-3.5" />
                </button>
            )}
            <span className="text-[9px] font-black uppercase tracking-wider shrink-0" style={{ color: cfg.color }}>{sev}</span>
        </div>
    )
}

function ConditionModal({ condition, onClose }) {
    if (!condition) return null
    const info = CONDITION_INFO[condition.condition] || {}
    const sev = condition.severity || 'Unknown'
    const cfg = SEVERITY_CONFIG[sev] || SEVERITY_CONFIG.Unknown

    return (
        <div className="absolute inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)' }}>
            <div className="w-full max-w-md rounded-2xl glass-over overflow-hidden shadow-2xl animate-fade-up border" style={{ borderColor: 'rgba(255,255,255,0.15)' }}>
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>
                    <div className="flex items-center gap-2">
                        <div className="w-2.5 h-2.5 rounded-full" style={{ background: cfg.color, boxShadow: `0 0 10px ${cfg.color}` }} />
                        <h3 className="text-white font-black text-lg">{condition.condition}</h3>
                    </div>
                    <button onClick={onClose} className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors">
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Body */}
                <div className="p-5 flex flex-col gap-4">
                    {/* Measurement vs Ideal */}
                    <div className="p-3 rounded-xl border" style={{ background: 'rgba(0,0,0,0.3)', borderColor: 'rgba(255,255,255,0.05)' }}>
                        <div className="flex items-center justify-between mb-2">
                            <span className="text-[10px] font-black uppercase tracking-widest text-gray-500">Your Measurement</span>
                            <span className="text-[10px] font-black uppercase tracking-widest" style={{ color: cfg.color }}>Severity: {sev}</span>
                        </div>
                        <p className="font-mono text-sm mb-3" style={{ color: cfg.color }}>{condition.measured || "Data unavailable"}</p>

                        <div className="mb-1"><span className="text-[10px] font-black uppercase tracking-widest text-emerald-500">Ideal Range</span></div>
                        <p className="text-xs text-gray-300">{info.ideal || "Within normal physiological limits."}</p>
                    </div>

                    {/* Description */}
                    <div>
                        <span className="text-[10px] font-black uppercase tracking-widest text-violet-400 mb-1 block">What is it?</span>
                        <p className="text-sm text-gray-300 leading-relaxed">{info.desc || "A structural deviation from normal spinal or leg alignment."}</p>
                    </div>

                    {/* Risks */}
                    <div>
                        <span className="text-[10px] font-black uppercase tracking-widest text-rose-400 mb-1 block">Potential Risks</span>
                        <p className="text-sm text-gray-300 leading-relaxed">{info.risks || "Varies depending on severity. Consult a professional."}</p>
                    </div>
                </div>

                {/* Footer */}
                <div className="bg-black/40 p-3 text-center border-t border-white/5">
                    <p className="text-[10px] text-gray-500 italic">This AI screening is not a clinical diagnosis. Always consult a physiotherapist or doctor.</p>
                </div>
            </div>
        </div>
    )
}

function SkeletonOverlay({ landmarks, highlightedIndices = [], width, height }) {
    if (!landmarks || landmarks.length === 0) return null
    const toXY = (lm) => ({ x: lm.x * width, y: lm.y * height })
    return (
        <svg style={{ position: 'absolute', inset: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
            viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="xMidYMid slice">
            {POSE_CONNECTIONS.map(([a, b], i) => {
                const la = landmarks[a], lb = landmarks[b]
                if (!la || !lb || la.visibility < 0.4 || lb.visibility < 0.4) return null
                const pa = toXY(la), pb = toXY(lb)
                const hi = highlightedIndices.includes(a) || highlightedIndices.includes(b)
                return <line key={i} x1={pa.x} y1={pa.y} x2={pb.x} y2={pb.y}
                    stroke={hi ? '#fb923c' : 'rgba(34,211,153,0.5)'} strokeWidth={hi ? 2.5 : 1.5} strokeLinecap="round" />
            })}
            {landmarks.map((lm, i) => {
                if (lm.visibility < 0.4) return null
                const { x, y } = toXY(lm)
                const hi = highlightedIndices.includes(i)
                return <circle key={i} cx={x} cy={y} r={hi ? 6 : 4}
                    fill={hi ? '#fb923c' : '#22d399'} stroke="rgba(0,0,0,0.6)" strokeWidth={1.5} />
            })}
        </svg>
    )
}

export default function PostureDeformity() {
    const webcamRef = useRef(null)
    const [isScanning, setIsScanning] = useState(true)
    const [result, setResult] = useState(null)
    const [error, setError] = useState(null)
    const [isAppReady, setIsAppReady] = useState(false)
    const [facingMode] = useState('user')
    const [isNerdMode, setIsNerdMode] = useState(false)
    const [isDrawerOpen, setIsDrawerOpen] = useState(false)
    const [selectedCondition, setSelectedCondition] = useState(null)
    const isProcessing = useRef(false)
    const lastRequestTime = useRef(0)
    const [lockedImage, setLockedImage] = useState(null)
    const [lockedLandmarks, setLockedLandmarks] = useState(null)
    const [videoSize, setVideoSize] = useState({ width: 1280, height: 720 })
    const videoSizeRef = useRef({ width: 1280, height: 720 })
    const SAMPLE_INTERVAL = 300

    // Health check
    useEffect(() => {
        const check = async () => {
            try { const res = await fetch(`${API_BASE}/health`); const d = await res.json(); if (d.models_ready) setIsAppReady(true) }
            catch { /* offline */ }
        }
        if (!isAppReady) { check(); const id = setInterval(check, 3000); return () => clearInterval(id) }
    }, [isAppReady])

    // Capture & predict
    const captureAndPredict = useCallback(async () => {
        if (!isAppReady || !isScanning || !webcamRef.current) return

        const video = webcamRef.current.video;
        if (video && video.videoWidth) {
            if (videoSizeRef.current.width !== video.videoWidth || videoSizeRef.current.height !== video.videoHeight) {
                videoSizeRef.current = { width: video.videoWidth, height: video.videoHeight };
                setVideoSize(videoSizeRef.current);
            }
        }

        const now = Date.now()
        if (now - lastRequestTime.current < SAMPLE_INTERVAL || isProcessing.current) return
        const imageSrc = webcamRef.current.getScreenshot()
        if (!imageSrc) return
        lastRequestTime.current = now
        isProcessing.current = true
        try {
            const resp = await axios.post(`${API_BASE}/predict`, { image: imageSrc, mode: 'POSTURE_DEFORMITY', debug: isNerdMode })
            pollResult(resp.data.task_id)
        } catch { setError('Connection lost'); isProcessing.current = false }
    }, [isScanning, isNerdMode, isAppReady])

    const pollResult = (taskId) => {
        let attempts = 0
        const check = async () => {
            if (attempts++ >= 20) { isProcessing.current = false; return }
            try {
                const res = await axios.get(`${API_BASE}/result/${taskId}`)
                if (res.data.state === 'SUCCESS') {
                    const data = res.data.result
                    if (data.status === 'error') { setError(data.error || 'Error'); setResult(null) }
                    else { setResult(data); setError(null) }
                    isProcessing.current = false
                } else if (['PENDING', 'STARTED'].includes(res.data.state)) setTimeout(check, 300)
                else { setError('Task failed'); isProcessing.current = false }
            } catch { setError('Poll error'); isProcessing.current = false }
        }
        setTimeout(check, 400)
    }

    useEffect(() => { const id = setInterval(captureAndPredict, SAMPLE_INTERVAL); return () => clearInterval(id) }, [captureAndPredict])

    const conditions = result?.conditions ?? []
    const landmarks = result?.debug_info?.landmarks ?? []
    const overall = result?.overall ?? null
    const worstSev = result?.debug_info?.analysis?.worst_severity ?? null
    const worstCond = result?.debug_info?.analysis?.worst_condition ?? ''
    const summaryLabel = result?.label ?? '–'
    const highlightedIndices = conditions.flatMap(c =>
        c.severity !== 'Normal' && c.severity !== 'Unknown' ? (c.landmarks || []) : []
    )

    // Cards to render — real data or skeleton placeholders
    const cards = conditions.length > 0
        ? conditions
        : SKELETON_NAMES.map(name => ({ condition: name, severity: 'Unknown' }))

    return (
        <div className="flex flex-col md:flex-row h-full overflow-hidden relative" style={{ background: 'var(--bg-1)' }}>
            {!isAppReady && <LoadingOverlay message="Warming up Pose Engine..." />}
            {selectedCondition && (
                <ConditionModal condition={selectedCondition} onClose={() => setSelectedCondition(null)} />
            )}

            {/* ── LEFT: Camera ─────────────────────────────────────────── */}
            <div className="relative flex-1 overflow-hidden flex items-center justify-center" style={{ background: '#000' }}>
                {lockedImage ? (
                    <img src={lockedImage} alt="Locked Frame" className="absolute inset-0 w-full h-full object-cover" />
                ) : (
                    <Webcam ref={webcamRef} audio={false} screenshotFormat="image/jpeg" screenshotQuality={0.82}
                        videoConstraints={{ width: 640, height: 480, facingMode }}
                        mirrored={facingMode === 'user'}
                        className="absolute inset-0 w-full h-full object-cover" />
                )}

                {((isScanning && landmarks.length > 0) || lockedLandmarks) && (
                    <div style={{ position: 'absolute', inset: 0 }}>
                        <SkeletonOverlay landmarks={lockedLandmarks || landmarks} highlightedIndices={highlightedIndices} width={videoSize.width} height={videoSize.height} />
                    </div>
                )}

                {/* Top HUD */}
                <div className="absolute top-4 left-4 right-4 flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl backdrop-blur-md"
                        style={{ background: 'rgba(0,0,0,0.5)', border: '1px solid rgba(255,255,255,0.1)' }}>
                        <Bone className="w-4 h-4 text-violet-400" />
                        <span className="text-[11px] font-black uppercase tracking-widest text-white">Spine &amp; Gait</span>
                    </div>
                    <div className="flex items-center gap-1">
                        <button onClick={() => setIsNerdMode(v => !v)}
                            className="px-2.5 py-1.5 rounded-xl text-[10px] font-bold uppercase tracking-wider transition-all"
                            style={isNerdMode
                                ? { background: 'rgba(167,139,250,0.2)', border: '1px solid rgba(167,139,250,0.4)', color: '#a78bfa' }
                                : { background: 'rgba(0,0,0,0.45)', border: '1px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.5)' }}>
                            Expert
                        </button>
                        <button onClick={() => {
                            if (isScanning) {
                                setLockedImage(webcamRef.current?.getScreenshot())
                                setLockedLandmarks(landmarks)
                            } else {
                                setLockedImage(null)
                                setLockedLandmarks(null)
                            }
                            setIsScanning(v => !v)
                        }}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl transition-all font-bold text-[10px] uppercase tracking-wider"
                            style={!isScanning
                                ? { background: 'rgba(251,113,133,0.2)', border: '1px solid rgba(251,113,133,0.4)', color: '#fb7185' }
                                : { background: 'rgba(0,0,0,0.45)', border: '1px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.8)' }}>
                            {isScanning ? (
                                <><Lock className="w-3.5 h-3.5" /> Lock Results</>
                            ) : (
                                <><Unlock className="w-3.5 h-3.5" /> Rescan</>
                            )}
                        </button>
                    </div>
                </div>

                {/* Guide overlay */}
                {!result && (
                    <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
                        <div className="rounded-2xl flex flex-col items-center gap-3 p-6"
                            style={{ border: '2px dashed rgba(167,139,250,0.35)' }}>
                            <Activity className="w-8 h-8 text-violet-400 animate-pulse" />
                            <p className="text-violet-300 text-sm font-bold text-center">
                                Stand facing the camera<br />
                                <span className="text-[11px] font-normal text-violet-500">Full body must be visible</span>
                            </p>
                        </div>
                    </div>
                )}

                {error && (
                    <div className="absolute bottom-4 left-4 right-4 text-center text-red-400 text-xs font-bold bg-red-900/30 border border-red-500/30 rounded-xl p-2 z-20">
                        ⚠ {error}
                    </div>
                )}
            </div>

            {/* ── RIGHT/BOTTOM: Results panel — app drawer on mobile, fixed on desktop ────── */}
            <div className="w-full md:w-80 flex flex-col shrink-0 overflow-y-auto md:overflow-hidden absolute md:relative bottom-0 left-0 right-0 z-30 bg-[#0f0f13] md:bg-[var(--bg-1)] rounded-t-[2rem] md:rounded-none max-h-[85vh] md:max-h-full transition-all duration-300 ease-[cubic-bezier(0.32,0.72,0,1)] shadow-[0_-10px_40px_rgba(0,0,0,0.5)] md:shadow-none"
                style={{ borderLeft: '1px solid var(--border-md)', borderTop: '1px solid var(--border-md)', padding: '12px 10px 8px' }}>

                {/* Mobile Drag Handle */}
                <div
                    className="w-full h-8 flex items-start justify-center md:hidden cursor-pointer"
                    onClick={() => setIsDrawerOpen(!isDrawerOpen)}
                >
                    <div className="w-12 h-1.5 bg-gray-600 rounded-full mt-1" />
                </div>

                {/* ── Overall assessment — compact fixed height ── */}
                <div className="rounded-xl px-3 py-2 shrink-0 mb-2 relative cursor-pointer md:cursor-auto group"
                    onClick={() => { if (window.innerWidth < 768) setIsDrawerOpen(!isDrawerOpen) }}
                    style={{
                        background: overall === 'Normal' ? 'rgba(34,211,153,0.07)' : overall === 'Abnormal' ? 'rgba(251,146,60,0.07)' : 'rgba(255,255,255,0.03)',
                        border: `1px solid ${overall === 'Normal' ? 'rgba(34,211,153,0.22)' : overall === 'Abnormal' ? 'rgba(251,146,60,0.22)' : 'rgba(255,255,255,0.07)'}`
                    }}>
                    <p className="text-[9px] font-black uppercase tracking-widest"
                        style={{ color: overall === 'Normal' ? '#22d399' : overall === 'Abnormal' ? '#fb923c' : '#475569' }}>
                        Overall
                    </p>
                    <p className="text-base font-black text-white leading-snug pr-8">{summaryLabel}</p>
                    {worstSev && worstSev !== 'Normal' && (
                        <p className="text-[9px] text-gray-500 truncate">Worst: {worstSev} — {worstCond}</p>
                    )}
                    {result && (
                        <p className="text-[9px] text-gray-600">
                            {result.abnormal_count === 0 ? 'All clear ✓' : `${result.abnormal_count} flagged`}
                        </p>
                    )}

                    {/* Arrow toggle for mobile */}
                    <div className="absolute right-4 top-1/2 -translate-y-1/2 md:hidden text-gray-400 group-hover:text-white transition-colors">
                        {isDrawerOpen ? <ChevronDown className="w-5 h-5" /> : <ChevronUp className="w-5 h-5" />}
                    </div>
                </div>

                {/* ── Drawer Content (Hidden on Mobile unless Open) ── */}
                <div className={`flex-col ${isDrawerOpen ? 'flex' : 'hidden'} md:flex flex-1 min-h-0 overflow-y-auto md:overflow-hidden`}>
                    {/* ── Conditions label ── */}
                    <p className="text-[9px] font-black uppercase tracking-widest text-gray-600 shrink-0 mt-2 mb-1 px-0.5">Conditions</p>

                    {/* ── 6 cards grow to fill ALL remaining space evenly (desktop) or stack (mobile) ── */}
                    <div className="flex flex-col gap-1 md:flex-1 md:min-h-0 min-h-[300px]">
                        {cards.map((cond, i) => (
                            <div key={i} className="flex-1 min-h-0">
                                <ConditionCard
                                    condition={cond}
                                    isLocked={!isScanning}
                                    onLearnMore={setSelectedCondition}
                                />
                            </div>
                        ))}
                    </div>

                    {/* ── Expert raw angles (only when nerd mode on + data exists) ── */}
                    {isNerdMode && conditions.length > 0 && (
                        <div className="rounded-xl px-3 py-2 mt-2 shrink-0"
                            style={{ background: 'rgba(0,0,0,0.35)', border: '1px solid rgba(167,139,250,0.15)' }}>
                            <p className="text-[9px] font-black uppercase tracking-widest text-violet-400 mb-1">Raw Angles</p>
                            {conditions.map((c, i) => (
                                <div key={i} className="flex justify-between text-[10px] font-mono py-0.5 border-b border-white/5 last:border-0">
                                    <span className="text-gray-500">{c.code}</span>
                                    <span className="text-violet-300">{c.angle != null ? `${c.angle} ${c.unit ?? ''}` : '–'}</span>
                                </div>
                            ))}
                        </div>
                    )}

                    {/* ── Disclaimer ── */}
                    <p className="text-[8px] text-gray-700 italic text-center mt-1.5 shrink-0 pb-4 md:pb-0">
                        Not a clinical diagnosis. Consult a physiotherapist.
                    </p>
                </div>
            </div>
        </div>
    )
}
