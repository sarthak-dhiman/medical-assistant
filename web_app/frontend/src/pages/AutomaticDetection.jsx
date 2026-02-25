import { useState, useEffect, useRef, useCallback } from 'react'
import Webcam from 'react-webcam'
import { Activity, Bug, Scan, RefreshCw, AlertCircle, Loader2, Lock, Unlock, Cpu, Info, X } from 'lucide-react'
import axios from 'axios'
import ResultDisplay from '../ResultDisplay'
import LoadingOverlay from '../components/LoadingOverlay'
import { useHealth } from '../context/HealthContext'

function AutomaticDetection() {
    const { isNerdMode, setIsNerdMode, gpuMemory, setGpuMemory } = useHealth()
    const webcamRef = useRef(null)
    const [isScanning, setIsScanning] = useState(true)
    const [result, setResult] = useState(null)
    const [isProcessing, setIsProcessing] = useState(false)
    const [error, setError] = useState(null)
    const [detectedMode, setDetectedMode] = useState(null)
    const [selectedConditionInfo, setSelectedConditionInfo] = useState(null)
    const [isAppReady, setIsAppReady] = useState(false)
    const API_BASE = `http://${window.location.hostname}:8000`
    const lastRequestTime = useRef(0)
    const isProcessingRef = useRef(false)
    const [facingMode, setFacingMode] = useState('user')
    const [lockedImage, setLockedImage] = useState(null)

    // Health polling
    useEffect(() => {
        const checkHealth = async () => {
            try {
                const res = await fetch(`${API_BASE}/health`)
                const data = await res.json()
                if (data.models_ready) setIsAppReady(true)
                if (data.gpu_memory) setGpuMemory(data.gpu_memory)
            } catch (e) { console.error('Backend offline:', e) }
        }
        if (!isAppReady) {
            checkHealth()
            const id = setInterval(checkHealth, 3000)
            return () => clearInterval(id)
        }
    }, [isAppReady, setGpuMemory, API_BASE])


    // 2 FPS Sampling (500ms interval)
    const SAMPLE_INTERVAL = 500;

    const captureAndPredict = useCallback(async () => {
        if (!isAppReady || !isScanning || !webcamRef.current) return
        const now = Date.now()
        if (now - lastRequestTime.current < SAMPLE_INTERVAL || isProcessingRef.current) return
        const imageSrc = webcamRef.current.getScreenshot()
        if (!imageSrc) return
        lastRequestTime.current = now
        isProcessingRef.current = true
        setIsProcessing(true)

        try {
            // Send to Diagnostic Gateway
            // Using axios for simpler async handling
            const response = await axios.post(`${API_BASE}/predict/auto`, {
                image: imageSrc.split(',')[1],
                mode: 'AUTO', // Ignored by backend but good for schema
                debug: isNerdMode
            });

            const taskId = response.data.task_id
            pollResult(taskId)

        } catch (err) {
            console.error('Auto Prediction Failed', err)
            setError('Connection Lost')
            isProcessingRef.current = false
            setIsProcessing(false)
        }
    }, [isScanning, isNerdMode, API_BASE, isAppReady])

    const pollResult = async (taskId) => {
        let attempts = 0
        const check = async () => {
            if (attempts++ >= 20) { isProcessingRef.current = false; setIsProcessing(false); return }
            try {
                const res = await axios.get(`${API_BASE}/result/${taskId}`)
                if (res.data.state === 'SUCCESS') {
                    const data = res.data.result
                    if (data.status === 'error') {
                        setError(data.error || 'Detection Error')
                        setResult(null)
                    } else {
                        setResult(data)
                        setDetectedMode(data.mode)
                        setError(null)
                    }
                    isProcessingRef.current = false
                    setIsProcessing(false)
                } else if (['PENDING', 'STARTED'].includes(res.data.state)) {
                    setTimeout(check, 100)
                } else {
                    setError('Detection Failed')
                    isProcessingRef.current = false
                    setIsProcessing(false)
                }
            } catch (e) {
                console.error('Polling failed', e)
                isProcessingRef.current = false
                setIsProcessing(false)
            }
        }
        check()
    }

    useEffect(() => {
        const interval = setInterval(captureAndPredict, SAMPLE_INTERVAL);
        return () => clearInterval(interval);
    }, [captureAndPredict]);

    const ACCENT_BY_MODE = {
        JAUNDICE_BODY: '#fbbf24', JAUNDICE_EYE: '#60a5fa',
        SKIN_DISEASE: '#f472b6', NAIL_DISEASE: '#818cf8',
        ORAL_CANCER: '#34d399', TEETH: '#fb923c',
    }
    const accent = (detectedMode && ACCENT_BY_MODE[detectedMode]) ?? '#22d3ee'

    return (
        <div className="relative w-full h-full overflow-hidden" style={{ background: '#000' }}>
            {!isAppReady && <LoadingOverlay />}

            {/* Modal */}
            {selectedConditionInfo && (
                <div className="absolute inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)' }}>
                    <div className="w-full max-w-md rounded-2xl glass-over overflow-hidden shadow-2xl animate-fade-up border" style={{ borderColor: 'rgba(255,255,255,0.15)' }}>
                        <div className="flex items-center justify-between p-4 border-b" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>
                            <div className="flex items-center gap-2">
                                <div className="w-2.5 h-2.5 rounded-full" style={{ background: accent, boxShadow: `0 0 10px ${accent}` }} />
                                <h3 className="text-white font-black text-lg">{selectedConditionInfo.label}</h3>
                            </div>
                            <button onClick={() => setSelectedConditionInfo(null)} className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors">
                                <X className="w-5 h-5" />
                            </button>
                        </div>
                        <div className="p-5 flex flex-col gap-4">
                            {selectedConditionInfo.triage && (
                                <div className="p-3 rounded-xl border" style={{ background: 'rgba(0,0,0,0.3)', borderColor: 'rgba(255,255,255,0.05)' }}>
                                    <span className="text-[10px] font-black uppercase tracking-widest block mb-1" style={{ color: accent }}>Triage Level</span>
                                    <p className="text-sm text-gray-300 font-bold">{selectedConditionInfo.triage.level}</p>
                                    <p className="text-xs text-gray-400 mt-1">{selectedConditionInfo.triage.message}</p>
                                </div>
                            )}
                            <div>
                                <span className="text-[10px] font-black uppercase tracking-widest text-emerald-400 mb-1 block">Clinical Reference</span>
                                <p className="text-sm text-gray-300 leading-relaxed">
                                    {selectedConditionInfo.label === 'Normal' || selectedConditionInfo.label === 'Healthy'
                                        ? "No significant morphological irregularities detected by the vision model."
                                        : "This condition requires clinical evaluation. Do not rely solely on AI screening."}
                                </p>
                            </div>
                        </div>
                    </div>
                </div>
            )}

            {/* ── Live camera feed or Frozen Image ───────────────────────────── */}
            {lockedImage ? (
                <img
                    src={lockedImage}
                    alt="Locked Frame"
                    className="absolute inset-0 w-full h-full object-cover"
                />
            ) : (
                <Webcam
                    ref={webcamRef}
                    audio={false}
                    screenshotFormat="image/jpeg"
                    videoConstraints={{ width: 1280, height: 720, facingMode }}
                    className="absolute inset-0 w-full h-full object-cover"
                    onUserMediaError={(e) => { console.error('Webcam error:', e); setError('Camera access denied.') }}
                />
            )}

            {/* â”€â”€ Scan-line â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
            {isScanning && (
                <div className="absolute left-0 right-0 h-px z-10 pointer-events-none animate-scan-line"
                    style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)`, opacity: 0.5 }} />
            )}

            {/* â”€â”€ Result / visualizer overlays â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
            <div className="absolute inset-0 pointer-events-none z-10">
                <ResultDisplay result={result} mode={detectedMode} isNerdMode={isNerdMode} />
            </div>

            {/* â”€â”€ Top HUD row â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
            <div className="absolute top-0 left-0 right-0 z-20 flex items-start justify-between p-4 md:p-5 pointer-events-none">

                {/* Mode badge */}
                <div className="glass-over rounded-2xl px-4 py-2.5 flex items-center gap-3">
                    <div className="w-2.5 h-2.5 rounded-full flex-shrink-0"
                        style={{
                            background: isScanning ? accent : '#4b5568',
                            boxShadow: isScanning ? `0 0 10px ${accent}` : 'none',
                            animation: isScanning ? 'pulse 1.2s infinite' : 'none',
                        }} />
                    <div>
                        <p className="label" style={{ color: accent }}>Auto-Pilot</p>
                        <p className="text-sm font-black text-white tracking-tight leading-none mt-0.5">
                            {detectedMode ? detectedMode.replace(/_/g, ' ') : 'Scanning'}
                        </p>
                    </div>
                    {isProcessing && <Loader2 className="w-4 h-4 animate-spin ml-1 flex-shrink-0" style={{ color: accent }} />}
                </div>

                {/* Controls */}
                <div className="flex items-center gap-2 pointer-events-auto">
                    {gpuMemory && (
                        <div className="hidden md:flex items-center gap-1.5 glass-over px-2.5 py-1.5 rounded-xl">
                            <Cpu className="w-3 h-3" style={{ color: '#34d399' }} />
                            <span className="mono text-[10px] font-bold" style={{ color: '#34d399' }}>{gpuMemory} MB</span>
                        </div>
                    )}
                    <button onClick={() => setFacingMode(m => m === 'user' ? 'environment' : 'user')}
                        className="p-2.5 rounded-xl transition-all glass-over hover:bg-white/10">
                        <RefreshCw className="w-4 h-4 text-white/70" />
                    </button>
                    <button onClick={() => setIsNerdMode(n => !n)}
                        className="p-2.5 rounded-xl transition-all"
                        style={isNerdMode ? { background: 'rgba(167,139,250,0.2)', border: '1px solid rgba(167,139,250,0.4)' } : {}}>
                        <Bug className="w-4 h-4" style={{ color: isNerdMode ? '#a78bfa' : 'rgba(255,255,255,0.5)' }} />
                    </button>
                </div>
            </div>

            {/* â”€â”€ Center reticle â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
            {!result && isScanning && (
                <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
                    <div className="relative flex items-center justify-center">
                        <div className="absolute w-48 h-48 rounded-full border animate-spin-slow"
                            style={{ borderColor: `${accent}20`, borderTopColor: `${accent}60` }} />
                        <div className="absolute w-36 h-36 rounded-full border"
                            style={{ borderColor: `${accent}15` }} />
                        <div className="w-2 h-2 rounded-full" style={{ background: accent, boxShadow: `0 0 12px ${accent}` }} />
                    </div>
                </div>
            )}

            {/* â”€â”€ Bottom result panel â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
            <div className="absolute bottom-0 left-0 right-0 z-20 p-4 md:p-5 pb-6 pointer-events-none">

                {/* Scan toggle (desktop) */}
                <div className="hidden md:flex justify-center mb-4 pointer-events-auto">
                    <button
                        onClick={() => {
                            if (isScanning) {
                                setLockedImage(webcamRef.current?.getScreenshot())
                            } else {
                                setLockedImage(null)
                            }
                            setIsScanning(!isScanning)
                        }}
                        className="flex items-center gap-2 px-5 py-2.5 rounded-full text-xs font-bold uppercase tracking-wider transition-all"
                        style={!isScanning ? {
                            background: 'rgba(251,113,133,0.15)', border: '1px solid rgba(251,113,133,0.3)', color: '#fb7185',
                        } : {
                            background: 'rgba(0,0,0,0.45)', border: '1px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.8)',
                        }}
                    >
                        {isScanning ? <><Lock className="w-3.5 h-3.5 text-gray-400" /> Lock Results</> : <><Unlock className="w-3.5 h-3.5" /> Unlock Session</>}
                    </button>
                </div>

                {/* Result card */}
                {(result || error) && (
                    <div className="glass-over rounded-2xl p-4 md:max-w-lg md:mx-auto animate-fade-up"
                        style={{ border: `1px solid ${accent}25` }}>
                        {error && !result ? (
                            <div className="flex items-center gap-3 text-sm" style={{ color: '#fb7185' }}>
                                <AlertCircle className="w-4 h-4 flex-shrink-0" />
                                {error}
                            </div>
                        ) : result && (
                            <div className="flex items-center gap-4">
                                {/* Confidence ring proxy */}
                                <div className="w-14 h-14 rounded-2xl flex items-center justify-center flex-shrink-0 border-2"
                                    style={{
                                        background: `${accent}10`,
                                        borderColor: `${accent}40`,
                                        boxShadow: `0 0 20px ${accent}20`,
                                    }}>
                                    <Activity className="w-6 h-6" style={{ color: accent }} />
                                </div>
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center justify-between mb-1">
                                        <p className="label" style={{ color: accent }}>Diagnosis</p>
                                        {!isScanning && result.label !== 'Normal' && result.label !== 'Healthy' && (
                                            <button
                                                onClick={() => setSelectedConditionInfo(result)}
                                                className="pointer-events-auto flex items-center gap-1.5 px-2 py-1 rounded-md text-[10px] font-bold tracking-widest uppercase transition-colors"
                                                style={{ background: `${accent}20`, color: accent, border: `1px solid ${accent}40` }}
                                            >
                                                <Info className="w-3 h-3" /> Learn More
                                            </button>
                                        )}
                                    </div>
                                    <h3 className="text-lg font-black text-white tracking-tight leading-none truncate">
                                        {result.label}
                                    </h3>
                                    {result.triage && (
                                        <p className="text-[11px] mt-1 leading-snug" style={{ color: 'var(--text-2)' }}>
                                            {result.triage.message}
                                        </p>
                                    )}
                                    {result.confidence != null && (
                                        <div className="flex items-center gap-2 mt-2">
                                            <div className="flex-1 h-1 rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.1)' }}>
                                                <div className="h-full rounded-full transition-all duration-700"
                                                    style={{ width: `${(result.confidence * 100).toFixed(0)}%`, background: accent }} />
                                            </div>
                                            <span className="mono text-[10px] font-bold flex-shrink-0" style={{ color: accent }}>
                                                {(result.confidence * 100).toFixed(1)}%
                                            </span>
                                        </div>
                                    )}
                                </div>
                            </div>
                        )}
                    </div>
                )}

                {/* Mobile scan toggle */}
                <div className="flex md:hidden justify-center mt-3 pointer-events-auto">
                    <button
                        onClick={() => {
                            if (isScanning) {
                                setLockedImage(webcamRef.current?.getScreenshot())
                            } else {
                                setLockedImage(null)
                            }
                            setIsScanning(!isScanning)
                        }}
                        className="flex items-center gap-2 px-5 py-2.5 rounded-full text-xs font-bold uppercase tracking-wider transition-all"
                        style={!isScanning ? {
                            background: 'rgba(251,113,133,0.15)', border: '1px solid rgba(251,113,133,0.3)', color: '#fb7185',
                        } : {
                            background: 'rgba(0,0,0,0.45)', border: '1px solid rgba(255,255,255,0.1)', color: 'rgba(255,255,255,0.8)',
                        }}
                    >
                        {isScanning ? <><Lock className="w-3.5 h-3.5 text-gray-400" /> Lock</> : <><Unlock className="w-3.5 h-3.5" /> Unlock</>}
                    </button>
                </div>
            </div>
        </div>
    )
}

export default AutomaticDetection
