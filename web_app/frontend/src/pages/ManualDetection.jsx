import { useState, useEffect } from 'react'
import WebcamCapture from '../WebcamCapture'
import ModelSelector, { getModeAccent, getModeLabel } from '../ModelSelector'
import { Upload, Activity, X, Bug, HelpCircle } from 'lucide-react'
import { useHealth } from '../context/HealthContext'

const MODE_TIPS = {
    JAUNDICE_BODY: 'Target the chest or abdominal skin area. Ensure the skin is clean and well-lit.',
    JAUNDICE_EYE: 'Focus on the sclera (white of the eye). Remove glasses. Good lighting is critical.',
    SKIN_DISEASE: 'Center the skin lesion in the frame. Eliminate shadows and reflections.',
    NAIL_DISEASE: 'Show the full nail clearly. Clean nails preferred. Include surrounding skin.',
    ORAL_CANCER: 'Open mouth wide for a full view. Focus on any abnormal tissue areas.',
    TEETH: 'Show both teeth and gums together. Ensure sharp focus on dental structures.',
}

function ManualDetection() {
    // ── Shared state from context ─────────────────────────────────────────────
    const { activeMode, setActiveMode, isNerdMode, setIsNerdMode, gpuMemory } = useHealth()

    // ── Local state ───────────────────────────────────────────────────────────
    const [uploadedImage, setUploadedImage] = useState(null)
    const [showHelp, setShowHelp] = useState(false)
    const [showMobileMenu, setShowMobileMenu] = useState(false)
    const [selectedConditionInfo, setSelectedConditionInfo] = useState(null)

    const API_BASE = `http://${window.location.hostname}:8000`

    // AppGate in App.jsx already blocks rendering until models are warm
    const isAppReady = true

    // Lock body scroll
    useEffect(() => {
        document.body.style.overflow = 'hidden'
        return () => { document.body.style.overflow = 'auto' }
    }, [])

    const handleUpload = (e) => {
        const file = e.target.files[0]
        if (file) {
            const reader = new FileReader()
            reader.onloadend = () => setUploadedImage(reader.result)
            reader.readAsDataURL(file)
        }
    }

    const accent = getModeAccent(activeMode)
    const modeLabel = getModeLabel(activeMode)
    const tipText = MODE_TIPS[activeMode] ?? ''

    return (
        <div className="relative w-full h-full flex overflow-hidden" style={{ background: 'var(--bg-base)' }}>

            {selectedConditionInfo && (
                <div className="absolute inset-0 z-50 flex items-center justify-center p-4" style={{ background: 'rgba(0,0,0,0.7)', backdropFilter: 'blur(8px)' }}>
                    <div className="w-full max-w-md rounded-2xl glass-over overflow-hidden shadow-2xl animate-fade-up border" style={{ borderColor: 'rgba(255,255,255,0.15)' }}>
                        <div className="flex items-center justify-between p-4 border-b" style={{ borderColor: 'rgba(255,255,255,0.1)' }}>
                            <h3 className="text-white font-black text-lg">{selectedConditionInfo.label}</h3>
                            <button onClick={() => setSelectedConditionInfo(null)} className="p-1 rounded-lg text-gray-400 hover:text-white hover:bg-white/10 transition-colors">
                                <X className="w-5 h-5" />
                            </button>
                        </div>
                        <div className="p-5">
                            <p className="text-sm text-gray-300">This AI screening is not a clinical diagnosis. Always consult a professional.</p>
                        </div>
                    </div>
                </div>
            )}

            {/* ── DESKTOP: Camera left / Controls right ───────────────────────────── */}
            <div className="flex-1 min-w-0 relative">
                {/* Mode accent strip at top of camera */}
                <div className="absolute top-0 left-0 right-0 h-0.5 z-30 transition-all duration-500"
                    style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)` }} />

                <WebcamCapture
                    mode={activeMode}
                    uploadedImage={uploadedImage}
                    isNerdMode={isNerdMode}
                    setIsNerdMode={setIsNerdMode}
                    setShowHelp={setShowHelp}
                    isAppReady={isAppReady}
                    setSelectedConditionInfo={setSelectedConditionInfo}
                />
            </div>

            {/* ── Right controls panel (hidden on mobile, use FAB instead) ──── */}
            <aside className="hidden lg:flex flex-col w-72 xl:w-80 flex-shrink-0 overflow-y-auto no-scrollbar"
                style={{ background: 'rgba(8,15,34,0.85)', borderLeft: '1px solid var(--border)', backdropFilter: 'blur(20px)' }}>

                {/* Mode indicator */}
                <div className="px-4 pt-4 pb-3" style={{ borderBottom: '1px solid var(--border)' }}>
                    <div className="flex items-center gap-3">
                        <div className="w-2 h-8 rounded-full flex-shrink-0" style={{ background: accent }} />
                        <div>
                            <p className="label mb-0.5">Active Mode</p>
                            <h3 className="text-sm font-black text-white tracking-tight">{modeLabel}</h3>
                        </div>
                        <div className="ml-auto w-2 h-2 rounded-full animate-pulse"
                            style={{ background: '#34d399', boxShadow: '0 0 6px #34d399' }} />
                    </div>
                </div>

                {/* Model selector */}
                <div className="px-3 py-3" style={{ borderBottom: '1px solid var(--border)' }}>
                    <p className="label mb-2 px-1">Detection Protocol</p>
                    <ModelSelector activeMode={activeMode} setActiveMode={setActiveMode} isMobile={false} />
                </div>

                {/* Input section */}
                <div className="px-3 py-3" style={{ borderBottom: '1px solid var(--border)' }}>
                    <p className="label mb-2 px-1">Input Source</p>

                    <label className="flex flex-col items-center justify-center gap-2 p-4 rounded-xl cursor-pointer transition-all duration-200 group"
                        style={{
                            border: `2px dashed ${uploadedImage ? accent : 'rgba(255,255,255,0.1)'}`,
                            background: uploadedImage ? `${accent}08` : 'transparent',
                        }}>
                        <Upload className="w-5 h-5 transition-colors" style={{ color: uploadedImage ? accent : 'var(--text-3)' }} />
                        <span className="text-[10px] font-bold uppercase tracking-widest"
                            style={{ color: uploadedImage ? accent : 'var(--text-3)' }}>
                            {uploadedImage ? 'Image Loaded' : 'Upload Static Frame'}
                        </span>
                        <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
                    </label>

                    {uploadedImage && (
                        <button onClick={() => setUploadedImage(null)}
                            className="mt-2 w-full flex items-center justify-center gap-2 py-2 rounded-xl text-[10px] font-bold uppercase tracking-wider transition-all"
                            style={{ background: 'rgba(251,113,133,0.1)', border: '1px solid rgba(251,113,133,0.25)', color: '#fb7185' }}>
                            <X className="w-3.5 h-3.5" /> Clear Upload
                        </button>
                    )}
                </div>

                {/* AI Tip */}
                {tipText && (
                    <div className="mx-3 my-3 p-3 rounded-xl" style={{ background: `${accent}07`, border: `1px solid ${accent}20` }}>
                        <p className="label mb-1" style={{ color: accent }}>AI Guidance</p>
                        <p className="text-[11px] leading-relaxed" style={{ color: 'var(--text-2)' }}>{tipText}</p>
                    </div>
                )}

                {/* Status footer */}
                <div className="mt-auto px-4 py-3" style={{ borderTop: '1px solid var(--border)' }}>
                    <div className="flex items-center gap-2">
                        <div className="w-1.5 h-1.5 rounded-full" style={{ background: '#34d399' }} />
                        <span className="mono text-[10px]" style={{ color: 'var(--text-3)' }}>Models ready</span>
                        {gpuMemory && (
                            <span className="ml-auto mono text-[10px]" style={{ color: 'var(--text-3)' }}>
                                GPU {gpuMemory} MB
                            </span>
                        )}
                    </div>
                </div>
            </aside>

            {/* ── MOBILE: Bottom FAB + drop-up drawer ──────────────────────────── */}
            <div className="lg:hidden fixed bottom-5 left-4 right-4 z-40 flex flex-col pointer-events-none">

                {/* Backdrop */}
                <div
                    className={`fixed inset-0 transition-opacity duration-300 pointer-events-auto ${showMobileMenu ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
                    style={{ background: 'rgba(0,0,0,0.65)', backdropFilter: 'blur(4px)' }}
                    onClick={() => setShowMobileMenu(false)}
                />

                {/* Drop-up drawer */}
                <div className={`absolute bottom-full left-0 right-0 mb-3 rounded-2xl p-4 shadow-2xl transition-all duration-300 pointer-events-auto origin-bottom glass ${showMobileMenu ? 'scale-100 opacity-100 translate-y-0' : 'scale-95 opacity-0 translate-y-8 pointer-events-none'}`}>
                    <div className="flex items-center justify-between mb-3">
                        <span className="text-xs font-black uppercase tracking-widest" style={{ color: 'var(--text-2)' }}>Detection Mode</span>
                        <button onClick={() => setShowMobileMenu(false)} className="p-1.5 rounded-lg hover:bg-white/10" style={{ color: 'var(--text-3)' }}>
                            <X className="w-4 h-4" />
                        </button>
                    </div>
                    <ModelSelector activeMode={activeMode} setActiveMode={(m) => { setActiveMode(m); setShowMobileMenu(false) }} isMobile={true} />
                </div>

                {/* FAB action bar */}
                <div className="flex items-center gap-2 p-2 rounded-2xl pointer-events-auto glass shadow-2xl">
                    <button onClick={() => setShowMobileMenu(!showMobileMenu)}
                        className="flex-1 flex items-center gap-3 px-3 py-2.5 rounded-xl transition-all"
                        style={{ background: 'rgba(255,255,255,0.05)' }}>
                        <div className="w-2 h-2 rounded-full" style={{ background: accent }} />
                        <div className="text-left leading-tight">
                            <p className="text-[9px] font-black uppercase tracking-widest" style={{ color: accent }}>Mode</p>
                            <p className="text-xs font-bold" style={{ color: 'var(--text-1)' }}>{modeLabel}</p>
                        </div>
                        <Activity className="w-4 h-4 ml-auto" style={{ color: 'var(--text-3)' }} />
                    </button>

                    <button onClick={() => setIsNerdMode(!isNerdMode)}
                        className="w-11 h-11 rounded-xl flex items-center justify-center transition-all flex-shrink-0"
                        style={isNerdMode ? { background: 'rgba(167,139,250,0.2)', border: '1px solid rgba(167,139,250,0.4)', color: 'var(--violet)' } : { background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.1)', color: 'var(--text-3)' }}>
                        <Bug className="w-4 h-4" />
                    </button>

                    <label className="w-11 h-11 rounded-xl flex items-center justify-center flex-shrink-0 cursor-pointer"
                        style={{ background: `${accent}20`, border: `1px solid ${accent}40`, color: accent }}>
                        <Upload className="w-4 h-4" />
                        <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
                    </label>
                </div>
            </div>

            {/* ── Help modal ─────────────────────────────────────────────────────── */}
            {showHelp && (
                <div className="fixed inset-0 z-[60] flex items-center justify-center p-6"
                    style={{ background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(12px)' }}>
                    <div className="glass rounded-3xl p-7 w-full max-w-sm shadow-2xl animate-fade-up" style={{ border: `1px solid ${accent}30` }}>
                        <div className="flex items-center justify-between mb-5">
                            <div className="flex items-center gap-3">
                                <div className="w-9 h-9 rounded-xl flex items-center justify-center"
                                    style={{ background: `${accent}18`, border: `1px solid ${accent}30` }}>
                                    <HelpCircle className="w-5 h-5" style={{ color: accent }} />
                                </div>
                                <h3 className="text-base font-black text-white tracking-tight">Protocol</h3>
                            </div>
                            <button onClick={() => setShowHelp(false)} className="p-1.5 rounded-lg hover:bg-white/10 transition-all">
                                <X className="w-4 h-4" style={{ color: 'var(--text-3)' }} />
                            </button>
                        </div>

                        <div className="p-4 rounded-xl mb-5" style={{ background: `${accent}08`, border: `1px solid ${accent}20` }}>
                            <p className="label mb-2" style={{ color: accent }}>Operational Focus</p>
                            <p className="text-xs leading-relaxed font-medium" style={{ color: 'var(--text-2)' }}>{tipText}</p>
                        </div>

                        <button onClick={() => setShowHelp(false)}
                            className="w-full py-3 rounded-xl text-xs font-black uppercase tracking-widest text-black transition-all active:scale-95"
                            style={{ background: accent }}>
                            Acknowledged
                        </button>
                    </div>
                </div>
            )}
        </div>
    )
}

export default ManualDetection
