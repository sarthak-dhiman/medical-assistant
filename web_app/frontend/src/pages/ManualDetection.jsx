import { useState, useEffect } from 'react'
import WebcamCapture from '../WebcamCapture'
import ModelSelector from '../ModelSelector'
import { Upload, X, Activity, Bug, HelpCircle } from 'lucide-react'

function ManualDetection() {
    const [activeMode, setActiveMode] = useState('JAUNDICE_BODY')
    const [uploadedImage, setUploadedImage] = useState(null)
    const [isNerdMode, setIsNerdMode] = useState(false)
    const [showHelp, setShowHelp] = useState(false);
    const [showMobileMenu, setShowMobileMenu] = useState(false);
    const [isAppReady, setIsAppReady] = useState(false);
    const API_BASE = `http://${window.location.hostname}:8000`;

    // Lock Body Scroll (Prevent Pull-to-Refresh & Scroll)
    useEffect(() => {
        document.body.style.overflow = 'hidden';
        return () => { document.body.style.overflow = 'auto'; };
    }, []);

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

    const handleUpload = (e) => {
        const file = e.target.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onloadend = () => setUploadedImage(reader.result);
            reader.readAsDataURL(file);
        }
    }


    return (
        <div className="relative w-full h-full flex flex-col bg-gray-950 text-white font-sans overflow-hidden">

            {/* Main Content Area */}
            <main className="flex-1 flex flex-col lg:flex-row lg:max-w-[1500px] lg:mx-auto w-full p-0 lg:px-4 lg:py-6 gap-0 lg:gap-6 overflow-hidden relative">

                {/* DESKTOP SIDEBAR */}
                <aside className="hidden lg:flex flex-col w-96 h-full justify-between pr-2 overflow-hidden">

                    {/* Top Section: Models */}
                    <div className="space-y-2 shrink-0">
                        <div className="flex items-center justify-between px-1">
                            <h3 className="text-gray-500 text-[10px] font-black uppercase tracking-[0.25em]">Specialized Units</h3>
                            <span className="text-[9px] font-bold text-cyan-500/50 uppercase tracking-widest">v2.4.1</span>
                        </div>
                        <ModelSelector activeMode={activeMode} setActiveMode={setActiveMode} isMobile={false} />
                    </div>

                    {/* Middle Section: Input (Compacted) */}
                    <div className="bg-white/[0.02] rounded-[1.5rem] p-3 border border-white/5 space-y-2 backdrop-blur-3xl shadow-xl shrink-0 my-2">
                        <h3 className="text-gray-500 text-[10px] font-black uppercase tracking-[0.25em]">Input Protocol</h3>
                        <div className="flex flex-col gap-2">
                            <div className="flex gap-2">
                                <label className="flex-1 h-14 flex flex-col items-center justify-center gap-1 rounded-2xl bg-white/[0.03] hover:bg-cyan-500/10 cursor-pointer transition-all border border-white/5 border-dashed hover:border-cyan-500/50 group">
                                    <Upload className="w-5 h-5 text-gray-500 group-hover:text-cyan-400 transition-all duration-300 group-hover:scale-110" />
                                    <span className="text-[7px] font-black text-gray-500 group-hover:text-cyan-400 uppercase tracking-widest transition-colors">Inject Static Frame</span>
                                    <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
                                </label>

                                <button
                                    onClick={() => setIsNerdMode(!isNerdMode)}
                                    className={`w-14 h-14 flex flex-col items-center justify-center gap-1 rounded-2xl transition-all border ${isNerdMode
                                        ? 'bg-purple-600 border-purple-500 text-white shadow-lg shadow-purple-900/40'
                                        : 'bg-white/[0.03] hover:bg-purple-500/10 border-white/5 border-dashed hover:border-purple-500/50 text-gray-500 hover:text-purple-400 group'
                                        }`}
                                    title="Toggle Nerd Mode (Dev Stats)"
                                >
                                    <Bug className={`w-5 h-5 transition-transform ${!isNerdMode && 'group-hover:scale-110'}`} />
                                    <span className="text-[7px] font-black uppercase tracking-widest">Nerd Mode</span>
                                </button>
                            </div>

                            {uploadedImage && (
                                <button
                                    onClick={() => setUploadedImage(null)}
                                    className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl bg-red-500/10 text-red-500 hover:bg-red-500/20 border border-red-500/20 transition-all text-[9px] font-black uppercase tracking-[0.15em]"
                                >
                                    <X className="w-4 h-4" /> Terminate Stream
                                </button>
                            )}
                        </div>
                    </div>

                    {/* Bottom Section: Hint */}
                    <div className="bg-blue-500/5 rounded-2xl p-3 border border-blue-500/10 shrink-0">
                        <p className="text-[8px] text-blue-300/60 leading-relaxed font-bold uppercase tracking-wider">
                            💡 ENSURE OPTIMAL LIGHTING FOR AI ACCURACY. KEEP CAMERA STEADY FOR {activeMode === 'JAUNDICE_BODY' ? 'SCLERA' : 'FOCUS'}.
                        </p>
                    </div>
                </aside>

                {/* CAMERA FEED (Full Screen on Mobile) */}
                <div className="flex-1 relative flex flex-col justify-center bg-black lg:rounded-[2.5rem] lg:border-[4px] lg:border-white/[0.03] overflow-hidden w-full h-full shadow-[0_0_50px_rgba(0,0,0,0.5)]">
                    <WebcamCapture
                        mode={activeMode}
                        uploadedImage={uploadedImage}
                        isNerdMode={isNerdMode}
                        setIsNerdMode={setIsNerdMode}
                        setShowHelp={setShowHelp}
                    />
                </div>

            </main>

            {/* MOBILE BOTTOM UI (Unified Action Bar + Drop-up Menu) */}
            <div className="lg:hidden fixed bottom-6 left-4 right-4 z-40 flex flex-col pointer-events-none">

                {/* Backdrop for open menu */}
                <div
                    className={`fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity duration-300 pointer-events-auto ${showMobileMenu ? 'opacity-100' : 'opacity-0 pointer-events-none'}`}
                    onClick={() => setShowMobileMenu(false)}
                />

                {/* Drop-up Model Selector Menu */}
                <div className={`absolute bottom-full left-0 right-0 mb-4 bg-gray-900/95 backdrop-blur-3xl border border-white/10 rounded-[2rem] p-4 shadow-2xl transition-all duration-300 pointer-events-auto origin-bottom ${showMobileMenu ? 'scale-100 opacity-100 translate-y-0' : 'scale-95 opacity-0 translate-y-10 pointer-events-none'}`}>
                    <div className="flex items-center justify-between mb-4 px-2">
                        <h3 className="text-white font-black uppercase tracking-widest text-sm">Select Protocol</h3>
                        <button onClick={() => setShowMobileMenu(false)} className="p-2 bg-white/5 rounded-full text-gray-400 hover:text-white">
                            <X className="w-5 h-5" />
                        </button>
                    </div>
                    {/* Render ModelSelector in Grid Mode for the drawer */}
                    <div className="max-h-[50vh] overflow-y-auto no-scrollbar pb-2">
                        <ModelSelector
                            activeMode={activeMode}
                            setActiveMode={(m) => { setActiveMode(m); setShowMobileMenu(false); }}
                            isMobile={false}
                        />
                    </div>
                </div>

                {/* Main Action Bar */}
                <div className="flex items-center justify-between bg-black/80 backdrop-blur-xl border border-white/10 p-2 rounded-full shadow-2xl shadow-black/50 pointer-events-auto">

                    {/* Model Select Toggle */}
                    <button
                        onClick={() => setShowMobileMenu(!showMobileMenu)}
                        className="flex-1 flex items-center justify-between bg-white/5 hover:bg-white/10 px-4 py-3 rounded-full transition-colors"
                    >
                        <div className="flex flex-col items-start leading-tight">
                            <span className="text-[8px] text-cyan-400 font-black uppercase tracking-widest mb-0.5">Active Mode</span>
                            <span className="text-sm text-white font-bold truncate max-w-[120px]">{activeMode.replace('_', ' ')}</span>
                        </div>
                        <div className="w-8 h-8 rounded-full bg-white/10 flex items-center justify-center">
                            <Activity className="w-4 h-4 text-white" />
                        </div>
                    </button>

                    {/* Quick Actions */}
                    <div className="flex items-center gap-2 pl-2 pr-1">
                        <button
                            onClick={() => setIsNerdMode(!isNerdMode)}
                            className={`w-12 h-12 rounded-full flex items-center justify-center transition-all ${isNerdMode ? 'bg-purple-600 text-white shadow-lg shadow-purple-900/40' : 'bg-white/5 text-gray-400'}`}
                        >
                            <Bug className="w-5 h-5" />
                        </button>
                        <label className="w-12 h-12 rounded-full bg-white text-black flex items-center justify-center shadow-lg active:scale-90 transition-transform cursor-pointer">
                            <Upload className="w-5 h-5" />
                            <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
                        </label>
                    </div>
                </div>
            </div>

            {/* MOBILE INSTRUCTIONS MODAL */}
            {
                showHelp && (
                    <div className="fixed inset-0 z-[60] flex items-center justify-center p-6 bg-black/90 backdrop-blur-md animate-in fade-in duration-300">
                        <div className="bg-gray-900/80 border border-white/10 rounded-[2.5rem] p-8 w-full max-w-sm shadow-2xl relative backdrop-blur-3xl">
                            <button
                                onClick={() => setShowHelp(false)}
                                className="absolute top-6 right-6 text-gray-500 hover:text-white transition-colors"
                            >
                                <X className="w-6 h-6" />
                            </button>

                            <h3 className="text-2xl font-black text-white mb-6 flex items-center gap-3 tracking-tighter uppercase">
                                <div className="w-10 h-10 rounded-2xl bg-blue-500/20 flex items-center justify-center border border-blue-500/30">
                                    <HelpCircle className="w-6 h-6 text-blue-400" />
                                </div>
                                Protocol
                            </h3>

                            <div className="bg-blue-600/10 rounded-2xl p-5 border border-blue-500/20 mb-6 shadow-inner">
                                <h4 className="text-white font-black mb-3 text-[10px] uppercase tracking-widest flex items-center gap-2">
                                    <div className="w-2 h-2 rounded-full bg-blue-500 shadow-[0_0_10px_#3b82f6]"></div>
                                    Operational Focus
                                </h4>
                                <p className="text-gray-300 text-xs leading-relaxed font-bold uppercase tracking-tight opacity-90">
                                    {activeMode === 'JAUNDICE_EYE' && "CALIBRATE FOR SCLERA. REMOVE EYEWEAR. MAINTAIN HIGH LUX ENV."}
                                    {activeMode === 'JAUNDICE_BODY' && "TARGET CHEST/ABDOMEN. SKIN SHOULD BE CLEAN & WELL LIT."}
                                    {activeMode === 'SKIN_DISEASE' && "ALIGN LESION WITHIN RETICLE. ELIMINATE EXTERNAL SHADOWS."}
                                    {activeMode === 'NAIL_DISEASE' && "ISOLATE UNGUIS AREA. CLEAN SURFACE STRONGLY PREFERRED."}
                                    {activeMode === 'ORAL_CANCER' && "MAXIMIZE ORAL APERTURE. TARGET ABNORMAL TISSUE DIRECTLY."}
                                    {activeMode === 'TEETH' && "EXPOSE GINGIVAL MARGINS. ENSURE SHARP FOCUS ON DENTITION."}
                                </p>
                            </div>

                            <div className="bg-gray-950 p-4 rounded-2xl border border-white/5 mb-8">
                                <p className="text-[10px] text-blue-400 font-black tracking-widest opacity-80 uppercase text-center animate-pulse">
                                    SYSTEM_READY_FOR_DIAGNOSIS
                                </p>
                            </div>

                            <button
                                onClick={() => setShowHelp(false)}
                                className="w-full bg-white text-black font-black py-4 rounded-2xl transition-all active:scale-95 shadow-lg shadow-white/10 uppercase tracking-widest text-xs"
                            >
                                Acknowledge
                            </button>
                        </div>
                    </div>
                )
            }

        </div >
    )
}

export default ManualDetection
