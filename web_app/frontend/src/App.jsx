import { useState, useEffect } from 'react'
import WebcamCapture from './WebcamCapture'
import { Upload, X, User, Eye, Stethoscope, Activity, Bug, HelpCircle, ChevronUp, ChevronDown } from 'lucide-react'

function App() {
  const [activeMode, setActiveMode] = useState('JAUNDICE_BODY')
  const [uploadedImage, setUploadedImage] = useState(null)
  const [isNerdMode, setIsNerdMode] = useState(false)
  const [showHelp, setShowHelp] = useState(false);
  const [isAppReady, setIsAppReady] = useState(false);
  const [showModeMenu, setShowModeMenu] = useState(false);

  // Lock Body Scroll (Prevent Pull-to-Refresh & Scroll)
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = 'auto'; };
  }, []);

  // Poll Backend Health until Models are Ready
  useEffect(() => {
    // Relative API path (proxied by Nginx)
    const HEALTH_URL = "/health";

    const checkHealth = async () => {
      try {
        const res = await fetch(HEALTH_URL);
        const data = await res.json();
        if (data.models_ready) {
          setIsAppReady(true);
        } else {
          console.log("Waiting for models...", data);
        }
      } catch (e) {
        console.error("Backend offline:", e);
      }
    };

    if (!isAppReady) {
      checkHealth(); // Initial check
      const interval = setInterval(checkHealth, 3000); // Poll every 3s
      return () => clearInterval(interval);
    }
  }, [isAppReady]);

  if (!isAppReady) {
    return (
      <div className="w-full h-[100dvh] bg-gray-950 flex flex-col items-center justify-center text-white p-4">
        <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-cyan-500 mb-6"></div>
        <h2 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500 mb-2">
          Medical Assistant
        </h2>
        <p className="text-gray-400 text-sm animate-pulse">Initializing AI Models...</p>
      </div>
    );
  }

  const modes = [
    { id: 'JAUNDICE_BODY', label: 'Jaundice Baby', shortLabel: 'Baby', icon: <User className="w-5 h-5" /> },
    { id: 'JAUNDICE_EYE', label: 'Jaundice Adult', shortLabel: 'Adult', icon: <Eye className="w-5 h-5" /> },
    { id: 'SKIN_DISEASE', label: 'Skin Disease', shortLabel: 'Skin', icon: <Stethoscope className="w-5 h-5" /> },
    { id: 'BURNS', label: 'Burns Detection', shortLabel: 'Burns', icon: <Stethoscope className="w-5 h-5" /> },
    { id: 'NAIL_DISEASE', label: 'Nail Disease', shortLabel: 'Nail', icon: <Stethoscope className="w-5 h-5" /> },
  ]

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setUploadedImage(reader.result);
      reader.readAsDataURL(file);
    }
  }


  return (
    <div className="relative w-full h-[100dvh] flex flex-col bg-gray-950 text-white font-sans overflow-hidden supports-[height:100cqh]:h-[100cqh] supports-[height:100svh]:h-[100svh]">


      {/* Navbar (Desktop Only or Minimal on Mobile) */}
      <nav className="hidden lg:block border-b border-gray-800 bg-gray-900/50 backdrop-blur-xl z-50">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center">
                <Activity className="w-5 h-5 text-white" />
              </div>
              <span className="font-bold text-xl bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500">
                Medical Assistant
              </span>
            </div>
            {/* Desktop Nerd Mode Toggle */}
            <button
              onClick={() => setIsNerdMode(!isNerdMode)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-bold transition-all border ${isNerdMode ? 'bg-purple-500/20 text-purple-400 border-purple-500/50' : 'bg-gray-800 text-gray-500 border-gray-700'
                }`}
            >
              <Bug className="w-4 h-4" />
              NERD MODE: {isNerdMode ? 'ON' : 'OFF'}
            </button>
          </div>
        </div>
      </nav>

      {/* Main Content Area */}
      <main className="flex-1 flex flex-col lg:flex-row lg:max-w-7xl lg:mx-auto w-full p-0 lg:px-4 lg:py-8 gap-0 lg:gap-6 overflow-hidden relative">

        {/* DESKTOP SIDEBAR (Hidden on Mobile) */}
        <aside className="hidden lg:flex flex-col w-80 space-y-4">
          {/* Mode Selection */}
          <div className="bg-gray-900/50 rounded-2xl p-4 border border-gray-800">
            <h3 className="text-gray-400 text-xs font-semibold uppercase tracking-wider mb-4">Detection Mode</h3>
            <div className="space-y-2">
              {modes.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => setActiveMode(mode.id)}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-all duration-200 ${activeMode === mode.id
                    ? 'bg-blue-600 text-white shadow-lg shadow-blue-900/20'
                    : 'hover:bg-gray-800 text-gray-400'
                    }`}
                >
                  {mode.icon}
                  <span className="font-medium">{mode.label}</span>
                </button>
              ))}
            </div>
          </div>

          {/* Input Source */}
          <div className="bg-gray-900/50 rounded-2xl p-4 border border-gray-800">
            <h3 className="text-gray-400 text-xs font-semibold uppercase tracking-wider mb-4">Input Source</h3>
            <div className="space-y-2">
              <label className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl bg-gray-800 hover:bg-gray-700 cursor-pointer transition-all border border-gray-700 border-dashed group">
                <Upload className="w-5 h-5 text-gray-400 group-hover:text-cyan-400" />
                <span className="text-gray-300 font-medium group-hover:text-cyan-400">Upload Image</span>
                <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
              </label>
              {uploadedImage && (
                <button
                  onClick={() => setUploadedImage(null)}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 rounded-xl bg-red-500/20 text-red-400 hover:bg-red-500/30 transition-all text-sm font-medium"
                >
                  <X className="w-4 h-4" /> Clear Image
                </button>
              )}
            </div>
          </div>

          {/* Instructions Button (Replaces Tab) */}
          <button
            onClick={() => setShowHelp(true)}
            className="w-full flex items-center justify-between px-4 py-3 rounded-xl bg-blue-900/10 text-blue-400 hover:bg-blue-900/20 active:scale-[0.98] transition-all border border-blue-500/10 group mt-auto"
          >
            <span className="font-semibold flex items-center gap-2">
              <HelpCircle className="w-5 h-5 group-hover:text-blue-300 transition-colors" />
              <span className="group-hover:text-gray-200 transition-colors">Instructions & Help</span>
            </span>
            <div className="bg-blue-500/10 p-1 rounded-full group-hover:bg-blue-500/20">
              <ChevronUp className="w-4 h-4 rotate-90 opacity-50 group-hover:opacity-100 transition-opacity" />
            </div>
          </button>
        </aside>

        {/* CAMERA FEED (Full Screen on Mobile) */}
        <div className="flex-1 relative flex flex-col justify-center bg-black lg:rounded-3xl lg:border-2 lg:border-gray-800 overflow-hidden w-full h-full">
          <div className="absolute inset-0">
            <WebcamCapture
              mode={activeMode}
              uploadedImage={uploadedImage}
              isNerdMode={isNerdMode}
              setIsNerdMode={setIsNerdMode}
              setShowHelp={setShowHelp}
            />
          </div>
        </div>

      </main>

      {/* MOBILE/TABLET BOTTOM NAVIGATION (DROP UP) */}
      <div className="lg:hidden fixed bottom-4 md:bottom-6 left-4 right-4 z-[50] flex flex-col justify-end pointer-events-none gap-3 pb-[env(safe-area-inset-bottom)]">

        {/* Drop Up Menu */}
        {showModeMenu && (
          <div className="w-full bg-gray-900/95 backdrop-blur-xl border border-gray-700/50 rounded-2xl shadow-2xl overflow-hidden animate-in slide-in-from-bottom-5 fade-in duration-200 pointer-events-auto origin-bottom mb-2">
            <div className="max-h-[50vh] overflow-y-auto p-2 space-y-1">
              {modes.map((mode) => (
                <button
                  key={mode.id}
                  onClick={() => { setActiveMode(mode.id); setShowModeMenu(false); }}
                  className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl transition-all active:scale-[0.98] ${activeMode === mode.id
                    ? 'bg-blue-600/20 border border-blue-500/30 text-white shadow-lg shadow-blue-900/10'
                    : 'bg-transparent text-gray-400 hover:bg-gray-800 hover:text-white'
                    }`}
                >
                  <div className={`p-2 rounded-lg ${activeMode === mode.id ? 'bg-blue-500 text-white' : 'bg-gray-800 text-gray-400'}`}>
                    {mode.icon}
                  </div>
                  <div className="flex flex-col items-start">
                    <span className="font-semibold text-sm tracking-wide">{mode.label}</span>
                  </div>
                  {activeMode === mode.id && <div className="ml-auto w-2 h-2 rounded-full bg-blue-500 shadow-[0_0_10px_#3b82f6]"></div>}
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Bottom Bar (Floating Capsule) */}
        <div className="w-full pointer-events-auto flex gap-3 h-16">

          {/* Main Mode Trigger */}
          <button
            onClick={() => setShowModeMenu(!showModeMenu)}
            className={`flex-1 bg-gray-900/90 backdrop-blur-xl border border-gray-700/50 rounded-2xl shadow-xl flex items-center justify-between px-4 transition-all active:scale-[0.98] group ${showModeMenu ? 'ring-2 ring-blue-500/50 border-blue-500/50' : ''}`}
          >
            <div className="flex items-center gap-3">
              <div className="h-10 w-10 flex items-center justify-center rounded-xl bg-gradient-to-br from-blue-600/20 to-cyan-500/20 text-blue-400 border border-blue-500/20">
                {modes.find(m => m.id === activeMode)?.icon}
              </div>
              <div className="flex flex-col items-start">
                <span className="text-[10px] text-gray-500 font-bold uppercase tracking-wider">Current Mode</span>
                <span className="text-sm font-bold text-white truncate max-w-[120px]">
                  {modes.find(m => m.id === activeMode)?.shortLabel}
                </span>
              </div>
            </div>
            <div className={`p-2 rounded-full transition-all duration-300 ${showModeMenu ? 'bg-blue-600 text-white rotate-180' : 'bg-gray-800 text-gray-400'}`}>
              <ChevronUp className="w-5 h-5" />
            </div>
          </button>

          {/* Tools */}
          <button
            onClick={() => setIsNerdMode(!isNerdMode)}
            className={`h-16 w-16 rounded-2xl flex items-center justify-center transition-all border shadow-xl active:scale-95 ${isNerdMode
              ? 'bg-purple-600 text-white border-purple-500 shadow-purple-900/30'
              : 'bg-gray-900/90 text-gray-400 border-gray-700/50 backdrop-blur-xl'
              }`}
          >
            <Bug className="w-6 h-6" />
          </button>

          <label className="h-16 w-16 rounded-2xl bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center text-white border border-blue-400/30 shadow-xl shadow-blue-900/30 cursor-pointer active:scale-95 transition-transform">
            <Upload className="w-6 h-6" />
            <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
          </label>

        </div>
      </div>

      {/* MOBILE INSTRUCTIONS MODAL */}
      {showHelp && (
        <div className="fixed inset-0 z-[60] flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
          <div className="bg-gray-900 border border-gray-800 rounded-2xl p-6 w-full max-w-sm shadow-2xl relative">
            <button
              onClick={() => setShowHelp(false)}
              className="absolute top-4 right-4 text-gray-400 hover:text-white"
            >
              <X className="w-6 h-6" />
            </button>

            <h3 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
              <HelpCircle className="w-6 h-6 text-blue-400" /> Instructions
            </h3>

            <div className="bg-blue-900/10 rounded-xl p-4 border border-blue-500/10 mb-4">
              <h4 className="text-blue-400 font-semibold mb-2 text-sm uppercase">Current Mode</h4>
              <p className="text-gray-300">
                {activeMode === 'JAUNDICE_EYE' && "Come closer & remove glasses. Best for adults."}
                {activeMode === 'JAUNDICE_BODY' && "Ensure good lighting. Show face or skin clearly. Best for babies."}
                {activeMode === 'SKIN_DISEASE' && "Focus camera on affected area. Keep steady."}
                {activeMode === 'BURNS' && "Ensure good lighting. Focus on the burn area."}
                {activeMode === 'NAIL_DISEASE' && "Focus specifically on the affected nail. Remove polish if possible."}
              </p>
            </div>

            <div className="border-t border-gray-800 pt-4">
              <p className="text-xs text-blue-300 opacity-80">
                💡 <strong>Tip:</strong> If the model doesn't load or gets stuck, try switching modes and back to refresh it.
              </p>
            </div>

            <button
              onClick={() => setShowHelp(false)}
              className="w-full mt-6 bg-blue-600 hover:bg-blue-500 text-white font-bold py-3 rounded-xl transition-colors"
            >
              Got it
            </button>
          </div>
        </div>
      )}

    </div>
  )
}

export default App
