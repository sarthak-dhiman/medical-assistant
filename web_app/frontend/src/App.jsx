import { useState, useEffect } from 'react'
import WebcamCapture from './WebcamCapture'
import { Upload, X, User, Eye, Stethoscope, Activity, Bug } from 'lucide-react'

function App() {
  const [activeMode, setActiveMode] = useState('JAUNDICE_BODY')
  const [uploadedImage, setUploadedImage] = useState(null)
  const [isNerdMode, setIsNerdMode] = useState(false)
  const [showStartupInfo, setShowStartupInfo] = useState(true);

  // Auto-dismiss startup info after 10 seconds
  useEffect(() => {
    const timer = setTimeout(() => setShowStartupInfo(false), 10000);
    return () => clearTimeout(timer);
  }, []);

  const modes = [
    { id: 'JAUNDICE_BODY', label: 'Body Jaundice', shortLabel: 'Body', icon: <User className="w-5 h-5" /> },
    { id: 'JAUNDICE_EYE', label: 'Eye Jaundice', shortLabel: 'Eye', icon: <Eye className="w-5 h-5" /> },
    { id: 'SKIN_DISEASE', label: 'Skin Disease', shortLabel: 'Skin', icon: <Stethoscope className="w-5 h-5" /> },
  ]

  const handleUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => setUploadedImage(reader.result);
      reader.readAsDataURL(file);
    }
  }

  // Lock Body Scroll (Prevent Pull-to-Refresh & Scroll)
  useEffect(() => {
    document.body.style.overflow = 'hidden';
    return () => { document.body.style.overflow = 'auto'; };
  }, []);

  return (
    <div className="relative w-full h-[100dvh] flex flex-col bg-gray-950 text-white font-sans overflow-hidden supports-[height:100cqh]:h-[100cqh] supports-[height:100svh]:h-[100svh]">

      {/* Startup Notification */}
      {showStartupInfo && (
        <div className="fixed top-20 left-1/2 transform -translate-x-1/2 z-50 w-11/12 max-w-md bg-blue-900/90 border border-blue-500/50 text-blue-100 px-4 py-3 rounded-xl shadow-2xl backdrop-blur-md flex items-start gap-3 animate-in fade-in slide-in-from-top-4 duration-500">
          <div className="mt-1 p-1 bg-blue-500/20 rounded-full">
            <Activity className="w-5 h-5 text-blue-300" />
          </div>
          <div>
            <h3 className="font-bold text-sm">First Time Setup</h3>
            <p className="text-xs opacity-90 mt-1">
              The AI models are warming up. The first prediction may take
              <span className="font-bold text-white"> 20-30 seconds</span>.
              Please be patient!
            </p>
          </div>
          <button onClick={() => setShowStartupInfo(false)} className="ml-auto text-blue-300 hover:text-white">
            <X className="w-4 h-4" />
          </button>
        </div>
      )}

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

          {/* Instructions */}
          <div className="bg-blue-900/10 rounded-2xl p-6 border border-blue-500/10">
            <h4 className="text-blue-400 font-semibold mb-2">Instructions</h4>
            <p className="text-sm text-gray-400 leading-relaxed">
              {uploadedImage ? "Analyzing static image." : "Position yourself in the camera frame."}
            </p>
          </div>
        </aside>

        {/* CAMERA FEED (Full Screen on Mobile) */}
        <div className="flex-1 relative flex flex-col justify-center bg-black lg:rounded-3xl lg:border-2 lg:border-gray-800 overflow-hidden w-full h-full">
          <div className="absolute inset-0">
            <WebcamCapture mode={activeMode} uploadedImage={uploadedImage} isNerdMode={isNerdMode} />
          </div>

          {/* MOBILE TOP BAR (Overlay) */}
          <div className="lg:hidden absolute top-0 left-0 right-0 p-4 bg-gradient-to-b from-black/80 to-transparent z-10 flex justify-between items-start pointer-events-none">
            <div className="flex flex-col">
              <span className="text-xs font-bold text-cyan-400 tracking-wider">MEDICAL AI</span>
              <span className="text-lg font-bold text-white">{modes.find(m => m.id === activeMode)?.label}</span>
            </div>

            <div className="flex gap-2 pointer-events-auto">
              {/* Mobile Nerd Mode Toggle */}
              <button
                onClick={() => setIsNerdMode(!isNerdMode)}
                className={`p-2 rounded-full backdrop-blur-md shadow-lg transition-all ${isNerdMode ? 'bg-purple-500/80 text-white' : 'bg-gray-800/80 text-gray-400'
                  }`}
              >
                <Bug className="w-5 h-5" />
              </button>

              {/* Clear Image Button Mobile */}
              {uploadedImage && (
                <button
                  onClick={() => setUploadedImage(null)}
                  className="bg-red-500/80 backdrop-blur-md p-2 rounded-full text-white shadow-lg"
                >
                  <X className="w-5 h-5" />
                </button>
              )}
            </div>
          </div>
        </div>

      </main>

      {/* MOBILE BOTTOM NAVIGATION */}
      <div className="lg:hidden bg-gray-900 border-t border-gray-800 pb-safe pt-2 px-4 z-50">
        <div className="flex justify-between items-center h-16">

          {/* Mode Tabs */}
          <div className="flex bg-gray-800 rounded-full p-1 gap-1">
            {modes.map((mode) => (
              <button
                key={mode.id}
                onClick={() => setActiveMode(mode.id)}
                className={`flex flex-col items-center justify-center px-4 py-2 rounded-full transition-all ${activeMode === mode.id
                  ? 'bg-blue-600 text-white shadow-md'
                  : 'text-gray-400'
                  }`}
              >
                {activeMode === mode.id ? <span className="text-xs font-bold">{mode.shortLabel}</span> : mode.icon}
              </button>
            ))}
          </div>

          {/* Upload FAB (Mini) */}
          <label className="flex items-center justify-center w-12 h-12 rounded-full bg-gray-800 text-cyan-400 border border-gray-700 shadow-lg cursor-pointer active:scale-95 transition-transform">
            <Upload className="w-6 h-6" />
            <input type="file" accept="image/*" className="hidden" onChange={handleUpload} />
          </label>
        </div>
      </div>

    </div>
  )
}

export default App
