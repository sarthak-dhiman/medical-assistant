import { useState, useEffect } from 'react'
import axios from 'axios'
import WebcamCapture from '../WebcamCapture'

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000'

const DetectionPage = () => {
    const [activeMode, setActiveMode] = useState('JAUNDICE_BODY')
    const [uploadedImage, setUploadedImage] = useState(null)
    const [isNerdMode, setIsNerdMode] = useState(false)
    const [modelsReady, setModelsReady] = useState(false)

    // Poll Backend Health until Models are Ready
    useEffect(() => {
        const checkHealth = async () => {
            try {
                const response = await axios.get(`${API_URL}/health`)
                const data = response.data
                if (data.models_ready) {
                    setModelsReady(true)
                } else {
                    console.log("Waiting for models...", data);
                    setTimeout(checkHealth, 2000)
                }
            } catch (error) {
                console.error("Health check failed:", error)
                setTimeout(checkHealth, 3000)
            }
        }
        checkHealth()
    }, [])

    if (!modelsReady) {
        return (
            <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black flex items-center justify-center">
                <div className="text-center">
                    <div className="w-16 h-16 border-4 border-blue-500 border-t-transparent rounded-full animate-spin mx-auto mb-4"></div>
                    <p className="text-gray-400 text-sm animate-pulse">Initializing AI Models...</p>
                </div>
            </div>
        )
    }

    const modes = [
        { id: 'JAUNDICE_EYE', label: 'Jaundice Eye', icon: '👁️' },
        { id: 'JAUNDICE_BODY', label: 'Jaundice Body', icon: '👶' },
        { id: 'SKIN_DISEASE', label: 'Skin Disease', icon: '🔬' }
    ]

    const handleFileUpload = (e) => {
        const file = e.target.files[0]
        if (file) {
            const reader = new FileReader()
            reader.onloadend = () => {
                setUploadedImage(reader.result)
            }
            reader.readAsDataURL(file)
        }
    }

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
            <div className="max-w-7xl mx-auto px-4 py-8">
                {/* Header */}
                <div className="flex items-center justify-between mb-8">
                    <div>
                        <h1 className="text-3xl font-bold mb-2">AI Detection</h1>
                        <p className="text-gray-400">Select a detection mode and start screening</p>
                    </div>

                    {/* Nerd Mode Toggle */}
                    <button
                        onClick={() => setIsNerdMode(!isNerdMode)}
                        className={`flex items-center gap-2 px-4 py-2 rounded-full text-sm font-bold transition-all border ${isNerdMode ? 'bg-purple-500/20 text-purple-400 border-purple-500/50' : 'bg-gray-800 text-gray-500 border-gray-700'}`}
                    >
                        🤓 NERD MODE: {isNerdMode ? 'ON' : 'OFF'}
                    </button>
                </div>

                <div className="grid lg:grid-cols-[300px_1fr] gap-6">
                    {/* Sidebar */}
                    <div className="space-y-6">
                        {/* Mode Selection */}
                        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl p-6">
                            <h3 className="text-gray-400 text-xs font-semibold uppercase tracking-wider mb-4">Detection Mode</h3>
                            <div className="space-y-2">
                                {modes.map((mode) => (
                                    <button
                                        key={mode.id}
                                        onClick={() => setActiveMode(mode.id)}
                                        className={`w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-all duration-200 ${activeMode === mode.id
                                            ? 'bg-blue-600 text-white shadow-lg shadow-blue-500/50'
                                            : 'bg-gray-700/50 text-gray-300 hover:bg-gray-700'
                                            }`}
                                    >
                                        <span className="text-2xl">{mode.icon}</span>
                                        <span className="font-medium">{mode.label}</span>
                                    </button>
                                ))}
                            </div>
                        </div>

                        {/* Upload */}
                        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl p-6">
                            <h3 className="text-gray-400 text-xs font-semibold uppercase tracking-wider mb-4">Upload Image</h3>
                            <label className="block w-full px-4 py-3 bg-gray-700 hover:bg-gray-600 rounded-xl text-center cursor-pointer transition-all">
                                <span className="font-medium">📁 Choose File</span>
                                <input
                                    type="file"
                                    accept="image/*"
                                    onChange={handleFileUpload}
                                    className="hidden"
                                />
                            </label>
                            {uploadedImage && (
                                <button
                                    onClick={() => setUploadedImage(null)}
                                    className="w-full mt-2 px-4 py-2 bg-red-600/20 text-red-400 border border-red-600/50 rounded-xl hover:bg-red-600/30 transition-all"
                                >
                                    Clear Upload
                                </button>
                            )}
                        </div>

                        {/* Instructions */}
                        <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 border border-blue-500/30 rounded-2xl p-6">
                            <h3 className="text-blue-400 text-xs font-semibold uppercase tracking-wider mb-3">💡 Tips</h3>
                            <ul className="text-sm text-gray-400 space-y-2">
                                {activeMode === 'JAUNDICE_EYE' && <li>• Remove glasses for best results</li>}
                                {activeMode === 'JAUNDICE_EYE' && <li>• Look directly at camera</li>}
                                {activeMode === 'JAUNDICE_BODY' && <li>• Ensure good lighting</li>}
                                {activeMode === 'JAUNDICE_BODY' && <li>• Show clear skin area</li>}
                                {activeMode === 'SKIN_DISEASE' && <li>• Focus on affected area</li>}
                                {activeMode === 'SKIN_DISEASE' && <li>• Avoid blur</li>}
                            </ul>
                        </div>
                    </div>

                    {/* Main Detection Area */}
                    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl overflow-hidden">
                        <WebcamCapture mode={activeMode} uploadedImage={uploadedImage} isNerdMode={isNerdMode} />
                    </div>
                </div>
            </div>
        </div>
    )
}

export default DetectionPage
