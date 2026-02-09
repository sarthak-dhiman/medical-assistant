import { Link } from 'react-router-dom'
import { Eye, User, Microscope, Zap, Target, Lock, Smartphone } from 'lucide-react'

const HomePage = () => {
    const capabilities = [
        {
            icon: 'Eye',
            title: 'Jaundice Eye Detection',
            description: 'Advanced sclera analysis for accurate jaundice detection in adults',
            accuracy: '85%',
            color: 'from-yellow-500 to-orange-500'
        },
        {
            icon: 'User',
            title: 'Jaundice Body Detection',
            description: 'Specialized neonatal jaundice screening for infants',
            accuracy: '75%',
            color: 'from-amber-500 to-yellow-500'
        },
        {
            icon: 'Microscope',
            title: 'Skin Disease Classification',
            description: 'AI-powered identification of 23 common skin conditions',
            accuracy: '82%',
            color: 'from-blue-500 to-cyan-500'
        }
    ]

    const features = [
        { icon: 'Zap', title: 'Real-Time Analysis', desc: 'Instant results using GPU acceleration' },
        { icon: 'Target', title: 'High Accuracy', desc: 'Medical-grade AI models' },
        { icon: 'Lock', title: 'Privacy First', desc: 'All processing happens locally' },
        { icon: 'Smartphone', title: 'Mobile Friendly', desc: 'Works on any device with a camera' }
    ]

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
            {/* Hero Section */}
            <div className="relative overflow-hidden">
                {/* Animated Background */}
                <div className="absolute inset-0 opacity-20">
                    <div className="absolute top-20 left-20 w-72 h-72 bg-blue-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse"></div>
                    <div className="absolute top-40 right-20 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse delay-1000"></div>
                    <div className="absolute bottom-20 left-1/2 w-72 h-72 bg-pink-500 rounded-full mix-blend-multiply filter blur-3xl animate-pulse delay-2000"></div>
                </div>

                <div className="relative max-w-7xl mx-auto px-6 py-24 sm:py-32">
                    {/* Main Hero */}
                    <div className="text-center mb-20">
                        <div className="inline-flex items-center gap-2 px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-full text-blue-400 text-sm font-semibold mb-6">
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-blue-400 opacity-75"></span>
                                <span className="relative inline-flex rounded-full h-2 w-2 bg-blue-500"></span>
                            </span>
                            AI-Powered Medical Screening
                        </div>

                        <h1 className="text-5xl sm:text-7xl font-bold mb-6 bg-gradient-to-r from-white via-blue-100 to-purple-200 bg-clip-text text-transparent">
                            Medical AI Assistant
                        </h1>

                        <p className="text-xl sm:text-2xl text-gray-400 max-w-3xl mx-auto mb-10">
                            Advanced computer vision for instant medical screening. Detect jaundice and skin conditions in seconds.
                        </p>

                        <div className="flex flex-col sm:flex-row gap-4 justify-center">
                            <Link
                                to="/detect"
                                className="px-8 py-4 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl font-bold text-lg hover:from-blue-500 hover:to-purple-500 transition-all transform hover:scale-105 shadow-lg shadow-blue-500/50"
                            >
                                Start Detection
                            </Link>
                            <Link
                                to="/how-it-works"
                                className="px-8 py-4 bg-gray-800 border border-gray-700 rounded-xl font-bold text-lg hover:bg-gray-700 transition-all"
                            >
                                How It Works
                            </Link>
                        </div>
                    </div>

                    {/* Capabilities Grid */}
                    <div className="grid md:grid-cols-3 gap-6 mb-20">
                        {capabilities.map((cap, idx) => (
                            <div
                                key={idx}
                                className="group relative bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl p-6 hover:border-gray-600 transition-all hover:transform hover:scale-105"
                            >
                                <div className={`absolute inset-0 bg-gradient-to-br ${cap.color} opacity-0 group-hover:opacity-10 rounded-2xl transition-opacity`}></div>

                                <div className="relative">
                                    <div className="text-5xl mb-4">
                                        {cap.icon === 'Eye' && <Eye className="w-12 h-12" />}
                                        {cap.icon === 'User' && <User className="w-12 h-12" />}
                                        {cap.icon === 'Microscope' && <Microscope className="w-12 h-12" />}
                                    </div>
                                    <h3 className="text-xl font-bold mb-2">{cap.title}</h3>
                                    <p className="text-gray-400 text-sm mb-4">{cap.description}</p>

                                    <div className="flex items-center gap-2">
                                        <div className="flex-1 bg-gray-700 rounded-full h-2 overflow-hidden">
                                            <div
                                                className={`h-full bg-gradient-to-r ${cap.color} rounded-full`}
                                                style={{ width: cap.accuracy }}
                                            ></div>
                                        </div>
                                        <span className="text-sm font-semibold text-gray-300">{cap.accuracy}</span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Features */}
                    <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-6">
                        {features.map((feature, idx) => (
                            <div key={idx} className="bg-gray-800/30 backdrop-blur-sm border border-gray-700 rounded-xl p-6 text-center">
                                <div className="text-4xl mb-3 flex justify-center">
                                    {feature.icon === 'Zap' && <Zap className="w-10 h-10" />}
                                    {feature.icon === 'Target' && <Target className="w-10 h-10" />}
                                    {feature.icon === 'Lock' && <Lock className="w-10 h-10" />}
                                    {feature.icon === 'Smartphone' && <Smartphone className="w-10 h-10" />}
                                </div>
                                <h4 className="font-bold mb-2">{feature.title}</h4>
                                <p className="text-sm text-gray-400">{feature.desc}</p>
                            </div>
                        ))}
                    </div>

                    {/* CTA Section */}
                    <div className="mt-20 text-center bg-gradient-to-r from-blue-900/30 to-purple-900/30 border border-blue-500/30 rounded-2xl p-12">
                        <h2 className="text-3xl font-bold mb-4">Ready to Get Started?</h2>
                        <p className="text-gray-400 mb-6 max-w-2xl mx-auto">
                            Our AI models are trained on thousands of medical images to provide accurate, instant screening results.
                        </p>
                        <Link
                            to="/detect"
                            className="inline-block px-8 py-4 bg-white text-gray-900 rounded-xl font-bold hover:bg-gray-100 transition-all transform hover:scale-105"
                        >
                            Try Detection Now
                        </Link>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default HomePage
