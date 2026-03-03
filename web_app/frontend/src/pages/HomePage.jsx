import { Link } from 'react-router-dom'
import { Eye, User, Microscope, Zap, Target, Lock, Smartphone } from 'lucide-react'

const HomePage = () => {
    const capabilities = [
        {
            icon: <Eye className="w-8 h-8" style={{ color: '#fbbf24' }} />,
            title: 'Jaundice Eye Detection',
            description: 'Advanced sclera analysis for accurate jaundice detection in adults.',
            accuracy: 85,
            accentColor: '#fbbf24',
        },
        {
            icon: <User className="w-8 h-8" style={{ color: '#a78bfa' }} />,
            title: 'Jaundice Body Detection',
            description: 'Specialized neonatal jaundice screening for infants.',
            accuracy: 75,
            accentColor: '#a78bfa',
        },
        {
            icon: <Microscope className="w-8 h-8" style={{ color: '#818cf8' }} />,
            title: 'Skin Disease Classification',
            description: 'AI-powered identification of 23 common skin conditions.',
            accuracy: 82,
            accentColor: '#818cf8',
        },
    ]

    const features = [
        { icon: <Zap className="w-7 h-7" style={{ color: '#fbbf24' }} />, title: 'Real-Time Analysis', desc: 'Instant results using GPU acceleration.' },
        { icon: <Target className="w-7 h-7" style={{ color: '#a78bfa' }} />, title: 'High Accuracy', desc: 'Medical-grade AI models.' },
        { icon: <Lock className="w-7 h-7" style={{ color: '#34d399' }} />, title: 'Privacy First', desc: 'All processing happens locally.' },
        { icon: <Smartphone className="w-7 h-7" style={{ color: '#818cf8' }} />, title: 'Mobile Friendly', desc: 'Works on any device with a camera.' },
    ]

    return (
        <div className="min-h-screen overflow-y-auto no-scrollbar" style={{ background: 'var(--bg-base)', color: 'var(--text-1)' }}>
            {/* Hero Section */}
            <div className="relative overflow-hidden">
                {/* Animated glows */}
                <div className="absolute inset-0 opacity-25 pointer-events-none">
                    <div className="absolute top-20 left-20 w-72 h-72 rounded-full blur-3xl animate-pulse"
                        style={{ background: '#7c3aed' }} />
                    <div className="absolute top-40 right-20 w-72 h-72 rounded-full blur-3xl animate-pulse"
                        style={{ background: '#4f46e5', animationDelay: '1s' }} />
                    <div className="absolute bottom-20 left-1/2 w-72 h-72 rounded-full blur-3xl animate-pulse"
                        style={{ background: '#a78bfa', animationDelay: '2s' }} />
                </div>

                <div className="relative max-w-7xl mx-auto px-6 py-24 sm:py-32">
                    {/* Main Hero */}
                    <div className="text-center mb-20">
                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full text-sm font-semibold mb-8"
                            style={{ background: 'rgba(124,58,237,0.1)', border: '1px solid rgba(124,58,237,0.3)', color: '#a78bfa' }}>
                            <span className="relative flex h-2 w-2">
                                <span className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
                                    style={{ background: '#a78bfa' }} />
                                <span className="relative inline-flex rounded-full h-2 w-2"
                                    style={{ background: '#7c3aed' }} />
                            </span>
                            AI-Powered Medical Screening
                        </div>

                        <h1 className="text-5xl sm:text-7xl font-black mb-6 tracking-tight"
                            style={{ background: 'linear-gradient(to right, #ffffff, #ddd6fe, #c7d2fe)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent', backgroundClip: 'text' }}>
                            Medical AI<br />Assistant
                        </h1>

                        <p className="text-xl sm:text-2xl max-w-3xl mx-auto mb-10" style={{ color: 'var(--text-2)' }}>
                            Advanced computer vision for instant medical screening. Detect jaundice and skin conditions in seconds.
                        </p>

                        <div className="flex flex-col sm:flex-row gap-4 justify-center">
                            <Link
                                to="/"
                                className="px-8 py-4 rounded-2xl font-bold text-lg text-white transition-all hover:scale-105 shadow-lg"
                                style={{ background: 'linear-gradient(135deg, #7c3aed, #4f46e5)', boxShadow: '0 8px 32px rgba(124,58,237,0.35)' }}
                            >
                                Start Detection
                            </Link>
                            <Link
                                to="/auto"
                                className="px-8 py-4 rounded-2xl font-bold text-lg transition-all hover:scale-105"
                                style={{ background: 'rgba(255,255,255,0.05)', border: '1px solid rgba(255,255,255,0.12)', color: 'var(--text-1)' }}
                            >
                                Auto-Pilot Mode
                            </Link>
                        </div>
                    </div>

                    {/* Capabilities Grid */}
                    <div className="grid md:grid-cols-3 gap-6 mb-20">
                        {capabilities.map((cap, idx) => (
                            <div key={idx} className="glass rounded-3xl p-7 hover:scale-[1.02] transition-all duration-300 group relative overflow-hidden">
                                {/* Hover glow */}
                                <div className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-300 rounded-3xl"
                                    style={{ background: `radial-gradient(ellipse at top left, ${cap.accentColor}12, transparent 60%)` }} />

                                <div className="relative">
                                    <div className="w-14 h-14 rounded-2xl flex items-center justify-center mb-5"
                                        style={{ background: `${cap.accentColor}15`, border: `1px solid ${cap.accentColor}30` }}>
                                        {cap.icon}
                                    </div>
                                    <h3 className="text-lg font-black mb-2 tracking-tight" style={{ color: 'var(--text-1)' }}>
                                        {cap.title}
                                    </h3>
                                    <p className="text-sm mb-5 leading-relaxed" style={{ color: 'var(--text-2)' }}>
                                        {cap.description}
                                    </p>
                                    <div className="flex items-center gap-3">
                                        <div className="flex-1 rounded-full h-1.5 overflow-hidden"
                                            style={{ background: 'rgba(255,255,255,0.08)' }}>
                                            <div
                                                className="h-full rounded-full"
                                                style={{ width: `${cap.accuracy}%`, background: `linear-gradient(90deg, ${cap.accentColor}80, ${cap.accentColor})` }}
                                            />
                                        </div>
                                        <span className="text-xs font-black mono" style={{ color: cap.accentColor }}>
                                            {cap.accuracy}%
                                        </span>
                                    </div>
                                </div>
                            </div>
                        ))}
                    </div>

                    {/* Features */}
                    <div className="grid sm:grid-cols-2 lg:grid-cols-4 gap-5">
                        {features.map((feature, idx) => (
                            <div key={idx} className="glass-light rounded-2xl p-6 text-center hover:scale-[1.02] transition-all duration-300">
                                <div className="flex justify-center mb-4">
                                    <div className="w-12 h-12 rounded-xl flex items-center justify-center"
                                        style={{ background: 'rgba(255,255,255,0.05)' }}>
                                        {feature.icon}
                                    </div>
                                </div>
                                <h4 className="font-black text-sm mb-1.5 tracking-tight" style={{ color: 'var(--text-1)' }}>
                                    {feature.title}
                                </h4>
                                <p className="text-xs leading-relaxed" style={{ color: 'var(--text-2)' }}>
                                    {feature.desc}
                                </p>
                            </div>
                        ))}
                    </div>

                    {/* CTA Section */}
                    <div className="mt-20 text-center glass rounded-3xl p-12 relative overflow-hidden">
                        <div className="absolute inset-0 opacity-30 pointer-events-none"
                            style={{ background: 'radial-gradient(ellipse at center, rgba(124,58,237,0.2), transparent 70%)' }} />
                        <h2 className="text-3xl font-black mb-4 tracking-tight relative" style={{ color: 'var(--text-1)' }}>
                            Ready to Get Started?
                        </h2>
                        <p className="mb-8 max-w-2xl mx-auto relative text-sm leading-relaxed" style={{ color: 'var(--text-2)' }}>
                            Our AI models are trained on thousands of medical images to provide accurate, instant screening results.
                        </p>
                        <Link
                            to="/"
                            className="inline-block px-8 py-3.5 rounded-2xl font-black text-white transition-all hover:scale-105 text-sm uppercase tracking-wider"
                            style={{ background: 'linear-gradient(135deg, #7c3aed, #4f46e5)', boxShadow: '0 8px 24px rgba(124,58,237,0.4)' }}
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
