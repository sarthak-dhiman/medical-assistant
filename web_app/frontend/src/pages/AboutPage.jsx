const AboutPage = () => {
    const team = [
        { role: 'AI Models', desc: 'EfficientNet-B4, SegFormer' },
        { role: 'Framework', desc: 'PyTorch, Transformers' },
        { role: 'Backend', desc: 'FastAPI, Celery, Redis' },
        { role: 'Frontend', desc: 'React, Vite, TailwindCSS' },
        { role: 'Deployment', desc: 'Docker, NVIDIA GPU' }
    ]

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
            <div className="max-w-6xl mx-auto px-6 py-16">
                {/* Header */}
                <div className="text-center mb-16">
                    <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-white to-green-200 bg-clip-text text-transparent">
                        About This Project
                    </h1>
                    <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                        An open-source medical AI screening tool built with cutting-edge deep learning
                    </p>
                </div>

                {/* Mission */}
                <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 border border-blue-500/30 rounded-2xl p-12 mb-16 text-center">
                    <h2 className="text-3xl font-bold mb-4">Our Mission</h2>
                    <p className="text-lg text-gray-300 max-w-3xl mx-auto leading-relaxed">
                        To democratize access to AI-powered medical screening tools, enabling early detection of jaundice and skin conditions through accessible, privacy-first technology.
                    </p>
                </div>

                {/* Tech Stack */}
                <div className="mb-16">
                    <h2 className="text-3xl font-bold mb-8 text-center">Technology Stack</h2>
                    <div className="grid sm:grid-cols-2 lg:grid-cols-3 gap-6">
                        {team.map((item, idx) => (
                            <div key={idx} className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 text-center">
                                <div className="text-blue-400 font-bold mb-2">{item.role}</div>
                                <div className="text-gray-400 text-sm">{item.desc}</div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Features */}
                <div className="mb-16">
                    <h2 className="text-3xl font-bold mb-8 text-center">Key Features</h2>
                    <div className="grid md:grid-cols-2 gap-6">
                        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
                            <h3 className="text-xl font-bold mb-3">Privacy First</h3>
                            <p className="text-gray-400">All image processing happens locally. No data is stored or transmitted to external servers.</p>
                        </div>
                        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
                            <h3 className="text-xl font-bold mb-3">Real-Time Detection</h3>
                            <p className="text-gray-400">GPU-accelerated inference delivers results in ~2 seconds with CUDA support.</p>
                        </div>
                        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
                            <h3 className="text-xl font-bold mb-3">High Accuracy</h3>
                            <p className="text-gray-400">Models trained on thousands of medical images achieve 75-85% accuracy.</p>
                        </div>
                        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6">
                            <h3 className="text-xl font-bold mb-3">Cross-Platform</h3>
                            <p className="text-gray-400">Works on desktop and mobile devices with camera support.</p>
                        </div>
                    </div>
                </div>

                {/* Disclaimer */}
                <div className="bg-gradient-to-br from-red-900/20 to-orange-900/20 border border-red-500/30 rounded-2xl p-8 text-center">
                    <h2 className="text-2xl font-bold mb-4 text-red-400">Important Disclaimer</h2>
                    <p className="text-gray-300 max-w-3xl mx-auto leading-relaxed">
                        This tool is designed for <strong>educational and screening purposes only</strong>. It is NOT a substitute for professional medical diagnosis, treatment, or advice. Always consult qualified healthcare professionals for medical concerns. The AI models may produce false positives or false negatives.
                    </p>
                </div>

                {/* Version */}
                <div className="mt-12 text-center text-gray-500 text-sm">
                    <p>Version 2.0.0 • Built with React & PyTorch</p>
                    <p className="mt-2">© 2026 Medical AI Assistant • Open Source Project</p>
                </div>
            </div>
        </div>
    )
}

export default AboutPage
