import { Camera, Brain, Scissors, Target, CheckCircle } from 'lucide-react'

const HowItWorksPage = () => {
    const steps = [
        {
            num: '01',
            title: 'Capture or Upload',
            desc: 'Use your device camera for real-time detection or upload an existing image',
            icon: 'Camera'
        },
        {
            num: '02',
            title: 'AI Processing',
            desc: 'Our GPU-accelerated models analyze the image using deep learning',
            icon: 'Brain'
        },
        {
            num: '03',
            title: 'Segmentation',
            desc: 'SegFormer isolates relevant regions (skin, eyes) for focused analysis',
            icon: 'Scissors'
        },
        {
            num: '04',
            title: 'Classification',
            desc: 'EfficientNet models classify the condition with confidence scores',
            icon: 'Target'
        },
        {
            num: '05',
            title: 'Results',
            desc: 'Get instant results with visual overlays and detailed metrics',
            icon: 'CheckCircle'
        }
    ]

    const models = [
        {
            name: 'SegFormer',
            purpose: 'Semantic Segmentation',
            details: 'Identifies skin, eyes, and other body parts with pixel-level precision',
            tech: 'Transformer-based architecture'
        },
        {
            name: 'EfficientNet-B4',
            purpose: 'Jaundice Detection',
            details: 'Specialized models for eye and body jaundice screening',
            tech: 'Convolutional Neural Network'
        },
        {
            name: 'EfficientNet-B4',
            purpose: 'Skin Disease Classification',
            details: 'Identifies 23 common skin conditions including eczema, psoriasis, melanoma',
            tech: 'Transfer learning from ImageNet'
        }
    ]

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
            <div className="max-w-6xl mx-auto px-6 py-16">
                {/* Header */}
                <div className="text-center mb-16">
                    <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-white to-blue-200 bg-clip-text text-transparent">
                        How It Works
                    </h1>
                    <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                        Our AI pipeline combines state-of-the-art computer vision models for accurate medical screening
                    </p>
                </div>

                {/* Process Steps */}
                <div className="mb-20">
                    <h2 className="text-3xl font-bold mb-8 text-center">Detection Pipeline</h2>
                    <div className="space-y-6">
                        {steps.map((step, idx) => (
                            <div key={idx} className="flex gap-6 items-start bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl p-6 hover:border-blue-500/50 transition-all">
                                <div className="flex-shrink-0">
                                    <div className="w-16 h-16 bg-gradient-to-br from-blue-600 to-purple-600 rounded-xl flex items-center justify-center text-2xl font-bold">
                                        {step.num}
                                    </div>
                                </div>
                                <div className="flex-1">
                                    <div className="flex items-center gap-3 mb-2">
                                        <span className="text-3xl">
                                            {step.icon === 'Camera' && <Camera className="w-8 h-8" />}
                                            {step.icon === 'Brain' && <Brain className="w-8 h-8" />}
                                            {step.icon === 'Scissors' && <Scissors className="w-8 h-8" />}
                                            {step.icon === 'Target' && <Target className="w-8 h-8" />}
                                            {step.icon === 'CheckCircle' && <CheckCircle className="w-8 h-8" />}
                                        </span>
                                        <h3 className="text-2xl font-bold">{step.title}</h3>
                                    </div>
                                    <p className="text-gray-400">{step.desc}</p>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* AI Models */}
                <div className="mb-20">
                    <h2 className="text-3xl font-bold mb-8 text-center">AI Models</h2>
                    <div className="grid md:grid-cols-3 gap-6">
                        {models.map((model, idx) => (
                            <div key={idx} className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl p-6">
                                <div className="text-blue-400 text-sm font-semibold uppercase tracking-wider mb-2">
                                    {model.purpose}
                                </div>
                                <h3 className="text-xl font-bold mb-3">{model.name}</h3>
                                <p className="text-gray-400 text-sm mb-4">{model.details}</p>
                                <div className="inline-block px-3 py-1 bg-purple-500/20 text-purple-400 rounded-full text-xs font-semibold">
                                    {model.tech}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>

                {/* Technical Details */}
                <div className="bg-gradient-to-br from-blue-900/20 to-purple-900/20 border border-blue-500/30 rounded-2xl p-8">
                    <h2 className="text-2xl font-bold mb-6">Performance</h2>
                    <div className="grid sm:grid-cols-3 gap-6">
                        <div>
                            <div className="text-3xl font-bold text-blue-400 mb-2">~2s</div>
                            <div className="text-gray-400">Average inference time</div>
                        </div>
                        <div>
                            <div className="text-3xl font-bold text-purple-400 mb-2">GPU</div>
                            <div className="text-gray-400">CUDA acceleration</div>
                        </div>
                        <div>
                            <div className="text-3xl font-bold text-pink-400 mb-2">Local</div>
                            <div className="text-gray-400">Privacy-first processing</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default HowItWorksPage
