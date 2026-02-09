import { useState } from 'react'

const SupportPage = () => {
    const [formData, setFormData] = useState({
        name: '',
        email: '',
        type: 'bug',
        description: ''
    })
    const [submitted, setSubmitted] = useState(false)

    const handleSubmit = (e) => {
        e.preventDefault()
        // In production, this would send to a backend
        console.log('Support request:', formData)
        setSubmitted(true)
        setTimeout(() => {
            setSubmitted(false)
            setFormData({ name: '', email: '', type: 'bug', description: '' })
        }, 3000)
    }

    const faqs = [
        {
            q: 'Is my data stored or shared?',
            a: 'No. All processing happens locally in your browser or on the server. No images are stored or transmitted to third parties.'
        },
        {
            q: 'How accurate are the AI models?',
            a: 'Our models achieve 75-85% accuracy on test datasets. However, this is NOT a replacement for professional medical diagnosis.'
        },
        {
            q: 'What browsers are supported?',
            a: 'Chrome, Firefox, Safari, and Edge (latest versions). Camera access is required for real-time detection.'
        },
        {
            q: 'Can I use this on mobile?',
            a: 'Yes! The app is fully responsive and works on mobile devices with camera support.'
        },
        {
            q: 'Why is the first detection slow?',
            a: 'Models are loaded on first use. Subsequent detections are much faster (~2 seconds).'
        }
    ]

    return (
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-black text-white">
            <div className="max-w-6xl mx-auto px-6 py-16">
                {/* Header */}
                <div className="text-center mb-16">
                    <h1 className="text-5xl font-bold mb-4 bg-gradient-to-r from-white to-purple-200 bg-clip-text text-transparent">
                        Support
                    </h1>
                    <p className="text-xl text-gray-400 max-w-3xl mx-auto">
                        Need help? Found a bug? We're here to assist you.
                    </p>
                </div>

                <div className="grid lg:grid-cols-2 gap-8 mb-16">
                    {/* Contact Form */}
                    <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl p-8">
                        <h2 className="text-2xl font-bold mb-6">Report an Issue</h2>

                        {submitted ? (
                            <div className="bg-green-500/20 border border-green-500/50 rounded-xl p-6 text-center">
                                <div className="text-4xl mb-3 text-green-400">✓</div>
                                <p className="text-green-400 font-semibold">Thank you! We'll review your report.</p>
                            </div>
                        ) : (
                            <form onSubmit={handleSubmit} className="space-y-4">
                                <div>
                                    <label className="block text-sm font-semibold text-gray-400 mb-2">Name</label>
                                    <input
                                        type="text"
                                        required
                                        value={formData.name}
                                        onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                                        className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl focus:border-blue-500 focus:outline-none transition-all"
                                        placeholder="Your name"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-semibold text-gray-400 mb-2">Email</label>
                                    <input
                                        type="email"
                                        required
                                        value={formData.email}
                                        onChange={(e) => setFormData({ ...formData, email: e.target.value })}
                                        className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl focus:border-blue-500 focus:outline-none transition-all"
                                        placeholder="your@email.com"
                                    />
                                </div>

                                <div>
                                    <label className="block text-sm font-semibold text-gray-400 mb-2">Type</label>
                                    <select
                                        value={formData.type}
                                        onChange={(e) => setFormData({ ...formData, type: e.target.value })}
                                        className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl focus:border-blue-500 focus:outline-none transition-all"
                                    >
                                        <option value="bug">Bug Report</option>
                                        <option value="feature">Feature Request</option>
                                        <option value="question">Question</option>
                                        <option value="other">Other</option>
                                    </select>
                                </div>

                                <div>
                                    <label className="block text-sm font-semibold text-gray-400 mb-2">Description</label>
                                    <textarea
                                        required
                                        value={formData.description}
                                        onChange={(e) => setFormData({ ...formData, description: e.target.value })}
                                        rows="5"
                                        className="w-full px-4 py-3 bg-gray-700 border border-gray-600 rounded-xl focus:border-blue-500 focus:outline-none transition-all resize-none"
                                        placeholder="Describe the issue or your request..."
                                    />
                                </div>

                                <button
                                    type="submit"
                                    className="w-full px-6 py-3 bg-gradient-to-r from-blue-600 to-purple-600 rounded-xl font-bold hover:from-blue-500 hover:to-purple-500 transition-all transform hover:scale-105"
                                >
                                    Submit Report
                                </button>
                            </form>
                        )}
                    </div>

                    {/* Quick Links */}
                    <div className="space-y-6">
                        <div className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-2xl p-8">
                            <h2 className="text-2xl font-bold mb-6">Quick Links</h2>
                            <div className="space-y-4">
                                <a href="https://github.com" target="_blank" rel="noopener noreferrer" className="flex items-center gap-3 p-4 bg-gray-700/50 rounded-xl hover:bg-gray-700 transition-all">
                                    <span className="text-2xl text-blue-400">⌘</span>
                                    <div>
                                        <div className="font-semibold">GitHub Repository</div>
                                        <div className="text-sm text-gray-400">View source code</div>
                                    </div>
                                </a>
                                <a href="mailto:support@medicalai.com" className="flex items-center gap-3 p-4 bg-gray-700/50 rounded-xl hover:bg-gray-700 transition-all">
                                    <span className="text-2xl text-blue-400">@</span>
                                    <div>
                                        <div className="font-semibold">Email Support</div>
                                        <div className="text-sm text-gray-400">support@medicalai.com</div>
                                    </div>
                                </a>
                                <a href="https://discord.com" target="_blank" rel="noopener noreferrer" className="flex items-center gap-3 p-4 bg-gray-700/50 rounded-xl hover:bg-gray-700 transition-all">
                                    <span className="text-2xl text-blue-400">#</span>
                                    <div>
                                        <div className="font-semibold">Community Discord</div>
                                        <div className="text-sm text-gray-400">Join the discussion</div>
                                    </div>
                                </a>
                            </div>
                        </div>

                        <div className="bg-gradient-to-br from-yellow-900/20 to-orange-900/20 border border-yellow-500/30 rounded-2xl p-6">
                            <h3 className="text-yellow-400 font-bold mb-2">Medical Disclaimer</h3>
                            <p className="text-sm text-gray-400">
                                This tool is for screening purposes only and should NOT replace professional medical diagnosis. Always consult a healthcare provider for medical advice.
                            </p>
                        </div>
                    </div>
                </div>

                {/* FAQs */}
                <div>
                    <h2 className="text-3xl font-bold mb-8 text-center">Frequently Asked Questions</h2>
                    <div className="space-y-4">
                        {faqs.map((faq, idx) => (
                            <details key={idx} className="bg-gray-800/50 backdrop-blur-sm border border-gray-700 rounded-xl p-6 group">
                                <summary className="font-bold cursor-pointer list-none flex items-center justify-between">
                                    {faq.q}
                                    <span className="text-gray-400 group-open:rotate-180 transition-transform">▼</span>
                                </summary>
                                <p className="mt-4 text-gray-400 leading-relaxed">{faq.a}</p>
                            </details>
                        ))}
                    </div>
                </div>
            </div>
        </div>
    )
}

export default SupportPage
