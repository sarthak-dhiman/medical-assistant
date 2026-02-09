import { Link, useLocation } from 'react-router-dom'

const Navigation = () => {
    const location = useLocation()

    const navItems = [
        { path: '/', label: 'Home', icon: '🏠' },
        { path: '/detect', label: 'Detection', icon: '🔬' },
        { path: '/how-it-works', label: 'How It Works', icon: '⚙️' },
        { path: '/about', label: 'About', icon: 'ℹ️' },
        { path: '/support', label: 'Support', icon: '💬' }
    ]

    return (
        <nav className="sticky top-0 z-50 bg-gray-900/80 backdrop-blur-lg border-b border-gray-800">
            <div className="max-w-7xl mx-auto px-4">
                <div className="flex items-center justify-between h-16">
                    {/* Logo */}
                    <Link to="/" className="flex items-center gap-2 font-bold text-xl">
                        <span className="text-2xl">🏥</span>
                        <span className="bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                            Medical AI
                        </span>
                    </Link>

                    {/* Desktop Navigation */}
                    <div className="hidden md:flex items-center gap-1">
                        {navItems.map((item) => (
                            <Link
                                key={item.path}
                                to={item.path}
                                className={`px-4 py-2 rounded-lg font-medium transition-all ${location.pathname === item.path
                                        ? 'bg-blue-600 text-white'
                                        : 'text-gray-400 hover:text-white hover:bg-gray-800'
                                    }`}
                            >
                                <span className="mr-2">{item.icon}</span>
                                {item.label}
                            </Link>
                        ))}
                    </div>

                    {/* Mobile Menu Button */}
                    <button className="md:hidden p-2 text-gray-400 hover:text-white">
                        <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
                        </svg>
                    </button>
                </div>

                {/* Mobile Navigation */}
                <div className="md:hidden pb-4 space-y-1">
                    {navItems.map((item) => (
                        <Link
                            key={item.path}
                            to={item.path}
                            className={`block px-4 py-2 rounded-lg font-medium transition-all ${location.pathname === item.path
                                    ? 'bg-blue-600 text-white'
                                    : 'text-gray-400 hover:text-white hover:bg-gray-800'
                                }`}
                        >
                            <span className="mr-2">{item.icon}</span>
                            {item.label}
                        </Link>
                    ))}
                </div>
            </div>
        </nav>
    )
}

export default Navigation
