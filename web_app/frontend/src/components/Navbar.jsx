import { Link, useLocation } from 'react-router-dom'
import { Activity, Scan, HelpCircle } from 'lucide-react'

function Navbar() {
    const location = useLocation();
    const isAuto = location.pathname === '/auto';

    return (
        <nav className="border-b border-white/5 bg-gray-950/50 backdrop-blur-xl z-50 sticky top-0 w-full">
            <div className="max-w-[1500px] mx-auto px-4 md:px-6 h-auto md:h-16 flex flex-col md:flex-row items-center justify-between py-2 md:py-0 gap-2 md:gap-0">

                {/* Logo */}
                <div className="flex items-center gap-3 w-full md:w-auto justify-between md:justify-start">
                    <div className="flex items-center gap-3">
                        <div className="w-8 h-8 md:w-9 md:h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center shadow-lg shadow-cyan-900/40 border border-white/10">
                            <Activity className="w-4 h-4 md:w-5 md:h-5 text-white" />
                        </div>
                        <span className="font-black text-lg md:text-xl bg-clip-text text-transparent bg-gradient-to-r from-cyan-400 to-blue-500 tracking-tighter">
                            MEDICAL<span className="text-white opacity-20 italic">AI</span>
                        </span>
                    </div>
                </div>

                {/* Navigation Switch - Horizontal Scroll on Mobile */}
                <div className="flex bg-white/5 rounded-xl p-1 border border-white/5 w-full md:w-auto overflow-x-auto scrollbar-hide">
                    <Link
                        to="/"
                        className={`flex-1 md:flex-none whitespace-nowrap px-4 md:px-6 py-2 rounded-lg text-[10px] md:text-xs font-black uppercase tracking-widest transition-all flex items-center justification-center gap-2 ${location.pathname === '/'
                            ? 'bg-cyan-500 text-white shadow-lg shadow-cyan-900/20'
                            : 'text-gray-500 hover:text-white hover:bg-white/5'
                            }`}
                    >
                        <Activity className="w-3 h-3" />
                        Manual
                    </Link>
                    <Link
                        to="/auto"
                        className={`flex-1 md:flex-none whitespace-nowrap px-4 md:px-6 py-2 rounded-lg text-[10px] md:text-xs font-black uppercase tracking-widest transition-all flex items-center justification-center gap-2 ${isAuto
                            ? 'bg-purple-500 text-white shadow-lg shadow-purple-900/20'
                            : 'text-gray-500 hover:text-white hover:bg-white/5'
                            }`}
                    >
                        <Scan className="w-3 h-3" />
                        Auto-Pilot
                    </Link>
                    <Link
                        to="/posture"
                        className={`flex-1 md:flex-none whitespace-nowrap px-4 md:px-6 py-2 rounded-lg text-[10px] md:text-xs font-black uppercase tracking-widest transition-all flex items-center justification-center gap-2 ${location.pathname === '/posture'
                            ? 'bg-orange-500 text-white shadow-lg shadow-orange-900/20'
                            : 'text-gray-500 hover:text-white hover:bg-white/5'
                            }`}
                    >
                        <Activity className="w-3 h-3" />
                        Posture
                    </Link>
                </div>

                {/* Right Actions */}
                <div className="hidden md:flex items-center gap-4">
                    {/* Add additional items here if needed */}
                </div>

            </div>
        </nav>
    )
}


export default Navbar
