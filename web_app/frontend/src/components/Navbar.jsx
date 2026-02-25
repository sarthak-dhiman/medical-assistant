import { Link, useLocation } from 'react-router-dom'
import { Activity, Scan, Cpu, FlaskConical, Stethoscope, Bone } from 'lucide-react'
import { useHealth } from '../context/HealthContext'

const ROUTES = [
    { path: '/', label: 'Manual', icon: Stethoscope, accent: '#22d3ee', bg: 'rgba(34,211,238,0.1)', border: 'rgba(34,211,238,0.3)', glow: 'rgba(34,211,238,0.2)' },
    { path: '/auto', label: 'Auto-Pilot', icon: Scan, accent: '#818cf8', bg: 'rgba(129,140,248,0.1)', border: 'rgba(129,140,248,0.3)', glow: 'rgba(129,140,248,0.2)' },
    { path: '/posture', label: 'Posture', icon: Activity, accent: '#34d399', bg: 'rgba(52,211,153,0.1)', border: 'rgba(52,211,153,0.3)', glow: 'rgba(52,211,153,0.2)' },
    { path: '/deformity', label: 'Spine & Gait', icon: Bone, accent: '#a78bfa', bg: 'rgba(167,139,250,0.1)', border: 'rgba(167,139,250,0.3)', glow: 'rgba(167,139,250,0.2)' },
]

function Navbar() {
    const location = useLocation()
    const { isNerdMode, setIsNerdMode, gpuMemory } = useHealth()

    return (
        <nav className="app-nav px-4 md:px-6" style={{ gap: 12 }}>

            {/* â”€â”€ Logo â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
            <div className="flex items-center gap-2.5 shrink-0">
                <div className="w-7 h-7 rounded-lg flex items-center justify-center shadow-lg shadow-cyan-900/30"
                    style={{ background: 'linear-gradient(135deg, #22d3ee 0%, #6366f1 100%)' }}>
                    <Stethoscope className="w-3.5 h-3.5 text-white" />
                </div>
                <span className="hidden sm:block font-black text-[13px] tracking-tight" style={{ color: 'var(--text-1)' }}>
                    MEDICAL<span style={{ color: 'var(--cyan)', opacity: 0.7, fontWeight: 300, fontStyle: 'italic' }}> AI</span>
                </span>
            </div>

            {/* Divider */}
            <div className="hidden md:block w-px h-4 shrink-0" style={{ background: 'var(--border-md)' }} />

            {/* â”€â”€ Route tabs â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
            <div className="flex items-center gap-1 rounded-xl p-1" style={{ background: 'rgba(255,255,255,0.04)', border: '1px solid rgba(255,255,255,0.06)' }}>
                {ROUTES.map(({ path, label, icon: Icon, accent, bg, border, glow }) => {
                    const active = location.pathname === path
                    return (
                        <Link
                            key={path}
                            to={path}
                            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-[11px] font-bold uppercase tracking-wider transition-all duration-200 whitespace-nowrap"
                            style={active ? {
                                background: bg,
                                border: `1px solid ${border}`,
                                color: accent,
                                boxShadow: `0 2px 16px ${glow}`,
                            } : {
                                border: '1px solid transparent',
                                color: 'var(--text-3)',
                            }}
                        >
                            <Icon className="w-3.5 h-3.5 shrink-0" />
                            <span className="hidden sm:block">{label}</span>
                        </Link>
                    )
                })}
            </div>

            {/* â”€â”€ Right: GPU + Expert â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */}
            <div className="ml-auto flex items-center gap-2 shrink-0">
                {gpuMemory && (
                    <div className="hidden md:flex items-center gap-1.5 px-2.5 py-1 rounded-lg"
                        style={{ background: 'rgba(52,211,153,0.08)', border: '1px solid rgba(52,211,153,0.2)' }}>
                        <Cpu className="w-3 h-3 shrink-0" style={{ color: 'var(--green)' }} />
                        <span className="mono text-[10px] font-bold" style={{ color: 'var(--green)' }}>{gpuMemory} MB</span>
                    </div>
                )}

                <button
                    onClick={() => setIsNerdMode(v => !v)}
                    className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-bold uppercase tracking-wider transition-all duration-200"
                    style={isNerdMode ? {
                        background: 'rgba(167,139,250,0.12)',
                        border: '1px solid rgba(167,139,250,0.35)',
                        color: 'var(--violet)',
                    } : {
                        background: 'transparent',
                        border: '1px solid rgba(255,255,255,0.09)',
                        color: 'var(--text-3)',
                    }}
                >
                    <FlaskConical className="w-3.5 h-3.5 shrink-0" />
                    <span className="hidden sm:block">Expert</span>
                </button>
            </div>
        </nav>
    )
}

export default Navbar
