import { Baby, Eye, ScanLine, Hand, Microscope, Smile } from 'lucide-react'

const MODES = [
    {
        id: 'JAUNDICE_BODY',
        label: 'Jaundice Body',
        sub: 'Skin pigmentation',
        icon: Baby,
        accent: '#fbbf24',
        bg: 'rgba(251,191,36,0.08)',
        border: 'rgba(251,191,36,0.22)',
    },
    {
        id: 'JAUNDICE_EYE',
        label: 'Jaundice Eye',
        sub: 'Scleral icterus',
        icon: Eye,
        accent: '#60a5fa',
        bg: 'rgba(96,165,250,0.08)',
        border: 'rgba(96,165,250,0.22)',
    },
    {
        id: 'SKIN_DISEASE',
        label: 'Skin Disease',
        sub: 'Dermatological AI',
        icon: ScanLine,
        accent: '#f472b6',
        bg: 'rgba(244,114,182,0.08)',
        border: 'rgba(244,114,182,0.22)',
    },
    {
        id: 'NAIL_DISEASE',
        label: 'Nail Disease',
        sub: 'Onychopathy scan',
        icon: Hand,
        accent: '#818cf8',
        bg: 'rgba(129,140,248,0.08)',
        border: 'rgba(129,140,248,0.22)',
    },
    {
        id: 'ORAL_CANCER',
        label: 'Oral Cancer',
        sub: 'Mucosal screening',
        icon: Microscope,
        accent: '#34d399',
        bg: 'rgba(52,211,153,0.08)',
        border: 'rgba(52,211,153,0.22)',
    },
    {
        id: 'TEETH',
        label: 'Dental Health',
        sub: 'Dentition analysis',
        icon: Smile,
        accent: '#fb923c',
        bg: 'rgba(251,146,60,0.08)',
        border: 'rgba(251,146,60,0.22)',
    },
]

export const getModeAccent = (id) => MODES.find(m => m.id === id)?.accent ?? '#22d3ee'
export const getModeLabel  = (id) => MODES.find(m => m.id === id)?.label  ?? id

const ModelSelector = ({ activeMode, setActiveMode, isMobile }) => {
    if (isMobile) {
        // Horizontal scroll chips for mobile
        return (
            <div className="flex gap-2 overflow-x-auto no-scrollbar pb-1 pt-0.5">
                {MODES.map(({ id, label, icon: Icon, accent, bg, border }) => {
                    const active = activeMode === id
                    return (
                        <button
                            key={id}
                            onClick={() => setActiveMode(id)}
                            className="flex-shrink-0 flex items-center gap-2 px-3 py-2 rounded-xl text-[11px] font-bold uppercase tracking-wide transition-all duration-200"
                            style={active ? { background: bg, border: `1px solid ${border}`, color: accent } : {
                                background: 'rgba(255,255,255,0.04)',
                                border: '1px solid rgba(255,255,255,0.07)',
                                color: 'var(--text-3)',
                            }}
                        >
                            <Icon style={{ width: 12, height: 12 }} />
                            {label.split(' â€” ')[0]}
                        </button>
                    )
                })}
            </div>
        )
    }

    // Desktop list view
    return (
        <div className="flex flex-col gap-1">
            {MODES.map(({ id, label, sub, icon: Icon, accent, bg, border }) => {
                const active = activeMode === id
                return (
                    <button
                        key={id}
                        onClick={() => setActiveMode(id)}
                        className="group relative flex items-center gap-3 w-full px-3 py-2.5 rounded-xl text-left transition-all duration-200 overflow-hidden"
                        style={active ? {
                            background: bg,
                            border: `1px solid ${border}`,
                        } : {
                            background: 'transparent',
                            border: '1px solid transparent',
                        }}
                    >
                        {/* Left accent bar */}
                        <div className="absolute left-0 top-2 bottom-2 w-0.5 rounded-full transition-all duration-200"
                            style={{ background: active ? accent : 'transparent' }} />

                        {/* Icon */}
                        <div className="w-7 h-7 rounded-lg flex items-center justify-center flex-shrink-0 transition-all duration-200"
                            style={{ background: active ? `${accent}22` : 'rgba(255,255,255,0.06)' }}>
                            <Icon style={{ width: 13, height: 13, color: active ? accent : 'var(--text-3)' }} />
                        </div>

                        {/* Labels */}
                        <div className="flex flex-col min-w-0 flex-1">
                            <span className="text-[11px] font-bold leading-tight truncate transition-colors duration-200"
                                style={{ color: active ? 'var(--text-1)' : 'var(--text-2)' }}>
                                {label}
                            </span>
                            <span className="text-[9px] leading-tight mt-0.5"
                                style={{ color: active ? accent : 'var(--text-3)', opacity: active ? 0.8 : 1 }}>
                                {sub}
                            </span>
                        </div>

                        {/* Active dot */}
                        {active && (
                            <div className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                                style={{ background: accent, boxShadow: `0 0 6px ${accent}` }} />
                        )}
                    </button>
                )
            })}
        </div>
    )
}

export default ModelSelector
