import { User, Eye, Stethoscope, ChevronRight } from 'lucide-react'

const ModelSelector = ({ activeMode, setActiveMode, isMobile }) => {
    const modes = [
        {
            id: 'JAUNDICE_BODY',
            label: 'Jaundice Baby',
            shortLabel: 'Baby',
            icon: <User className="w-4 h-4" />,
            color: 'from-yellow-400 to-orange-500',
            textColor: 'text-yellow-400',
            bgColor: 'bg-yellow-500/10',
            borderColor: 'border-yellow-500/20'
        },
        {
            id: 'JAUNDICE_EYE',
            label: 'Jaundice Adult',
            shortLabel: 'Adult',
            icon: <Eye className="w-4 h-4" />,
            color: 'from-amber-400 to-yellow-600',
            textColor: 'text-amber-400',
            bgColor: 'bg-amber-500/10',
            borderColor: 'border-amber-500/20'
        },
        {
            id: 'SKIN_DISEASE',
            label: 'Skin Disease',
            shortLabel: 'Skin',
            icon: <Stethoscope className="w-4 h-4" />,
            color: 'from-pink-400 to-rose-600',
            textColor: 'text-rose-400',
            bgColor: 'bg-rose-500/10',
            borderColor: 'border-rose-500/20'
        },
        {
            id: 'BURNS',
            label: 'Burns Detection',
            shortLabel: 'Burns',
            icon: <Stethoscope className="w-4 h-4" />,
            color: 'from-red-500 to-orange-700',
            textColor: 'text-red-400',
            bgColor: 'bg-red-500/10',
            borderColor: 'border-red-500/20'
        },
        {
            id: 'NAIL_DISEASE',
            label: 'Nail Disease',
            shortLabel: 'Nail',
            icon: <Stethoscope className="w-4 h-4" />,
            color: 'from-indigo-400 to-blue-600',
            textColor: 'text-indigo-400',
            bgColor: 'bg-indigo-500/10',
            borderColor: 'border-indigo-500/20'
        },
        // Cataract removed
        {
            id: 'ORAL_CANCER',
            label: 'Oral Cancer',
            shortLabel: 'Oral',
            icon: <Stethoscope className="w-4 h-4" />,
            color: 'from-emerald-400 to-teal-600',
            textColor: 'text-emerald-400',
            bgColor: 'bg-emerald-500/10',
            borderColor: 'border-emerald-500/20'
        },
        {
            id: 'TEETH',
            label: 'Teeth Disease',
            shortLabel: 'Teeth',
            icon: <Stethoscope className="w-4 h-4" />,
            color: 'from-slate-300 to-gray-500',
            textColor: 'text-slate-300',
            bgColor: 'bg-slate-500/10',
            borderColor: 'border-slate-500/20'
        },
    ]

    if (isMobile) {
        return (
            <div className="w-full h-24 pointer-events-auto">
                <div className="flex gap-3 overflow-x-auto pb-4 pt-1 px-4 no-scrollbar scroll-smooth snap-x">
                    {modes.map((m) => (
                        <button
                            key={m.id}
                            onClick={() => setActiveMode(m.id)}
                            className={`flex-shrink-0 snap-center w-28 h-20 rounded-2xl flex flex-col items-center justify-center gap-1.5 transition-all duration-300 border ${activeMode === m.id
                                ? `${m.borderColor} ${m.bgColor} scale-105 shadow-lg shadow-black/40`
                                : 'bg-gray-900/40 border-white/5 text-gray-500'
                                }`}
                        >
                            <div className={`p-1 rounded-xl ${activeMode === m.id ? `bg-gradient-to-tr ${m.color} text-white` : 'bg-white/5'}`}>
                                {m.icon}
                            </div>
                            <span className={`text-[9px] font-black uppercase tracking-tighter ${activeMode === m.id ? m.textColor : ''}`}>
                                {m.shortLabel}
                            </span>
                            {activeMode === m.id && (
                                <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-8 h-1 bg-white rounded-full blur-[1px]"></div>
                            )}
                        </button>
                    ))}
                </div>
            </div>
        )
    }

    // Desktop Grid View
    return (
        <div className="grid grid-cols-2 gap-3">
            {modes.map((m) => (
                <button
                    key={m.id}
                    onClick={() => setActiveMode(m.id)}
                    className={`group relative flex flex-col items-start p-3 rounded-xl transition-all duration-300 border overflow-hidden ${activeMode === m.id
                        ? `${m.borderColor} ${m.bgColor} shadow-xl shadow-black/20 translate-y-[-1px]`
                        : 'bg-gray-900/10 border-white/5 hover:border-white/10 hover:bg-white/5'
                        }`}
                >
                    {/* Active Gradient Background */}
                    <div className={`absolute inset-0 opacity-0 group-hover:opacity-5 transition-opacity duration-300 bg-gradient-to-br ${m.color}`} />

                    <div className="flex justify-between items-center w-full mb-2">
                        <div className={`p-1.5 rounded-lg transition-all duration-300 ${activeMode === m.id ? `bg-gradient-to-tr ${m.color} text-white shadow-lg` : 'bg-gray-800 text-gray-400 group-hover:text-gray-200'}`}>
                            {m.icon}
                        </div>
                        {activeMode === m.id && (
                            <div className="flex items-center gap-1 bg-white/5 px-2 py-0.5 rounded-full">
                                <span className={`text-[7px] font-black uppercase tracking-widest ${m.textColor}`}>Active</span>
                                <div className={`w-0.5 h-0.5 rounded-full animate-pulse bg-current ${m.textColor}`} />
                            </div>
                        )}
                    </div>

                    <div className="space-y-0 z-10">
                        <span className={`text-[8px] font-bold uppercase tracking-widest opacity-40 ${activeMode === m.id ? m.textColor : 'text-white'}`}>Mode</span>
                        <h4 className={`text-xs font-black tracking-tight transition-colors ${activeMode === m.id ? 'text-white' : 'text-gray-400 group-hover:text-gray-200'}`}>
                            {m.label}
                        </h4>
                    </div>

                    <ChevronRight className={`absolute bottom-4 right-4 w-4 h-4 transition-all duration-300 ${activeMode === m.id ? `opacity-100 ${m.textColor} translate-x-0` : 'opacity-0 -translate-x-2'}`} />
                </button>
            ))}
        </div>
    )
}

export default ModelSelector
