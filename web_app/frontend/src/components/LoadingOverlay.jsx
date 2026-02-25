import { Activity, AlertTriangle, RefreshCw, Zap } from 'lucide-react'

/**
 * Full-screen overlay that blocks interaction until AI models are warmed up.
 * Also shows a crash screen if the worker dies.
 */
export default function LoadingOverlay({ statusMessage, hasCrashed }) {
    if (hasCrashed) {
        return (
            <div className="absolute inset-0 z-50 bg-gray-950 flex flex-col items-center justify-center gap-6 px-6">
                {/* Crash icon */}
                <div className="relative w-24 h-24">
                    <div className="absolute inset-0 rounded-full border-2 border-red-500/20 animate-ping" />
                    <div className="absolute inset-0 flex items-center justify-center">
                        <AlertTriangle className="w-10 h-10 text-red-500" />
                    </div>
                </div>

                {/* Status */}
                <div className="flex flex-col items-center gap-3 max-w-sm text-center">
                    <h2 className="text-xl font-black text-red-400 uppercase tracking-widest">
                        Worker Crashed
                    </h2>
                    <p className="text-sm text-gray-400 leading-relaxed">
                        {statusMessage || 'The AI backend worker process has stopped responding.'}
                    </p>

                    <div className="w-full mt-2 p-4 rounded-xl text-left space-y-2"
                        style={{ background: 'rgba(239,68,68,0.06)', border: '1px solid rgba(239,68,68,0.2)' }}>
                        <p className="text-[10px] font-black uppercase tracking-widest text-red-400 mb-2">Recovery Steps</p>
                        <p className="text-xs text-gray-400"><span className="text-red-400 font-bold">1.</span> Check Docker logs for the worker container</p>
                        <p className="text-xs text-gray-400"><span className="text-red-400 font-bold">2.</span> Run <span className="font-mono bg-black/40 px-1 rounded text-red-300">docker compose restart worker-light worker-heavy</span></p>
                        <p className="text-xs text-gray-400"><span className="text-red-400 font-bold">3.</span> If crash persists, check available GPU memory</p>
                    </div>
                </div>

                {/* Retry button */}
                <button
                    onClick={() => window.location.reload()}
                    className="flex items-center gap-2 px-6 py-3 rounded-xl text-sm font-black uppercase tracking-widest transition-all active:scale-95"
                    style={{ background: 'rgba(239,68,68,0.15)', border: '1px solid rgba(239,68,68,0.4)', color: '#f87171' }}
                >
                    <RefreshCw className="w-4 h-4" />
                    Retry Connection
                </button>
            </div>
        )
    }

    return (
        <div className="absolute inset-0 z-50 bg-gray-950 flex flex-col items-center justify-center gap-6">
            {/* Pulsing ring */}
            <div className="relative w-24 h-24">
                <div className="absolute inset-0 rounded-full border-2 border-cyan-500/20 animate-ping" />
                <div className="absolute inset-2 rounded-full border-2 border-cyan-500/30 animate-pulse" />
                <div className="absolute inset-0 flex items-center justify-center">
                    <Activity className="w-8 h-8 text-cyan-500 animate-pulse" />
                </div>
            </div>

            {/* Status text */}
            <div className="flex flex-col items-center gap-2">
                <h2 className="text-lg font-black text-white uppercase tracking-widest">
                    Warming Up
                </h2>
                <p className="text-xs text-gray-500 font-mono tracking-wider animate-pulse">
                    {statusMessage || 'Loading AI models on GPU…'}
                </p>
                <div className="flex gap-1 mt-2">
                    {[0, 1, 2, 3, 4].map(i => (
                        <div
                            key={i}
                            className="w-1.5 h-1.5 rounded-full bg-cyan-500"
                            style={{
                                animation: `pulse 1.4s ease-in-out ${i * 0.2}s infinite`,
                                opacity: 0.3,
                            }}
                        />
                    ))}
                </div>
            </div>

            {/* GPU badge */}
            <div className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg"
                style={{ background: 'rgba(52,211,153,0.07)', border: '1px solid rgba(52,211,153,0.2)' }}>
                <Zap className="w-3 h-3 text-emerald-400" />
                <span className="text-[10px] font-bold uppercase tracking-widest text-emerald-400">CUDA GPU Active</span>
            </div>

            {/* Bottom hint */}
            <p className="absolute bottom-8 text-[10px] text-gray-600 font-mono uppercase tracking-widest">
                Models warm up once per session — typically 30–60 s
            </p>
        </div>
    )
}
