import { Activity } from 'lucide-react'

/**
 * Full-screen overlay that blocks interaction until AI models are warmed up.
 * Shows a pulsing animation with status text.
 */
export default function LoadingOverlay() {
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
                    Loading AI models...
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

            {/* Bottom hint */}
            <p className="absolute bottom-8 text-[10px] text-gray-600 font-mono uppercase tracking-widest">
                This usually takes 15-30 seconds on first boot
            </p>
        </div>
    )
}
