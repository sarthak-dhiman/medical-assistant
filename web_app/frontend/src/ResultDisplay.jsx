import { useState, useEffect } from 'react'

const ResultDisplay = ({ result, mode, isNerdMode }) => {
    // Check what we receive
    if (isNerdMode && result) {
        console.log("🤓 Nerd Mode Result:", {
            mode,
            label: result.label,
            bbox: result.bbox,
            debug_info_keys: Object.keys(result.debug_info || {})
        });
    }
    const [viewMode, setViewMode] = useState('MASKS');

    // Auto-switch viewMode only when the target mode changes
    useEffect(() => {
        if (result?.debug_info?.grad_cam && !result?.debug_info?.masks) {
            setViewMode('HEATMAP');
        } else {
            setViewMode('MASKS');
        }
    }, [mode]); // Triggers on mode change, not every result update
    if (!result) return null;

    if (result.status === 'error') {
        return (
            <div className="absolute top-24 left-1/2 transform -translate-x-1/2 bg-red-900/80 border border-red-500 text-white px-4 py-2 rounded-lg backdrop-blur-sm flex items-center gap-2 shadow-lg z-50">
                <span className="text-xs font-bold font-mono">⚠️ {result.error}</span>
            </div>
        );
    }

    // Check for warnings (e.g. Blurry Image / Fallback Mode)
    // We check if ANY eye has a warning
    const warning = result.eyes?.find(e => e.warning)?.warning;

    // Safety check for array existence
    const eyes = result.eyes || [];
    const bbox = result.bbox;

    // Helper to draw box style
    const Box = ({ x1, y1, x2, y2, label, color, conf }) => {
        // Backend now returns NORMALIZED coordinates (0.0 to 1.0)
        const width = x2 - x1;
        const height = y2 - y1;

        return (
            <div
                className={`absolute border-4 ${color} rounded-lg flex flex-col items-start z-50`}
                style={{
                    left: `${x1 * 100}%`,
                    top: `${y1 * 100}%`,
                    width: `${width * 100}%`,
                    height: `${height * 100}%`,
                }}
            >
                <div className={`px-2 py-0.5 text-[10px] font-bold text-black ${color.replace('border', 'bg')} rounded shadow-lg`}
                    style={{
                        transform: y1 < 0.1 ? 'translateY(100%)' : 'translateY(-110%)',
                        whiteSpace: 'nowrap'
                    }}>
                    {label} {conf && `(${Math.round(conf * 100)}%)`}
                </div>
            </div>
        );
    };

    return (
        <>
            {/* Show Bounding Boxes Only when NOT showing Masks/Heatmap (to avoid clutter) OR if user prefers both */}
            {isNerdMode && (
                <>
                    {/* --- VISUALIZATION LAYERS --- */}

                    {/* LAYER 1: SEGMENTATION MASKS */}
                    {viewMode === 'MASKS' && (
                        <>
                            {/* Overall Skin Mask */}
                            {result.debug_info?.masks?.skin_mask && (
                                <div className="absolute inset-0 z-10 opacity-30 pointer-events-none">
                                    <img src={result.debug_info.masks.skin_mask} className="w-full h-full object-cover" alt="Skin Mask" />
                                </div>
                            )}
                            {/* Person Mask (for Jaundice Body) */}
                            {result.debug_info?.masks?.person_mask && (
                                <div className="absolute inset-0 z-0 opacity-20 pointer-events-none mix-blend-overlay">
                                    <img src={result.debug_info.masks.person_mask} className="w-full h-full object-cover" alt="Person Mask" />
                                </div>
                            )}
                        </>
                    )}

                    {/* LAYER 2: GRAD-CAM HEATMAP */}
                    {viewMode === 'HEATMAP' && result.debug_info?.grad_cam && (
                        <div className="absolute inset-0 z-10 opacity-60 pointer-events-none mix-blend-screen">
                            <img src={result.debug_info.grad_cam} className="w-full h-full object-cover" alt="AI Attention Map" />
                        </div>
                    )}

                    {/* Single BBox (Body/Skin) */}
                    {bbox && (
                        <Box
                            x1={bbox[0]} y1={bbox[1]} x2={bbox[2]} y2={bbox[3]}
                            label={(result.label || '').replace(/unknown_normal/gi, 'Normal')}
                            conf={result.confidence}
                            color={(result.label || '').includes('Jaundice') ? 'border-red-500' : 'border-green-500'}
                        />
                    )}

                    {/* Multiple Eyes */}
                    {eyes.map((eye, idx) => (
                        <Box
                            key={idx}
                            x1={eye.bbox[0]} y1={eye.bbox[1]} x2={eye.bbox[2]} y2={eye.bbox[3]}
                            label={eye.label}
                            conf={eye.confidence}
                            color={(eye.label || '').includes('Jaundice') ? 'border-red-500' : 'border-green-500'}
                        />
                    ))}
                </>
            )}

            {/* NERD MODE: Stats Overlay */}
            {isNerdMode && result.debug_info && (
                <div className="hidden md:block absolute top-20 left-4 w-64 bg-black/80 border border-purple-500/30 rounded-xl p-3 backdrop-blur-md z-50 text-[10px] font-mono text-gray-300 shadow-xl pointer-events-auto">
                    <div className="flex justify-between items-center mb-2 border-b border-purple-500/20 pb-1">
                        <h4 className="text-purple-400 font-bold uppercase tracking-wider">Nerd Stats</h4>
                        {/* View Toggle */}
                        <div className="flex bg-gray-900 rounded p-0.5 gap-1">
                            <button
                                onClick={() => setViewMode('MASKS')}
                                className={`px-2 py-0.5 rounded text-[9px] font-bold transition-colors ${viewMode === 'MASKS' ? 'bg-purple-600 text-white' : 'text-gray-500 hover:text-gray-300'}`}
                            >
                                MASKS
                            </button>
                            <button
                                onClick={() => setViewMode('HEATMAP')}
                                className={`px-2 py-0.5 rounded text-[9px] font-bold transition-colors ${viewMode === 'HEATMAP' ? 'bg-red-600 text-white' : 'text-gray-500 hover:text-gray-300'}`}
                            >
                                HEAT
                            </button>
                        </div>
                    </div>

                    {/* Result Info */}
                    <div className="mb-3 space-y-1">
                        <div className="flex justify-between">
                            <span className="text-gray-500 italic">Label:</span>
                            <span className="text-white font-bold">{result.label}</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-500 italic">Task/Debug Keys:</span>
                            <span className="text-cyan-400">[{Object.keys(result).slice(0, 3).join(',')}] / [{Object.keys(result.debug_info || {}).slice(0, 3).join(',')}]</span>
                        </div>
                        <div className="flex justify-between">
                            <span className="text-gray-500 italic">Raw BBox:</span>
                            <span className="text-yellow-400">{result.bbox ? `[${result.bbox.map(b => b.toFixed(2)).join(',')}]` : 'None'}</span>
                        </div>
                    </div>

                    {/* Color Stats */}
                    {result.debug_info.color_stats && (
                        <div className="mb-3">
                            <h5 className="text-gray-400 font-bold mb-1">Mean Color (Sclera/Skin)</h5>
                            <div className="flex gap-2 items-center bg-gray-900 p-2 rounded border border-gray-700">
                                <div
                                    className="w-8 h-8 rounded border border-white/20 shadow-inner"
                                    style={{
                                        backgroundColor: `rgb(${result.debug_info.color_stats.mean_rgb.join(',')})`
                                    }}
                                />
                                <div className="space-y-0.5">
                                    <div>RGB: {result.debug_info.color_stats.mean_rgb.join(', ')}</div>
                                    <div>HSV: {result.debug_info.color_stats.mean_hsv.join(', ')}</div>
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Raw Probability (Binary Models) */}
                    {result.debug_info.raw_probability !== undefined && (
                        <div className="mb-3">
                            <h5 className="text-gray-400 font-bold mb-1">Raw Confidence</h5>
                            <div className="text-gray-300 font-mono text-xs">
                                <div className="flex justify-between mb-1">
                                    <span>{result.debug_info.raw_probability.toFixed(4)}</span>
                                    <span className={result.debug_info.raw_probability > 0.5 ? "text-red-400" : "text-green-400"}>
                                        {result.debug_info.raw_probability > 0.5 ? "POSITIVE" : "NEGATIVE"}
                                    </span>
                                </div>
                                <div className="w-full bg-gray-700 h-2 rounded overflow-hidden">
                                    <div
                                        className={`h-full ${result.debug_info.raw_probability > 0.5 ? 'bg-red-500' : 'bg-green-500'}`}
                                        style={{ width: `${result.debug_info.raw_probability * 100}%` }}
                                    />
                                </div>
                            </div>
                        </div>
                    )}

                    {/* Top-3 Props (Skin Disease) */}
                    {result.debug_info.top_3 && (
                        <div>
                            <h5 className="text-gray-400 font-bold mb-1">Top 3 Classes</h5>
                            <div className="space-y-1">
                                {result.debug_info.top_3.map((item, i) => (
                                    <div key={i} className="flex justify-between items-center text-gray-300">
                                        <span>{item.label}</span>
                                        <span className={i === 0 ? "text-green-400 font-bold" : "text-gray-500"}>
                                            {(item.confidence * 100).toFixed(1)}%
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    )}
                </div >
            )}

            {/* DEBUG: Show Processed Eye Input (Only in Nerd Mode) */}
            {
                isNerdMode && result.debug_image && (
                    <div className="hidden md:block absolute bottom-20 right-4 w-24 h-24 bg-black/50 border border-white/20 rounded-lg overflow-hidden backdrop-blur-sm shadow-lg z-50">
                        <img src={result.debug_image} alt="Debug AI View" className="w-full h-full object-contain" />
                        <span className="absolute bottom-0 left-0 w-full text-[9px] font-bold text-center bg-black/80 text-white py-0.5">AI Input View</span>
                    </div>
                )
            }

            {/* Contextual Warnings (Blurry, etc) */}
            {
                warning && (
                    <div className="absolute top-24 left-1/2 transform -translate-x-1/2 bg-yellow-900/80 border border-yellow-500/50 text-yellow-100 px-4 py-2 rounded-lg backdrop-blur-sm flex items-center gap-2 shadow-lg z-40">
                        <svg className="w-5 h-5 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                        </svg>
                        <span className="text-xs font-medium">{warning}</span>
                    </div>
                )
            }
        </>
    );
};

export default ResultDisplay;
