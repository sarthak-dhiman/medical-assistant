const ResultDisplay = ({ result, mode, isNerdMode }) => {
    if (!result) return null;

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
                className={`absolute border-2 ${color} rounded-lg flex flex-col items-start`}
                style={{
                    left: `${x1 * 100}%`,
                    top: `${y1 * 100}%`,
                    width: `${width * 100}%`,
                    height: `${height * 100}%`,
                }}
            >
                <div className={`-mt-6 px-2 py-0.5 text-xs font-bold text-black ${color.replace('border', 'bg')} rounded`}>
                    {label} {conf && `(${Math.round(conf * 100)}%)`}
                </div>
            </div>
        );
    };

    return (
        <>
            {/* Show Bounding Boxes ONLY in Nerd Mode */}
            {isNerdMode && (
                <>
                    {/* Single BBox (Body/Skin) */}
                    {bbox && (
                        <Box
                            x1={bbox[0]} y1={bbox[1]} x2={bbox[2]} y2={bbox[3]}
                            label={result.label}
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

            {/* DEBUG: Show Processed Eye Input (Only in Nerd Mode) */}
            {isNerdMode && result.debug_image && (
                <div className="absolute bottom-20 right-4 w-24 h-24 bg-black/50 border border-white/20 rounded-lg overflow-hidden backdrop-blur-sm">
                    <img src={result.debug_image} alt="Debug AI View" className="w-full h-full object-contain" />
                    <span className="absolute bottom-0 left-0 w-full text-[10px] text-center bg-black/70 text-gray-300">AI View</span>
                </div>
            )}

            {/* Contextual Warnings (Blurry, etc) */}
            {warning && (
                <div className="absolute top-24 left-1/2 transform -translate-x-1/2 bg-yellow-900/80 border border-yellow-500/50 text-yellow-100 px-4 py-2 rounded-lg backdrop-blur-sm flex items-center gap-2 shadow-lg z-40">
                    <svg className="w-5 h-5 text-yellow-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
                    </svg>
                    <span className="text-xs font-medium">{warning}</span>
                </div>
            )}
        </>
    );
};

export default ResultDisplay;
