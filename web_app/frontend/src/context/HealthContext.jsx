import { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react'

const HealthContext = createContext(null)

const API_BASE = `http://${window.location.hostname}:8000`
const POLL_INTERVAL_MS = 5000   // Check every 5s
const CRASH_MISS_THRESHOLD = 4  // 4 consecutive failures = crashed

export function HealthProvider({ children }) {
    const [activeMode, setActiveMode] = useState('JAUNDICE_BODY')
    const [isNerdMode, setIsNerdMode] = useState(false)
    const [isInfant, setIsInfant] = useState(false)
    const [gpuMemory, setGpuMemory] = useState(null)
    const [lastResult, setLastResult] = useState(null)

    // Global warmup / crash state
    const [isWarm, setIsWarm] = useState(false)
    const [hasCrashed, setHasCrashed] = useState(false)
    const [healthStatus, setHealthStatus] = useState('loading') // 'loading' | 'ready' | 'crashed'
    const [statusMessage, setStatusMessage] = useState('Connecting to backend…')

    const failCount = useRef(0)
    const wasWarm = useRef(false)

    const checkHealth = useCallback(async () => {
        try {
            const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(15000) })
            const data = await res.json()

            if (data.gpu_memory) setGpuMemory(data.gpu_memory)

            if (data.models_ready) {
                failCount.current = 0
                wasWarm.current = true
                setIsWarm(true)
                setHasCrashed(false)
                setHealthStatus('ready')
                setStatusMessage('All systems ready')
            } else {
                // Backend up but workers still warming
                setStatusMessage(data.status === 'loading' ? 'Loading AI models on GPU…' : 'Workers starting…')
                // If we were previously warm and now lost it — potential crash
                if (wasWarm.current) {
                    failCount.current += 1
                    if (failCount.current >= CRASH_MISS_THRESHOLD) {
                        setHasCrashed(true)
                        setHealthStatus('crashed')
                        setStatusMessage('Worker process lost. Models may have crashed.')
                    }
                }
            }
        } catch {
            failCount.current += 1
            if (wasWarm.current && failCount.current >= CRASH_MISS_THRESHOLD) {
                setHasCrashed(true)
                setHealthStatus('crashed')
                setStatusMessage('Backend unreachable. Container may have exited.')
            } else if (!wasWarm.current) {
                setStatusMessage('Waiting for backend to start…')
            }
        }
    }, [])

    useEffect(() => {
        checkHealth()
        const id = setInterval(checkHealth, POLL_INTERVAL_MS)
        return () => clearInterval(id)
    }, [checkHealth])

    return (
        <HealthContext.Provider value={{
            activeMode, setActiveMode,
            isNerdMode, setIsNerdMode,
            isInfant, setIsInfant,
            gpuMemory, setGpuMemory,
            lastResult, setLastResult,
            isWarm, hasCrashed, healthStatus, statusMessage,
        }}>
            {children}
        </HealthContext.Provider>
    )
}

export function useHealth() {
    const ctx = useContext(HealthContext)
    if (!ctx) throw new Error('useHealth must be used inside <HealthProvider>')
    return ctx
}
