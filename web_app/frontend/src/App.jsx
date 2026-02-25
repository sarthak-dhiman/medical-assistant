import { BrowserRouter, Routes, Route } from 'react-router-dom'
import { HealthProvider, useHealth } from './context/HealthContext'
import Navbar from './components/Navbar'
import LoadingOverlay from './components/LoadingOverlay'
import ManualDetection from './pages/ManualDetection'
import AutomaticDetection from './pages/AutomaticDetection'
import PostureDetection from './pages/PostureDetection'
import PostureDeformity from './pages/PostureDeformity'

/** Global warmup/crash gate — sits *inside* HealthProvider so it can read context */
function AppGate() {
  const { isWarm, hasCrashed, statusMessage } = useHealth()

  // Block ALL routes until models are warm
  if (!isWarm || hasCrashed) {
    return (
      <div style={{ height: '100dvh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
        <div className="relative flex-1">
          <LoadingOverlay statusMessage={statusMessage} hasCrashed={hasCrashed} />
        </div>
      </div>
    )
  }

  return (
    <div style={{ height: '100dvh', display: 'flex', flexDirection: 'column', overflow: 'hidden' }}>
      <Navbar />
      <main className="app-content">
        <Routes>
          <Route path="/" element={<ManualDetection />} />
          <Route path="/auto" element={<AutomaticDetection />} />
          <Route path="/posture" element={<PostureDetection />} />
          <Route path="/deformity" element={<PostureDeformity />} />
        </Routes>
      </main>
    </div>
  )
}

function App() {
  return (
    <HealthProvider>
      <BrowserRouter>
        <AppGate />
      </BrowserRouter>
    </HealthProvider>
  )
}

export default App
