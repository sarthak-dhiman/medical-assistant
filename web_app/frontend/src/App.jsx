import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar from './components/Navbar'
import ManualDetection from './pages/ManualDetection'
import AutomaticDetection from './pages/AutomaticDetection'
import PostureDetection from './pages/PostureDetection'

function App() {
  return (
    <BrowserRouter>
      <div className="flex flex-col h-[100dvh] bg-gray-950 text-white font-sans overflow-hidden">
        {/* Navigation Bar */}
        <Navbar />

        {/* Content Area */}
        <div className="flex-1 overflow-hidden relative">
          <Routes>
            <Route path="/" element={<ManualDetection />} />
            <Route path="/auto" element={<AutomaticDetection />} />
            <Route path="/posture" element={<PostureDetection />} />
          </Routes>
        </div>
      </div>
    </BrowserRouter>
  )
}

export default App
