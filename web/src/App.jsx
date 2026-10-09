import { Navigate, Route, Routes } from 'react-router-dom'
import Home from './pages/Home.jsx'
import ClassPage from './pages/ClassPage.jsx'

// URL → page. Each <Route> shows one page component.
export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route path="/class/:classId" element={<ClassPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}
