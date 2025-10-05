
import React from 'react';
import PoseAnalyzer from './components/PoseAnalyzer';
import { GithubIcon } from './components/Icons';
import { BrowserRouter, Routes, Route, Link, Navigate } from 'react-router-dom';

const MonitorCashiers = React.lazy(() => import('./components/pages/MonitorCashiers'));
const AdminResults = React.lazy(() => import('./components/pages/AdminResults'));

const App: React.FC = () => {
  return (
    <BrowserRouter>
      <div className="bg-gray-900 text-gray-100 min-h-screen flex flex-col font-sans">
        <header className="bg-gray-800/50 backdrop-blur-sm shadow-lg p-4 w-full z-10 border-b border-gray-700">
          <div className="container mx-auto flex justify-between items-center">
            <h1 className="text-2xl font-bold text-cyan-400">
              <span className="animate-pulse">🏃</span> Human Pose Analyzer
            </h1>
            <nav className="flex items-center gap-4 text-sm">
              <Link className="hover:text-cyan-400" to="/">Home</Link>
              <Link className="hover:text-cyan-400" to="/monitor">Cashier Pose</Link>
              <Link className="hover:text-cyan-400" to="/admin">Admin</Link>
            </nav>
          </div>
        </header>

        <main className="flex-grow container mx-auto p-4 md:p-8 flex flex-col">
          <React.Suspense fallback={<div>Loading...</div>}>
            <Routes>
              <Route path="/" element={<PoseAnalyzer />} />
              <Route path="/monitor" element={<MonitorCashiers />} />
              <Route path="/admin" element={<AdminResults />} />
              <Route path="*" element={<Navigate to="/" replace />} />
            </Routes>
          </React.Suspense>
        </main>

        <footer className="bg-gray-800/30 text-center p-4 text-gray-500 text-sm border-t border-gray-700">
          <p>Powered by React, Tailwind CSS, and Google's MediaPipe</p>
        </footer>
      </div>
    </BrowserRouter>
  );
};

export default App;
