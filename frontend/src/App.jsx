import { useState, useEffect, useRef } from 'react'
import './App.css'

function App() {
  const [connected, setConnected] = useState(false)
  const [cameraActive, setCameraActive] = useState(false)
  const [predictions, setPredictions] = useState(null)
  const [gallery, setGallery] = useState([])
  const [smileThreshold, setSmileThreshold] = useState(0.15)
  const [systemInfo, setSystemInfo] = useState(null)
  const [error, setError] = useState(null)
  const [frameCount, setFrameCount] = useState(0)
  
  const wsRef = useRef(null)
  const videoRef = useRef(null)

  // Fetch gallery
  const fetchGallery = async () => {
    try {
      const response = await fetch('/api/gallery')
      const data = await response.json()
      if (data.success) {
        setGallery(data.images)
      }
    } catch (error) {
      console.error('Error fetching gallery:', error)
    }
  }

  // Fetch system info
  const fetchSystemInfo = async () => {
    try {
      const response = await fetch('/api/system-info')
      const data = await response.json()
      setSystemInfo(data)
    } catch (error) {
      console.error('Error fetching system info:', error)
    }
  }

  // Start camera
  const startCamera = () => {
    if (wsRef.current) {
      console.log('WebSocket already connected')
      return
    }

    console.log('Connecting to WebSocket...')
    setError(null)
    
    const ws = new WebSocket('ws://localhost:8000/ws')
    
    ws.onopen = () => {
      console.log('✅ WebSocket connected successfully')
      setConnected(true)
      setCameraActive(true)
      setError(null)
    }

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data)
        
        if (data.type === 'frame') {
          setFrameCount(prev => prev + 1)
          
          if (videoRef.current && data.frame) {
            videoRef.current.src = `data:image/jpeg;base64,${data.frame}`
          }
          
          if (data.predictions) {
            setPredictions(data.predictions)
          }
        } else if (data.type === 'auto_capture' || data.type === 'capture_success') {
          console.log('Image captured!')
          fetchGallery()
        } else if (data.type === 'error') {
          console.error('Backend error:', data.message)
          setError(data.message)
        }
      } catch (error) {
        console.error('Error parsing WebSocket message:', error)
      }
    }

    ws.onerror = (error) => {
      console.error('❌ WebSocket error:', error)
      setError('WebSocket connection error')
    }

    ws.onclose = (event) => {
      console.log('WebSocket disconnected', event.code, event.reason)
      setConnected(false)
      setCameraActive(false)
      wsRef.current = null
      setFrameCount(0)
      setPredictions(null)
      
      // Clear video feed on disconnect
      if (videoRef.current) {
        videoRef.current.src = ''
      }
      
      if (event.code !== 1000) {
        setError('Connection closed unexpectedly')
      }
    }

    wsRef.current = ws
  }

  // Stop camera
  const stopCamera = () => {
    if (wsRef.current) {
      console.log('Stopping camera...')
      wsRef.current.send(JSON.stringify({ type: 'stop' }))
      wsRef.current.close()
      wsRef.current = null
    }
    
    // Clear video feed immediately
    if (videoRef.current) {
      videoRef.current.src = ''
    }
    
    // Reset state
    setPredictions(null)
    setFrameCount(0)
    setError(null)
  }

  // Capture photo
  const capturePhoto = () => {
    if (wsRef.current && connected) {
      console.log('Capturing photo...')
      wsRef.current.send(JSON.stringify({ type: 'capture' }))
    }
  }

  // Update smile threshold
  const updateSmileThreshold = async (value) => {
    setSmileThreshold(value)
    
    try {
      await fetch('/api/settings/smile-threshold', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ threshold: value })
      })
      
      if (wsRef.current && connected) {
        wsRef.current.send(JSON.stringify({
          type: 'settings',
          smile_threshold: value
        }))
      }
    } catch (error) {
      console.error('Error updating threshold:', error)
    }
  }

  // Delete image
  const deleteImage = async (filename) => {
    try {
      await fetch(`/api/gallery/${filename}`, { method: 'DELETE' })
      fetchGallery()
    } catch (error) {
      console.error('Error deleting image:', error)
    }
  }

  // Clear gallery
  const clearGallery = async () => {
    if (confirm('Delete all captured images?')) {
      try {
        await fetch('/api/gallery', { method: 'DELETE' })
        fetchGallery()
      } catch (error) {
        console.error('Error clearing gallery:', error)
      }
    }
  }

  // Load gallery and system info on mount
  useEffect(() => {
    fetchGallery()
    fetchSystemInfo()
    
    const interval = setInterval(fetchSystemInfo, 2000)
    return () => clearInterval(interval)
  }, [])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (wsRef.current) {
        wsRef.current.close()
      }
    }
  }, [])

  return (
    <div className="app">
      <header className="app-header">
        <h1>Smilage Selfie Capture</h1>
        <p>AI-Based Image Analysis Tool for Smile & Age Prediction</p>
      </header>

      <div className="main-content">
        <div className="camera-section">
          <div className="video-container">
            {cameraActive ? (
              <img ref={videoRef} className="video-feed" alt="Video feed" />
            ) : (
              <div style={{
                width: '100%',
                aspectRatio: '4/3',
                background: '#000',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                color: '#666',
                fontSize: '1.2rem',
                borderRadius: '10px'
              }}>
                📹 Camera Inactive - Click "Start Camera" to begin
              </div>
            )}
            <div className="video-overlay">
              <span className={`status-indicator ${cameraActive ? 'active' : 'inactive'}`}></span>
              {cameraActive ? `Camera Active (${frameCount} frames)` : 'Camera Inactive'}
            </div>
            {error && (
              <div style={{
                position: 'absolute',
                top: '50%',
                left: '50%',
                transform: 'translate(-50%, -50%)',
                background: 'rgba(255, 0, 0, 0.9)',
                padding: '20px',
                borderRadius: '10px',
                color: 'white',
                zIndex: 10
              }}>
                ⚠️ {error}
              </div>
            )}
          </div>

          <div className="controls">
            {!cameraActive ? (
              <button className="btn btn-primary" onClick={startCamera}>
                ▶ Start Camera
              </button>
            ) : (
              <>
                <button className="btn btn-danger" onClick={stopCamera}>
                  ⏹ Stop Camera
                </button>
                <button className="btn btn-success" onClick={capturePhoto}>
                  📸 Capture
                </button>
              </>
            )}
          </div>

          {predictions && predictions.faces && predictions.faces.length > 0 && (
            <div className="predictions">
              <h3>Live Predictions</h3>
              {predictions.faces.map((face, idx) => (
                <div key={idx}>
                  <div className="prediction-item">
                    <span className="prediction-label">Age:</span>
                    <span className="prediction-value">{face.age}</span>
                  </div>
                  <div className="prediction-item">
                    <span className="prediction-label">Gender:</span>
                    <span className="prediction-value">{face.gender}</span>
                  </div>
                  <div className="prediction-item">
                    <span className="prediction-label">Emotion:</span>
                    <span className="prediction-value">{face.emotion}</span>
                  </div>
                  <div className="prediction-item">
                    <span className="prediction-label">Smile Score:</span>
                    <span className={`prediction-value ${face.is_smiling ? 'smiling' : ''}`}>
                      {face.smile_score ? face.smile_score.toFixed(2) : '0.00'}
                    </span>
                  </div>
                  <div className="prediction-item">
                    <span className="prediction-label">Image Quality:</span>
                    <span className={`prediction-value ${face.is_clear ? 'quality-good' : 'quality-bad'}`}>
                      {face.is_clear ? 'Clear' : 'Blurry'}
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="sidebar">
          <div className="gallery">
            <h2>
              Gallery
              {gallery.length > 0 && (
                <button className="btn btn-secondary btn-sm" onClick={clearGallery} style={{fontSize: '0.8rem', padding: '4px 8px'}}>
                  🗑️ Clear All
                </button>
              )}
            </h2>
            {gallery.length > 0 ? (
              <div className="gallery-grid">
                {gallery.map((img) => (
                  <div key={img.filename} className="gallery-item">
                    <img src={img.url} alt={img.filename} />
                    <div className="gallery-item-overlay">
                      <button 
                        className="btn btn-danger btn-sm"
                        onClick={() => deleteImage(img.filename)}
                      >
                        Delete
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="gallery-empty">
                <p>No captured images yet</p>
              </div>
            )}
          </div>

          <div className="settings">
            <h2>⚙️ Settings</h2>
            <div className="setting-item">
              <label>Smile Threshold</label>
              <input
                type="range"
                min="0"
                max="0.5"
                step="0.05"
                value={smileThreshold}
                onChange={(e) => updateSmileThreshold(parseFloat(e.target.value))}
              />
              <span className="setting-value">{smileThreshold.toFixed(2)}</span>
            </div>
          </div>

          {systemInfo && (
            <div className="benchmark">
              <h2>📊 Benchmark</h2>
              <div className="benchmark-item">
                <span className="benchmark-label">Avg CPU Usage:</span>
                <span className="benchmark-value">{systemInfo.cpu_usage}</span>
              </div>
              <div className="benchmark-item">
                <span className="benchmark-label">Avg Memory Usage:</span>
                <span className="benchmark-value">{systemInfo.memory_usage}</span>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export default App
