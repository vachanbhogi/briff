import { useState, useEffect, useRef } from 'react';
import { PoseLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// Predefined skeleton connections list (Standard MediaPipe Pose Topology)
const POSE_CONNECTIONS = [
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // torso sides
  [23, 24], // hips
  [23, 25], [25, 27], // left leg
  [24, 26], [26, 28], // right leg
  // Face contours
  [0, 1], [1, 2], [2, 3], [3, 7], // left eye-ear
  [0, 4], [4, 5], [5, 6], [6, 8], // right eye-ear
  [9, 10] // mouth
];

function App() {
  const [stream, setStream] = useState(null);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [isActive, setIsActive] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  
  // Model loading states
  const [poseLandmarker, setPoseLandmarker] = useState(null);
  const [modelStatus, setModelStatus] = useState('LOADING_MODEL');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const requestRef = useRef(null);

  // Initialize MediaPipe PoseLandmarker on WASM
  useEffect(() => {
    const initPose = async () => {
      try {
        setModelStatus('LOADING_WASM');
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
        );
        
        setModelStatus('LOADING_MODEL_FILE');
        const landmarker = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          outputSegmentationMasks: false
        });

        setPoseLandmarker(landmarker);
        setModelStatus('READY');
      } catch (err) {
        console.error('Error initializing MediaPipe Pose:', err);
        setErrorMsg('MEDIAPIPE_INIT_ERROR: WASM failed to load.');
        setModelStatus('FAILED');
      }
    };

    initPose();

    // Fetch camera input devices
    const getDevices = async () => {
      try {
        await navigator.mediaDevices.getUserMedia({ video: true }).then(s => {
          s.getTracks().forEach(track => track.stop());
        }).catch(() => {});

        const deviceList = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = deviceList.filter(device => device.kind === 'videoinput');
        setDevices(videoDevices);
        if (videoDevices.length > 0) {
          setSelectedDeviceId(videoDevices[0].deviceId);
        }
      } catch (err) {
        console.error('Error listing devices:', err);
      }
    };
    getDevices();

    // Cleanup on unmount
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, []);

  // Main real-time WASM pose detection and canvas drawing loop
  const detectPose = () => {
    if (!videoRef.current || !canvasRef.current || !poseLandmarker || !isActive) {
      requestRef.current = requestAnimationFrame(detectPose);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Ensure video is actively playing and has dimensions
    if (video.readyState >= 2 && video.videoWidth > 0) {
      // Set canvas size to match raw video dimensions
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      // Clear previous frames
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Run pose landmark estimation
      const startTimeMs = performance.now();
      const results = poseLandmarker.detectForVideo(video, startTimeMs);

      if (results.landmarks && results.landmarks.length > 0) {
        results.landmarks.forEach((landmarks) => {
          
          // 1. Draw Skeleton Connection Lines (Clearly Visible Neon Cyan)
          ctx.beginPath();
          ctx.strokeStyle = '#00f0ff';
          ctx.lineWidth = 4;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.shadowBlur = 8;
          ctx.shadowColor = '#00f0ff';

          POSE_CONNECTIONS.forEach(([startIdx, endIdx]) => {
            const startLandmark = landmarks[startIdx];
            const endLandmark = landmarks[endIdx];

            // Only draw lines if they exist and are sufficiently visible
            if (
              startLandmark && endLandmark && 
              startLandmark.visibility > 0.5 && 
              endLandmark.visibility > 0.5
            ) {
              const sx = startLandmark.x * canvas.width;
              const sy = startLandmark.y * canvas.height;
              const ex = endLandmark.x * canvas.width;
              const ey = endLandmark.y * canvas.height;
              
              ctx.moveTo(sx, sy);
              ctx.lineTo(ex, ey);
            }
          });
          ctx.stroke();
          ctx.shadowBlur = 0; // reset shadow

          // 2. Draw Concentric Joint Circles (Super Clean Visible Design)
          landmarks.forEach((landmark, idx) => {
            if (landmark.visibility > 0.5) {
              const x = landmark.x * canvas.width;
              const y = landmark.y * canvas.height;

              // Distinguish face joints from main skeleton body joints
              const isFaceJoint = idx < 11;

              if (isFaceJoint) {
                // Face: Tiny clean subtle grey/white dots
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fillStyle = '#ffffff';
                ctx.fill();
              } else {
                // Body Joints: Concentric white-cyan rings
                // Outer Cyan Ring
                ctx.beginPath();
                ctx.arc(x, y, 6, 0, 2 * Math.PI);
                ctx.fillStyle = '#00f0ff';
                ctx.fill();
                
                // Inner White Ring
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fillStyle = '#ffffff';
                ctx.fill();
              }
            }
          });

        });
      }
    }

    requestRef.current = requestAnimationFrame(detectPose);
  };

  // Run or pause animation loop based on camera stream activity
  useEffect(() => {
    if (isActive && poseLandmarker) {
      requestRef.current = requestAnimationFrame(detectPose);
    } else {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isActive, poseLandmarker]);

  const startCamera = async (deviceId = selectedDeviceId) => {
    try {
      setErrorMsg('');
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }

      const constraints = {
        video: deviceId ? { deviceId: { exact: deviceId } } : true,
        audio: false
      };

      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(mediaStream);
      setIsActive(true);
      
      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
      }
    } catch (err) {
      console.error(err);
      setErrorMsg('CAMERA_ERROR: permissions denied or hardware in use.');
      setIsActive(false);
    }
  };

  const stopCamera = () => {
    if (stream) {
      stream.getTracks().forEach(track => track.stop());
      setStream(null);
    }
    setIsActive(false);
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    // Clear canvas when disconnecting feed
    if (canvasRef.current) {
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  const handleDeviceChange = (e) => {
    const newId = e.target.value;
    setSelectedDeviceId(newId);
    if (isActive) {
      startCamera(newId);
    }
  };

  return (
    <div className="w-screen h-screen bg-black text-white font-mono flex flex-col items-center justify-center relative overflow-hidden select-none">
      
      {/* 90% Screen Space Camera Viewport */}
      <div className="w-[90vw] h-[90vh] bg-black border border-neutral-800 flex items-center justify-center relative overflow-hidden rounded-3xl">
        
        {isActive ? (
          <div className="w-full h-full relative">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover rounded-3xl scale-x-[-1]"
            />
            {/* Absolute Skeleton Overlay Canvas */}
            <canvas
              ref={canvasRef}
              className="absolute inset-0 w-full h-full object-cover pointer-events-none rounded-3xl scale-x-[-1]"
            />
          </div>
        ) : (
          <div className="flex flex-col items-center gap-4 text-center">
            {errorMsg ? (
              <span className="text-red-500 text-xs">{errorMsg}</span>
            ) : modelStatus !== 'READY' ? (
              <span className="text-neutral-500 text-xs animate-pulse">
                {modelStatus === 'LOADING_MODEL' && '[ LOADING WASM MEDIA_PIPE ]'}
                {modelStatus === 'LOADING_WASM' && '[ RESOLVING WASM ENVIRONMENT ]'}
                {modelStatus === 'LOADING_MODEL_FILE' && '[ COMPILING POSE LANDMARKER TASK ]'}
                {modelStatus === 'FAILED' && '[ INITIALIZATION_FAILED ]'}
              </span>
            ) : (
              <span className="text-neutral-500 text-xs">POSE_FEED_STANDBY</span>
            )}

            <button 
              onClick={() => startCamera()}
              disabled={modelStatus !== 'READY'}
              className={`px-4 py-2 border text-xs tracking-widest transition-all uppercase cursor-pointer ${
                modelStatus === 'READY' 
                  ? 'border-neutral-700 hover:border-white text-white' 
                  : 'border-neutral-900 text-neutral-600 cursor-not-allowed'
              }`}
            >
              {modelStatus === 'READY' ? '[ connect stream ]' : '[ calibrating systems ]'}
            </button>
          </div>
        )}

        {/* Small Monospaced Toggle Overlay */}
        {isActive && (
          <div className="absolute bottom-4 left-4 right-4 flex justify-between items-center bg-black/60 backdrop-blur-sm px-4 py-2 border border-neutral-800 text-[10px] text-neutral-400 rounded-xl">
            <div className="flex items-center gap-2">
              <span className="w-1.5 h-1.5 bg-[#00f0ff] rounded-full animate-ping"></span>
              <span>WASM_POSE_ACTIVE</span>
            </div>
            
            <div className="flex items-center gap-4">
              {devices.length > 1 && (
                <select 
                  value={selectedDeviceId}
                  onChange={handleDeviceChange}
                  className="bg-transparent text-neutral-400 border-none outline-none cursor-pointer pr-2 hover:text-white"
                >
                  {devices.map((device, idx) => (
                    <option key={device.deviceId} value={device.deviceId} className="bg-black text-white">
                      {device.label || `CAM_${idx + 1}`}
                    </option>
                  ))}
                </select>
              )}
              
              <button 
                onClick={stopCamera}
                className="hover:text-white uppercase tracking-wider cursor-pointer"
              >
                [ disconnect ]
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Tiny minimal subtle status footer outside the 90% frame */}
      <div className="h-[5vh] flex items-center text-[9px] text-neutral-600 uppercase tracking-widest">
        BRIFF // WASM_MEDIAPIPE_POSE
      </div>

    </div>
  );
}

export default App;
