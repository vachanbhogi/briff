import { useState, useEffect, useRef } from 'react';
import { PoseLandmarker, HandLandmarker, FilesetResolver } from '@mediapipe/tasks-vision';

// Predefined skeleton connections topology (Standard MediaPipe Pose model)
const POSE_CONNECTIONS = [
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // torso sides
  [23, 24], // hips
  [23, 25], [25, 27], // left leg
  [24, 26], [26, 28], // right leg
  [0, 1], [1, 2], [2, 3], [3, 7], // left face profile
  [0, 4], [4, 5], [5, 6], [6, 8], // right face profile
  [9, 10] // mouth
];

// Hand connections topology
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], // thumb
  [0, 5], [5, 6], [6, 7], [7, 8], // index
  [0, 9], [9, 10], [10, 11], [11, 12], // middle
  [0, 13], [13, 14], [14, 15], [15, 16], // ring
  [0, 17], [17, 18], [18, 19], [19, 20], // pinky
  [5, 9], [9, 13], [13, 17] // knuckles
];

function App() {
  const [stream, setStream] = useState(null);
  const [devices, setDevices] = useState([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState('');
  const [isActive, setIsActive] = useState(false);
  const [errorMsg, setErrorMsg] = useState('');
  
  // MediaPipe Vision Tasks
  const [poseLandmarker, setPoseLandmarker] = useState(null);
  const [handLandmarker, setHandLandmarker] = useState(null);
  const [modelStatus, setModelStatus] = useState('LOADING_MODEL');
  const [pulseCheckState, setPulseCheckState] = useState('OFFLINE'); // OFFLINE, PLACE_FINGERS, DETECTING, CORRECT
  const [showSkeleton, setShowSkeleton] = useState(true);
  const [activeMode, setActiveMode] = useState('PULSE');
  const showSkeletonRef = useRef(true);

  // Sync ref with showSkeleton state to avoid requestAnimationFrame closure traps
  useEffect(() => {
    showSkeletonRef.current = showSkeleton;
  }, [showSkeleton]);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const requestRef = useRef(null);

  // Moving Average coordinates for target smoothing (Low-Pass Filter)
  const leftTargetRef = useRef({ x: 0.5, y: 0.5 });
  const rightTargetRef = useRef({ x: 0.5, y: 0.5 });
  const initializedTargetsRef = useRef(false);

  // Initialize both Pose and Hand trackers inside the browser via WASM task bundles
  useEffect(() => {
    const initVision = async () => {
      try {
        setModelStatus('LOADING_WASM');
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.35/wasm"
        );
        
        setModelStatus('LOADING_POSE_MODEL');
        const poseDetector = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          outputSegmentationMasks: false
        });

        setModelStatus('LOADING_HAND_MODEL');
        const handDetector = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2
        });

        setPoseLandmarker(poseDetector);
        setHandLandmarker(handDetector);
        setModelStatus('READY');
      } catch (err) {
        console.error('Error initializing MediaPipe:', err);
        setErrorMsg('MEDIAPIPE_INIT_ERROR: WASM vision systems failed to load.');
        setModelStatus('FAILED');
      }
    };

    initVision();

    // Fetch cameras
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

    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, []);

  // Main real-time multi-model tracking and rendering loop
  const processFrame = () => {
    if (!videoRef.current || !canvasRef.current || !poseLandmarker || !handLandmarker || !isActive) {
      requestRef.current = requestAnimationFrame(processFrame);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    if (video.readyState >= 2 && video.videoWidth > 0) {
      if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
      }

      ctx.clearRect(0, 0, canvas.width, canvas.height);

      const startTimeMs = performance.now();
      
      // 1. Run Pose & Hand Landmarkers sequentially on WASM
      const poseResults = poseLandmarker.detectForVideo(video, startTimeMs);
      const handResults = handLandmarker.detectForVideo(video, startTimeMs);

      let neckTargets = null;
      let leftPulseTriggered = false;
      let rightPulseTriggered = false;

      // 2. Draw Body Pose Skeleton & Calculate Neck Coordinates
      if (poseResults.landmarks && poseResults.landmarks.length > 0) {
        const landmarks = poseResults.landmarks[0];
        
        // Extract key anatomical points
        const nose = landmarks[0];
        const leftEar = landmarks[7];
        const rightEar = landmarks[8];
        const leftShoulder = landmarks[11];
        const rightShoulder = landmarks[12];

        if (leftShoulder && rightShoulder && nose) {
          // Anatomical Carotid points lie directly beneath the ears, on the sides of the windpipe.
          // Because checking pulse involves putting hands on the neck, hands will frequently obscure the ears.
          // Fallback ratios based on nose-shoulder lines keep calculations steady when ears are blocked.
          let rawLeftX, rawLeftY;
          if (leftEar && leftEar.visibility > 0.5) {
            rawLeftX = leftEar.x;
            rawLeftY = leftEar.y + (leftShoulder.y - leftEar.y) * 0.38;
          } else {
            rawLeftX = nose.x + (leftShoulder.x - nose.x) * 0.24;
            rawLeftY = nose.y + (leftShoulder.y - nose.y) * 0.48;
          }

          let rawRightX, rawRightY;
          if (rightEar && rightEar.visibility > 0.5) {
            rawRightX = rightEar.x;
            rawRightY = rightEar.y + (rightShoulder.y - rightEar.y) * 0.38;
          } else {
            rawRightX = nose.x + (rightShoulder.x - nose.x) * 0.24;
            rawRightY = nose.y + (rightShoulder.y - nose.y) * 0.48;
          }

          // 2b. Apply exponential low-pass filter smoothing (85% historic, 15% new)
          // This eliminates jitter completely and anchors the target perfectly even during hand occlusions.
          if (!initializedTargetsRef.current) {
            leftTargetRef.current = { x: rawLeftX, y: rawLeftY };
            rightTargetRef.current = { x: rawRightX, y: rawRightY };
            initializedTargetsRef.current = true;
          } else {
            leftTargetRef.current.x = leftTargetRef.current.x * 0.88 + rawLeftX * 0.12;
            leftTargetRef.current.y = leftTargetRef.current.y * 0.88 + rawLeftY * 0.12;
            
            rightTargetRef.current.x = rightTargetRef.current.x * 0.88 + rawRightX * 0.12;
            rightTargetRef.current.y = rightTargetRef.current.y * 0.88 + rawRightY * 0.12;
          }

          neckTargets = {
            left: { ...leftTargetRef.current },
            right: { ...rightTargetRef.current }
          };
        }

        // Draw Pose Connections (Neon Cyan)
        if (showSkeletonRef.current) {
          ctx.beginPath();
          ctx.strokeStyle = 'rgba(0, 240, 255, 0.3)'; // Semi-transparent body skeleton
          ctx.lineWidth = 3;
          ctx.lineCap = 'round';
          ctx.lineJoin = 'round';
          ctx.shadowBlur = 4;
          ctx.shadowColor = '#00f0ff';

          POSE_CONNECTIONS.forEach(([startIdx, endIdx]) => {
            const start = landmarks[startIdx];
            const end = landmarks[endIdx];
            if (start && end && start.visibility > 0.5 && end.visibility > 0.5) {
              ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
              ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
            }
          });
          ctx.stroke();
          ctx.shadowBlur = 0; // reset
        }
      }

      // 3. Process Hands and Check Intersection with Neck Targets
      if (handResults.landmarks && handResults.landmarks.length > 0) {
        
        // Draw Hand skeletons in sharp white
        handResults.landmarks.forEach((handLandmarks) => {
          if (showSkeletonRef.current) {
            ctx.beginPath();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 3.5;
            ctx.lineCap = 'round';
            ctx.lineJoin = 'round';
            ctx.shadowBlur = 6;
            ctx.shadowColor = '#ffffff';

            HAND_CONNECTIONS.forEach(([startIdx, endIdx]) => {
              const start = handLandmarks[startIdx];
              const end = handLandmarks[endIdx];
              if (start && end) {
                ctx.moveTo(start.x * canvas.width, start.y * canvas.height);
                ctx.lineTo(end.x * canvas.width, end.y * canvas.height);
              }
            });
            ctx.stroke();
            ctx.shadowBlur = 0;

            // Draw hand joints
            handLandmarks.forEach((lm) => {
              ctx.beginPath();
              ctx.arc(lm.x * canvas.width, lm.y * canvas.height, 4, 0, 2 * Math.PI);
              ctx.fillStyle = '#ffffff';
              ctx.fill();
            });
          }

          // Check pulse intersection!
          // Carotid pulse is checked using the Index Finger Tip (8) & Middle Finger Tip (12)
          const indexTip = handLandmarks[8];
          const middleTip = handLandmarks[12];

          if (indexTip && middleTip && neckTargets) {
            // Proximity tests (both fingers must be within 5% distance of the target anchor)
            const threshold = 0.052;

            const lDistIndex = Math.hypot(indexTip.x - neckTargets.left.x, indexTip.y - neckTargets.left.y);
            const lDistMiddle = Math.hypot(middleTip.x - neckTargets.left.x, middleTip.y - neckTargets.left.y);
            
            const rDistIndex = Math.hypot(indexTip.x - neckTargets.right.x, indexTip.y - neckTargets.right.y);
            const rDistMiddle = Math.hypot(middleTip.x - neckTargets.right.x, middleTip.y - neckTargets.right.y);

            if (lDistIndex < threshold && lDistMiddle < threshold) {
              leftPulseTriggered = true;
            }
            if (rDistIndex < threshold && rDistMiddle < threshold) {
              rightPulseTriggered = true;
            }
          }
        });
      }

      // 4. Draw Neck Targets & Heartbeat Ripples
      if (neckTargets) {
        const pulseVal = Math.sin(performance.now() / 150) * 3; // subtle breathing scale

        // Draw Left Carotid Target
        const lx = neckTargets.left.x * canvas.width;
        const ly = neckTargets.left.y * canvas.height;
        
        ctx.beginPath();
        ctx.arc(lx, ly, 16 + (leftPulseTriggered ? pulseVal * 2.5 : pulseVal), 0, 2 * Math.PI);
        ctx.lineWidth = leftPulseTriggered ? 4 : 2;
        ctx.strokeStyle = leftPulseTriggered ? '#00ff66' : '#ff3366';
        ctx.shadowBlur = leftPulseTriggered ? 12 : 4;
        ctx.shadowColor = leftPulseTriggered ? '#00ff66' : '#ff3366';
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(lx, ly, 4, 0, 2 * Math.PI);
        ctx.fillStyle = leftPulseTriggered ? '#00ff66' : '#ff3366';
        ctx.fill();

        // Target label string
        ctx.font = 'bold 9px monospace';
        ctx.fillStyle = leftPulseTriggered ? '#00ff66' : '#ff3366';
        ctx.shadowBlur = 0;
        ctx.fillText('CAROTID_PULSE_L', lx + 24, ly + 3);

        // Draw Right Carotid Target
        const rx = neckTargets.right.x * canvas.width;
        const ry = neckTargets.right.y * canvas.height;
        
        ctx.beginPath();
        ctx.arc(rx, ry, 16 + (rightPulseTriggered ? pulseVal * 2.5 : pulseVal), 0, 2 * Math.PI);
        ctx.lineWidth = rightPulseTriggered ? 4 : 2;
        ctx.strokeStyle = rightPulseTriggered ? '#00ff66' : '#ff3366';
        ctx.shadowBlur = rightPulseTriggered ? 12 : 4;
        ctx.shadowColor = rightPulseTriggered ? '#00ff66' : '#ff3366';
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(rx, ry, 4, 0, 2 * Math.PI);
        ctx.fillStyle = rightPulseTriggered ? '#00ff66' : '#ff3366';
        ctx.fill();

        ctx.fillStyle = rightPulseTriggered ? '#00ff66' : '#ff3366';
        ctx.shadowBlur = 0;
        ctx.fillText('CAROTID_PULSE_R', rx - 120, ry + 3);

        // 5. Heartbeat Radar Ripple Animation on Active Match
        // Renders expanding circular shockwaves radiating outward to visualize heartbeat check activity!
        if (leftPulseTriggered || rightPulseTriggered) {
          const activeX = leftPulseTriggered ? lx : rx;
          const activeY = leftPulseTriggered ? ly : ry;
          
          const time = performance.now() / 1000;
          const count = 3;
          for (let i = 0; i < count; i++) {
            const progress = (time + i / count) % 1; // normalized expand time 0 to 1
            const radius = 16 + progress * 55;
            const alpha = 1 - progress; // fade out
            
            ctx.beginPath();
            ctx.arc(activeX, activeY, radius, 0, 2 * Math.PI);
            ctx.strokeStyle = `rgba(0, 255, 102, ${alpha * 0.75})`;
            ctx.lineWidth = 1.5;
            ctx.shadowBlur = 4;
            ctx.shadowColor = '#00ff66';
            ctx.stroke();
          }
          ctx.shadowBlur = 0;
        }

        // Determine overall status
        if (leftPulseTriggered || rightPulseTriggered) {
          setPulseCheckState('CORRECT');
        } else if (handResults.landmarks && handResults.landmarks.length > 0) {
          setPulseCheckState('ALIGNING');
        } else {
          setPulseCheckState('PLACE_FINGERS');
        }
      } else {
        setPulseCheckState('ALIGN_BODY');
      }
    }

    requestRef.current = requestAnimationFrame(processFrame);
  };

  // Toggle loops
  useEffect(() => {
    if (isActive && poseLandmarker && handLandmarker) {
      requestRef.current = requestAnimationFrame(processFrame);
    } else {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    }
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [isActive, poseLandmarker, handLandmarker]);

  // Bulletproof camera stream binding watcher to avoid React mount race conditions
  useEffect(() => {
    if (stream && videoRef.current) {
      videoRef.current.srcObject = stream;
    }
  }, [stream]);

  const startCamera = async (deviceId = selectedDeviceId) => {
    try {
      setErrorMsg('');
      if (stream) {
        stream.getTracks().forEach(track => track.stop());
      }
      initializedTargetsRef.current = false; // Reset smoothing tracker on start

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
    setPulseCheckState('OFFLINE');
    initializedTargetsRef.current = false;
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
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
    <div className="w-screen h-screen bg-neutral-dark text-white font-mono flex flex-col items-center justify-center relative overflow-hidden select-none">
      
      {/* 90% Screen Space Camera Viewport */}
      <div className="w-[90vw] h-[90vh] bg-[#0c0d12] border border-border-dark flex items-center justify-center relative overflow-hidden rounded-3xl">
        
        {/* Render the video and canvas elements permanently in the DOM to avoid React mount race conditions */}
        <div className={`w-full h-full relative ${isActive ? 'block' : 'hidden'}`}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover rounded-3xl scale-x-[-1]"
          />
          {/* Absolute Skeleton + Hands + Targets Drawing Overlay */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full object-cover pointer-events-none rounded-3xl scale-x-[-1] z-10"
          />

          {/* Hyper-Visible Pulse Check Status HUD */}
          <div className="absolute top-6 inset-x-6 flex flex-col items-center justify-center pointer-events-none z-20">
            <div className={`px-6 py-3 border rounded-2xl backdrop-blur-md transition-all text-xs font-bold text-center flex flex-col items-center gap-1.5 shadow-xl max-w-lg ${
              pulseCheckState === 'CORRECT'
                ? 'border-[#00ff66] bg-[#08090c]/85 text-[#00ff66] scale-105'
                : pulseCheckState === 'ALIGNING'
                ? 'border-yellow-500 bg-[#08090c]/85 text-yellow-400'
                : pulseCheckState === 'PLACE_FINGERS'
                ? 'border-[#ff3366] bg-[#08090c]/85 text-[#ff3366]'
                : 'border-border-dark bg-[#08090c]/85 text-neutral-400'
            }`}>
              
              <span className="tracking-widest font-extrabold uppercase text-[10px]">
                {pulseCheckState === 'CORRECT' && '▶ [ PULSE_VERIFICATION: CORRECT ]'}
                {pulseCheckState === 'ALIGNING' && '▷ [ ALIGNING INDEX & MIDDLE FINGERS ]'}
                {pulseCheckState === 'PLACE_FINGERS' && '▷ [ CAROTID_PULSE_STANDBY ]'}
                {pulseCheckState === 'ALIGN_BODY' && '▷ [ PLEASE ALIGN HEAD & SHOULDERS ]'}
              </span>

              <span className="text-[10px] opacity-75 font-medium lowercase">
                {pulseCheckState === 'CORRECT' && 'perfect positioning! check carotid pulse correctly.'}
                {pulseCheckState === 'ALIGNING' && 'fingertips detected. place directly onto neck target circle.'}
                {pulseCheckState === 'PLACE_FINGERS' && 'place index & middle fingertips on side of neck below jaw.'}
                {pulseCheckState === 'ALIGN_BODY' && 'stand centered. head and shoulders must be fully visible.'}
              </span>

            </div>
          </div>
        </div>

        {/* Render the Offline Standby UI only when camera is inactive */}
        {!isActive && (
          <div className="flex flex-col items-center gap-4 text-center">
            {errorMsg ? (
              <span className="text-red-500 text-xs">{errorMsg}</span>
            ) : modelStatus !== 'READY' ? (
              <div className="flex flex-col gap-2 items-center">
                <span className="text-neutral-500 text-xs animate-pulse">
                  {modelStatus === 'LOADING_MODEL' && '[ LOADING WASM VISION BUNDLES ]'}
                  {modelStatus === 'LOADING_WASM' && '[ RESOLVING WASM COMPILATION ENV ]'}
                  {modelStatus === 'LOADING_POSE_MODEL' && '[ REGISTERING POSE ESTIMATION TASKS ]'}
                  {modelStatus === 'LOADING_HAND_MODEL' && '[ INITIALIZING MULTI-HAND TRACKERS ]'}
                  {modelStatus === 'FAILED' && '[ SENSORS_INITIALIZATION_FAILED ]'}
                </span>
                <span className="text-[9px] text-neutral-700">Downloading MediaPipe model weights...</span>
              </div>
            ) : (
              <span className="text-neutral-500 text-xs">BRIFF_POSE_HAND_STANDBY</span>
            )}

            <button 
              onClick={() => startCamera()}
              disabled={modelStatus !== 'READY'}
              aria-label="Connect live camera feed stream"
              className={`px-4 py-2 border text-xs tracking-widest transition-all uppercase cursor-pointer ${
                modelStatus === 'READY' 
                  ? 'border-neutral-700 hover:border-white text-white' 
                  : 'border-neutral-900 text-neutral-600 cursor-not-allowed'
              }`}
            >
              {modelStatus === 'READY' ? '[ connect stream ]' : '[ calibrating sensors ]'}
            </button>
          </div>
        )}

        {/* Small Monospaced Toggle Overlay */}
        {isActive && (
          <div className="absolute bottom-4 left-4 right-4 flex justify-between items-center bg-[#08090c]/70 backdrop-blur-sm px-4 py-2 border border-border-dark text-[10px] text-neutral-400 rounded-xl z-20">
            <div 
              onClick={() => setShowSkeleton(!showSkeleton)}
              className="flex items-center gap-2 cursor-pointer hover:text-white transition-all select-none"
              title="Click to toggle skeletal overlay wireframe lines"
            >
              <span className={`w-1.5 h-1.5 rounded-full ${showSkeleton ? 'bg-[#00f0ff] animate-ping' : 'bg-neutral-600'}`}></span>
              <span className={showSkeleton ? 'text-neutral-200 font-semibold' : 'text-neutral-500 font-normal'}>
                {showSkeleton ? 'WASM_SENSORS_SHOWN' : 'WASM_SENSORS_MUTED'}
              </span>
            </div>
            
            <div className="flex items-center gap-4">
              
              {/* First-Aid Training Program Selector */}
              <select 
                value={activeMode}
                onChange={(e) => setActiveMode(e.target.value)}
                aria-label="Select active first-aid training mode"
                className="bg-transparent text-neutral-400 border-none outline-none cursor-pointer pr-2 hover:text-white uppercase font-bold tracking-wider"
              >
                <option value="PULSE" className="bg-[#08090c] text-white">MODE_PULSE</option>
                <option value="CPR" className="bg-[#08090c] text-white">MODE_CPR</option>
                <option value="HEIMLICH" className="bg-[#08090c] text-white">MODE_HEIMLICH</option>
              </select>

              <div className="w-px h-3 bg-neutral-800"></div>

              {devices.length > 1 && (
                <>
                  <select 
                    value={selectedDeviceId}
                    onChange={handleDeviceChange}
                    aria-label="Select camera hardware input source"
                    className="bg-transparent text-neutral-400 border-none outline-none cursor-pointer pr-2 hover:text-white"
                  >
                    {devices.map((device, idx) => (
                      <option key={device.deviceId} value={device.deviceId} className="bg-[#08090c] text-white">
                        {device.label || `CAM_${idx + 1}`}
                      </option>
                    ))}
                  </select>
                  <div className="w-px h-3 bg-neutral-800"></div>
                </>
              )}
              
              <button 
                onClick={stopCamera}
                aria-label="Disconnect active camera stream"
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
        BRIFF // WASM_POSE_AND_HAND_SENSORS
      </div>

    </div>
  );
}

export default App;
