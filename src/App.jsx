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
  const [cprState, setCprState] = useState('CPR_OFFLINE');
  const [activeMode, setActiveMode] = useState('PULSE');
  const showSkeletonRef = useRef(true);

  // Sync ref with showSkeleton state to avoid requestAnimationFrame closure traps
  useEffect(() => {
    showSkeletonRef.current = showSkeleton;
  }, [showSkeleton]);

  const activeModeRef = useRef('PULSE');

  // Sync ref with activeMode state to avoid loop closures
  useEffect(() => {
    activeModeRef.current = activeMode;
  }, [activeMode]);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const requestRef = useRef(null);

  // Moving Average coordinates for target smoothing (Low-Pass Filter)
  const leftTargetRef = useRef({ x: 0.5, y: 0.5 });
  const rightTargetRef = useRef({ x: 0.5, y: 0.5 });
  const initializedTargetsRef = useRef(false);

  const sternumTargetRef = useRef({ x: 0.5, y: 0.5 });
  const sternumInitializedRef = useRef(false);
  const shoulderYHistoryRef = useRef([]);
  const compressionPeaksRef = useRef([]);
  const cprBpmRef = useRef(0);
  const cprDepthRatioRef = useRef(0);
  const cprStateRef = useRef('CPR_OFFLINE');
  const cprPlacementValidRef = useRef(false);
  const cprPhaseRef = useRef('UP');
  const cprMinYRef = useRef(0);
  const cprMaxYRef = useRef(0);
  
  const audioCtxRef = useRef(null);
  const lastTickRef = useRef(0);

  useEffect(() => {
    cprStateRef.current = cprState;
  }, [cprState]);

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
        const deviceList = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = deviceList.filter(device => device.kind === 'videoinput');
        setDevices(videoDevices);
        if (videoDevices.length > 0 && !selectedDeviceId) {
          setSelectedDeviceId(videoDevices[0].deviceId);
        }
      } catch (err) {
        console.error('Error listing devices:', err);
      }
    };

    const initCameras = async () => {
      try {
        const s = await navigator.mediaDevices.getUserMedia({ video: true });
        s.getTracks().forEach(track => track.stop());
      } catch (e) {}
      await getDevices();
    };
    initCameras();

    navigator.mediaDevices.addEventListener('devicechange', getDevices);

    return () => {
      navigator.mediaDevices.removeEventListener('devicechange', getDevices);
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
      let cprCompressionTriggered = false;

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

        // CPR Sternum Target & Compression Detection
        if (activeModeRef.current === 'CPR') {
          const leftShoulder = landmarks[11];
          const rightShoulder = landmarks[12];
          const leftHip = landmarks[23];
          const rightHip = landmarks[24];
          if (leftShoulder && rightShoulder && leftHip && rightHip) {
            const midShoulder = {
              x: (leftShoulder.x + rightShoulder.x) / 2,
              y: (leftShoulder.y + rightShoulder.y) / 2
            };
            const midHip = {
              x: (leftHip.x + rightHip.x) / 2,
              y: (leftHip.y + rightHip.y) / 2
            };
            
            // Fixed ground target for dummy placement
            const rawSternumX = 0.5;
            const rawSternumY = 0.65;

            if (!sternumInitializedRef.current) {
              sternumTargetRef.current = { x: rawSternumX, y: rawSternumY };
              sternumInitializedRef.current = true;
            } else {
              sternumTargetRef.current.x = sternumTargetRef.current.x * 0.88 + rawSternumX * 0.12;
              sternumTargetRef.current.y = sternumTargetRef.current.y * 0.88 + rawSternumY * 0.12;
            }

            // Metronome
            if (audioCtxRef.current) {
              if (performance.now() - lastTickRef.current >= 545.45) { // ~110 BPM
                lastTickRef.current = performance.now();
                const osc = audioCtxRef.current.createOscillator();
                const gain = audioCtxRef.current.createGain();
                osc.connect(gain);
                gain.connect(audioCtxRef.current.destination);
                osc.frequency.setValueAtTime(800, audioCtxRef.current.currentTime);
                gain.gain.setValueAtTime(0.05, audioCtxRef.current.currentTime);
                gain.gain.exponentialRampToValueAtTime(0.001, audioCtxRef.current.currentTime + 0.05);
                osc.start();
                osc.stop(audioCtxRef.current.currentTime + 0.05);
              }
            }

            // Shoulder Oscillation Tracking (2D Scale-Invariant)
            const history = shoulderYHistoryRef.current;
            const rawShoulderY = midShoulder.y;
            
            // Apply low-pass filter to remove tracking jitter (50/50 mix for stable 2D landmarks)
            let smoothedShoulderY = rawShoulderY;
            if (history.length > 0) {
              smoothedShoulderY = history[history.length - 1].y * 0.5 + rawShoulderY * 0.5;
            }
            
            // Check CPR Posture (wrists together and below shoulders)
            const leftWrist = landmarks[15];
            const rightWrist = landmarks[16];
            let isCprPosture = false;
            
            if (leftWrist && rightWrist) {
              const wristDist = Math.hypot(leftWrist.x - rightWrist.x, leftWrist.y - rightWrist.y);
              const wristsBelowShoulders = leftWrist.y > midShoulder.y && rightWrist.y > midShoulder.y;
              // Wrists should be close together (clasped) and below the shoulders (leaning down)
              // We use 0.25 to be lenient, as hands can overlap and confuse the model
              if (wristDist < 0.25 && wristsBelowShoulders) {
                isCprPosture = true;
              }
            } else if (leftWrist || rightWrist) {
              // If one hand completely occludes the other, MediaPipe might only see one wrist.
              // As long as it is below the shoulder, we'll allow it.
              const visibleWrist = leftWrist || rightWrist;
              if (visibleWrist.y > midShoulder.y) {
                isCprPosture = true;
              }
            }

            if (isCprPosture) {
              history.push({ y: smoothedShoulderY, time: startTimeMs });
              if (history.length > 90) history.shift();
              
              // 2D Peak Detection via UP/DOWN Phase State Machine (Scale-Invariant)
              const torsoLength = Math.hypot(midHip.x - midShoulder.x, midHip.y - midShoulder.y);
              const currY = smoothedShoulderY;

              if (cprPhaseRef.current === 'UP') {
                // Initialize anchor on first frame
                if (cprMinYRef.current === 0) cprMinYRef.current = currY;
                
                // Track the highest physical point in the air (minimum Y value)
                if (currY < cprMinYRef.current) {
                  cprMinYRef.current = currY;
                }
                
                // To enter DOWN phase, they must compress (Y increases) by at least 4.5% of torso length
                if (currY - cprMinYRef.current > torsoLength * 0.045) {
                  cprPhaseRef.current = 'DOWN';
                  cprMaxYRef.current = currY; // initialize max Y for the bottom of stroke
                }
              } else if (cprPhaseRef.current === 'DOWN') {
                // Track the lowest physical point towards the ground (maximum Y value)
                if (currY > cprMaxYRef.current) {
                  cprMaxYRef.current = currY;
                }
                
                // To complete the stroke and trigger a beat, they must recoil (Y decreases) by at least 2% of torso length
                if (cprMaxYRef.current - currY > torsoLength * 0.02) {
                  const amplitude = cprMaxYRef.current - cprMinYRef.current;
                  cprDepthRatioRef.current = amplitude / torsoLength; // Relative depth ratio
                  
                  const peaks = compressionPeaksRef.current;
                  const peakTime = startTimeMs;
                  
                  // Debounce to prevent double counting (max 240 BPM)
                  if (peaks.length === 0 || peakTime - peaks[peaks.length - 1] > 250) { 
                    peaks.push(peakTime);
                    cprCompressionTriggered = true;
                    if (peaks.length > 6) peaks.shift(); // Keep 6 recent peaks for a stable rolling average
                  }
                  
                  cprPhaseRef.current = 'UP';
                  cprMinYRef.current = currY; // reset for next stroke
                }
              }
            } else {
              // If posture is broken (hands off chest, not in frame), freeze the state machine 
              // so that camera jitter doesn't trigger false beats.
            }

            // Calculate and purge BPM
            const peaks = compressionPeaksRef.current;
            const nowMs = startTimeMs;
            
            // If they haven't done a compression in the last 1.5 seconds, they stopped.
            if (peaks.length > 0 && nowMs - peaks[peaks.length - 1] > 1500) {
              peaks.length = 0;
            }

            if (peaks.length >= 3) {
              const avgInterval = (peaks[peaks.length - 1] - peaks[0]) / (peaks.length - 1);
              cprBpmRef.current = Math.round(60000 / avgInterval);
            } else if (peaks.length === 0) {
              cprBpmRef.current = 0; // Reset BPM when stopped
            }
          }
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
          if (activeModeRef.current === 'PULSE') {
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
          } else if (activeModeRef.current === 'CPR') {
            const palm1 = handLandmarks[0];
            const target = sternumTargetRef.current;
            if (palm1 && sternumInitializedRef.current) {
              const dist = Math.hypot(palm1.x - target.x, palm1.y - target.y);
              if (dist < 0.08) {
                // If this is the second hand, check if it's close to the first hand
                // For simplicity we just see if ANY palm is within 8% of target.
                cprPlacementValidRef.current = true;
              }
            }
          }
        });
      }

      // Determine overall CPR placement status outside loop (needs to clear if hands gone)
      if (activeModeRef.current === 'CPR' && (!handResults.landmarks || handResults.landmarks.length === 0)) {
        cprPlacementValidRef.current = false;
      }

      // 4. Draw Neck Targets & Heartbeat Ripples (Pulse Mode only)
      if (activeModeRef.current === 'PULSE') {
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
          ctx.save();
          ctx.scale(-1, 1);
          ctx.fillText('CAROTID_PULSE_L', -(lx - 24), ly + 3);
          ctx.restore();

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
          ctx.save();
          ctx.scale(-1, 1);
          ctx.fillText('CAROTID_PULSE_R', -(rx + 90), ry + 3);
          ctx.restore();

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
      } else if (activeModeRef.current === 'CPR') {
        if (sternumInitializedRef.current) {
          const target = sternumTargetRef.current;
          const sx = target.x * canvas.width;
          const sy = target.y * canvas.height;
          
          const pulseVal = Math.sin(performance.now() / 150) * 3;
          const isValid = cprPlacementValidRef.current;

          // Draw Dummy Outline Guide
          const dummyWidth = canvas.width * 0.15;
          const dummyHeight = canvas.height * 0.20;
          
          ctx.beginPath();
          ctx.ellipse(sx, sy - dummyHeight * 0.2, dummyWidth, dummyHeight, 0, 0, 2 * Math.PI);
          ctx.lineWidth = 2;
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
          ctx.setLineDash([8, 6]);
          ctx.stroke();
          ctx.setLineDash([]);
          
          ctx.beginPath();
          ctx.arc(sx, sy - dummyHeight * 1.5, dummyWidth * 0.6, 0, 2 * Math.PI);
          ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
          ctx.setLineDash([8, 6]);
          ctx.stroke();
          ctx.setLineDash([]);
          
          ctx.font = 'bold 14px monospace';
          ctx.fillStyle = 'rgba(255, 255, 255, 0.4)';
          ctx.textAlign = 'center';
          ctx.save();
          ctx.scale(-1, 1);
          ctx.fillText('ALIGN DUMMY HERE', -sx, sy - dummyHeight * 0.8);
          ctx.restore();
          ctx.textAlign = 'left'; // reset

          ctx.beginPath();
          ctx.arc(sx, sy, 20 + (isValid ? pulseVal * 2 : pulseVal), 0, 2 * Math.PI);
          ctx.lineWidth = isValid ? 4 : 2;
          ctx.strokeStyle = isValid ? '#00ff66' : '#ffaa00';
          ctx.shadowBlur = isValid ? 12 : 4;
          ctx.shadowColor = isValid ? '#00ff66' : '#ffaa00';
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(sx, sy, 5, 0, 2 * Math.PI);
          ctx.fillStyle = isValid ? '#00ff66' : '#ffaa00';
          ctx.fill();

          ctx.font = 'bold 9px monospace';
          ctx.fillStyle = isValid ? '#00ff66' : '#ffaa00';
          ctx.shadowBlur = 0;
          ctx.save();
          ctx.scale(-1, 1);
          ctx.fillText('STERNUM_TARGET', -(sx - 30), sy + 3);
          ctx.restore();

          if (cprCompressionTriggered) {
            const time = performance.now() / 1000;
            const count = 2;
            for (let i = 0; i < count; i++) {
              const progress = (time + i / count) % 1;
              const radius = 20 + progress * 80;
              const alpha = 1 - progress;
              
              ctx.beginPath();
              ctx.arc(sx, sy, radius, 0, 2 * Math.PI);
              ctx.strokeStyle = `rgba(0, 255, 102, ${alpha * 0.8})`;
              ctx.lineWidth = 2;
              ctx.stroke();
            }
          }

          // State Machine
          if (!isValid) {
            setCprState('CPR_POSITION_HANDS');
          } else {
            const bpm = cprBpmRef.current;
            if (bpm === 0) setCprState('CPR_COMPRESSING');
            else if (bpm < 100) setCprState('CPR_RATE_SLOW');
            else if (bpm <= 120) setCprState('CPR_RATE_GOOD');
            else setCprState('CPR_RATE_FAST');
          }
        } else {
          setCprState('CPR_ALIGN_BODY');
        }
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
      sternumInitializedRef.current = false;
      shoulderYHistoryRef.current = [];
      compressionPeaksRef.current = [];
      cprBpmRef.current = 0;
      cprDepthRatioRef.current = 0;
      cprPlacementValidRef.current = false;
      cprPhaseRef.current = 'UP';
      cprMinYRef.current = 0;
      cprMaxYRef.current = 0;

      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtxRef.current.state === 'suspended') {
        audioCtxRef.current.resume();
      }

      const constraints = {
        video: deviceId ? { deviceId: { exact: deviceId } } : true,
        audio: false
      };

      const mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      setStream(mediaStream);
      setIsActive(true);
      
      // Re-fetch devices now that a stream is active, as macOS/Safari 
      // often hides laptop cameras until an active stream is opened
      try {
        const deviceList = await navigator.mediaDevices.enumerateDevices();
        const videoDevices = deviceList.filter(device => device.kind === 'videoinput');
        setDevices(videoDevices);
      } catch (e) {}
      
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
    setCprState('CPR_OFFLINE');
    initializedTargetsRef.current = false;
    sternumInitializedRef.current = false;
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
            className="w-full h-full object-contain rounded-3xl scale-x-[-1]"
          />
          {/* Absolute Skeleton + Hands + Targets Drawing Overlay */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full object-contain pointer-events-none rounded-3xl scale-x-[-1] z-10"
          />

          {/* Hyper-Visible Pulse Check Status HUD */}
          <div className="absolute top-6 inset-x-6 flex flex-col items-center justify-center pointer-events-none z-20">
            {activeMode === 'PULSE' && (
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
            )}

            {activeMode === 'CPR' && (
              <div className={`px-6 py-3 border rounded-2xl backdrop-blur-md transition-all text-xs font-bold text-center flex flex-col items-center gap-1.5 shadow-xl max-w-lg min-w-[320px] ${
                cprState === 'CPR_RATE_GOOD'
                  ? 'border-[#00ff66] bg-[#08090c]/85 text-[#00ff66]'
                  : cprState === 'CPR_RATE_SLOW' || cprState === 'CPR_RATE_FAST' || cprState === 'CPR_POSITION_HANDS' || cprState === 'CPR_ALIGN_BODY'
                  ? 'border-[#ffaa00] bg-[#08090c]/85 text-[#ffaa00]'
                  : 'border-[#00f0ff] bg-[#08090c]/85 text-[#00f0ff]'
              }`}>
                <span className="tracking-widest font-extrabold uppercase text-[10px]">
                  ▶ [ CPR_COMPRESSION_TRAINING: ACTIVE ]
                </span>
                
                <div className="w-full flex justify-between items-center text-[10px] uppercase tracking-wider py-1 border-y border-opacity-30 border-current my-1">
                  <div className="flex flex-col items-start">
                    <span>BPM: <span className="text-sm font-black">{cprBpmRef.current || '--'}</span></span>
                    <span className="text-[8px] opacity-80">
                      {cprState === 'CPR_RATE_SLOW' ? 'INCREASE RATE' : cprState === 'CPR_RATE_FAST' ? 'DECREASE RATE' : cprState === 'CPR_RATE_GOOD' ? 'RATE CORRECT' : 'AWAITING DATA'}
                    </span>
                  </div>
                  
                  <div className="flex flex-col items-end text-right">
                    <span>DEPTH: {
                      cprDepthRatioRef.current < 0.045 ? 'SHALLOW' : 
                      cprDepthRatioRef.current > 0.06 ? 'DEEP' : 
                      cprDepthRatioRef.current > 0 ? 'GOOD' : '--'
                    }</span>
                    <span className="text-[8px] opacity-80">
                      PLACEMENT: {cprPlacementValidRef.current ? '✓ CENTERED' : '✕ OFF TARGET'}
                    </span>
                  </div>
                </div>

                <span className="text-[9px] opacity-75 font-medium lowercase">
                  {cprState === 'CPR_ALIGN_BODY' ? 'align body — kneel centered in frame' :
                   cprState === 'CPR_POSITION_HANDS' ? 'position hands — place heel of palms on target' :
                   'compress at 100-120/min • push hard, push fast'}
                </span>
              </div>
            )}

            {activeMode === 'HEIMLICH' && (
              <div className="px-6 py-3 border border-border-dark bg-[#08090c]/85 text-neutral-400 rounded-2xl backdrop-blur-md transition-all text-xs font-bold text-center flex flex-col items-center gap-1.5 shadow-xl max-w-lg">
                <span className="tracking-widest font-extrabold uppercase text-[10px] text-[#00f0ff]">
                  ▶ [ HEIMLICH_TRAINING_MODE: ACTIVE ]
                </span>
                <span className="text-[10px] opacity-75 font-medium lowercase">
                  heimlich maneuver module active. stand behind patient and wrap hands around upper abdomen. (simulation)
                </span>
              </div>
            )}
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
                id="program-selector"
                value={activeMode}
                onChange={(e) => {
                  const mode = e.target.value;
                  setActiveMode(mode);
                  if (mode !== 'PULSE') {
                    setPulseCheckState('OFFLINE');
                  }
                }}
                aria-label="Select active first-aid training mode"
                className="bg-transparent text-neutral-400 border-none outline-none cursor-pointer pr-2 hover:text-white uppercase font-bold tracking-wider"
              >
                <option value="PULSE" className="bg-[#08090c] text-white">MODE_PULSE</option>
                <option value="CPR" className="bg-[#08090c] text-white">MODE_CPR</option>
                <option value="HEIMLICH" className="bg-[#08090c] text-white">MODE_HEIMLICH</option>
              </select>

              <div className="w-px h-3 bg-neutral-800"></div>

              {devices.length > 0 && (
                <>
                  <select 
                    id="camera-selector"
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
