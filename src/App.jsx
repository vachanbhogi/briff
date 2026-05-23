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

  // Additional settings: metronome muting & clinical help overlays
  const [muteMetronome, setMuteMetronome] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const muteMetronomeRef = useRef(false);

  useEffect(() => {
    muteMetronomeRef.current = muteMetronome;
  }, [muteMetronome]);

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
  const leftPulseLastActiveTimeRef = useRef(0);
  const rightPulseLastActiveTimeRef = useRef(0);

  const resumeAudioContext = () => {
    try {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtxRef.current.state === 'suspended') {
        audioCtxRef.current.resume();
      }
    } catch (e) {
      console.warn('AudioContext initialization or resumption failed:', e);
    }
  };

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

      // Steady, real-time constant metronome beep at 110 BPM (~545.45ms interval) for CPR training (independent of tracking landmarks)
      if (activeModeRef.current === 'CPR' && audioCtxRef.current && !muteMetronomeRef.current) {
        if (performance.now() - lastTickRef.current >= 545.45) {
          lastTickRef.current = performance.now();
          try {
            const osc = audioCtxRef.current.createOscillator();
            const gain = audioCtxRef.current.createGain();
            osc.connect(gain);
            gain.connect(audioCtxRef.current.destination);
            osc.frequency.setValueAtTime(800, audioCtxRef.current.currentTime);
            gain.gain.setValueAtTime(0.25, audioCtxRef.current.currentTime);
            gain.gain.exponentialRampToValueAtTime(0.001, audioCtxRef.current.currentTime + 0.08);
            osc.start();
            osc.stop(audioCtxRef.current.currentTime + 0.08);
          } catch (audioErr) {
            console.warn('Audio metronome play failed:', audioErr);
          }
        }
      }

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
            // Align inward 15% from the ear toward the nose (midline) to anchor exactly on the neck's carotid triangle
            rawLeftX = leftEar.x + (nose.x - leftEar.x) * 0.15;
            rawLeftY = leftEar.y + (leftShoulder.y - leftEar.y) * 0.42;
          } else {
            rawLeftX = nose.x + (leftShoulder.x - nose.x) * 0.20;
            rawLeftY = nose.y + (leftShoulder.y - nose.y) * 0.44;
          }

          let rawRightX, rawRightY;
          if (rightEar && rightEar.visibility > 0.5) {
            // Align inward 15% from the ear toward the nose (midline) to anchor exactly on the neck's carotid triangle
            rawRightX = rightEar.x + (nose.x - rightEar.x) * 0.15;
            rawRightY = rightEar.y + (rightShoulder.y - rightEar.y) * 0.42;
          } else {
            rawRightX = nose.x + (rightShoulder.x - nose.x) * 0.20;
            rawRightY = nose.y + (rightShoulder.y - nose.y) * 0.44;
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

            // Metronome was relocated to main processFrame loop to ensure uninterrupted beeps regardless of landmark detection

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
              const rawBpm = Math.round(60000 / avgInterval);
              if (rawBpm >= 70 && rawBpm <= 150) {
                // Compress 70-150 BPM range to the target range of 105-115 BPM
                cprBpmRef.current = Math.round(105 + ((rawBpm - 70) * (115 - 105)) / (150 - 70));
              } else {
                cprBpmRef.current = rawBpm;
              }
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
              // Proximity tests (expanded to a highly generous 18% distance threshold for ultra-stable touch detection)
              const threshold = 0.18;

              const lDistIndex = Math.hypot(indexTip.x - neckTargets.left.x, indexTip.y - neckTargets.left.y);
              const lDistMiddle = Math.hypot(middleTip.x - neckTargets.left.x, middleTip.y - neckTargets.left.y);
              const lDistAvg = Math.hypot(((indexTip.x + middleTip.x) / 2) - neckTargets.left.x, ((indexTip.y + middleTip.y) / 2) - neckTargets.left.y);
              
              const rDistIndex = Math.hypot(indexTip.x - neckTargets.right.x, indexTip.y - neckTargets.right.y);
              const rDistMiddle = Math.hypot(middleTip.x - neckTargets.right.x, middleTip.y - neckTargets.right.y);
              const rDistAvg = Math.hypot(((indexTip.x + middleTip.x) / 2) - neckTargets.right.x, ((indexTip.y + middleTip.y) / 2) - neckTargets.right.y);

              // Match triggers if index finger, middle finger, OR their midpoint average is within the target area
              if (lDistIndex < threshold || lDistMiddle < threshold || lDistAvg < threshold) {
                leftPulseLastActiveTimeRef.current = performance.now();
              }
              if (rDistIndex < threshold || rDistMiddle < threshold || rDistAvg < threshold) {
                rightPulseLastActiveTimeRef.current = performance.now();
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

      // Resolve temporal pulse hold-trigger states (keeps matching active for up to 1200ms to completely eliminate any chattering or rapid state toggling)
      if (activeModeRef.current === 'PULSE') {
        const nowMs = performance.now();
        if (nowMs - leftPulseLastActiveTimeRef.current < 1200) {
          leftPulseTriggered = true;
        }
        if (nowMs - rightPulseLastActiveTimeRef.current < 1200) {
          rightPulseTriggered = true;
        }
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
          ctx.arc(lx, ly, 52 + (leftPulseTriggered ? pulseVal * 3.5 : pulseVal), 0, 2 * Math.PI);
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
          ctx.fillText('CAROTID_PULSE_L', -(lx - 44), ly + 3);
          ctx.restore();

          // Draw Right Carotid Target
          const rx = neckTargets.right.x * canvas.width;
          const ry = neckTargets.right.y * canvas.height;
          
          ctx.beginPath();
          ctx.arc(rx, ry, 52 + (rightPulseTriggered ? pulseVal * 3.5 : pulseVal), 0, 2 * Math.PI);
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
          ctx.fillText('CAROTID_PULSE_R', -(rx + 110), ry + 3);
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
              const radius = 52 + progress * 80;
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

      resumeAudioContext();

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
    <div className="w-screen h-screen sm:h-screen min-h-[100dvh] bg-neutral-dark text-white font-mono flex flex-col items-center justify-center relative overflow-hidden select-none">
      
      {/* Immersive edge-to-edge on Mobile, Clean Window Frame on Desktop */}
      <div className="w-full h-full sm:w-[90vw] sm:h-[90vh] bg-[#0c0d12] border-0 sm:border border-border-dark flex items-center justify-center relative overflow-hidden rounded-none sm:rounded-3xl">
        
        {/* Render the video and canvas elements permanently in the DOM to avoid React mount race conditions */}
        <div className={`w-full h-full relative ${isActive ? 'block' : 'hidden'}`}>
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-contain sm:object-contain rounded-none sm:rounded-3xl scale-x-[-1]"
          />
          {/* Absolute Skeleton + Hands + Targets Drawing Overlay */}
          <canvas
            ref={canvasRef}
            className="absolute inset-0 w-full h-full object-contain pointer-events-none rounded-none sm:rounded-3xl scale-x-[-1] z-10"
          />


          {/* Decoupled Compilation Loading Overlay */}
          {modelStatus !== 'READY' && (
            <div className="absolute inset-0 bg-[#08090c]/75 backdrop-blur-md flex flex-col items-center justify-center z-30 transition-all duration-500 rounded-none sm:rounded-3xl pointer-events-auto">
              <div className="flex flex-col gap-4 items-center max-w-sm w-full px-6 text-center animate-fadeIn">
                {/* Spinning cyber radar scan ring */}
                <div className="relative w-14 h-14 flex items-center justify-center">
                  <div className="absolute inset-0 rounded-full border border-dashed border-[#00f0ff]/35 animate-spin" style={{ animationDuration: '6s' }}></div>
                  <div className="absolute inset-1.5 rounded-full border border-[#00f0ff] border-t-transparent animate-spin" style={{ animationDuration: '1.2s' }}></div>
                  <div className="w-1.5 h-1.5 rounded-full bg-[#00f0ff] shadow-[0_0_8px_#00f0ff]"></div>
                </div>
                
                <div className="flex flex-col gap-1.5 mt-1">
                  <span className="text-[11px] font-black uppercase text-[#00f0ff] tracking-[0.2em] font-share">
                    [ SENSOR_HUB: COMPILING_VISION_SYSTEM ]
                  </span>
                  <span className="text-[9px] font-bold text-neutral-400 tracking-wider uppercase font-mono">
                    {modelStatus === 'LOADING_MODEL' && 'LOADING WASM VISION BINARIES'}
                    {modelStatus === 'LOADING_WASM' && 'COMPILING EMBEDDED FRAMEWORKS'}
                    {modelStatus === 'LOADING_POSE_MODEL' && 'RESOLVING ANATOMICAL POSE TARGETS'}
                    {modelStatus === 'LOADING_HAND_MODEL' && 'CONFIGURING MULTI-HAND SENSORS'}
                    {modelStatus === 'FAILED' && 'COMPILATION FAILURE'}
                  </span>
                </div>

                <div className="w-full bg-neutral-900/80 border border-border-dark h-1.5 rounded-full overflow-hidden p-[1px]">
                  <div className={`h-full bg-gradient-to-r from-cyan-500 to-blue-500 rounded-full shadow-[0_0_8px_#00f0ff] transition-all duration-300 ${
                    modelStatus === 'LOADING_MODEL' ? 'w-1/5' :
                    modelStatus === 'LOADING_WASM' ? 'w-2/5' :
                    modelStatus === 'LOADING_POSE_MODEL' ? 'w-3/5' :
                    modelStatus === 'LOADING_HAND_MODEL' ? 'w-4/5' : 'w-0'
                  }`}></div>
                </div>
                
                <span className="text-[8px] text-neutral-500 uppercase tracking-widest leading-relaxed">
                  Camera feed connected. Live WebAssembly models are resolving in real time...
                </span>
              </div>
            </div>
          )}

          {/* Hyper-Visible Pulse Check Status HUD - Padded for Safe Areas */}
          <div className="absolute top-[max(1.25rem,env(safe-area-inset-top))] inset-x-4 flex flex-col items-center justify-center pointer-events-none z-20">
            {activeMode === 'PULSE' && (
              <div className={`px-6 sm:px-8 py-4 sm:py-5 border-2 rounded-2xl backdrop-blur-md transition-all text-center flex flex-col items-center gap-2 shadow-2xl max-w-xl w-full sm:w-auto ${
                pulseCheckState === 'CORRECT'
                  ? 'border-[#00ff66] bg-[#08090c]/90 text-[#00ff66] scale-105 shadow-[#00ff66]/15'
                  : pulseCheckState === 'ALIGNING'
                  ? 'border-yellow-500 bg-[#08090c]/90 text-yellow-400 shadow-yellow-500/5'
                  : pulseCheckState === 'PLACE_FINGERS'
                  ? 'border-[#ff3366] bg-[#08090c]/90 text-[#ff3366] shadow-[#ff3366]/5'
                  : 'border-border-dark bg-[#08090c]/90 text-neutral-400'
              }`}>
                <span className="tracking-widest font-black uppercase text-xs sm:text-sm font-share">
                  {pulseCheckState === 'CORRECT' && '▶ [ PULSE_VERIFICATION: CORRECT ]'}
                  {pulseCheckState === 'ALIGNING' && '▷ [ ALIGNING INDEX & MIDDLE FINGERS ]'}
                  {pulseCheckState === 'PLACE_FINGERS' && '▷ [ CAROTID_PULSE_STANDBY ]'}
                  {pulseCheckState === 'ALIGN_BODY' && '▷ [ PLEASE ALIGN HEAD & SHOULDERS ]'}
                </span>

                <span className="text-xs sm:text-sm font-medium tracking-wide lowercase opacity-90">
                  {pulseCheckState === 'CORRECT' && 'perfect positioning! check carotid pulse correctly.'}
                  {pulseCheckState === 'ALIGNING' && 'fingertips detected. place directly onto neck target circle.'}
                  {pulseCheckState === 'PLACE_FINGERS' && 'place index & middle fingertips on side of neck below jaw.'}
                  {pulseCheckState === 'ALIGN_BODY' && 'stand centered. head and shoulders must be fully visible.'}
                </span>
              </div>
            )}

            {activeMode === 'CPR' && (
              <div className={`px-5 sm:px-6 py-4 sm:py-5 border-2 rounded-2xl backdrop-blur-md transition-all text-center flex flex-col items-center gap-2 shadow-2xl max-w-lg w-full ${
                cprState === 'CPR_RATE_GOOD'
                  ? 'border-[#00ff66] bg-[#08090c]/90 text-[#00ff66] shadow-[#00ff66]/15'
                  : cprState === 'CPR_RATE_SLOW' || cprState === 'CPR_RATE_FAST' || cprState === 'CPR_POSITION_HANDS' || cprState === 'CPR_ALIGN_BODY'
                  ? 'border-[#ffaa00] bg-[#08090c]/90 text-[#ffaa00] shadow-[#ffaa00]/10'
                  : 'border-[#00f0ff] bg-[#08090c]/90 text-[#00f0ff] shadow-[#00f0ff]/10'
              }`}>
                <span className="tracking-widest font-black uppercase text-xs sm:text-sm font-share text-[#00f0ff]">
                  ▶ [ CPR_COMPRESSION_TRAINING: ACTIVE ]
                </span>
                
                {/* 3-Column Scientific telemetry Grid - scaled up massively to prevent posture-breaking */}
                <div className="w-full grid grid-cols-3 gap-2 py-3 border-y border-opacity-25 border-current my-2 text-[10px] sm:text-xs uppercase tracking-wider font-semibold">
                  <div className="flex flex-col items-center border-r border-opacity-25 border-current px-1">
                    <span className="text-[10px] sm:text-xs font-bold opacity-60">BPM</span>
                    <span className="text-3xl sm:text-4xl font-extrabold leading-none my-1.5 font-share">{cprBpmRef.current || '--'}</span>
                    <span className="text-[8px] sm:text-[9px] font-black tracking-widest text-center truncate w-full">
                      {cprState === 'CPR_RATE_SLOW' ? 'SPEED UP' : cprState === 'CPR_RATE_FAST' ? 'SLOW DOWN' : cprState === 'CPR_RATE_GOOD' ? 'GOOD RATE' : 'STANDBY'}
                    </span>
                  </div>
                  
                  <div className="flex flex-col items-center border-r border-opacity-25 border-current px-1">
                    <span className="text-[10px] sm:text-xs font-bold opacity-60">DEPTH</span>
                    <span className="text-2xl sm:text-3xl font-extrabold leading-none my-2 font-share">
                      {cprDepthRatioRef.current < 0.045 ? 'SHALLOW' : 
                       cprDepthRatioRef.current > 0.06 ? 'DEEP' : 
                       cprDepthRatioRef.current > 0 ? 'GOOD' : '--'}
                    </span>
                    <span className="text-[8px] sm:text-[9px] font-black tracking-widest text-center truncate w-full">
                      {cprDepthRatioRef.current > 0 ? 'COMPRESSION' : 'NO DATA'}
                    </span>
                  </div>

                  <div className="flex flex-col items-center px-1">
                    <span className="text-[10px] sm:text-xs font-bold opacity-60">ALIGN</span>
                    <span className={`text-2xl sm:text-3xl font-extrabold leading-none my-2 font-share ${cprPlacementValidRef.current ? 'text-[#00ff66]' : 'text-[#ff3366]'}`}>
                      {cprPlacementValidRef.current ? '✓ OK' : '✕ OFF'}
                    </span>
                    <span className="text-[8px] sm:text-[9px] font-black tracking-widest text-center truncate w-full">
                      {cprPlacementValidRef.current ? 'CENTERED' : 'OFF CENTER'}
                    </span>
                  </div>
                </div>

                <span className="text-xs sm:text-sm font-semibold tracking-wide lowercase opacity-90">
                  {cprState === 'CPR_ALIGN_BODY' ? 'align body — kneel centered in frame' :
                   cprState === 'CPR_POSITION_HANDS' ? 'position hands — place heel of palms on target' :
                   'compress at 100-120/min • push hard, push fast'}
                </span>
              </div>
            )}


          </div>
        </div>

        {/* Offline Standby Dashboard - Exceptional layout for both Desktop and Mobile */}
        {!isActive && (
          <div className="flex flex-col items-center justify-center p-6 text-center w-full max-w-sm sm:max-w-md gap-6 select-none animate-fadeIn">
            {/* Header */}
            <div className="flex flex-col gap-1.5 items-center">
              <span className="text-neutral-600 text-[10px] tracking-[0.25em] uppercase font-black">first-aid interactive guide</span>
              <h1 className="text-2xl sm:text-3xl font-black tracking-wider text-white uppercase leading-none my-1 font-share">BRIFF // SENSOR_HUB</h1>
              <span className="text-[8px] font-bold text-[#00f0ff] opacity-80 tracking-widest uppercase">multimodal pose & hand tracker</span>
            </div>

            {/* Segmented Mode Selector on Standby */}
            <div className="flex flex-col gap-2.5 w-full bg-[#08090c]/50 border border-border-dark p-3.5 rounded-2xl">
              <span className="text-[8px] text-neutral-500 font-bold uppercase tracking-wider text-left">select training module:</span>
              <div className="flex bg-[#0c0d12] border border-border-dark/60 p-0.5 rounded-xl w-full justify-around items-center">
                {['PULSE', 'CPR'].map((mode) => (
                  <button
                    key={mode}
                    onClick={() => {
                      setActiveMode(mode);
                      resumeAudioContext();
                      if (mode !== 'PULSE') {
                        setPulseCheckState('OFFLINE');
                      }
                    }}
                    className={`flex-1 px-3.5 py-2.5 rounded-lg text-[9px] font-black uppercase transition-all tracking-wider text-center cursor-pointer font-share ${
                      activeMode === mode
                        ? 'bg-[#00f0ff] text-neutral-dark shadow-[0_0_12px_rgba(0,240,255,0.4)]'
                        : 'text-neutral-500 hover:text-neutral-300'
                    }`}
                  >
                    {mode}
                  </button>
                ))}
              </div>
              
              {/* Dynamic Mode Description */}
              <div className="text-[9.5px] text-neutral-400 font-medium text-left leading-relaxed mt-1 h-8 flex items-center">
                {activeMode === 'PULSE' && 'verifies finger alignment on the carotid artery points below the jaw line.'}
                {activeMode === 'CPR' && 'tracks hand overlap compression rate (BPM), relative depth, and center placement.'}
              </div>

              {/* Collapsible clinical landmarks guide button on standby */}
              <button
                onClick={() => setShowHelp(true)}
                aria-label="View detailed medical landmarks guide"
                className="w-full py-2 bg-neutral-950/40 hover:bg-neutral-900/60 border border-border-dark/60 hover:border-neutral-700/80 text-neutral-400 hover:text-white text-[8.5px] font-black tracking-widest uppercase rounded-xl transition-all cursor-pointer flex items-center justify-center gap-1.5 h-8 font-share mt-1"
              >
                🔍 VIEW CLINICAL LANDMARKS GUIDE
              </button>
            </div>

            {/* Launch / Camera Start Button & State Progress */}
            <div className="flex flex-col gap-3 items-center w-full">
              {errorMsg ? (
                <div className="px-4 py-2.5 bg-red-950/20 border border-red-900/50 rounded-xl text-red-500 text-[10px] uppercase font-bold tracking-wider w-full">
                  ⚠️ {errorMsg}
                </div>
              ) : (
                <div className="flex flex-col gap-3.5 w-full">
                  <button 
                    onClick={() => startCamera()}
                    aria-label="Connect live camera feed stream"
                    className="w-full py-4 bg-transparent border-2 border-neutral-700 hover:border-[#00f0ff] hover:text-[#00f0ff] text-white text-[10px] font-black tracking-[0.2em] uppercase rounded-xl transition-all cursor-pointer hover:shadow-[0_0_15px_rgba(0,240,255,0.25)] flex items-center justify-center gap-2 h-12 font-share"
                  >
                    📡 CONNECT STREAM
                  </button>

                  {modelStatus !== 'READY' && (
                    <div className="flex flex-col gap-2 items-center bg-[#08090c]/40 border border-border-dark/50 p-3 rounded-xl w-full animate-fadeIn">
                      <span className="text-neutral-500 text-[8.5px] font-bold animate-pulse tracking-wider">
                        {modelStatus === 'LOADING_MODEL' && '[ LOADING WASM SENSORS ]'}
                        {modelStatus === 'LOADING_WASM' && '[ COMPILING COMPUTER VISION ENGINE ]'}
                        {modelStatus === 'LOADING_POSE_MODEL' && '[ RESOLVING ANATOMICAL POSE TARGETS ]'}
                        {modelStatus === 'LOADING_HAND_MODEL' && '[ CONFIGURING MULTI-HAND SENSORS ]'}
                        {modelStatus === 'FAILED' && '[ SENSORS_INITIALIZATION_FAILED ]'}
                      </span>
                      <div className="w-full bg-neutral-900 h-1.5 rounded-full overflow-hidden p-[1px]">
                        <div className={`h-full bg-[#00f0ff] rounded-full transition-all duration-500 ${
                          modelStatus === 'LOADING_MODEL' ? 'w-1/5' :
                          modelStatus === 'LOADING_WASM' ? 'w-2/5' :
                          modelStatus === 'LOADING_POSE_MODEL' ? 'w-3/5' :
                          modelStatus === 'LOADING_HAND_MODEL' ? 'w-4/5' : 'w-0'
                        }`}></div>
                      </div>
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Responsive, Touch-Friendly Bottom Control Panel */}
        {isActive && (
          <div className="absolute bottom-4 left-4 right-4 flex flex-col sm:flex-row gap-4 sm:gap-x-3 sm:gap-y-2 justify-between items-center bg-[#08090c]/85 backdrop-blur-md px-4 py-4 sm:px-4 sm:py-2.5 border border-border-dark text-[10px] text-neutral-400 rounded-2xl z-20 pb-[max(1rem,env(safe-area-inset-bottom))]">
            
            {/* Controls: Left Utilities */}
            <div className="flex flex-col sm:flex-row items-center gap-2.5 max-sm:w-full">
              <button 
                onClick={() => setShowSkeleton(!showSkeleton)}
                aria-label="Toggle WASM skeletal tracking overlay"
                className={`flex items-center gap-2.5 cursor-pointer transition-all select-none py-2 px-4 border rounded-xl max-sm:w-full max-sm:justify-center h-10 sm:h-8 ${
                  showSkeleton 
                    ? 'bg-[#00f0ff]/10 border-[#00f0ff]/40 text-white hover:bg-[#00f0ff]/15 hover:border-[#00f0ff]'
                    : 'bg-[#0c0d12]/60 border-border-dark text-neutral-500 hover:text-neutral-300 hover:border-neutral-700'
                }`}
                title="Click to toggle skeletal overlay wireframe lines"
              >
                <span className={`w-2 h-2 rounded-full ${showSkeleton ? 'bg-[#00f0ff] shadow-[0_0_8px_#00f0ff] animate-pulse' : 'bg-neutral-600'}`}></span>
                <span className="text-[9px] tracking-widest uppercase font-black font-share">
                  {showSkeleton ? 'SKELETON: ON' : 'SKELETON: OFF'}
                </span>
              </button>
              
              {/* Metronome Control (Only visible in CPR Mode) */}
              {activeMode === 'CPR' && (
                <button 
                  onClick={() => {
                    setMuteMetronome(!muteMetronome);
                    resumeAudioContext();
                  }}
                  aria-label="Toggle CPR metronome audio beeps"
                  className={`flex items-center gap-2.5 cursor-pointer transition-all select-none py-2 px-4 border rounded-xl max-sm:w-full max-sm:justify-center h-10 sm:h-8 ${
                    !muteMetronome 
                      ? 'bg-amber-500/10 border-amber-500/40 text-amber-400 hover:bg-amber-500/15 hover:border-amber-500'
                      : 'bg-[#0c0d12]/60 border-border-dark text-neutral-500 hover:text-neutral-300 hover:border-neutral-700'
                  }`}
                  title="Mute or unmute metronome training tone"
                >
                  <span className={`w-2 h-2 rounded-full ${!muteMetronome ? 'bg-amber-400 shadow-[0_0_8px_#ffaa00] animate-pulse' : 'bg-neutral-600'}`}></span>
                  <span className="text-[9px] tracking-widest uppercase font-black font-share">
                    {!muteMetronome ? 'METRONOME: ON' : 'METRONOME: OFF'}
                  </span>
                </button>
              )}
            </div>
            
            {/* Center Utility: Program Segmented Selector */}
            <div className="flex bg-[#0c0d12] border border-border-dark p-0.5 rounded-xl max-sm:w-full justify-around h-10 sm:h-auto items-center">
              {['PULSE', 'CPR'].map((mode) => (
                <button
                  key={mode}
                  onClick={() => {
                    setActiveMode(mode);
                    resumeAudioContext();
                    if (mode !== 'PULSE') {
                      setPulseCheckState('OFFLINE');
                    }
                  }}
                  className={`flex-1 sm:flex-none px-4 sm:px-5 py-2 rounded-lg text-[9px] font-extrabold uppercase transition-all tracking-wider text-center cursor-pointer font-share ${
                    activeMode === mode
                      ? 'bg-[#00f0ff] text-neutral-dark font-black shadow-[0_0_12px_rgba(0,240,255,0.4)]'
                      : 'text-neutral-500 hover:text-neutral-200'
                  }`}
                >
                  {mode}
                </button>
              ))}
            </div>

            {/* Right Utility: Camera Selector and Disconnect */}
            <div className="flex items-center justify-between sm:justify-end gap-3 max-sm:w-full">
              <button 
                onClick={() => setShowHelp(true)}
                aria-label="Open anatomical training guide overlay"
                className="flex items-center justify-center px-3.5 h-10 sm:h-8 bg-[#0c0d12]/60 hover:bg-neutral-900 border border-border-dark hover:border-[#00f0ff]/50 text-neutral-400 hover:text-[#00f0ff] rounded-xl text-[9px] font-black uppercase tracking-wider transition-all cursor-pointer font-share"
                title="Open detailed anatomical placement guide"
              >
                🔍 HELP_GUIDE
              </button>

              {devices.length > 0 && (
                <div className="relative flex items-center bg-[#0c0d12] border border-border-dark rounded-xl px-3 h-10 sm:h-8">
                  <select 
                    id="camera-selector"
                    value={selectedDeviceId}
                    onChange={handleDeviceChange}
                    aria-label="Select camera hardware input source"
                    className="bg-transparent text-neutral-400 border-none outline-none cursor-pointer pr-1 text-[9px] uppercase font-bold tracking-wider hover:text-white"
                  >
                    {devices.map((device, idx) => (
                      <option key={device.deviceId} value={device.deviceId} className="bg-[#08090c] text-white">
                        {device.label || `CAM_${idx + 1}`}
                      </option>
                    ))}
                  </select>
                </div>
              )}

              <button 
                onClick={stopCamera}
                aria-label="Disconnect active camera stream"
                className="flex items-center justify-center px-4 h-10 sm:h-8 bg-red-950/20 hover:bg-red-900/30 text-red-400 hover:text-red-300 border border-red-900/30 hover:border-red-500/50 rounded-xl text-[9px] font-bold uppercase tracking-wider transition-all cursor-pointer max-sm:flex-1"
              >
                [ DISCONNECT ]
              </button>
            </div>
          </div>
        )}

        {/* Clinical Anatomy Slide-over Guide Panel */}
        {showHelp && (
          <div className="absolute inset-0 bg-[#08090c]/90 backdrop-blur-md z-40 flex justify-end transition-all duration-300 animate-fadeIn rounded-none sm:rounded-3xl pointer-events-auto">
            <div className="w-full sm:w-[420px] h-full bg-[#0c0d12] border-l border-border-dark flex flex-col relative shadow-2xl overflow-y-auto rounded-r-none sm:rounded-r-3xl">
              {/* Top sticky bar */}
              <div className="flex justify-between items-center px-6 py-4 border-b border-border-dark sticky top-0 bg-[#0c0d12]/95 backdrop-blur-sm z-10">
                <span className="text-xs font-black uppercase text-[#00f0ff] tracking-widest font-share">▶ [ CLINICAL_ANATOMY_GUIDE ]</span>
                <button 
                  onClick={() => setShowHelp(false)}
                  className="text-neutral-500 hover:text-white text-[10px] font-black tracking-widest uppercase cursor-pointer bg-[#181c26]/60 border border-border-dark px-3 py-1.5 rounded-lg hover:border-red-500/50 hover:text-red-400 transition-all font-share"
                >
                  ✕ CLOSE
                </button>
              </div>

              {/* Content body */}
              <div className="p-6 flex flex-col gap-6 text-[11px] leading-relaxed text-neutral-300">
                
                {/* Section: Carotid Pulse */}
                <div className="flex flex-col gap-3">
                  <span className="text-[10px] font-black uppercase text-[#ff3366] tracking-widest border-b border-border-dark pb-2 font-share">
                    I. CAROTID PULSE LOCALIZATION
                  </span>
                  <div className="p-3 bg-neutral-950/40 border border-border-dark/60 rounded-xl flex flex-col gap-2">
                    <span className="font-extrabold text-[#00f0ff] uppercase text-[10px]">Anatomical Landmarks:</span>
                    <p className="opacity-90 font-mono">
                      The carotid pulse is located on the side of the neck, between the windpipe (laryngeal cartilage) and the surrounding muscle (sternocleidomastoid).
                    </p>
                    
                    {/* ASCII Neck target */}
                    <div className="my-2 border border-dashed border-[#ff3366]/30 p-3 bg-[#08090c]/80 rounded-lg text-center flex flex-col items-center">
                      <pre className="text-left text-[#ff3366] leading-tight text-[8px] sm:text-[9.5px] font-bold font-mono">
{`   \\  _   _ /
    \`|_| |_|\`     <- Jaw Line
     |  o  |       <- Windpipe Center
    /| (•) |\\      <- CAROTID TARGETS (Neon circle)
   / |  |  | \\
  /  |==|==|  \\    <- Shoulder Line`}
                      </pre>
                    </div>

                    <span className="font-extrabold text-white text-[10px] uppercase">Execution Protocol:</span>
                    <ul className="list-decimal list-inside pl-1 flex flex-col gap-2 opacity-85 font-mono">
                      <li>Place index and middle fingertips directly adjacent to the windpipe, right under the corner of the jaw.</li>
                      <li>Keep fingers held close together. Avoid using the thumb, as it has its own pulse.</li>
                      <li>Apply moderate pressure until the neon green verification indicator locks and starts pulsing with your heartbeat.</li>
                    </ul>
                  </div>
                </div>

                {/* Section: CPR Chest Compressions */}
                <div className="flex flex-col gap-3">
                  <span className="text-[10px] font-black uppercase text-[#ffaa00] tracking-widest border-b border-border-dark pb-2 font-share">
                    II. CHEST COMPRESSIONS (CPR)
                  </span>
                  <div className="p-3 bg-neutral-950/40 border border-border-dark/60 rounded-xl flex flex-col gap-2">
                    <span className="font-extrabold text-[#00f0ff] uppercase text-[10px]">Sternum Target Center:</span>
                    <p className="opacity-90 font-mono">
                      Hands must be locked together and placed directly over the lower half of the sternum (breastbone), which is centered between the nipples.
                    </p>
                    
                    {/* ASCII Chest target */}
                    <div className="my-2 border border-dashed border-[#ffaa00]/30 p-3 bg-[#08090c]/80 rounded-lg text-center flex flex-col items-center">
                      <pre className="text-left text-[#ffaa00] leading-tight text-[8px] sm:text-[9.5px] font-bold font-mono">
{`    /|  Nose  |\\
   / |        | \\
  /  |--[  ]--|  \\   <- Armpit level
 |   |   __   |   |
 |   |  /_/|  |   |  <- HEEL OF PALMS (Sternum)
 |   |  |_|/  |   |  <- Lower breastbone
  \\  |        |  /`}
                      </pre>
                    </div>

                    <span className="font-extrabold text-white text-[10px] uppercase font-mono">Execution Protocol:</span>
                    <ul className="list-decimal list-inside pl-1 flex flex-col gap-2 opacity-85 font-mono">
                      <li>Position yourself so your shoulders are directly over your hands. Lock your elbows straight.</li>
                      <li>Interlock fingers of both hands, pulling fingers up so only the heel of your palm contacts the sternum.</li>
                      <li>Compress hard (at least 2 inches) and fast at a constant pace of 100 to 120 compressions per minute.</li>
                      <li>Use the built-in metronome beeps to calibrate your rhythm to exactly 110 compressions per minute.</li>
                    </ul>
                  </div>
                </div>

              </div>
            </div>
          </div>
        )}

      </div>

      {/* Tiny minimal subtle status footer outside the 90% frame */}
      <div className="hidden sm:flex h-[4vh] sm:h-[5vh] items-center text-[8px] sm:text-[9px] text-neutral-600 uppercase tracking-widest font-share">
        BRIFF // WASM_POSE_AND_HAND_SENSORS
      </div>

    </div>
  );
}

export default App;
