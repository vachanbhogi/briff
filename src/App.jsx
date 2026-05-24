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

  // Throttled CPR Telemetry Sync States
  const [displayCprBpm, setDisplayCprBpm] = useState(0);
  const [displayCprDepthRatio, setDisplayCprDepthRatio] = useState(0);
  const [displayCprPlacementValid, setDisplayCprPlacementValid] = useState(false);
  const lastStateSyncTimeRef = useRef(0);

  // Heimlich Training States
  const [displayHeimlichPhase, setDisplayHeimlichPhase] = useState('STANCE');
  const [displayHeimlichJHookValid, setDisplayHeimlichJHookValid] = useState(false);
  const [displayHeimlichHandsClasped, setDisplayHeimlichHandsClasped] = useState(false);
  const heimlichPhaseRef = useRef('STANCE');
  const heimlichJHookValidRef = useRef(false);
  const heimlichHandsClaspedRef = useRef(false);
  const heimlichTrajectoryRef = useRef([]);
  const heimlichAnchorRef = useRef(null);
  const heimlichSuccessTimeRef = useRef(0);

  // Web Serial API States for Arduino
  const [arduinoConnected, setArduinoConnected] = useState(false);
  const serialWriterRef = useRef(null);
  const lastSentCommandRef = useRef(null);

  // Additional settings: metronome muting & clinical help overlays
  const [muteMetronome, setMuteMetronome] = useState(false);
  const [showHelp, setShowHelp] = useState(false);
  const muteMetronomeRef = useRef(false);

  useEffect(() => {
    muteMetronomeRef.current = muteMetronome;
  }, [muteMetronome]);

  // Dialog Close on Esc
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Escape' && showHelp) setShowHelp(false);
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [showHelp]);

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
        } else if (activeModeRef.current === 'HEIMLICH') {
          const leftShoulder = landmarks[11];
          const rightShoulder = landmarks[12];
          const leftHip = landmarks[23];
          const rightHip = landmarks[24];
          const leftWrist = landmarks[15];
          const rightWrist = landmarks[16];

          if (leftShoulder && rightShoulder && leftHip && rightHip) {
            const midShoulderY = (leftShoulder.y + rightShoulder.y) / 2;
            const midShoulderX = (leftShoulder.x + rightShoulder.x) / 2;
            const midHipY = (leftHip.y + rightHip.y) / 2;

            heimlichAnchorRef.current = {
              x: midShoulderX,
              y: midShoulderY + (midHipY - midShoulderY) * 0.5
            };

            // Phase 1: Stance Check (Sideways)
            // If the horizontal distance between left and right shoulder is very small, they are sideways
            const shoulderWidth = Math.abs(leftShoulder.x - rightShoulder.x);
            const isSideways = shoulderWidth < 0.15; // threshold for profile view

            if (heimlichPhaseRef.current === 'STANCE') {
              if (isSideways) {
                heimlichPhaseRef.current = 'HANDS';
              }
            }

            // Phase 2: Hand Placement
            if (heimlichPhaseRef.current === 'HANDS' || heimlichPhaseRef.current === 'THRUST') {
              let isClasped = false;
              let isCorrectHeight = false;
              let handsCenter = null;

              if (leftWrist && rightWrist) {
                // Check if wrists are close together
                const wristDist = Math.hypot(leftWrist.x - rightWrist.x, leftWrist.y - rightWrist.y);
                isClasped = wristDist < 0.15;

                // Check if hands are between chest and hips
                const wristY = (leftWrist.y + rightWrist.y) / 2;
                isCorrectHeight = wristY > midShoulderY && wristY < midHipY;
                
                handsCenter = {
                  x: (leftWrist.x + rightWrist.x) / 2,
                  y: wristY
                };
              } else if (leftWrist || rightWrist) {
                // From a side profile, one hand might occlude the other
                isClasped = true; // assume clasped if sideways and one hand is tracking
                const wrist = leftWrist || rightWrist;
                isCorrectHeight = wrist.y > midShoulderY && wrist.y < midHipY;
                handsCenter = { x: wrist.x, y: wrist.y };
              }

              heimlichHandsClaspedRef.current = isClasped && isCorrectHeight;

              if (heimlichPhaseRef.current === 'HANDS' && isClasped && isCorrectHeight && isSideways) {
                heimlichPhaseRef.current = 'THRUST';
              } else if (!isSideways) {
                // If they turn back around, reset to stance, unless we are in the 3-second victory window
                if (startTimeMs - heimlichSuccessTimeRef.current > 3000) {
                  heimlichPhaseRef.current = 'STANCE';
                  heimlichJHookValidRef.current = false;
                  heimlichTrajectoryRef.current = [];
                }
              }

              // Phase 3: Thrust Trajectory Tracking
              if (heimlichPhaseRef.current === 'THRUST' && handsCenter && isClasped) {
                const trajectory = heimlichTrajectoryRef.current;
                trajectory.push({ x: handsCenter.x, y: handsCenter.y, time: startTimeMs });
                
                // Keep the last 1.5 seconds of trajectory
                while (trajectory.length > 0 && startTimeMs - trajectory[0].time > 1500) {
                  trajectory.shift();
                }

                // Analyze trajectory for J-Hook
                if (trajectory.length > 10) {
                  const startP = trajectory[0];
                  const endP = trajectory[trajectory.length - 1];
                  
                  // Need to see rapid movement: Inward (X changes based on facing) and Upward (Y decreases)
                  // From a side profile, "Inward" is horizontal movement toward the spine
                  const deltaY = startP.y - endP.y; // Positive means upward
                  const deltaX = Math.abs(startP.x - endP.x); // Absolute horizontal movement
                  
                  const velocityY = deltaY / (endP.time - startP.time);
                  
                  // If they moved up significantly and fast enough, and had some horizontal motion
                  if (deltaY > 0.08 && deltaX > 0.02 && velocityY > 0.0001) {
                    heimlichJHookValidRef.current = true;
                    heimlichSuccessTimeRef.current = startTimeMs;
                  }
                }
              } else if (heimlichPhaseRef.current === 'THRUST' && !isClasped) {
                 // Reset thrust validation if hands un-clasp, unless we are in the 3-second victory window
                 if (startTimeMs - heimlichSuccessTimeRef.current > 3000) {
                   heimlichJHookValidRef.current = false;
                   heimlichTrajectoryRef.current = [];
                   heimlichPhaseRef.current = 'HANDS';
                 }
              }
              
              // Force state lock for 3 seconds after success to let Arduino play the victory chime
              if (heimlichJHookValidRef.current && (startTimeMs - heimlichSuccessTimeRef.current < 3000)) {
                 heimlichPhaseRef.current = 'THRUST'; // Keep in thrust phase
              } else if (heimlichJHookValidRef.current && (startTimeMs - heimlichSuccessTimeRef.current >= 3000)) {
                 // Reset after 3 seconds
                 heimlichJHookValidRef.current = false;
                 heimlichTrajectoryRef.current = [];
                 if (!isClasped) {
                   heimlichPhaseRef.current = 'HANDS';
                 }
              }
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

          // State Machine & Arduino Serial Transmission
          let newCprState = 'CPR_POSITION_HANDS';
          let ardCommand = 'S';
          
          if (!isValid) {
            newCprState = 'CPR_POSITION_HANDS';
          } else {
            const bpm = cprBpmRef.current;
            if (bpm === 0) {
               newCprState = 'CPR_COMPRESSING';
            } else if (bpm < 100) {
               newCprState = 'CPR_RATE_SLOW';
               ardCommand = 'L';
            } else if (bpm <= 120) {
               newCprState = 'CPR_RATE_GOOD';
               ardCommand = 'G';
            } else {
               newCprState = 'CPR_RATE_FAST';
               ardCommand = 'F';
            }
          }
          setCprState(newCprState);
          
          if (serialWriterRef.current && ardCommand !== lastSentCommandRef.current) {
            lastSentCommandRef.current = ardCommand;
            serialWriterRef.current.write(ardCommand).catch(e => console.error(e));
          }
        } else {
          setCprState('CPR_ALIGN_BODY');
          if (serialWriterRef.current && lastSentCommandRef.current !== 'S') {
            lastSentCommandRef.current = 'S';
            serialWriterRef.current.write('S').catch(e => console.error(e));
          }
        }
      } else if (activeModeRef.current === 'HEIMLICH') {
        const valid = heimlichJHookValidRef.current;
        const ardCommand = valid ? 'V' : 'S';
        
        if (serialWriterRef.current && lastSentCommandRef.current !== ardCommand) {
          lastSentCommandRef.current = ardCommand;
          serialWriterRef.current.write(ardCommand).catch(e => console.error(e));
        }
        
        const phase = heimlichPhaseRef.current;
        
        let cx = canvas.width / 2;
        let cy = canvas.height / 2;
        
        if (heimlichAnchorRef.current) {
          cx = heimlichAnchorRef.current.x * canvas.width;
          cy = heimlichAnchorRef.current.y * canvas.height;
        }

        ctx.save();
        // The canvas is scale(-1, 1) earlier in CSS, but for drawing text/arrows we need to account for the canvas context scale if any
        ctx.scale(-1, 1);
        ctx.font = 'bold 16px monospace';
        ctx.textAlign = 'center';
        
        if (phase === 'STANCE') {
          // Draw arrows indicating to turn sideways
          ctx.fillStyle = 'rgba(255, 170, 0, 0.8)';
          ctx.fillText('TURN 90° (PROFILE VIEW)', -cx, cy - 100);
          
          // Draw rotation arrows
          ctx.beginPath();
          ctx.ellipse(-cx, cy, 60, 20, 0, Math.PI, 2 * Math.PI);
          ctx.strokeStyle = '#ffaa00';
          ctx.lineWidth = 4;
          ctx.stroke();
          
          ctx.beginPath();
          ctx.moveTo(-cx - 60, cy);
          ctx.lineTo(-cx - 70, cy - 15);
          ctx.lineTo(-cx - 50, cy - 15);
          ctx.fill();
        } else if (phase === 'HANDS') {
          // Draw target zone for hands (between chest and navel)
          ctx.fillStyle = 'rgba(0, 255, 102, 0.8)';
          ctx.fillText('CLASP HANDS HERE', -cx, cy - 60);
          
          ctx.beginPath();
          ctx.rect(-cx - 50, cy - 40, 100, 80);
          ctx.strokeStyle = '#00ff66';
          ctx.lineWidth = 2;
          ctx.setLineDash([8, 6]);
          ctx.stroke();
          ctx.setLineDash([]);
        } else if (phase === 'THRUST') {
          ctx.fillStyle = valid ? '#00ff66' : 'rgba(0, 240, 255, 0.8)';
          ctx.fillText(valid ? 'J-HOOK SUCCESS!' : 'EXECUTE J-HOOK THRUST', -cx, cy - 100);
          
          // Draw J-Hook arrow
          ctx.beginPath();
          ctx.moveTo(-cx + 40, cy + 40);
          ctx.lineTo(-cx - 20, cy + 40); // inward
          ctx.lineTo(-cx - 20, cy - 40); // upward
          
          ctx.strokeStyle = valid ? '#00ff66' : '#00f0ff';
          ctx.lineWidth = 8;
          ctx.lineJoin = 'round';
          ctx.stroke();
          
          // Arrowhead
          ctx.beginPath();
          ctx.moveTo(-cx - 20, cy - 40);
          ctx.lineTo(-cx - 35, cy - 20);
          ctx.lineTo(-cx - 5, cy - 20);
          ctx.fillStyle = valid ? '#00ff66' : '#00f0ff';
          ctx.fill();

          if (valid) {
             const pulseVal = Math.sin(performance.now() / 150) * 5;
             ctx.beginPath();
             ctx.arc(-cx, cy - 60, 80 + pulseVal, 0, 2*Math.PI);
             ctx.fillStyle = 'rgba(0, 255, 102, 0.2)';
             ctx.fill();
          }
        }
        ctx.restore();
      }
    }

    const nowMs = performance.now();
    if (nowMs - lastStateSyncTimeRef.current > 200) {
      lastStateSyncTimeRef.current = nowMs;
      setDisplayCprBpm(cprBpmRef.current);
      setDisplayCprDepthRatio(cprDepthRatioRef.current);
      setDisplayCprPlacementValid(cprPlacementValidRef.current);
      setDisplayHeimlichPhase(heimlichPhaseRef.current);
      setDisplayHeimlichJHookValid(heimlichJHookValidRef.current);
      setDisplayHeimlichHandsClasped(heimlichHandsClaspedRef.current);
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

  const connectArduino = async () => {
    try {
      console.log('Requesting Arduino port...');
      if (!('serial' in navigator)) {
        setErrorMsg('WEB SERIAL API NOT SUPPORTED. USE CHROME/EDGE.');
        return;
      }
      const port = await navigator.serial.requestPort();
      await port.open({ baudRate: 9600 });
      const textEncoder = new TextEncoderStream();
      textEncoder.readable.pipeTo(port.writable);
      serialWriterRef.current = textEncoder.writable.getWriter();
      setArduinoConnected(true);
      console.log('Arduino connected successfully!');
    } catch (err) {
      console.error('Serial connection error:', err);
      // User cancelled or port in use
    }
  };

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
      
      heimlichPhaseRef.current = 'STANCE';
      heimlichJHookValidRef.current = false;
      heimlichHandsClaspedRef.current = false;
      heimlichTrajectoryRef.current = [];
      heimlichAnchorRef.current = null;
      heimlichSuccessTimeRef.current = 0;

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
    heimlichPhaseRef.current = 'STANCE';
    heimlichAnchorRef.current = null;
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
    <main className="w-full h-screen bg-[#000000] text-[#8ab4f8] font-mono text-xs overflow-hidden flex flex-col p-1 sm:p-2 border-2 sm:border-8 border-[#000000] box-border relative select-none">
      <div className="mri-scanline"></div>
      
      {/* HEADER */}
      <div className="flex justify-between items-center px-4 py-2 border border-[#1a2f3d] mb-1 sm:mb-2 bg-[#030b14] shrink-0">
        <div className="flex items-center gap-4">
          <span className="font-bold tracking-widest text-[#e8f0fe] hidden sm:inline">BRIFF_DIAGNOSTICS</span>
          <span className="font-bold tracking-widest text-[#e8f0fe] sm:hidden">BRIFF</span>
        </div>
        <div className="flex items-center gap-2">
          {arduinoConnected ? (
            <span className="text-[#34d399] text-[10px] sm:text-xs border border-[#34d399] px-2 py-1 mr-2 bg-[#34d399]/10">USB_LINKED</span>
          ) : null}
          <select 
            className="bg-[#030b14] border border-[#1a2f3d] text-[#8ab4f8] text-[10px] p-1 outline-none w-20 sm:w-24 truncate"
            value={selectedDeviceId || ''}
            onChange={handleDeviceChange}
          >
            {devices.map(device => (
              <option key={device.deviceId} value={device.deviceId}>
                {device.label || `CAM_${device.deviceId.slice(0,4)}`}
              </option>
            ))}
          </select>
          <button 
            onClick={() => setShowHelp(true)}
            className="border border-[#8ab4f8] text-[#8ab4f8] px-2 py-1 text-[10px] hover:bg-[#1a2f3d]"
          >
            ?_MANUAL
          </button>
          {errorMsg ? (
            <span className="text-[#f87171] animate-pulse">SYS_ERR</span>
          ) : modelStatus !== 'READY' ? (
            <span className="text-[#fbbf24] animate-pulse">INIT</span>
          ) : (
            <span className="text-[#34d399]">ONLINE</span>
          )}
          <span className="w-2 h-2 bg-[#34d399] rounded-none animate-pulse"></span>
        </div>
      </div>

      {/* 6-PANEL GRID LAYOUT */}
      <div className="flex-1 flex flex-col sm:grid sm:grid-cols-3 sm:grid-rows-2 gap-1 sm:gap-2 min-h-0 overflow-y-auto sm:overflow-hidden">
        
        {/* VIEWPORT 1 (MAIN CAMERA FEED) - Spans 2 rows and 2 columns */}
        <div className="bg-[#000000] relative flex-none h-[75vh] sm:h-auto sm:col-span-2 sm:row-span-2 border border-[#1a2f3d] flex items-center justify-center overflow-hidden shrink-0">
          {/* Axis Labels */}
          <div className="absolute top-2 left-2 text-[10px] text-[#45627a] pointer-events-none z-20">AXIAL</div>
          <div className="absolute bottom-2 left-2 text-[10px] text-[#45627a] pointer-events-none z-20">A</div>
          <div className="absolute top-2 right-2 text-[10px] text-[#45627a] pointer-events-none z-20">L</div>
          <div className="absolute bottom-2 right-2 text-[10px] text-[#45627a] pointer-events-none z-20">P</div>
          <div className="absolute top-1/2 right-2 -translate-y-1/2 text-[10px] text-[#45627a] pointer-events-none z-20">R</div>

          {/* Crosshairs */}
          <div className="absolute top-0 bottom-0 left-1/2 border-l border-dashed border-[#1a2f3d] opacity-50 z-10 pointer-events-none"></div>
          <div className="absolute left-0 right-0 top-1/2 border-t border-dashed border-[#1a2f3d] opacity-50 z-10 pointer-events-none"></div>

          {/* Video Feed */}
          <div className={`w-full h-full relative ${isActive ? 'block' : 'hidden'}`}>
            <video ref={videoRef} autoPlay playsInline muted className="w-full h-full object-contain scale-x-[-1]" />
            <canvas ref={canvasRef} className="absolute inset-0 w-full h-full object-contain pointer-events-none scale-x-[-1] z-20" />
            
            {/* Real-time Telemetry Data Overlay */}
            <div className="absolute bottom-4 left-4 z-30 flex flex-col gap-1 font-mono pointer-events-none bg-[#000000]/60 p-2 border border-[#1a2f3d]">
              {activeMode === 'PULSE' && (
                <>
                  <span className={`text-xs ${pulseCheckState === 'CORRECT' ? 'text-[#34d399]' : 'text-[#8ab4f8]'}`}>
                    &gt; TRG_ACQ: {pulseCheckState}
                  </span>
                  <div className="flex items-baseline gap-2 mt-1">
                    <span className="text-2xl sm:text-5xl font-bold text-[#e8f0fe]">{isActive ? 'CALC...' : '--'}</span>
                    <span className="text-[#45627a] text-[10px] sm:text-xs">BPM</span>
                  </div>
                </>
              )}
              {activeMode === 'CPR' && (
                <>
                  <span className={`text-xs ${cprState === 'CPR_RATE_GOOD' ? 'text-[#34d399]' : 'text-[#fbbf24]'}`}>
                    &gt; CPR_STAT: {cprState}
                  </span>
                  <div className="flex items-baseline gap-2 mt-1">
                    <span className="text-2xl sm:text-5xl font-bold text-[#e8f0fe]">{isActive ? displayCprBpm : '--'}</span>
                    <span className="text-[#45627a] text-[10px] sm:text-xs">BPM</span>
                  </div>
                  <span className="text-[#45627a] text-xs">DPTH: {isActive ? displayCprDepthRatio.toFixed(3) : '--'} </span>
                  <span className="text-[#45627a] text-xs">ALGN: {displayCprPlacementValid ? 'TRUE' : 'FALSE'}</span>
                </>
              )}
              {activeMode === 'HEIMLICH' && (
                <>
                  <span className={`text-xs ${displayHeimlichJHookValid ? 'text-[#34d399]' : 'text-[#fbbf24]'}`}>
                    &gt; TRAINING_PHASE: {displayHeimlichPhase}
                  </span>
                  <div className="flex flex-col gap-1 mt-2 text-xs">
                    <span className="text-[#45627a]">
                      SIDWAYS: {displayHeimlichPhase !== 'STANCE' ? <span className="text-[#34d399]">TRUE</span> : <span className="text-[#f87171]">FALSE</span>}
                    </span>
                    <span className="text-[#45627a]">
                      HANDS: {displayHeimlichHandsClasped ? <span className="text-[#34d399]">CLASPED</span> : <span className="text-[#f87171]">WAITING</span>}
                    </span>
                    <span className="text-[#45627a]">
                      THRUST: {displayHeimlichJHookValid ? <span className="text-[#34d399] font-bold animate-pulse">J-HOOK DETECTED</span> : '--'}
                    </span>
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Offline/Loading States */}
          {!isActive && !errorMsg && modelStatus === 'READY' && (
             <div className="absolute inset-0 flex items-center justify-center z-20 flex-col gap-4">
               <span className="animate-pulse text-lg tracking-widest">&gt; STANDBY_MODE</span>
               <button onClick={() => startCamera()} className="border border-[#8ab4f8] px-4 py-2 hover:bg-[#1a2f3d] transition-colors uppercase">
                 INITIALIZE SCANNER
               </button>
             </div>
          )}

          {modelStatus !== 'READY' && !errorMsg && (
             <div className="absolute inset-0 flex items-center justify-center z-20 flex-col gap-4">
               <div className="w-16 h-16 border-2 border-dashed border-[#8ab4f8] rounded-full animate-spin"></div>
               <span className="animate-pulse text-[#fbbf24] uppercase">&gt; CALIBRATING_SYSTEM: {modelStatus}</span>
             </div>
          )}

          {errorMsg && (
            <div className="absolute inset-0 flex items-center justify-center z-20">
               <span className="text-[#f87171] border border-[#f87171] px-4 py-2 bg-[#f87171]/10 uppercase">
                 &gt; ERR: {errorMsg}
               </span>
            </div>
          )}
        </div>

        {/* VIEWPORT 2 (DIAGNOSTIC CHARTS/INFO) */}
        <div className="bg-[#000000] relative p-4 flex flex-col border border-[#1a2f3d] justify-between min-h-[200px] sm:min-h-0">
          <div className="absolute top-2 left-2 text-[10px] text-[#45627a] pointer-events-none">CORONAL (DATA_STREAM)</div>
          
          <div className="mt-6 flex flex-col gap-4">
            
            <div className="flex flex-col gap-1 text-[10px] sm:text-xs">
               <span className="text-[#45627a]">SYS_MSG:</span>
               {activeMode === 'PULSE' && (
                 <span className="text-[#e8f0fe]">&gt; ALIGN INDEX & MIDDLE FINGERS TO JAW TARGET.</span>
               )}
               {activeMode === 'CPR' && (
                 <span className="text-[#e8f0fe]">&gt; LOCK HANDS. MAINTAIN 100-120 BPM. CENTER CHEST.</span>
               )}
               {activeMode === 'HEIMLICH' && (
                 <div className="flex flex-col">
                   {displayHeimlichPhase === 'STANCE' && <span className="text-[#e8f0fe]">&gt; TURN 90 DEGREES (PROFILE VIEW) TO CAMERA.</span>}
                   {displayHeimlichPhase === 'HANDS' && <span className="text-[#e8f0fe]">&gt; MAKE A FIST. CLASP HANDS TOGETHER ABOVE NAVEL.</span>}
                   {displayHeimlichPhase === 'THRUST' && <span className="text-[#e8f0fe]">&gt; EXECUTE SHARP INWARD AND UPWARD THRUST.</span>}
                 </div>
               )}
            </div>
          </div>
        </div>

        {/* VIEWPORT 3 (CONTROLS & SETTINGS) */}
        <div className="bg-[#000000] relative p-4 flex flex-col border border-[#1a2f3d] justify-between min-h-[200px] sm:min-h-0">
          <div className="absolute top-2 left-2 text-[10px] text-[#45627a] pointer-events-none">SAGITTAL (SYSTEM_CTRL)</div>
          
          <div className="flex flex-col gap-1 mt-6 text-[10px] sm:text-xs">
            <span className="text-[#45627a] mb-2 border-b border-[#1a2f3d] pb-1 uppercase">Select Protocol:</span>
            
            <button 
              className={`text-left flex items-center justify-between p-2 cursor-pointer transition-colors ${activeMode === 'PULSE' ? 'bg-[#8ab4f8] text-[#000000]' : 'text-[#8ab4f8] hover:bg-[#1a2f3d]'}`} 
              onClick={() => { setActiveMode('PULSE'); resumeAudioContext(); }}
            >
              <span>[1] PULSE VERIFICATION</span>
              {activeMode === 'PULSE' && <span>■</span>}
            </button>
            <button 
              className={`text-left flex items-center justify-between p-2 cursor-pointer transition-colors ${activeMode === 'CPR' ? 'bg-[#8ab4f8] text-[#000000]' : 'text-[#8ab4f8] hover:bg-[#1a2f3d]'}`} 
              onClick={() => { setActiveMode('CPR'); resumeAudioContext(); }}
            >
              <span>[2] CPR COMPRESSIONS</span>
              {activeMode === 'CPR' && <span>■</span>}
            </button>
            <button 
              className={`text-left flex items-center justify-between p-2 cursor-pointer transition-colors ${activeMode === 'HEIMLICH' ? 'bg-[#8ab4f8] text-[#000000]' : 'text-[#8ab4f8] hover:bg-[#1a2f3d]'}`} 
              onClick={() => { setActiveMode('HEIMLICH'); resumeAudioContext(); }}
            >
              <span>[3] HEIMLICH TRAINING</span>
              {activeMode === 'HEIMLICH' && <span>■</span>}
            </button>
            <button 
              className={`text-left flex items-center justify-between p-2 cursor-pointer transition-colors border border-dashed mt-2 ${arduinoConnected ? 'text-[#00ff66] border-[#00ff66] bg-[#00ff66]/10' : 'text-[#fbbf24] border-[#1a2f3d] hover:bg-[#1a2f3d]'}`}
              onClick={connectArduino}
            >
              <span>{arduinoConnected ? '[ ARDUINO CONNECTED ]' : '[ LINK ARDUINO ]'}</span>
            </button>

            {arduinoConnected && (
              <button 
                className="text-left flex items-center justify-between p-2 cursor-pointer transition-colors text-[#00f0ff] border border-dashed border-[#00f0ff] hover:bg-[#00f0ff]/10 mt-2" 
                onClick={() => {
                  if (serialWriterRef.current) {
                    console.log('Manual Test: Sending V');
                    serialWriterRef.current.write('V').catch(e => console.error(e));
                    // Auto stop after 3s like the Heimlich logic
                    setTimeout(() => {
                      if (serialWriterRef.current) {
                        console.log('Manual Test: Sending S');
                        serialWriterRef.current.write('S').catch(e => console.error(e));
                      }
                    }, 3000);
                  }
                }}
              >
                <span>[ TEST ARDUINO BUZZER ]</span>
              </button>
            )}
          </div>

          <div className="mt-4 flex flex-col gap-2 border-t border-[#1a2f3d] pt-2">
            <button 
              onClick={isActive ? stopCamera : () => startCamera()} 
              className={`border p-2 text-center text-[10px] font-bold uppercase transition-colors ${isActive ? 'border-[#f87171] text-[#f87171] hover:bg-[#f87171]/10' : 'border-[#34d399] text-[#34d399] hover:bg-[#34d399]/10'}`}
            >
              {isActive ? 'HALT_SCAN' : 'ENGAGE_SCAN'}
            </button>
          </div>
        </div>

      </div>

      {/* SLIDE-OVER HELP MANUAL - MRI THEMED */}      {showHelp && (
        <div className="absolute inset-0 z-50 flex justify-end">
          <div className="absolute inset-0 bg-[#000000]/80 backdrop-blur-sm" onClick={() => setShowHelp(false)}></div>
          <div className="w-full sm:w-[500px] h-full bg-[#000000] border-l border-[#8ab4f8] flex flex-col relative z-10 overflow-y-auto">
            
            <div className="p-4 border-b border-[#1a2f3d] flex justify-between items-center bg-[#030b14] sticky top-0">
              <h2 className="text-lg text-[#e8f0fe] font-bold tracking-widest">&gt; DIAGNOSTIC_MANUAL.txt</h2>
              <button onClick={() => setShowHelp(false)} className="text-[#f87171] border border-[#f87171] px-2 hover:bg-[#f87171]/20">X_CLOSE</button>
            </div>

            <div className="p-4 flex flex-col gap-6 text-xs text-[#8ab4f8]">
                {/* Section 1 */}
                <div className="flex flex-col gap-2 border border-[#1a2f3d] p-4">
                  <span className="text-[#fbbf24] border-b border-[#1a2f3d] pb-2 font-bold">&gt; MODULE_01: PULSE_ACQ</span>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
                    <div>
                      <p className="text-[#45627a] mb-2">Align index & middle fingers to carotid artery (groove between windpipe and neck muscle).</p>
                      <ul className="text-white list-disc pl-4 space-y-1 ml-2">
                        <li>System tracks hand positioning.</li>
                        <li>Do not use thumb (contains own pulse).</li>
                        <li>Ensure camera sees head & shoulders.</li>
                      </ul>
                    </div>
                    <div className="border border-dashed border-[#1a2f3d] p-2 text-center flex flex-col justify-center items-center text-[#45627a] bg-[#030b14]">
                      <pre className="text-left text-[8px] sm:text-[10px] leading-tight font-mono">
{`   \  _   _ /
    \`|_| |_|\`     <- Jaw Line
     |  o  |       <- Windpipe Center
    /| (•) |\      <- TARGET
   / |  |  | \
  /  |==|==|  \    <- Shoulder Line`}
                      </pre>
                    </div>
                  </div>
                </div>

                {/* Section 2 */}
                <div className="flex flex-col gap-2 border border-[#1a2f3d] p-4">
                  <span className="text-[#fbbf24] border-b border-[#1a2f3d] pb-2 font-bold">&gt; MODULE_02: COMPRESSIONS_CPR</span>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mt-2">
                    <div>
                      <p className="text-[#45627a] mb-2">Lock hands over lower half of sternum. Shoulders directly over hands. Straight elbows.</p>
                      <ul className="text-white list-disc pl-4 space-y-1 ml-2">
                        <li>Only palm heel contacts sternum.</li>
                        <li>Depth: &gt; 2 inches.</li>
                        <li>Rate: 100-120 BPM.</li>
                        <li>Use metronome for sync.</li>
                      </ul>
                    </div>
                    <div className="border border-dashed border-[#1a2f3d] p-2 text-center flex flex-col justify-center items-center text-[#45627a] bg-[#030b14]">
                      <pre className="text-left text-[8px] sm:text-[10px] leading-tight font-mono">
{`    /|  Nose  |\
   / |        | \
  /  |--[  ]--|  \ 
 |   |   __   |   |
 |   |  /_/|  |   |  <- PALMS
 |   |  |_|/  |   |  
  \  |        |  /`}
                      </pre>
                    </div>
                  </div>
                </div>
            </div>
          </div>
        </div>
      )}
    </main>
  );
}

export default App;
