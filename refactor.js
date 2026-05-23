const fs = require('fs');
let code = fs.readFileSync('src/App.jsx', 'utf8');

// 1. Color Palette Swaps
code = code.replace(/neon-cyan/g, 'medical-blue');
code = code.replace(/neon-pink/g, 'medical-red');
code = code.replace(/neon-green/g, 'medical-green');
code = code.replace(/neon-amber/g, 'medical-amber');

// 2. Remove glowing shadows
code = code.replace(/shadow-\[0_0_12px_rgba\(0,240,255,0\.4\)\]/g, 'shadow-sm');
code = code.replace(/shadow-\[0_0_15px_rgba\(0,240,255,0\.25\)\]/g, 'shadow-sm');
code = code.replace(/shadow-\[0_0_8px_#00f0ff\]/g, '');
code = code.replace(/shadow-\[0_0_8px_#ffaa00\]/g, '');
code = code.replace(/shadow-medical-green\/15/g, 'shadow-sm');
code = code.replace(/shadow-medical-amber\/10/g, 'shadow-sm');
code = code.replace(/shadow-medical-blue\/10/g, 'shadow-sm');
code = code.replace(/shadow-medical-red\/5/g, 'shadow-sm');
code = code.replace(/shadow-yellow-500\/5/g, 'shadow-sm');

// 3. Remove Slop Fonts and Tracking
code = code.replace(/font-share/g, '');
code = code.replace(/tracking-\[0\.2em\]/g, 'tracking-normal');
code = code.replace(/tracking-\[0\.25em\]/g, 'tracking-normal');
code = code.replace(/tracking-widest/g, 'tracking-normal');
code = code.replace(/tracking-wider/g, 'tracking-normal');

// 4. Remove font-mono from the root main wrapper so it uses standard sans-serif
code = code.replace(/font-mono flex flex-col/g, 'flex flex-col');

// 5. Explicitly add font-mono back to the telemetry numbers and logs
code = code.replace(/text-3xl sm:text-4xl font-extrabold/g, 'text-3xl sm:text-4xl font-extrabold font-mono');
code = code.replace(/text-2xl sm:text-3xl font-extrabold/g, 'text-2xl sm:text-3xl font-extrabold font-mono');
code = code.replace(/font-bold animate-pulse/g, 'font-bold animate-pulse font-mono');

// 6. De-slop the copy (Remove brackets, fix casing)
code = code.replace(/▶ \[ CLINICAL_ANATOMY_GUIDE \]/g, 'Clinical anatomy guide');
code = code.replace(/\[ SENSOR_HUB: COMPILING_VISION_SYSTEM \]/g, 'Sensor Hub: Compiling vision system');
code = code.replace(/\[ LOADING WASM SENSORS \]/g, 'Loading WASM sensors');
code = code.replace(/\[ COMPILING COMPUTER VISION ENGINE \]/g, 'Compiling computer vision engine');
code = code.replace(/\[ RESOLVING ANATOMICAL POSE TARGETS \]/g, 'Resolving anatomical pose targets');
code = code.replace(/\[ CONFIGURING MULTI-HAND SENSORS \]/g, 'Configuring multi-hand sensors');
code = code.replace(/\[ SENSORS_INITIALIZATION_FAILED \]/g, 'Sensors initialization failed');
code = code.replace(/▶ \[ PULSE_VERIFICATION: CORRECT \]/g, 'Pulse verification: Correct');
code = code.replace(/▶ \[ PULSE_VERIFICATION: ALIGNING \]/g, 'Pulse verification: Aligning');
code = code.replace(/▶ \[ PULSE_VERIFICATION: PLACE_FINGERS \]/g, 'Pulse verification: Place fingers');
code = code.replace(/▶ \[ CPR_COMPRESSION_TRAINING: ACTIVE \]/g, 'CPR compression training: Active');
code = code.replace(/BRIFF \/\/ SENSOR_HUB/g, 'Briff Sensor Hub');
code = code.replace(/\[ DISCONNECT \]/g, 'Disconnect');
code = code.replace(/SKELETON: ON/g, 'Skeleton On');
code = code.replace(/SKELETON: OFF/g, 'Skeleton Off');
code = code.replace(/METRONOME: ON/g, 'Metronome On');
code = code.replace(/METRONOME: OFF/g, 'Metronome Off');
code = code.replace(/PULSE/g, 'Pulse');
code = code.replace(/CPR/g, 'CPR'); // Leave CPR uppercase as it's an acronym
code = code.replace(/🔍 HELP_GUIDE/g, 'Help Guide');
code = code.replace(/✕ CLOSE/g, 'Close');
code = code.replace(/📡 CONNECT STREAM/g, 'Connect Stream');

// 7. Simplify the loading animation
const oldLoader = `<div className="relative w-14 h-14 flex items-center justify-center">
                  <div className="absolute inset-0 rounded-full border border-dashed border-medical-blue/35 animate-spin" style={{ animationDuration: '6s' }}></div>
                  <div className="absolute inset-1.5 rounded-full border border-medical-blue border-t-transparent animate-spin" style={{ animationDuration: '1.2s' }}></div>
                  <div className="w-1.5 h-1.5 rounded-full bg-medical-blue "></div>
                </div>`;
const newLoader = `<div className="relative w-8 h-8 flex items-center justify-center">
                  <div className="w-3 h-3 rounded-full bg-medical-blue animate-pulse"></div>
                </div>`;
code = code.replace(oldLoader, newLoader);

fs.writeFileSync('src/App.jsx', code);
