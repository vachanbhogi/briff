# BRIFF - AI Emergency Response Coach

**Briff** is an automated, real-time AI coaching platform for emergency response. It converts passive visual data from any standard web camera into an interactive lifesaving instructor. 

In an emergency, 54% of bystanders experience cognitive panic, and 9 out of 10 adults lack practical emergency preparedness. However, immediate bystander action can multiply survival rates by 3X. Briff bridges this gap by democratizing immediate crisis readiness through zero-latency, in-browser computer vision.

## Features

Briff provides live diagnostic coaching across three critical emergency protocols:

1. **Pulse Check**
   - Carotid Nerve Evaluation
   - Arterial Placement Tracking
2. **CPR Coach**
   - Torso Alignment Matrix
   - Real-time BPM Pacing (Metronome at 110 BPM)
   - Compression Vector & Depth Tracking
3. **Heimlich Guide**
   - Posture Correction
   - Thrust Angle Geometry
   - Abdominal Focal Tracking

## How It Works

Briff uses a **3D Landmark Engine** powered by Google's MediaPipe. It tracks **75 dynamical coordinates** (Pose and Hand landmarks) in real-time. 

- **Edge Computing:** Rendering and spatial geometry calculations occur entirely inside the browser frame. No video data is sent to a server.
- **Zero Latency:** Operates at 30+ FPS with <10ms latency on the edge.
- **Clinical Sources:** Guided by data from the AHA Heart & Stroke Statistics and American Red Cross.

## Tech Stack

- **Frontend:** React, Vite
- **Styling:** Tailwind CSS v4 (Custom UI "MRI" Theme)
- **Computer Vision:** Google MediaPipe (Pose & Hand Landmarker WASM models)
- **Hardware Integration:** Web Serial API (for optional Arduino CPR dummy telemetry)

## Getting Started

### Prerequisites
- Node.js (v18+)
- [Bun](https://bun.sh/) (Package Manager)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/mockup.git
   cd mockup
   ```

2. Install dependencies:
   ```bash
   bun install
   ```

3. Start the development server:
   ```bash
   bun run dev
   ```

4. Open your browser and navigate to the local URL (usually `http://localhost:5173`). 
*Note: Camera permissions must be granted for the AI tracking to function.*

## Hardware Integration (Optional)

Briff includes support for connecting to a custom Arduino-based CPR training dummy via the Web Serial API. When connected, it can send and receive telemetry data for compression depth and rate validation.

1. Connect the Arduino via USB.
2. Click `[ LINK ARDUINO ]` in the Sagittal System Control panel.
3. Select the serial port in the browser prompt.

## License

This project is intended as a proof-of-concept / MVP for emergency response coaching.