---
name: Briff First-Aid
description: A high-energy, super minimal first-aid tracker
colors:
  bg-base: "#111b21"
  bg-panel: "#202f36"
  border-subtle: "#37464f"
  text-primary: "#ffffff"
  text-secondary: "#afbac0"
  brand-green: "#58cc02"
  brand-blue: "#1cb0f6"
  brand-red: "#ff4b4b"
  brand-yellow: "#ffc800"
typography:
  display:
    fontFamily: "'Nunito', 'Varela Round', system-ui, sans-serif"
    fontWeight: 900
  body:
    fontFamily: "'Nunito', 'Varela Round', system-ui, sans-serif"
    fontWeight: 400
rounded:
  full: "9999px"
  3xl: "1.5rem"
spacing:
  md: "1rem"
components:
  button-primary:
    backgroundColor: "transparent"
    textColor: "{colors.brand-blue}"
    rounded: "{rounded.full}"
---

# Design System: Briff First-Aid

## 1. Overview

**Creative North Star: "The MRI Scan"**

The system blends the high-energy urgency of an arcade challenge with the hyper-minimalist, high-contrast aesthetic of a medical MRI scan. The UI should strip away unnecessary decorations, relying on pure flat shapes, strong glowing outlines against deep dark backgrounds, and heavily rounded sleek elements. It rejects the chunky, literal gamification of Duolingo in favor of a sleek, technical, and urgent medical diagnostic feel.

**Key Characteristics:**
- Super minimal and data-driven.
- High-contrast against dark backgrounds.
- Urgent and high-energy.
- Sleek, heavily rounded components with flat graphic outlines.

## 2. Colors

The "MRI Palette" relies on stark, bright diagnostic colors against a deep void.

### Primary
- **Diagnostic Blue** (#1cb0f6): The core active state color. Used for primary UI outlines and active scan states.
- **Vital Green** (#58cc02): Indicates correct form, successful pulse detection, and positive health metrics.

### Secondary
- **Critical Red** (#ff4b4b): Used for disconnection, failures, and urgent warnings.
- **Warning Yellow** (#ffc800): Used for alignment corrections and rate warnings during CPR.

### Neutral
- **Deep Void** (#111b21): The absolute background, mimicking the darkness of an MRI scan.
- **Pure White** (#ffffff): Primary text and high-contrast elements.
- **Muted Data** (#afbac0): Secondary text and inactive states.

**The Absolute Void Rule.** The background must remain deep and dark. Colors should act as glowing lines or flat stark fills against this void.

## 3. Typography

**Display Font:** 'Nunito', 'Varela Round', system-ui, sans-serif
**Body Font:** 'Nunito', 'Varela Round', system-ui, sans-serif
**Label/Mono Font:** 'Nunito', monospace

**Character:** Highly legible, clean, and rounded sans-serif that balances the stark medical look with approachable geometry.

### Hierarchy
- **Display** (900, 3rem, 1): Large readouts and critical system statuses.
- **Headline** (800, 1.5rem, 1.2): Section headers and primary active modes.
- **Body** (400, 1rem, 1.5): Standard instructional text.
- **Label** (800, 0.875rem, uppercase): UI buttons, diagnostic tags, and telemetry labels.

**The Data Readout Rule.** Typography should feel like clean, urgent data. Use uppercase labels with heavy tracking for system modes. No cards, no backgrounds behind text—the data floats pure and stark.

## 4. Elevation

The system uses a completely flat approach. Depth is conveyed purely through stark outlines, typography scale, and vivid color contrasts (drop shadows are allowed only to create a glowing effect, not 3D elevation).

**The Graphic Flatness Rule.** Pure flat typography and shapes. No drop shadows for elevation. No background panels or "cards". If an element needs to stand out, give it a high-contrast border or a pure solid glowing drop-shadow.

## 5. Components

### Buttons
- **Shape:** Heavily rounded, sleek pill shapes (9999px).
- **Primary:** Transparent background with a strong `Diagnostic Blue` border and text.
- **Hover / Focus:** Fill becomes solid `Diagnostic Blue` with `Pure White` text.
- **Secondary:** Flat `Panel Dark` backgrounds with `Muted Data` text.

### Telemetry HUD (Pulse/CPR)
- **Style:** Large, centered typography floating directly on the void.
- **Background:** Absolutely transparent. NO CARDS or panels.
- **Focus:** Relies entirely on color (Green/Blue/Yellow/Red) and drop-shadows on the text to indicate state.

## 6. Do's and Don'ts

### Do:
- **Do** make text and data float directly on the `Deep Void` background.
- **Do** maintain a super minimal, dark background (The MRI Scan look).
- **Do** make buttons sleek and heavily rounded (pill shapes) but keep layouts card-less.
- **Do** use vivid, high-contrast colors (Green, Blue, Red) to indicate system states via text and borders.

### Don't:
- **Don't** use cards, panels, or elevated containers with backgrounds.
- **Don't** use basic bootstrap dashboards.
- **Don't** use 3D thick borders or physical pressable button metaphors.
- **Don't** clutter the screen with academic medical textbook diagrams.
