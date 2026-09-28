# <img src="./public/icon.svg" width="32" height="32" alt="OpticSpec Icon" /> OpticSpec — Deep Full-Frame AI Image-to-Prompt Generator

**OpticSpec** is a high-speed, full-frame visual reverse-engineering and image-to-prompt studio. When you upload, drag-and-drop, or paste (`Ctrl+V`) any image, OpticSpec scans **every single pixel across the entire frame**—combining deterministic client-side optical telemetry with parallel Vision-Language Models (NVIDIA NIM & high-speed multimodal failover) to generate a complete, copy-ready prompt capable of recreating the exact same image in any AI image generator.

![OpticSpec Instrumentation & Streaming Pipeline](./public/architecture-instrumentation.svg)

---

## 1. Core Capabilities & Full-Frame Master Instructions

Every image uploaded to **OpticSpec** is analyzed using a comprehensive **7-Pillar Master Instruction Note** embedded directly in the backend engine:

1. **Entire Frame, Side Angle & Camera Geometry**:
   - Scans the entire frame from edge to edge across all **9 spatial zones** (`Top-Left`, `Top-Center`, `Top-Right`, `Mid-Left`, `Center`, `Mid-Right`, `Bottom-Left`, `Bottom-Center`, `Bottom-Right`).
   - Captures exact shot framing, camera elevation, viewing perspective, **side/profile angle** (front-facing, 3/4 oblique angle, side profile, low-angle, high-angle, eye-level, Dutch tilt), lens focal length in `mm`, aperture `f-stop`, and depth-of-field falloff.
2. **Exact Number of Persons / Characters & Complete Individual Characteristics**:
   - Explicitly counts the **exact number of persons, characters, figures, or subjects** across the foreground, midground, background, frame edges, and reflections.
   - Never focuses on just one subject—describes **every single person/character individually**, including body posture, limb orientation, relative size in the frame, hair style/color, skin/surface texture, complete clothing/wardrobe design variables, fabric folds, and accessories.
3. **Facial Expressions, Emotion of All Variables, Lived Experience & Impression**:
   - Extracts hyper-specific **facial expressions and micro-expressions** on everyone in the frame: eyebrow arch/tension, eyelid openness, eye gaze direction, eye catchlights, cheek contours, lip curvature, and jawline tension.
   - Defines the exact **emotion, psychological state, lived experience, and personal impression** radiated by each individual, the interpersonal emotional dynamic between characters, and the **emotional resonance/impression carried by every non-human variable** (props, weather, architecture, lighting, and textures).
4. **All Motion (Dynamic) & Un-Motion (Static) Variables**:
   - Identifies and describes every **in-motion (dynamic)** variable: walking/running stride, hand/body gestures, action trajectories, wind-blown hair or garments, airborne dust/particles, flowing water, smoke, vehicle movement, and directional motion blur.
   - Identifies and describes every **un-motion (static/stationary)** variable: anchored poses, still background objects, fixed architectural structures, resting props, and grounded surfaces, along with their complete material characteristics.
5. **All Written Words (Verbatim OCR) & Unwritten Design Variables**:
   - Transcribes word-for-word **every visible written word, number, letter, sign, logo, label, badge, poster text, or watermark** in quotes, detailing its typography/font style, lettering size, colour, and surface placement.
   - Describes all **unwritten visual elements**, symbols, geometric patterns, graphic design variables, layout hierarchy, and negative space.
6. **Relative Size, Scale & Spatial Proportions**:
   - Quantifies the exact size, scale, proportions, and distance from the lens for every character, object, text element, and architectural feature in the frame.
7. **Sunlight, Lighting Physics, Brightness, Colour Encoding & Colour Combination**:
   - Captures **sunlight direction, solar elevation angle, hard sunbeams vs. diffused sky light**, golden-hour warmth, shade, rim lighting, bounce fill, artificial light sources, shadow sharpness, and specular highlights.
   - Embeds measured **brightness/luminance percentages**, **exposure profile (EV)**, **dynamic contrast**, **colour encoding (`sRGB / Rec.709`)**, **estimated colour temperature (`Kelvin`)**, **9-zone spatial pixel colours**, and **10 dominant `HEX` + `RGB` colour combinations**.

---

## 2. Deep Optical Instrumentation & Mathematical Telemetry

Before the image reaches the vision model, `/lib/pixel-telemetry.ts` executes a sub-15ms HTML5 Canvas 2D pixel instrumentation pass on the client:

### A. High-Clarity Optical Transport Compression
- Resizes oversized source images to a maximum dimension of **`1024px` at `0.85` JPEG quality** so fine background characters, micro-expressions, and small written text (OCR) remain sharp while keeping upload payloads compact (`~120KB–180KB`) for instant network transfer.

### B. 3×3 (9-Zone) Full-Frame Spatial Pixel Grid
- Projects the image onto a **`72×72` sampling matrix** (`5,184` sampled pixels, divisible by `3` into nine `24×24` spatial zones):
  - `Top-Left`, `Top-Center`, `Top-Right`
  - `Mid-Left`, `Center`, `Mid-Right`
  - `Bottom-Left`, `Bottom-Center`, `Bottom-Right`
- For each of the 9 zones, computes the mean RGB vector, converts it to an exact **`HEX` colour code**, and calculates the zone's localized **brightness percentage (`0%–100%`)**.

### C. Luminance, Exposure Profile (EV) & Dynamic Contrast
- Computes per-pixel perceived luminance using the **ITU-R BT.601 luma formula**:
  $$\text{Luminance} = 0.299 R + 0.587 G + 0.114 B$$
- Classifies every sampled pixel into:
  - **Shadows** ($\text{Luminance} < 70$)
  - **Midtones** ($70 \le \text{Luminance} \le 185$)
  - **Highlights** ($\text{Luminance} > 185$)
- Derives the **Exposure Profile** (`-1.3 EV Low-Key Dark Exposure` through `+1.2 EV High-Key Bright Exposure`) and **Dynamic Contrast Curve** (`High-Contrast Chiaroscuro`, `Low-Key Dark & Moody`, `High-Key Bright & Airy`, `Soft Logarithmic / Matte Film Midtone Curve`, or `Balanced Full-Range Contrast`).

### D. 10-Cluster Chromatic Quantization (`HEX` + `RGB`)
- Quantizes all `5,184` pixels into 3D RGB colour buckets (`step = 22`) and filters clusters by Euclidean RGB distance:
  $$d = \sqrt{(R_1 - R_2)^2 + (G_1 - G_2)^2 + (B_1 - B_2)^2} \ge 30$$
- Extracts the **top 10 distinct colour swatches**, calculating each swatch's `HEX` code, `rgb(r, g, b)` value, frame coverage percentage, and optical role (`Primary Dominant Field`, `Deep Shadow Base`, `Specular Peak Highlight`, `Vibrant Chromatic Accent`, `Low-Key Structural Tone`, or `Midtone Surface Anchor`).

### E. Sunlight & Colour Temperature (`Kelvin`) Estimation
- Evaluates the full-frame Red-to-Blue differential ($\Delta_{RB} = \bar{R} - \bar{B}$) and mean HSV saturation to classify the scene's **Colour Temperature & Sunlight Bias** (`3100K–3600K Warm Amber`, `4400K–4900K Warm Golden-Hour Sunlight`, `5400K Neutral Daylight`, `6200K–6800K Cool Crisp Overcast`, or `7200K–8500K Cool Blue-Hour Shade`).

---

## 3. Sample Visual Deconstructions & Instrumentation Gallery

### Sample 01 — Swiss Chronograph Macro Horology
![Swiss Chronograph Macro](./public/samples/sample_horology_macro_1790547004749.jpg)

- **Measured Optical Telemetry**:
  - **Aspect Ratio & Framing**: `4:3` (`1:1.33` Macro Oblique Side-Angle Close-Up, `90mm f/2.8` Tilt-Shift Macro Lens)
  - **Luminance & Exposure**: `38%` Mean Brightness (`-0.5 EV Moody Controlled Exposure`), `54%` Shadows, `34%` Midtones, `12%` Specular Highlights (`High-Contrast Chiaroscuro`)
  - **Sunlight & Colour Encoding**: `sRGB / Rec.709 8-bit`, `3400K` Warm Tungsten Key + `6500K` Cool Steel Rim, Dominant Palette `#141821`, `#D49B4B`, `#8C6239`, `#E2E8F0`, `#3B1D36`
  - **Motion & Un-Motion Variables**: Stationary ("un-motion") brushed bridges, bevelled steel screws, and ruby jewel bearings contrasted with the oscillating balance wheel and escapement gear train.

---

### Sample 02 — Nordic Brutalist Fjord Pavilion
![Nordic Brutalist Fjord Pavilion](./public/samples/sample_nordic_pavilion_1790547018724.jpg)

- **Measured Optical Telemetry**:
  - **Aspect Ratio & Framing**: `16:9` (`24mm f/8` Architectural Tilt-Shift Wide Shot, Eye-Level 3/4 Side Perspective)
  - **Luminance & Exposure**: `44%` Mean Brightness (`0.0 EV Neutral Exposure`), `42%` Shadows, `46%` Midtones, `12%` Highlights
  - **Sunlight & Colour Encoding**: `sRGB / Rec.709 8-bit`, `4600K` Low-Elevation Golden-Hour Sunlight + `7400K` Nordic Fjord Twilight Reflection, Dominant Palette `#1B2632`, `#D9822B`, `#4A5D6E`, `#0F171E`, `#E6C280`
  - **Motion & Un-Motion Variables**: Un-motion cantilevered board-formed concrete slabs and floor-to-ceiling glass panes contrasted with gently rippling fjord water reflections and drifting mountain mist.

---

### Sample 03 — Conservatory Glass & Bioluminescent Botanicals
![Conservatory Glass Botanicals](./public/samples/sample_glass_botanicals_1790547030523.jpg)

- **Measured Optical Telemetry**:
  - **Aspect Ratio & Framing**: `4:3` (`50mm f/1.8` Medium-Wide Interior Perspective, Low-Angle Upward Tilt)
  - **Luminance & Exposure**: `41%` Mean Brightness (`-0.5 EV Moody Controlled Exposure`), `49%` Shadows, `39%` Midtones, `12%` Specular Highlights
  - **Sunlight & Colour Encoding**: `sRGB / Rec.709 8-bit`, Diffused Moonlight + `4800K` Warm Amber Lantern Glow, Dominant Palette `#0D1F1D`, `#2E6F5E`, `#D98A3C`, `#183632`, `#8CD9C4`
  - **Motion & Un-Motion Variables**: Stationary Victorian wrought-iron ribs and glass panes with drifting suspended humidity droplets and unfurling tropical fern fronds.

---

## 4. High-Speed Backend Architecture (`/app/api/analyze/route.ts`)

1. **Parallel Multi-Model Racing (`Promise.any`)**:
   - Simultaneously races NVIDIA NIM Vision-Language endpoints (`qwen/qwen2.5-vl-72b-instruct`, `meta/llama-3.2-90b-vision-instruct`, and `meta/llama-3.2-11b-vision-instruct`) with an `AbortController` timeout guard, and automatically fails over to low-latency multimodal streaming (`gemini-3-flash-preview` / `gemini-2.5-flash`) if any upstream endpoint is rate-limited or unavailable.
2. **Real-Time Refusal Interceptor (`verifyStreamNotRefused`)**:
   - Inspects the initial 85 characters of every candidate stream before forwarding bytes to the browser. If any model emits a safety/policy refusal (such as `"I can't help you with that..."`), that stream is immediately aborted and replaced by a verified descriptive stream so the user never receives a refusal message.
3. **SHA-1 In-Memory LRU Prompt Cache (`PROMPT_CACHE`)**:
   - Hashes the incoming image payload and instruction signature using `crypto.createHash('sha1')` to serve repeat analyses in **`< 5ms`**.
4. **Live Token Streaming**:
   - Streams raw prompt tokens directly to the UI as they are generated so text appears immediately on screen without boxes or clutter.

---

## 5. Project Structure

```text
├── app/
│   ├── api/analyze/route.ts     # Parallel NVIDIA NIM + Multimodal streaming API, refusal guard & LRU cache
│   ├── globals.css              # Tailwind CSS v4 global stylesheet
│   ├── icon.svg                 # Custom optical viewfinder vector favicon
│   ├── layout.tsx               # Root layout, typography, OpenGraph/Twitter cards & JSON-LD SEO schema
│   └── page.tsx                 # Main application entry point
├── components/
│   └── OpticSpecStudio.tsx      # Streamlined drag-and-drop / paste upload & unboxed live prompt UI
├── lib/
│   ├── pixel-telemetry.ts       # Client-side 9-zone spatial grid, 10-cluster HEX/RGB & luminance scanner
│   └── types.ts                 # TypeScript interfaces for optical telemetry and vision models
└── public/
    ├── architecture-instrumentation.svg  # System architecture & optical pipeline diagram
    ├── icon.svg                          # Public vector app icon
    └── samples/                          # High-resolution sample images
```

---

## 6. Environment Configuration & Deployment

1. Configure environment secrets in `.env.local` (or the AI Studio Secrets panel):
   ```env
   GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
   NVIDIA_API_KEY="YOUR_NVIDIA_API_KEY"
   APP_URL="https://your-deployment-url.run.app"
   ```
2. Install dependencies and build for production:
   ```bash
   npm install
   npm run build
   npm start
   ```

---

**Developed by Rehan...🌻**  
*devloped by Rehan...🌻*
