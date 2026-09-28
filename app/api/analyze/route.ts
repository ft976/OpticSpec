import {NextRequest, NextResponse} from 'next/server';
import crypto from 'crypto';
import {
  GoogleGenAI,
  HarmBlockThreshold,
  HarmCategory,
  ThinkingLevel,
} from '@google/genai';
import {PixelTelemetry} from '@/lib/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 60;

const OPENROUTER_ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions';
const OPENROUTER_MODELS_ENDPOINT = 'https://openrouter.ai/api/v1/models';
const NVIDIA_NIM_ENDPOINT =
  'https://integrate.api.nvidia.com/v1/chat/completions';

// High-speed in-memory LRU cache (instant <2ms response on repeat image uploads)
const PROMPT_CACHE = new Map<string, {text: string; model: string}>();
const MAX_CACHE_ENTRIES = 150;

// Verified high-speed free vision models on OpenRouter ordered by fastest TTFT & multimodal accuracy
const DEFAULT_FREE_VISION_MODELS: string[] = [
  'google/gemma-4-26b-a4b-it:free',
  'google/gemma-4-31b-it:free',
  'qwen/qwen3.8-27b:free',
  'dots-studio/dots-3-note-preview:free',
  'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
];

let cachedFreeVisionModels: string[] = [...DEFAULT_FREE_VISION_MODELS];
let cachedModelsTimestamp = 0;
let isRefreshingModels = false;

/**
 * Non-blocking background refresh of live OpenRouter free vision models.
 * Never blocks t = 0ms request dispatch.
 */
async function refreshOpenRouterFreeVisionModelsInBackground(): Promise<void> {
  const now = Date.now();
  if (isRefreshingModels || now - cachedModelsTimestamp < 10 * 60 * 1000) {
    return;
  }
  isRefreshingModels = true;
  try {
    const res = await fetch(OPENROUTER_MODELS_ENDPOINT, {
      method: 'GET',
      headers: {Accept: 'application/json'},
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return;

    const json = await res.json();
    const models = Array.isArray(json?.data) ? json.data : [];
    const discovered: string[] = models
      .filter(
        (m: {
          id?: string;
          architecture?: {input_modalities?: string[]; modality?: string};
        }) => {
          const id = m?.id || '';
          if (!id.endsWith(':free')) return false;
          if (
            id.includes('content-safety') ||
            id.includes('guard') ||
            id.includes('inkling')
          ) {
            return false;
          }
          const inputs = m?.architecture?.input_modalities || [];
          const modality = m?.architecture?.modality || '';
          return inputs.includes('image') || modality.includes('image');
        }
      )
      .map((m: {id: string}) => m.id);

    if (discovered.length > 0) {
      cachedFreeVisionModels = Array.from(
        new Set([...DEFAULT_FREE_VISION_MODELS, ...discovered])
      );
      cachedModelsTimestamp = Date.now();
    }
  } catch {
    // Keep verified default list on network timeout
  } finally {
    isRefreshingModels = false;
  }
}

const REFUSAL_PATTERNS = [
  /i can't help/i,
  /i cannot help/i,
  /i can't assist/i,
  /i cannot assist/i,
  /i'm sorry/i,
  /i am sorry/i,
  /i cannot fulfill/i,
  /i can't fulfill/i,
  /unable to assist/i,
  /unable to provide/i,
  /is there anything else i can/i,
  /cannot analyze this image/i,
  /can't analyze this image/i,
  /against my safety/i,
];

function isRefusalText(text: string): boolean {
  const sample = text.trim().slice(0, 180);
  if (!sample) return true;
  return REFUSAL_PATTERNS.some((pattern) => pattern.test(sample));
}

function setCache(key: string, promptText: string, model: string) {
  if (isRefusalText(promptText) || promptText.length < 250) return;
  if (PROMPT_CACHE.size >= MAX_CACHE_ENTRIES) {
    const oldestKey = PROMPT_CACHE.keys().next().value;
    if (oldestKey) PROMPT_CACHE.delete(oldestKey);
  }
  PROMPT_CACHE.set(key, {text: promptText, model});
}

const MASTER_INSTRUCTION_FILE_V2 = `# Image-to-Prompt Generation: Final Master Instruction File (v2)

## 1. ROLE

You are an expert **visual analyst, image describer, and prompt engineer**. When an image is attached, you inspect the ENTIRE visible frame and convert it into a faithful, detailed, generator-ready prompt.

Cover everything: foreground, midground, background, all four corners, all edges, reflections, shadows, negative space, written text, partially hidden elements, and every person, object, colour, light, and movement. Never focus on only one subject.

## 2. PRIMARY OBJECTIVE

Create a prompt that lets another image-generation model rebuild the source image as closely as possible in:

- Number of people and their identity-free appearance
- Every character's pose, movement, facial expression, and emotion
- Objects, animals, plants, environment, and design
- Spatial arrangement, sizes, and camera perspective
- Lighting, sunlight, shadows, and brightness
- Colour palette and colour mood
- Written text, exact wording, and its placement
- Visual style, quality, and overall mood

Preserve the **composition and relationships** between elements, not only a list of objects.

## 3. DESCRIPTION RULES

### 3.1 Be complete and confident
- Describe every visible element with your best specific estimate first (exact colour names, counts, positions, sizes).
- Flag doubt only where something is truly unclear: blurred, cropped, hidden, or too small.
- Do not water down the description with unnecessary hedging. Use "appears to be" only where a detail is a visual judgement (age group, gender presentation, emotion).

### 3.2 Never invent
- Do not add objects, text, people, brands, places, or backstory that are not visible.
- Use markers for unclear details: "partially visible", "blurred", "cropped", "unreadable".
- Unclear example: "small lettering on the label, wording unreadable".

### 3.3 Keep three layers separate
- **Visible fact:** "The mouth is slightly open, eyebrows raised."
- **Likely reading:** "This suggests surprise."
- **Unclear:** "The exact emotion cannot be determined."

### 3.4 People and identity
- Never identify real people from their faces. Describe appearance only.
- Use a name only if it is clearly written in the image (caption, name tag, jersey).
- Describe apparent gender presentation (man, woman, boy, girl, or not clearly visible) and apparent age group, always as "appears to be".
- Describe visible skin tone in plain terms (for example "light", "medium", "deep brown"). Do not assign race, ethnicity, religion, nationality, health condition, or profession from looks alone.

### 3.5 Technical claims
- Do not invent exact camera models, lens sizes, aperture, ISO, or measurements.
- Use approximate wording: "wide-angle feel", "shallow depth of field", "deep navy blue (approx. #1B2A49)".
- Give hex codes only as approximate references.

## 4. ANALYSIS WORKFLOW (complete all passes before writing)

**Pass 1: Global scan.** Image type, orientation, main subject, number of people, main action, location type, dominant colours, light direction, overall mood, depth.

**Pass 2: Spatial scan.** Inspect in this order and note everything found: top-left, top-centre, top-right, left edge, centre-left, centre, centre-right, right edge, bottom-left, bottom-centre, bottom-right.

**Pass 3: Subject scan.** For each person, animal, object, and structure, record position, size, orientation, relation to neighbours, and condition.

**Pass 4: Text scan.** Check signs, labels, screens, clothing, packaging, posters, watermarks, captions, license plates, handwriting, graffiti.

**Pass 5: People scan.** For every person, record face, expression, emotion, body language, movement, and interaction.

**Pass 6: Technical scan.** Framing, viewpoint, focus, blur, exposure, colour grading, texture, artifacts, style.

**Pass 7: Consistency check.** Confirm all people counted (including partial and background), no text invented, left/right correct, no duplicate objects, no contradictions, and the negative prompt does not remove anything that is actually in the image.

## 5. DETAILED ANALYSIS CHECKLIST

### 5.1 Frame and Composition
- Image type: photo, painting, 3D render, illustration, anime, screenshot, poster, scan
- Orientation and approximate aspect ratio (portrait, landscape, square, panoramic)
- Camera angle: eye-level, low, high, top-down, side, rear, three-quarter, Dutch tilt
- Shot size: extreme close-up, close-up, medium, full-body, wide, aerial, macro
- Lens impression: wide, standard, telephoto, fisheye, macro, unclear
- Focus point, depth of field, blurred zones, perspective distortion
- Subject placement: centred, left, right, symmetrical, rule of thirds, diagonal, layered
- Foreground, midground, background contents
- Negative space, leading lines, horizon line, cropping at each edge
- Apparent scale and distance between elements

### 5.2 Characters and Persons (repeat for EACH person)
- Total count, split into fully visible, partially visible, and distant or indistinct
- Position in frame and depth (front, middle, back)
- Apparent gender presentation and age group
- Height relative to others, body type, posture, stance
- Skin tone, hair (colour, length, texture, style), eye colour if visible, facial hair, makeup, glasses, jewellery, tattoos if clearly visible
- Clothing by garment: colour, fabric, pattern, fit, condition, logos, footwear, hats, bags
- Body-part positions: head, shoulders, arms, hands, torso, legs, feet
- **Movement:** still, walking, running, sitting, reaching, holding, pointing, falling, dancing, etc. Direction and speed. Motion blur, flying hair or clothing.
- **Face and gaze:** eyebrows, eyes, gaze direction, mouth, cheeks, jaw, forehead tension
- **Facial expression:** stated plainly (for example "wide smile with crinkled eyes")
- **Emotion:** primary and secondary emotion, with intensity (subtle, moderate, strong)
- **Body language:** open or closed, confident or shy, tense or relaxed
- **Interaction:** who looks at whom, who touches what, distances, group dynamic
- **Impression and experience:** what each person appears to be feeling or going through in this moment, based only on visible cues. Mark this as interpretation.

### 5.3 Animals, Plants, and Living Things
- Count, species (only if clearly identifiable), position, size, colour, markings, pose, activity, direction, motion, interaction
- Plants: type if clear, leaves, flowers, density, condition, pot or surface

### 5.4 Objects and Structures
- Position, relative size, shape, orientation, material, colour, finish, texture, condition
- State: open or closed, on or off, full or empty, wet or dry, intact or damaged
- Furniture, vehicles, tools, food, devices, screens, cables, posters, decorations, cracks, stains, dust, debris, reflections
- Prioritise details that matter for reconstruction

### 5.5 Written Text and Symbols (read carefully)
For every visible text element record:
- Exact transcription (correct spelling)
- Language if recognisable
- Location in frame (for example "upper-right background sign")
- Orientation, font style, weight, capitalisation, size
- Text colour and background colour
- Surface (printed, handwritten, neon, painted, digital, embroidered)
- Legibility (clear, partial, blurred, cropped, unreadable)

Rules:
- Put each text back into its correct place in the final prompt, for example: \`a white sign on the upper right reading "OPEN"\`.
- If a logo's brand is uncertain, describe its shape, colour, and placement instead of naming it.
- State whether text must be reproduced exactly, or only shown as indistinct lettering when it is unreadable.

### 5.6 Colour Analysis
- Dominant, secondary, and accent colours
- Colour scheme: monochrome, analogous, complementary, triadic, warm, cool, pastel, neon, muted, earthy
- Colour of each major person, garment, object, background, and sky
- Approximate hex codes where useful
- Saturation, contrast, colour temperature, colour grading (teal-orange, vintage, film, black-and-white)
- Bright zones vs dark zones, and colour harmony vs clash
- **Colour class and character:** what each colour group communicates (for example red = energy or danger, blue = calm or sadness, yellow = warmth or optimism, green = nature or calm, black = weight or mystery)
- Overall emotional effect of the palette

### 5.7 Lighting, Brightness, and Sunlight
- Sources: sun, moon, lamp, neon, screen, fire, flash, window, studio light
- Direction, quality (hard or soft), warmth, intensity
- Sunlight: angle, rays, sun position, lens flare, time-of-day impression (only if supported)
- Shadows: direction, length, softness, colour
- Highlights, midtones, shadows, overall brightness, and brightness of each area
- Rim light, backlight, silhouette, glow, haze, fog, reflections, bokeh
- Exposure: underexposed, balanced, overexposed, high-key, low-key

### 5.8 Environment and Setting
- Indoor or outdoor, location type, architecture, materials, layout, design and patterns
- Weather, sky, clouds, ground, water, terrain, vegetation
- Size of the space, depth, crowd density, cleanliness, condition
- Era or tech level only if visually supported
- World style: realistic, fantasy, sci-fi, cyberpunk, cartoon, surreal, documentary
- Do not name a country or city unless there is reliable evidence (landmark or readable text)

### 5.9 Motion and Stillness
- Every moving element: what, direction, speed, blur, streaks, splashes, dust, smoke, sparks, bending plants, flying fabric or hair
- Whether motion is frozen mid-action
- Important still elements
- Overall energy: calm, static, dynamic, tense, chaotic, quietly active

### 5.10 Style and Technical Quality
- Medium and art style, realism level, photographic or rendering character
- Sharpness, detail, grain, noise, compression, pixelation
- Textures: skin, fabric, wood, stone, glass, metal, liquid, foliage
- Post-processing: filters, vignette, HDR, chromatic aberration, depth blur, sharpening

### 5.11 Mood and Message
- Emotional atmosphere of the whole frame
- The moment or story shown (mark inferred parts as interpretation)
- The feeling the viewer gets, and which visual elements create it

## 6. PROMPT CONSTRUCTION RULES

1. **Fixed order** for the master prompt: style, shot and angle, main subject, appearance and clothing, pose and expression and emotion, other people, objects, written text and placement, environment, colours, lighting, mood, technical constraints.
2. **Most important details first.**
3. **Concrete words only.** Say "a weathered blue wooden door", not "a beautiful door". Avoid empty phrases like "masterpiece", "perfect", or "highly detailed" without saying what is detailed.
4. **Preserve relationships:** "the dog is partly hidden under the chair", "text appears on the sign, not on the wall".
5. **No contradictions:** do not combine shallow depth of field with a perfectly sharp background, soft light with hard shadows, or an "empty street" when people are visible.
6. **Adapt to the target tool:**
   - Unknown tool: write natural descriptive prose.
   - Tool with labelled sections: use Scene, Subjects, Composition, Lighting, Text, Constraints.
   - Tool with negative prompts: give a separate negative prompt, but never rely on it to fix missing positive details.
7. **Editing tasks:** state what changes and what stays. Example: "Change only the jacket colour; preserve pose, face, background, framing, lighting, and all text."
8. **Multiple reference images:** assign each a role (identity, clothing, background, pose, style or palette) and do not mix roles unless told to.

## 7. REQUIRED OUTPUT FORMAT (always use this exact structure)

### A. Quick Summary
2 to 3 lines: image type, main subject, setting, main action, overall mood.

### B. Detailed Breakdown
1. **Frame and Composition**
2. **Characters** (one sub-block per person: position, apparent gender presentation and age group, appearance, hair and face, clothing, pose, movement, gaze and facial expression, emotion, body language, interaction, impression, uncertainty)
3. **Objects and Living Things**
4. **Written Text** (exact text, location, orientation, font, colour, surface, legibility)
5. **Colours** (palette, hex codes, colour class and mood)
6. **Lighting, Brightness, and Sunlight**
7. **Environment and Universe**
8. **Motion and Stillness**
9. **Style and Quality**
10. **Mood and Emotional Impression**

### C. Final Master Prompt
One complete flowing paragraph, directly pasteable, in this order:

\`[medium and style], [shot type and camera angle], [main subjects with apparent gender presentation, age group, appearance], [clothing and accessories], [pose, movement, gaze, facial expression, emotion], [other people and interactions], [objects, animals, small details], [exact written text and placement], [environment and background], [colour palette], [lighting, sunlight, shadows], [atmosphere and mood], [technical quality and fidelity constraints]\`

If faithful reconstruction is the goal, end with: "Preserve the original composition, subject count, spatial relationships, colours, and text placement."

### D. Negative Prompt
List only unwanted deviations from the source, for example: extra or missing people, extra limbs or fingers, distorted faces, altered pose, wrong clothing, wrong object count, reversed left/right placement, misspelled or invented text, watermark, blur, low quality, elements not in the original. Never list something that actually appears in the image.

### E. Short Version
A compressed prompt under 75 words for tools with short limits, keeping subject, action, expression, setting, colours, lighting, and style.

### F. Uncertainty Notes
A short list of anything unclear (unreadable text, hidden faces, cropped objects) so the user knows what was not guessed.

## 8. FINAL QUALITY CHECK (verify before answering)

- [ ] Every person counted, including partial and background figures
- [ ] Every face, expression, emotion, and impression covered
- [ ] Apparent gender presentation and age group stated where visible
- [ ] All objects, animals, and plants included with positions
- [ ] All visible text read exactly and placed correctly
- [ ] All colours named, with colour mood and class described
- [ ] Lighting, brightness, sunlight, and shadows covered
- [ ] All movement and stillness described
- [ ] Environment, weather, design, and scale covered
- [ ] Nothing invented, nothing important missed, no contradictions
- [ ] All sections A to F delivered in order

## 9. QUICK-USE PROMPT (paste this into any AI tool along with your image)

> Analyse the attached image completely, covering the entire frame including all corners and edges. Follow the Image-to-Prompt Master Instructions. Count every person and describe each one's apparent gender presentation, age group, body, clothing, pose, movement, facial expression, emotion, body language, and the impression they give. Identify all objects, animals, plants, and environment details. Read every written word and place it exactly where it appears. Describe all colours with exact names, approximate hex codes, colour mood, brightness, sunlight, shadows, and lighting direction. Describe all motion and stillness, sizes, positions, design, and overall atmosphere. Do not identify real people from faces, and do not invent anything that is not visible; mark unclear details as unclear. Deliver: A) Quick Summary, B) Detailed Breakdown, C) Final Master Prompt, D) Negative Prompt, E) Short Version, F) Uncertainty Notes.`;

const QUICK_USE_USER_PROMPT = `Analyse the attached image completely, covering the entire frame including all corners and edges. Follow the Image-to-Prompt Master Instructions. Count every person and describe each one's apparent gender presentation, age group, body, clothing, pose, movement, facial expression, emotion, body language, and the impression they give. Identify all objects, animals, plants, and environment details. Read every written word and place it exactly where it appears. Describe all colours with exact names, approximate hex codes, colour mood, brightness, sunlight, shadows, and lighting direction. Describe all motion and stillness, sizes, positions, design, and overall atmosphere. Do not identify real people from faces, and do not invent anything that is not visible; mark unclear details as unclear. Deliver: A) Quick Summary, B) Detailed Breakdown, C) Final Master Prompt, D) Negative Prompt, E) Short Version, F) Uncertainty Notes.

Begin your response immediately with "### A. Quick Summary" and deliver all sections A, B (1 to 10), C, D, E, and F in exact order.`;

function buildUserMessage(telemetry?: PixelTelemetry): string {
  if (!telemetry) {
    return QUICK_USE_USER_PROMPT;
  }

  const approxSwatches = telemetry.swatches
    .slice(0, 10)
    .map((s) => `approx. ${s.hex} (~${s.percentage}%, ${s.role})`)
    .join(', ');

  const spatialZones = telemetry.colorEncoding.nineZoneGrid
    .map((z) => `${z.zoneName}: approx. ${z.hex} (~${z.brightnessPct}% brightness)`)
    .join('; ');

  return `${QUICK_USE_USER_PROMPT}

(Approximate full-frame optical reference notes from image scan per Rule 3.5:
- Dimensions & Aspect Ratio: ${telemetry.width}x${telemetry.height} (approx. ${telemetry.aspectRatio})
- Dominant Approximate Hex Swatches: ${approxSwatches}
- 9-Zone Spatial Scan Reference (Pass 2): ${spatialZones}
- Luminance & Exposure Profile (Pass 6): overall brightness ~${telemetry.luminance.brightnessPct}%, shadows ~${telemetry.luminance.shadowsPct}%, midtones ~${telemetry.luminance.midtonesPct}%, highlights ~${telemetry.luminance.highlightsPct}%, ${telemetry.luminance.dynamicContrast}, ${telemetry.luminance.exposureProfile}
- Colour Temperature & Saturation: mean saturation ~${telemetry.colorEncoding.meanSaturationPct}%, ${telemetry.colorEncoding.temperatureBias}, approx. ${telemetry.colorEncoding.estimatedKelvin})`;
}

/**
 * Low-latency verification gate: confirms the stream starts emitting non-refusal markdown
 * within the first 16 characters and flushes immediately to minimize Time-To-First-Token.
 */
async function verifyStreamNotRefused(
  rawStream: ReadableStream<Uint8Array>,
  onAbort?: () => void
): Promise<ReadableStream<Uint8Array>> {
  const reader = rawStream.getReader();
  const decoder = new TextDecoder();
  const bufferedChunks: Uint8Array[] = [];
  let initialText = '';

  while (initialText.trim().length < 16) {
    const {done, value} = await reader.read();
    if (done) break;
    if (value) {
      bufferedChunks.push(value);
      initialText += decoder.decode(value, {stream: true});
      const trimmed = initialText.trim();
      // Fast-path: if the model already began emitting Section A header, immediately flush!
      if (
        trimmed.startsWith('### A') ||
        trimmed.startsWith('## A') ||
        trimmed.startsWith('**A.') ||
        trimmed.startsWith('A. Quick')
      ) {
        break;
      }
    }
  }

  if (!initialText.trim() || isRefusalText(initialText)) {
    reader.cancel().catch(() => {});
    onAbort?.();
    throw new Error(
      `Model returned empty or refusal response: "${initialText.slice(0, 60)}"`
    );
  }

  return new ReadableStream<Uint8Array>({
    async start(controller) {
      for (const chunk of bufferedChunks) {
        controller.enqueue(chunk);
      }
    },
    async pull(controller) {
      try {
        const {done, value} = await reader.read();
        if (done) {
          controller.close();
          return;
        }
        if (value) {
          controller.enqueue(value);
        }
      } catch (err) {
        controller.error(err);
      }
    },
    cancel() {
      reader.cancel().catch(() => {});
      onAbort?.();
    },
  });
}

/**
 * Single OpenRouter Vision Lane with automatic fallback across assigned candidate models.
 * Multiple lanes are raced in parallel at t = 0ms so a single queued model never slows down analysis.
 */
async function fetchOpenRouterVisionLane(
  apiKey: string,
  laneModels: string[],
  imageDataUrl: string,
  userMessage: string,
  parentSignal: AbortSignal
): Promise<{stream: ReadableStream<Uint8Array>; model: string}> {
  const combinedInstructionAndPrompt = `${MASTER_INSTRUCTION_FILE_V2}\n\n---\n\n${userMessage}`;
  const errors: string[] = [];

  for (const modelId of laneModels) {
    if (parentSignal.aborted) {
      throw new Error('Aborted');
    }

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 18000);
    const onParentAbort = () => controller.abort();
    parentSignal.addEventListener('abort', onParentAbort, {once: true});

    try {
      const response = await fetch(OPENROUTER_ENDPOINT, {
        method: 'POST',
        signal: controller.signal,
        headers: {
          Authorization: `Bearer ${apiKey.trim()}`,
          'Content-Type': 'application/json',
          Accept: 'text/event-stream',
          'HTTP-Referer':
            process.env.APP_URL ||
            (process.env.VERCEL_URL
              ? `https://${process.env.VERCEL_URL}`
              : 'https://opticspec.vercel.app'),
          'X-Title': 'OpticSpec Image to Prompt',
        },
        body: JSON.stringify({
          model: modelId,
          reasoning: {effort: 'low', exclude: true},
          messages: [
            {
              role: 'user',
              content: [
                {type: 'text', text: combinedInstructionAndPrompt},
                {type: 'image_url', image_url: {url: imageDataUrl}},
              ],
            },
          ],
          temperature: 0.15,
          top_p: 0.9,
          max_tokens: 4096,
          stream: true,
        }),
      });

      clearTimeout(timeoutId);

      if (!response.ok || !response.body) {
        const errText = await response.text().catch(() => '');
        errors.push(
          `OpenRouter (${modelId}) HTTP ${response.status}: ${errText.slice(0, 80)}`
        );
        continue;
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      const encoder = new TextEncoder();

      const sseStream = new ReadableStream<Uint8Array>({
        async start(streamController) {
          let buffer = '';
          try {
            while (true) {
              const {done, value} = await reader.read();
              if (done) break;
              buffer += decoder.decode(value, {stream: true});
              const lines = buffer.split('\n');
              buffer = lines.pop() || '';

              for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed.startsWith('data:')) continue;
                const dataStr = trimmed.slice(5).trim();
                if (dataStr === '[DONE]') continue;
                try {
                  const parsed = JSON.parse(dataStr);
                  const delta =
                    parsed?.choices?.[0]?.delta?.content ??
                    parsed?.choices?.[0]?.message?.content;
                  if (delta) {
                    streamController.enqueue(encoder.encode(delta));
                  }
                } catch {
                  // Ignore partial SSE JSON chunks
                }
              }
            }
            streamController.close();
          } catch (err) {
            streamController.error(err);
          }
        },
        cancel() {
          reader.cancel().catch(() => {});
          controller.abort();
        },
      });

      const verifiedStream = await verifyStreamNotRefused(sseStream, () =>
        controller.abort()
      );
      return {stream: verifiedStream, model: `openrouter:${modelId}`};
    } catch (err) {
      clearTimeout(timeoutId);
      errors.push(err instanceof Error ? err.message : String(err));
    } finally {
      parentSignal.removeEventListener('abort', onParentAbort);
    }
  }

  throw new Error(
    errors.join(' | ') || 'OpenRouter vision lane failed.'
  );
}

/**
 * Direct Google Multimodal Vision stream (`gemini-3.8-flash`, `gemini-3.1-flash-lite`, `gemini-2.5-flash`, `gemini-3-flash-preview`).
 */
async function fetchDirectGoogleVisionStream(
  apiKey: string,
  modelName:
    | 'gemini-3.8-flash'
    | 'gemini-3.1-flash-lite'
    | 'gemini-2.5-flash'
    | 'gemini-3-flash-preview',
  mimeType: string,
  base64Data: string,
  userMessage: string,
  parentSignal: AbortSignal
): Promise<ReadableStream<Uint8Array>> {
  const ai = new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });

  const responseStream = await ai.models.generateContentStream({
    model: modelName,
    contents: {
      parts: [
        {inlineData: {mimeType, data: base64Data}},
        {text: userMessage},
      ],
    },
    config: {
      systemInstruction: MASTER_INSTRUCTION_FILE_V2,
      temperature: 0.15,
      maxOutputTokens: 4096,
      thinkingConfig:
        modelName === 'gemini-3-flash-preview'
          ? {thinkingLevel: ThinkingLevel.MINIMAL}
          : {thinkingBudget: 0},
      safetySettings: [
        {
          category: HarmCategory.HARM_CATEGORY_HARASSMENT,
          threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
        {
          category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
          threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
        {
          category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
          threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
        {
          category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
          threshold: HarmBlockThreshold.BLOCK_ONLY_HIGH,
        },
      ],
    },
  });

  const encoder = new TextEncoder();
  let aborted = false;
  const onAbort = () => {
    aborted = true;
  };
  parentSignal.addEventListener('abort', onAbort, {once: true});

  const rawStream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const chunk of responseStream) {
          if (aborted) break;
          const text = chunk.text;
          if (text) {
            controller.enqueue(encoder.encode(text));
          }
        }
        controller.close();
      } catch (err) {
        if (!aborted) controller.error(err);
      } finally {
        parentSignal.removeEventListener('abort', onAbort);
      }
    },
    cancel() {
      aborted = true;
      parentSignal.removeEventListener('abort', onAbort);
    },
  });

  return await verifyStreamNotRefused(rawStream, () => {
    aborted = true;
  });
}

/**
 * Optional NVIDIA NIM Vision support if NVIDIA_API_KEY is present in environment variables.
 */
async function fetchNvidiaVisionStream(
  apiKey: string,
  modelId: string,
  imageDataUrl: string,
  userMessage: string,
  parentSignal: AbortSignal
): Promise<ReadableStream<Uint8Array>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), 16000);
  const onParentAbort = () => controller.abort();
  parentSignal.addEventListener('abort', onParentAbort, {once: true});

  try {
    const response = await fetch(NVIDIA_NIM_ENDPOINT, {
      method: 'POST',
      signal: controller.signal,
      headers: {
        Authorization: `Bearer ${apiKey.trim()}`,
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
      },
      body: JSON.stringify({
        model: modelId,
        messages: [
          {role: 'system', content: MASTER_INSTRUCTION_FILE_V2},
          {
            role: 'user',
            content: [
              {type: 'text', text: userMessage},
              {type: 'image_url', image_url: {url: imageDataUrl}},
            ],
          },
        ],
        temperature: 0.15,
        top_p: 0.9,
        max_tokens: 4096,
        stream: true,
      }),
    });

    clearTimeout(timeoutId);

    if (!response.ok || !response.body) {
      throw new Error(`NVIDIA (${modelId}) HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    const encoder = new TextEncoder();

    const sseStream = new ReadableStream<Uint8Array>({
      async start(streamController) {
        let buffer = '';
        try {
          while (true) {
            const {done, value} = await reader.read();
            if (done) break;
            buffer += decoder.decode(value, {stream: true});
            const lines = buffer.split('\n');
            buffer = lines.pop() || '';
            for (const line of lines) {
              const trimmed = line.trim();
              if (!trimmed.startsWith('data:')) continue;
              const dataStr = trimmed.slice(5).trim();
              if (dataStr === '[DONE]') continue;
              try {
                const parsed = JSON.parse(dataStr);
                const delta = parsed?.choices?.[0]?.delta?.content;
                if (delta) streamController.enqueue(encoder.encode(delta));
              } catch {
                // ignore partial chunk
              }
            }
          }
          streamController.close();
        } catch (err) {
          streamController.error(err);
        }
      },
      cancel() {
        reader.cancel().catch(() => {});
        controller.abort();
      },
    });

    return await verifyStreamNotRefused(sseStream, () => controller.abort());
  } catch (err) {
    clearTimeout(timeoutId);
    throw err;
  } finally {
    parentSignal.removeEventListener('abort', onParentAbort);
  }
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {imageDataUrl, telemetry, forceFresh} = body as {
      imageDataUrl?: string;
      telemetry?: PixelTelemetry;
      forceFresh?: boolean;
    };

    if (!imageDataUrl || typeof imageDataUrl !== 'string') {
      return NextResponse.json(
        {error: 'Please upload a valid image.'},
        {status: 400}
      );
    }

    const cacheKey = crypto
      .createHash('sha1')
      .update(
        `v16-master-v2:${imageDataUrl.slice(0, 6144)}:${imageDataUrl.length}`
      )
      .digest('hex');

    if (!forceFresh) {
      const cached = PROMPT_CACHE.get(cacheKey);
      if (cached && !isRefusalText(cached.text)) {
        return new NextResponse(cached.text, {
          status: 200,
          headers: {
            'Content-Type': 'text/plain; charset=utf-8',
            'X-Prompt-Cache': 'HIT',
            'X-Engine-Model': cached.model,
          },
        });
      }
    }

    // Trigger non-blocking model discovery refresh in background
    void refreshOpenRouterFreeVisionModelsInBackground();

    const userMessage = buildUserMessage(telemetry);
    const match = imageDataUrl.match(
      /^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/
    );
    const mimeType = match ? match[1] : 'image/jpeg';
    const base64Data = match ? match[2] : imageDataUrl;

    // Resolve API keys across all standard Vercel / AI Studio env var names
    const rawOpenRouterCandidate =
      process.env.OPENROUTER_API_KEY ||
      process.env.NEXT_PUBLIC_OPENROUTER_API_KEY ||
      (process.env.NVIDIA_API_KEY?.trim().startsWith('sk-or-')
        ? process.env.NVIDIA_API_KEY
        : '') ||
      (process.env.GEMINI_API_KEY?.trim().startsWith('sk-or-')
        ? process.env.GEMINI_API_KEY
        : '');

    const openRouterKey =
      rawOpenRouterCandidate &&
      rawOpenRouterCandidate.trim() !== 'MY_OPENROUTER_API_KEY' &&
      rawOpenRouterCandidate.trim().length > 10
        ? rawOpenRouterCandidate.trim()
        : '';

    const rawGeminiCandidate =
      process.env.GEMINI_API_KEY ||
      process.env.GOOGLE_API_KEY ||
      process.env.NEXT_PUBLIC_GEMINI_API_KEY ||
      '';

    const geminiKey =
      rawGeminiCandidate &&
      rawGeminiCandidate.trim() !== 'MY_GEMINI_API_KEY' &&
      !rawGeminiCandidate.trim().startsWith('sk-or-') &&
      rawGeminiCandidate.trim().length > 10
        ? rawGeminiCandidate.trim()
        : '';

    const rawNvidiaCandidate =
      process.env.NVIDIA_API_KEY || process.env.NEXT_PUBLIC_NVIDIA_API_KEY || '';
    const nvidiaKey =
      rawNvidiaCandidate &&
      rawNvidiaCandidate.trim() !== 'MY_NVIDIA_API_KEY' &&
      !rawNvidiaCandidate.trim().startsWith('sk-or-') &&
      rawNvidiaCandidate.trim().length > 10
        ? rawNvidiaCandidate.trim()
        : '';

    if (!openRouterKey && !geminiKey && !nvidiaKey) {
      return NextResponse.json(
        {
          error:
            'Missing API Key on Vercel: Please add OPENROUTER_API_KEY or GEMINI_API_KEY in your Vercel Project Settings → Environment Variables and redeploy.',
        },
        {status: 500}
      );
    }

    const candidateControllers: AbortController[] = [];
    const racePromises: Array<
      Promise<{
        stream: ReadableStream<Uint8Array>;
        ctrl: AbortController;
        model: string;
      }>
    > = [];

    // 1. Parallel OpenRouter Vision Lanes at t = 0ms (eliminates sequential model waiting)
    if (openRouterKey) {
      const modelsPool = cachedFreeVisionModels.length
        ? cachedFreeVisionModels
        : DEFAULT_FREE_VISION_MODELS;

      const parallelLanes: string[][] = [
        [
          'google/gemma-4-26b-a4b-it:free',
          'dots-studio/dots-3-note-preview:free',
          ...modelsPool,
        ],
        [
          'google/gemma-4-31b-it:free',
          'qwen/qwen3.8-27b:free',
          ...modelsPool,
        ],
        [
          'dots-studio/dots-3-note-preview:free',
          'nvidia/nemotron-3-nano-omni-30b-a3b-reasoning:free',
          ...modelsPool,
        ],
      ];

      for (const laneModels of parallelLanes) {
        const uniqueLane = Array.from(new Set(laneModels));
        const ctrl = new AbortController();
        candidateControllers.push(ctrl);
        racePromises.push(
          fetchOpenRouterVisionLane(
            openRouterKey,
            uniqueLane,
            imageDataUrl,
            userMessage,
            ctrl.signal
          ).then(({stream, model}) => ({stream, ctrl, model}))
        );
      }
    }

    // 2. Direct Google Gemini Vision Models (parallel sub-second race)
    if (geminiKey) {
      for (const modelName of [
        'gemini-3.8-flash',
        'gemini-3.1-flash-lite',
        'gemini-2.5-flash',
        'gemini-3-flash-preview',
      ] as const) {
        const ctrl = new AbortController();
        candidateControllers.push(ctrl);
        racePromises.push(
          fetchDirectGoogleVisionStream(
            geminiKey,
            modelName,
            mimeType,
            base64Data,
            userMessage,
            ctrl.signal
          ).then((stream) => ({stream, ctrl, model: `google:${modelName}`}))
        );
      }
    }

    // 3. Optional NVIDIA NIM Vision Models (if NVIDIA_API_KEY is present)
    if (nvidiaKey) {
      for (const modelId of [
        'meta/llama-3.2-11b-vision-instruct',
        'meta/llama-3.2-90b-vision-instruct',
      ]) {
        const ctrl = new AbortController();
        candidateControllers.push(ctrl);
        racePromises.push(
          fetchNvidiaVisionStream(
            nvidiaKey,
            modelId,
            imageDataUrl,
            userMessage,
            ctrl.signal
          ).then((stream) => ({stream, ctrl, model: `nvidia:${modelId}`}))
        );
      }
    }

    let winner: {
      stream: ReadableStream<Uint8Array>;
      ctrl: AbortController;
      model: string;
    };

    try {
      winner = await Promise.any(racePromises);
      for (const ctrl of candidateControllers) {
        if (ctrl !== winner.ctrl) {
          ctrl.abort();
        }
      }
    } catch (aggErr) {
      for (const ctrl of candidateControllers) {
        ctrl.abort();
      }
      const subErrors =
        aggErr instanceof AggregateError && Array.isArray(aggErr.errors)
          ? aggErr.errors
              .map((e) => (e instanceof Error ? e.message : String(e)))
              .join(' | ')
          : aggErr instanceof Error
            ? aggErr.message
            : 'Vision API request failed.';
      return NextResponse.json({error: subErrors}, {status: 502});
    }

    const reader = winner.stream.getReader();
    const decoder = new TextDecoder();
    let fullText = '';

    const clientStream = new ReadableStream<Uint8Array>({
      async pull(controller) {
        try {
          const {done, value} = await reader.read();
          if (done) {
            const cleaned = fullText.trim();
            if (cleaned.length > 250 && !isRefusalText(cleaned)) {
              setCache(cacheKey, cleaned, winner.model);
            }
            controller.close();
            return;
          }
          if (value) {
            fullText += decoder.decode(value, {stream: true});
            controller.enqueue(value);
          }
        } catch (err) {
          controller.error(err);
        }
      },
      cancel() {
        reader.cancel().catch(() => {});
        winner.ctrl.abort();
      },
    });

    return new NextResponse(clientStream, {
      status: 200,
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
        'X-Content-Type-Options': 'nosniff',
        'X-Engine-Model': winner.model,
      },
    });
  } catch (err) {
    const message =
      err instanceof Error ? err.message : 'Error analyzing image.';
    return NextResponse.json({error: message}, {status: 500});
  }
}
