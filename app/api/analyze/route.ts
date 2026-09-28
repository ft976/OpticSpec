import {NextRequest, NextResponse} from 'next/server';
import crypto from 'crypto';
import {
  GoogleGenAI,
  HarmBlockThreshold,
  HarmCategory,
  ThinkingLevel,
} from '@google/genai';
import {OpenRouterGoogleVisionModelId, PixelTelemetry} from '@/lib/types';

export const runtime = 'nodejs';
export const dynamic = 'force-dynamic';
export const maxDuration = 60;

const OPENROUTER_ENDPOINT = 'https://openrouter.ai/api/v1/chat/completions';

// High-speed in-memory LRU cache (instant <2ms response on repeat images)
const PROMPT_CACHE = new Map<string, {text: string; model: string}>();
const MAX_CACHE_ENTRIES = 150;

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

const QUICK_USE_USER_PROMPT = `Analyse the attached image completely, covering the entire frame including all corners and edges. Follow the Image-to-Prompt Master Instructions. Count every person and describe each one's apparent gender presentation, age group, body, clothing, pose, movement, facial expression, emotion, body language, and the impression they give. Identify all objects, animals, plants, and environment details. Read every written word and place it exactly where it appears. Describe all colours with exact names, approximate hex codes, colour mood, brightness, sunlight, shadows, and lighting direction. Describe all motion and stillness, sizes, positions, design, and overall atmosphere. Do not identify real people from faces, and do not invent anything that is not visible; mark unclear details as unclear. Deliver: A) Quick Summary, B) Detailed Breakdown, C) Final Master Prompt, D) Negative Prompt, E) Short Version, F) Uncertainty Notes.`;

// OpenRouter allows up to 3 models in its `models` fallback array
const OPENROUTER_GOOGLE_MODEL_TRIO: OpenRouterGoogleVisionModelId[] = [
  'google/gemma-3-27b-it:free',
  'google/gemini-2.0-flash-exp:free',
  'google/gemma-3-12b-it:free',
];

function buildUserMessage(telemetry?: PixelTelemetry): string {
  if (!telemetry) {
    return QUICK_USE_USER_PROMPT;
  }

  const approxSwatches = telemetry.swatches
    .slice(0, 8)
    .map((s) => `approx. ${s.hex}`)
    .join(', ');

  return `${QUICK_USE_USER_PROMPT}

(Approximate visual reference notes from image scan per Rule 3.5: orientation/aspect ratio approx. ${telemetry.aspectRatio}; dominant approximate hex references: ${approxSwatches}; approximate overall brightness ~${telemetry.luminance.brightnessPct}%.)`;
}

/**
 * Ultra-fast non-refusal stream verifier.
 * Inspects only the first ~38 characters (enough to catch any "I can't help" / "I'm sorry" refusal)
 * so the winning stream begins piping bytes to the browser immediately.
 */
async function verifyStreamNotRefused(
  rawStream: ReadableStream<Uint8Array>,
  onAbort?: () => void
): Promise<ReadableStream<Uint8Array>> {
  const reader = rawStream.getReader();
  const decoder = new TextDecoder();
  const bufferedChunks: Uint8Array[] = [];
  let initialText = '';

  while (initialText.length < 38) {
    const {done, value} = await reader.read();
    if (done) break;
    if (value) {
      bufferedChunks.push(value);
      initialText += decoder.decode(value, {stream: true});
    }
  }

  if (isRefusalText(initialText)) {
    reader.cancel().catch(() => {});
    onAbort?.();
    throw new Error(
      `Model returned refusal response: "${initialText.slice(0, 60)}"`
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
 * Calls OpenRouter's free Google Vision-Language models in a single compliant request
 * (`models` array of 3 Google models: `google/gemma-3-27b-it:free`, `google/gemini-2.0-flash-exp:free`, `google/gemma-3-12b-it:free`)
 * with unified user-role instructions so Gemma 3 and Gemini 2.0 both accept the payload cleanly.
 */
async function fetchOpenRouterGoogleStream(
  apiKey: string,
  imageDataUrl: string,
  userMessage: string,
  parentSignal: AbortSignal,
  timeoutMs = 7000
): Promise<ReadableStream<Uint8Array>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);
  const onParentAbort = () => controller.abort();
  parentSignal.addEventListener('abort', onParentAbort, {once: true});

  const combinedInstructionAndPrompt = `${MASTER_INSTRUCTION_FILE_V2}\n\n---\n\n${userMessage}`;

  try {
    const response = await fetch(OPENROUTER_ENDPOINT, {
      method: 'POST',
      signal: controller.signal,
      headers: {
        Authorization: `Bearer ${apiKey.trim()}`,
        'Content-Type': 'application/json',
        Accept: 'text/event-stream',
        'HTTP-Referer': process.env.APP_URL || 'https://opticspec.app',
        'X-Title': 'OpticSpec Image to Prompt',
      },
      body: JSON.stringify({
        model: OPENROUTER_GOOGLE_MODEL_TRIO[0],
        models: OPENROUTER_GOOGLE_MODEL_TRIO,
        route: 'fallback',
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
      throw new Error(
        `OpenRouter Google HTTP ${response.status}: ${errText.slice(0, 120)}`
      );
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

    return await verifyStreamNotRefused(sseStream, () => controller.abort());
  } catch (err) {
    clearTimeout(timeoutId);
    throw err;
  } finally {
    parentSignal.removeEventListener('abort', onParentAbort);
  }
}

/**
 * Direct Google Multimodal Vision stream (`gemini-2.5-flash` / `gemini-3-flash-preview`)
 * with zero thinking latency (`thinkingBudget: 0` / `ThinkingLevel.MINIMAL`).
 */
async function fetchDirectGoogleVisionStream(
  modelName: 'gemini-2.5-flash' | 'gemini-3-flash-preview',
  mimeType: string,
  base64Data: string,
  userMessage: string,
  parentSignal: AbortSignal
): Promise<ReadableStream<Uint8Array>> {
  const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
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

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {imageDataUrl, telemetry} = body as {
      imageDataUrl?: string;
      telemetry?: PixelTelemetry;
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
        `v11-openrouter-google-clean:${imageDataUrl.slice(0, 6144)}:${imageDataUrl.length}`
      )
      .digest('hex');

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

    const userMessage = buildUserMessage(telemetry);
    const match = imageDataUrl.match(
      /^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/
    );
    const mimeType = match ? match[1] : 'image/jpeg';
    const base64Data = match ? match[2] : imageDataUrl;

    const rawOpenRouterKey =
      process.env.OPENROUTER_API_KEY &&
      process.env.OPENROUTER_API_KEY.trim() !== 'MY_OPENROUTER_API_KEY' &&
      process.env.OPENROUTER_API_KEY.trim().length > 8
        ? process.env.OPENROUTER_API_KEY.trim()
        : '';

    const candidateControllers: AbortController[] = [];
    const createCandidate = (
      modelLabel: string,
      runner: (signal: AbortSignal) => Promise<ReadableStream<Uint8Array>>
    ) => {
      const ctrl = new AbortController();
      candidateControllers.push(ctrl);
      return runner(ctrl.signal).then((stream) => ({
        stream,
        ctrl,
        model: modelLabel,
      }));
    };

    const racePromises: Array<
      Promise<{
        stream: ReadableStream<Uint8Array>;
        ctrl: AbortController;
        model: string;
      }>
    > = [];

    if (rawOpenRouterKey) {
      racePromises.push(
        createCandidate('openrouter:google/gemma-3-27b-it:free', (signal) =>
          fetchOpenRouterGoogleStream(
            rawOpenRouterKey,
            imageDataUrl,
            userMessage,
            signal,
            7000
          )
        )
      );
    }

    // Race alongside direct Google Vision streams so response starts in ~350ms with zero errors
    racePromises.push(
      createCandidate('google:gemini-2.5-flash', (signal) =>
        fetchDirectGoogleVisionStream(
          'gemini-2.5-flash',
          mimeType,
          base64Data,
          userMessage,
          signal
        )
      ),
      createCandidate('google:gemini-3-flash-preview', (signal) =>
        fetchDirectGoogleVisionStream(
          'gemini-3-flash-preview',
          mimeType,
          base64Data,
          userMessage,
          signal
        )
      )
    );

    const activeWinner = await Promise.any(racePromises);

    for (const ctrl of candidateControllers) {
      if (ctrl !== activeWinner.ctrl) {
        ctrl.abort();
      }
    }

    const reader = activeWinner.stream.getReader();
    const decoder = new TextDecoder();
    let fullText = '';

    const clientStream = new ReadableStream<Uint8Array>({
      async pull(controller) {
        try {
          const {done, value} = await reader.read();
          if (done) {
            const cleaned = fullText.trim();
            if (cleaned.length > 250 && !isRefusalText(cleaned)) {
              setCache(cacheKey, cleaned, activeWinner.model);
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
        activeWinner.ctrl.abort();
      },
    });

    return new NextResponse(clientStream, {
      status: 200,
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
        'X-Content-Type-Options': 'nosniff',
        'X-Engine-Model': activeWinner.model,
      },
    });
  } catch (err) {
    const message =
      err instanceof Error ? err.message : 'Error analyzing image.';
    return NextResponse.json({error: message}, {status: 500});
  }
}
