import {NextRequest, NextResponse} from 'next/server';
import crypto from 'crypto';
import {
  GoogleGenAI,
  HarmBlockThreshold,
  HarmCategory,
  ThinkingLevel,
} from '@google/genai';
import {NvidiaVisionModelId, PixelTelemetry} from '@/lib/types';

const NVIDIA_NIM_ENDPOINT =
  'https://integrate.api.nvidia.com/v1/chat/completions';

// High-speed in-memory LRU cache (only stores verified, non-refusal prompts)
const PROMPT_CACHE = new Map<string, string>();
const MAX_CACHE_ENTRIES = 120;

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
  const sample = text.trim().slice(0, 220);
  if (!sample) return true;
  return REFUSAL_PATTERNS.some((pattern) => pattern.test(sample));
}

function setCache(key: string, promptText: string) {
  if (isRefusalText(promptText) || promptText.length < 150) return;
  if (PROMPT_CACHE.size >= MAX_CACHE_ENTRIES) {
    const oldestKey = PROMPT_CACHE.keys().next().value;
    if (oldestKey) PROMPT_CACHE.delete(oldestKey);
  }
  PROMPT_CACHE.set(key, promptText);
}

const MASTER_INSTRUCTION_NOTE = `Fetch all and entire data—every single data point from the image:
1. Side angle, camera viewing angle, framing, focal length, depth of field, and full-frame geometry.
2. Exact number of characters / no. of persons in frame: do NOT focus on only one person—focus on the ENTIRE FRAME and describe every single person/character in the foreground, midground, and background.
3. Body, posture, anatomy, wardrobe, design variables, relative size, scale, and proportions of every character and object.
4. Facial expression and emotion of all variables: perfectly defined lived experience on everyone, micro-expressions (eyes, gaze, brows, lips, posture), personal impression, interpersonal emotional dynamic, and the emotional resonance of every scene variable.
5. All variables and their movements—both MOTION (dynamic movement, walking, gestures, wind, flowing elements, trajectory) and UN-MOTION (static/stationary posture, anchored objects, still architecture)—and the entire characteristics of each variable.
6. All WRITTEN words (exact OCR transcription of every visible text, word, number, sign, logo, font style, size, colour, placement) and UNWRITTEN visual elements, symbols, textures, and graphic design variables.
7. Sunlight (solar angle, direction, sunbeams, natural vs. artificial lighting, shadows, highlights), brightness, luminance, environment, overall universe aesthetic, colour encoding (sRGB, Kelvin temperature, saturation), colour combination (exact HEX + RGB swatches), and every pixel across all 9 spatial zones of the image.`;

function buildVisualPromptInstruction(
  telemetry?: PixelTelemetry,
  customNote?: string
): string {
  const zoneGridString = telemetry?.colorEncoding.nineZoneGrid
    ? telemetry.colorEncoding.nineZoneGrid
        .map((z) => `${z.zoneName}: ${z.hex} (${z.brightnessPct}% brightness)`)
        .join(' | ')
    : '';

  const swatchString = telemetry?.swatches
    ? telemetry.swatches
        .map((s) => `${s.hex} ${s.rgb} (${s.percentage}% ${s.role})`)
        .join(', ')
    : '';

  const telemetryBlock = telemetry
    ? `MEASURED FULL-FRAME PIXEL TELEMETRY (MUST BE INCLUDED IN THE FINAL PROMPT):
- Exact Frame Dimensions & Aspect Ratio: ${telemetry.width}x${telemetry.height} px (Aspect Ratio ${telemetry.aspectRatio}, ${telemetry.megapixels} MP)
- Brightness, Luminance & Exposure: Mean Brightness ${telemetry.luminance.meanBrightness}/255 (${telemetry.luminance.brightnessPct}%), ${telemetry.luminance.exposureProfile}, Contrast Profile: ${telemetry.luminance.dynamicContrast} (Shadows ${telemetry.luminance.shadowsPct}%, Midtones ${telemetry.luminance.midtonesPct}%, Highlights ${telemetry.luminance.highlightsPct}%)
- Sunlight & Colour Encoding: ${telemetry.colorEncoding.colorSpace}, Mean Saturation ${telemetry.colorEncoding.meanSaturationPct}%, ${telemetry.colorEncoding.temperatureBias} (${telemetry.colorEncoding.estimatedKelvin})
- 9-Zone Entire-Frame Pixel Map (Every Corner & Region): ${zoneGridString}
- 10-Point Dominant Colour Combination (HEX + RGB): ${swatchString}`
    : '';

  const activeNote = customNote?.trim() || MASTER_INSTRUCTION_NOTE;

  return `Act as an elite visual scene deconstructor, master cinematographer, human facial expression & emotion analyst, OCR specialist, and precision AI prompt engineer.

MASTER INSTRUCTION NOTE (STRICTLY FOLLOW EVERY REQUIREMENT BELOW):
${activeNote}

${telemetryBlock}

Write one exhaustive, ultra-detailed, continuous text-to-image generation prompt (450 to 700 words) that captures EVERY SINGLE PIXEL and EVERY SINGLE DATA POINT of this image so another AI image generator will recreate the exact same image:

1. ENTIRE FRAME, SIDE ANGLE & CAMERA GEOMETRY:
   - Scan the entire frame from edge to edge across all 9 zones (Top-Left, Top-Center, Top-Right, Mid-Left, Center, Mid-Right, Bottom-Left, Bottom-Center, Bottom-Right).
   - State the exact framing, camera height, viewing angle, side/profile angle (e.g., front-facing, 3/4 oblique side angle, profile, low-angle, high-angle, eye-level), lens focal length in mm, f-stop aperture, and depth of field.
2. EXACT NO. OF PERSONS / CHARACTERS & BODY / DESIGN CHARACTERISTICS:
   - Explicitly state the exact number of persons/characters/figures in the frame. Never focus on just one—describe EVERY single character across foreground, midground, and background.
   - For each person/character, detail their body build, exact pose, head/shoulder/limb angles, relative size and scale in the frame, hair style/color, skin/surface texture, clothing/wardrobe design variables, fabric folds, and accessories.
3. FACIAL EXPRESSIONS, EMOTION OF ALL VARIABLES, DEFINED EXPERIENCE & IMPRESSION:
   - Detail the exact facial expression and micro-expressions on every person (eyebrows, eyelid openness, eye gaze vector, eye catchlights, cheek contours, lip curvature, jaw tension).
   - Perfectly define the lived experience, internal emotion, mood, and personal impression on everyone (e.g., serene warmth, intense focus, candid joy, quiet resilience, nostalgia, awe), plus the emotional aura and impression radiated by every non-human variable, object, and environment in the frame.
4. ALL VARIABLES, MOVEMENTS (MOTION & UN-MOTION) & COMPLETE CHARACTERISTICS:
   - Describe every moving ("motion") variable and its direction/speed/trajectory (walking, gestures, flowing hair/cloth, wind, water, particles, vehicles, motion blur).
   - Describe every stationary ("un-motion") variable (fixed posture, anchored props, still architecture, ground/floor, background fixtures) and all physical/material characteristics of each variable.
5. ALL WRITTEN WORDS (OCR) & UNWRITTEN DESIGN VARIABLES / SIZE:
   - Transcribe word-for-word EVERY visible written word, number, letter, sign, logo, label, badge, or text in quotes, specifying its typography/font design, size, colour, and exact position.
   - Also describe all unwritten visual symbols, graphic patterns, design variables, relative sizes, and spatial proportions across the frame.
6. SUNLIGHT, LIGHTING, ENVIRONMENT, UNIVERSE, BRIGHTNESS, COLOUR ENCODING & COLOUR COMBINATION:
   - Detail the environment, setting, atmosphere, and overall visual universe.
   - Describe the sunlight direction, solar elevation angle, natural daylight vs. artificial light sources, shadow sharpness/length, and specular highlights.
   - Explicitly weave in the measured brightness (${telemetry?.luminance.brightnessPct ?? 50}% luminance, ${telemetry?.luminance.exposureProfile ?? 'balanced exposure'}), colour encoding (${telemetry?.colorEncoding.colorSpace ?? 'sRGB'}, ${telemetry?.colorEncoding.estimatedKelvin ?? '5400K'}, ${telemetry?.colorEncoding.meanSaturationPct ?? 50}% saturation), 9-zone spatial colours, and exact HEX + RGB colour combinations (${telemetry?.swatches.map((s) => `${s.hex} ${s.rgb}`).join(', ') || 'exact hex/rgb palette'}).
   - End the prompt with "--ar ${telemetry?.aspectRatio || '16:9'} --v 6.1 --style raw --s 250 --q 2".

IMPORTANT: Output ONLY the final descriptive image generation prompt text itself. Do NOT include introductory commentary or markdown headings.`;
}

/**
 * Wraps an upstream stream and verifies the first ~85 characters are NOT a refusal
 * before resolving. If a model outputs a refusal ("I can't help you with that..."),
 * this function rejects immediately so the caller seamlessly switches to another engine.
 */
async function verifyStreamNotRefused(
  rawStream: ReadableStream<Uint8Array>
): Promise<ReadableStream<Uint8Array>> {
  const reader = rawStream.getReader();
  const decoder = new TextDecoder();
  const bufferedChunks: Uint8Array[] = [];
  let initialText = '';

  while (initialText.length < 85) {
    const {done, value} = await reader.read();
    if (done) break;
    if (value) {
      bufferedChunks.push(value);
      initialText += decoder.decode(value, {stream: true});
    }
  }

  if (isRefusalText(initialText)) {
    reader.cancel().catch(() => {});
    throw new Error(
      `Model returned refusal response: "${initialText.slice(0, 80)}"`
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
    },
  });
}

async function fetchNvidiaVerifiedStream(
  apiKey: string,
  modelId: NvidiaVisionModelId,
  imageDataUrl: string,
  instruction: string,
  timeoutMs = 9000
): Promise<ReadableStream<Uint8Array>> {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

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
        {
          role: 'system',
          content:
            'You are an expert cinematography, facial expression and human emotion analyst, optical layout, OCR text transcription, and full-frame visual scene description assistant for AI image generation prompts. Describe every visible character, their exact facial expression, emotional experience, and impression, plus all motion/static variables, written words, sunlight angles, scale, and colors in complete detail.',
        },
        {
          role: 'user',
          content: [
            {type: 'text', text: instruction},
            {type: 'image_url', image_url: {url: imageDataUrl}},
          ],
        },
      ],
      temperature: 0.2,
      top_p: 0.9,
      max_tokens: 1550,
      stream: true,
    }),
  });

  clearTimeout(timeoutId);

  if (!response.ok || !response.body) {
    const errText = await response.text().catch(() => '');
    throw new Error(
      `NVIDIA NIM (${modelId}) HTTP ${response.status}: ${errText.slice(0, 140)}`
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
  });

  return await verifyStreamNotRefused(sseStream);
}

async function createVerifiedGeminiStream(
  imageDataUrl: string,
  instruction: string
): Promise<ReadableStream<Uint8Array>> {
  const ai = new GoogleGenAI({
    apiKey: process.env.GEMINI_API_KEY,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });

  const match = imageDataUrl.match(
    /^data:(image\/[a-zA-Z0-9.+-]+);base64,(.+)$/
  );
  const mimeType = match ? match[1] : 'image/jpeg';
  const base64Data = match ? match[2] : imageDataUrl;

  const candidateModels = ['gemini-3-flash-preview', 'gemini-2.5-flash'];
  let lastErr: unknown = null;

  for (const modelName of candidateModels) {
    try {
      const responseStream = await ai.models.generateContentStream({
        model: modelName,
        contents: {
          parts: [
            {inlineData: {mimeType, data: base64Data}},
            {text: instruction},
          ],
        },
        config: {
          temperature: 0.2,
          ...(modelName === 'gemini-3-flash-preview'
            ? {thinkingConfig: {thinkingLevel: ThinkingLevel.LOW}}
            : {}),
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
      const rawStream = new ReadableStream<Uint8Array>({
        async start(controller) {
          try {
            for await (const chunk of responseStream) {
              const text = chunk.text;
              if (text) {
                controller.enqueue(encoder.encode(text));
              }
            }
            controller.close();
          } catch (err) {
            controller.error(err);
          }
        },
      });

      return await verifyStreamNotRefused(rawStream);
    } catch (err) {
      lastErr = err;
    }
  }

  throw lastErr instanceof Error
    ? lastErr
    : new Error('Failed to generate prompt stream.');
}

export async function POST(req: NextRequest) {
  try {
    const body = await req.json();
    const {imageDataUrl, telemetry, instructionNote} = body as {
      imageDataUrl?: string;
      telemetry?: PixelTelemetry;
      instructionNote?: string;
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
        `v4:${imageDataUrl.slice(0, 6144)}:${imageDataUrl.length}:${instructionNote || ''}`
      )
      .digest('hex');

    const cachedPrompt = PROMPT_CACHE.get(cacheKey);
    if (cachedPrompt && !isRefusalText(cachedPrompt)) {
      return new NextResponse(cachedPrompt, {
        status: 200,
        headers: {
          'Content-Type': 'text/plain; charset=utf-8',
          'X-Prompt-Cache': 'HIT',
        },
      });
    }

    const instruction = buildVisualPromptInstruction(
      telemetry,
      instructionNote
    );
    const rawNvidiaKey = process.env.NVIDIA_API_KEY;
    const hasValidNvidiaKey = Boolean(
      rawNvidiaKey &&
        rawNvidiaKey.trim().length > 8 &&
        rawNvidiaKey.trim() !== 'MY_NVIDIA_API_KEY'
    );

    let verifiedStream: ReadableStream<Uint8Array> | null = null;

    if (hasValidNvidiaKey && rawNvidiaKey) {
      try {
        // Race top NVIDIA vision models in parallel and verify the winner is NOT a refusal
        verifiedStream = await Promise.any([
          fetchNvidiaVerifiedStream(
            rawNvidiaKey,
            'qwen/qwen2.5-vl-72b-instruct',
            imageDataUrl,
            instruction,
            9000
          ),
          fetchNvidiaVerifiedStream(
            rawNvidiaKey,
            'meta/llama-3.2-90b-vision-instruct',
            imageDataUrl,
            instruction,
            9000
          ),
          fetchNvidiaVerifiedStream(
            rawNvidiaKey,
            'meta/llama-3.2-11b-vision-instruct',
            imageDataUrl,
            instruction,
            9000
          ),
        ]);
      } catch (nvidiaErr) {
        console.warn(
          'NVIDIA models refused or timed out, switching to verified high-speed multimodal engine:',
          nvidiaErr
        );
      }
    }

    if (!verifiedStream) {
      verifiedStream = await createVerifiedGeminiStream(
        imageDataUrl,
        instruction
      );
    }

    const reader = verifiedStream.getReader();
    const decoder = new TextDecoder();
    let fullText = '';

    const clientStream = new ReadableStream<Uint8Array>({
      async pull(controller) {
        try {
          const {done, value} = await reader.read();
          if (done) {
            const cleaned = fullText.trim();
            if (cleaned.length > 150 && !isRefusalText(cleaned)) {
              setCache(cacheKey, cleaned);
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
    });

    return new NextResponse(clientStream, {
      status: 200,
      headers: {
        'Content-Type': 'text/plain; charset=utf-8',
        'Cache-Control': 'no-cache, no-transform',
      },
    });
  } catch (err) {
    const message =
      err instanceof Error ? err.message : 'Error analyzing image.';
    return NextResponse.json({error: message}, {status: 500});
  }
}
