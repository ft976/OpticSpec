import {PixelTelemetry, ColorSwatch, ZoneTelemetry} from './types';

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

export function computeStandardAspectRatio(width: number, height: number): string {
  const ratio = width / height;
  const standards: Array<{label: string; val: number}> = [
    {label: '1:1', val: 1},
    {label: '16:9', val: 16 / 9},
    {label: '9:16', val: 9 / 16},
    {label: '4:3', val: 4 / 3},
    {label: '3:4', val: 3 / 4},
    {label: '3:2', val: 3 / 2},
    {label: '2:3', val: 2 / 3},
    {label: '21:9', val: 21 / 9},
    {label: '4:5', val: 4 / 5},
    {label: '5:4', val: 5 / 4},
  ];

  let closest = standards[0];
  let minDiff = Math.abs(ratio - closest.val);
  for (const s of standards) {
    const diff = Math.abs(ratio - s.val);
    if (diff < minDiff) {
      minDiff = diff;
      closest = s;
    }
  }
  if (minDiff < 0.06) {
    return closest.label;
  }
  const divisor = gcd(width, height);
  const rw = Math.round(width / divisor);
  const rh = Math.round(height / divisor);
  if (rw > 32 || rh > 32) {
    return closest.label;
  }
  return `${rw}:${rh}`;
}

function rgbToHex(r: number, g: number, b: number): string {
  return (
    '#' +
    [r, g, b]
      .map((x) => Math.max(0, Math.min(255, Math.round(x))).toString(16).padStart(2, '0'))
      .join('')
      .toUpperCase()
  );
}

function colorDistance(c1: [number, number, number], c2: [number, number, number]): number {
  const dr = c1[0] - c2[0];
  const dg = c1[1] - c2[1];
  const db = c1[2] - c2[2];
  return Math.sqrt(dr * dr + dg * dg + db * db);
}

function classifyColorRole(r: number, g: number, b: number, index: number): string {
  const lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255;
  const max = Math.max(r, g, b) / 255;
  const min = Math.min(r, g, b) / 255;
  const sat = max === 0 ? 0 : (max - min) / max;

  if (index === 0) return 'Primary Dominant Field';
  if (lum < 0.16) return 'Deep Shadow Base';
  if (lum > 0.84) return 'Specular Peak Highlight';
  if (sat > 0.48) return 'Vibrant Chromatic Accent';
  if (lum < 0.42) return 'Low-Key Structural Tone';
  return 'Midtone Surface Anchor';
}

const ZONE_LABELS = [
  'Top-Left',
  'Top-Center',
  'Top-Right',
  'Mid-Left',
  'Center',
  'Mid-Right',
  'Bottom-Left',
  'Bottom-Center',
  'Bottom-Right',
];

export async function extractImageTelemetryAndCompress(
  sourceUrlOrDataUrl: string,
  maxDimension = 1024
): Promise<{
  compressedDataUrl: string;
  telemetry: PixelTelemetry;
}> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const origWidth = img.naturalWidth || img.width || 1024;
      const origHeight = img.naturalHeight || img.height || 1024;

      // 1. High-clarity resize for vision model ingestion (1024px @ 0.85 JPEG to preserve small text & background characters)
      let targetW = origWidth;
      let targetH = origHeight;
      if (origWidth > maxDimension || origHeight > maxDimension) {
        if (origWidth >= origHeight) {
          targetW = maxDimension;
          targetH = Math.round((origHeight / origWidth) * maxDimension);
        } else {
          targetH = maxDimension;
          targetW = Math.round((origWidth / origHeight) * maxDimension);
        }
      }

      const transportCanvas = document.createElement('canvas');
      transportCanvas.width = targetW;
      transportCanvas.height = targetH;
      const tCtx = transportCanvas.getContext('2d');
      if (!tCtx) {
        reject(new Error('Canvas 2D context unavailable'));
        return;
      }
      tCtx.drawImage(img, 0, 0, targetW, targetH);
      const compressedDataUrl = transportCanvas.toDataURL('image/jpeg', 0.85);

      // 2. 72x72 full-frame sampling grid (divisible by 3 for exact 3x3 9-zone spatial grid)
      const sampleSize = 72;
      const sampleCanvas = document.createElement('canvas');
      sampleCanvas.width = sampleSize;
      sampleCanvas.height = sampleSize;
      const sCtx = sampleCanvas.getContext('2d');
      if (!sCtx) {
        reject(new Error('Sample canvas context unavailable'));
        return;
      }
      sCtx.drawImage(img, 0, 0, sampleSize, sampleSize);
      const imageData = sCtx.getImageData(0, 0, sampleSize, sampleSize).data;

      let shadows = 0;
      let midtones = 0;
      let highlights = 0;
      let lumSum = 0;
      let satSum = 0;
      let rTotal = 0;
      let bTotal = 0;
      const totalPixels = sampleSize * sampleSize;

      // 3x3 9-zone accumulators
      const gridAccum = Array.from({length: 9}, () => ({
        r: 0,
        g: 0,
        b: 0,
        lum: 0,
        n: 0,
      }));

      const buckets = new Map<
        string,
        {count: number; rSum: number; gSum: number; bSum: number}
      >();

      const third = sampleSize / 3;

      for (let y = 0; y < sampleSize; y++) {
        const rowIdx = Math.min(2, Math.floor(y / third));
        for (let x = 0; x < sampleSize; x++) {
          const colIdx = Math.min(2, Math.floor(x / third));
          const zoneIdx = rowIdx * 3 + colIdx;

          const idx = (y * sampleSize + x) * 4;
          const r = imageData[idx];
          const g = imageData[idx + 1];
          const b = imageData[idx + 2];

          const lum = 0.299 * r + 0.587 * g + 0.114 * b;
          lumSum += lum;
          rTotal += r;
          bTotal += b;

          gridAccum[zoneIdx].r += r;
          gridAccum[zoneIdx].g += g;
          gridAccum[zoneIdx].b += b;
          gridAccum[zoneIdx].lum += lum;
          gridAccum[zoneIdx].n++;

          if (lum < 70) shadows++;
          else if (lum > 185) highlights++;
          else midtones++;

          const maxC = Math.max(r, g, b) / 255;
          const minC = Math.min(r, g, b) / 255;
          satSum += maxC === 0 ? 0 : (maxC - minC) / maxC;

          const qr = Math.floor(r / 22) * 22;
          const qg = Math.floor(g / 22) * 22;
          const qb = Math.floor(b / 22) * 22;
          const key = `${qr},${qg},${qb}`;
          const existing = buckets.get(key);
          if (existing) {
            existing.count++;
            existing.rSum += r;
            existing.gSum += g;
            existing.bSum += b;
          } else {
            buckets.set(key, {count: 1, rSum: r, gSum: g, bSum: b});
          }
        }
      }

      const sortedClusters = Array.from(buckets.values())
        .sort((a, b) => b.count - a.count)
        .map((c) => ({
          count: c.count,
          rgb: [
            Math.round(c.rSum / c.count),
            Math.round(c.gSum / c.count),
            Math.round(c.bSum / c.count),
          ] as [number, number, number],
        }));

      const distinctClusters: Array<{count: number; rgb: [number, number, number]}> = [];
      for (const cluster of sortedClusters) {
        if (distinctClusters.length >= 10) break;
        const tooClose = distinctClusters.some(
          (existing) => colorDistance(existing.rgb, cluster.rgb) < 30
        );
        if (!tooClose) {
          distinctClusters.push(cluster);
        }
      }

      for (const cluster of sortedClusters) {
        if (distinctClusters.length >= 10) break;
        if (!distinctClusters.includes(cluster)) {
          distinctClusters.push(cluster);
        }
      }

      const clusterTotal =
        distinctClusters.reduce((acc, c) => acc + c.count, 0) || 1;

      const swatches: ColorSwatch[] = distinctClusters.map((c, idx) => {
        const [r, g, b] = c.rgb;
        return {
          hex: rgbToHex(r, g, b),
          rgb: `rgb(${r}, ${g}, ${b})`,
          percentage: Math.max(1, Math.round((c.count / clusterTotal) * 100)),
          role: classifyColorRole(r, g, b, idx),
        };
      });

      const meanBrightness = Math.round(lumSum / totalPixels);
      const brightnessPct = Math.round((meanBrightness / 255) * 100);
      const shadowsPct = Math.round((shadows / totalPixels) * 100);
      const highlightsPct = Math.round((highlights / totalPixels) * 100);
      const midtonesPct = Math.max(0, 100 - shadowsPct - highlightsPct);

      let dynamicContrast = 'Balanced Full-Range Contrast';
      if (shadowsPct > 52 && highlightsPct > 10) {
        dynamicContrast = 'High-Contrast Chiaroscuro (Deep Blacks + Specular Peaks)';
      } else if (shadowsPct > 58) {
        dynamicContrast = 'Low-Key Dark & Moody Tonal Range';
      } else if (highlightsPct > 48) {
        dynamicContrast = 'High-Key Bright & Airy Tonal Range';
      } else if (midtonesPct > 66) {
        dynamicContrast = 'Soft Logarithmic / Matte Film Midtone Curve';
      }

      let exposureProfile = '0.0 EV Neutral Studio Exposure';
      if (brightnessPct < 28) exposureProfile = '-1.3 EV Low-Key Dark Exposure';
      else if (brightnessPct < 42) exposureProfile = '-0.5 EV Moody Controlled Exposure';
      else if (brightnessPct > 72) exposureProfile = '+1.2 EV High-Key Bright Exposure';
      else if (brightnessPct > 58) exposureProfile = '+0.5 EV Luminous Exposure';

      const meanSaturationPct = Math.round((satSum / totalPixels) * 100);
      const rbDiff = (rTotal - bTotal) / totalPixels;
      let temperatureBias = 'Neutral Daylight Balanced';
      let estimatedKelvin = '5400K Neutral';
      if (rbDiff > 28) {
        temperatureBias = 'Warm Golden Sunlight / Tungsten Bias';
        estimatedKelvin = '3100K–3600K Warm Amber';
      } else if (rbDiff > 10) {
        temperatureBias = 'Warm Golden-Hour Sunlight Bias';
        estimatedKelvin = '4400K–4900K Warm Neutral';
      } else if (rbDiff < -25) {
        temperatureBias = 'Cool Cyan / Blue-Hour Shade Bias';
        estimatedKelvin = '7200K–8500K Cool Slate Blue';
      } else if (rbDiff < -8) {
        temperatureBias = 'Cool Crisp Overcast Daylight Bias';
        estimatedKelvin = '6200K–6800K Cool Daylight';
      }

      const nineZoneGrid: ZoneTelemetry[] = gridAccum.map((z, idx) => ({
        zoneName: ZONE_LABELS[idx],
        hex: z.n > 0 ? rgbToHex(z.r / z.n, z.g / z.n, z.b / z.n) : '#000000',
        brightnessPct:
          z.n > 0 ? Math.round((z.lum / z.n / 255) * 100) : 0,
      }));

      const aspectRatio = computeStandardAspectRatio(origWidth, origHeight);
      const megapixels = ((origWidth * origHeight) / 1_000_000).toFixed(2);

      resolve({
        compressedDataUrl,
        telemetry: {
          width: origWidth,
          height: origHeight,
          aspectRatio,
          megapixels,
          luminance: {
            meanBrightness,
            brightnessPct,
            shadowsPct,
            midtonesPct,
            highlightsPct,
            dynamicContrast,
            exposureProfile,
          },
          colorEncoding: {
            colorSpace: 'sRGB / Rec.709 8-bit Per Channel',
            meanSaturationPct,
            temperatureBias,
            estimatedKelvin,
            nineZoneGrid,
          },
          swatches,
        },
      });
    };
    img.onerror = () => {
      reject(new Error('Failed to decode image'));
    };
    img.src = sourceUrlOrDataUrl;
  });
}
