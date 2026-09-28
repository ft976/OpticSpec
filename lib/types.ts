export type OpenRouterGoogleVisionModelId =
  | 'google/gemma-4-31b-it:free'
  | 'google/gemma-4-26b-a4b-it:free'
  | 'qwen/qwen3.8-27b:free'
  | 'thinkingmachines/inkling:free'
  | 'thinkingmachines/inkling-small:free'
  | 'dots-studio/dots-3-note-preview:free'
  | 'openrouter/free';

export interface ColorSwatch {
  hex: string;
  rgb: string;
  r?: number;
  g?: number;
  b?: number;
  percentage: number;
  role: string;
}

export interface SpatialZoneTelemetry {
  zoneName: string;
  hex: string;
  brightnessPct: number;
}

export type ZoneTelemetry = SpatialZoneTelemetry;

export interface LuminanceProfile {
  meanBrightness: number; // 0 - 255
  brightnessPct: number; // 0 - 100
  shadowsPct: number;
  midtonesPct: number;
  highlightsPct: number;
  dynamicContrast: string;
  exposureProfile: string;
}

export interface ColorEncodingProfile {
  colorSpace: string;
  meanSaturationPct: number;
  temperatureBias: string;
  estimatedKelvin: string;
  nineZoneGrid: SpatialZoneTelemetry[];
}

export interface PixelTelemetry {
  width: number;
  height: number;
  aspectRatio: string;
  megapixels: string;
  swatches: ColorSwatch[];
  luminance: LuminanceProfile;
  colorEncoding: ColorEncodingProfile;
}
