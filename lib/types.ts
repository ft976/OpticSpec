export type OpenRouterGoogleVisionModelId =
  | 'google/gemma-3-27b-it:free'
  | 'google/gemini-2.0-flash-exp:free'
  | 'google/gemma-3-12b-it:free'
  | 'google/gemma-3-4b-it:free';

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
