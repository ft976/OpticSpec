export type NvidiaVisionModelId =
  | 'meta/llama-3.2-90b-vision-instruct'
  | 'meta/llama-3.2-11b-vision-instruct'
  | 'qwen/qwen2.5-vl-72b-instruct'
  | 'microsoft/phi-3.5-vision-instruct';

export interface ColorSwatch {
  hex: string;
  rgb: string;
  percentage: number;
  role: string;
}

export interface ZoneTelemetry {
  zoneName: string;
  hex: string;
  brightnessPct: number;
}

export interface PixelTelemetry {
  width: number;
  height: number;
  aspectRatio: string;
  megapixels: string;
  luminance: {
    meanBrightness: number;
    brightnessPct: number;
    shadowsPct: number;
    midtonesPct: number;
    highlightsPct: number;
    dynamicContrast: string;
    exposureProfile: string;
  };
  colorEncoding: {
    colorSpace: string;
    meanSaturationPct: number;
    temperatureBias: string;
    estimatedKelvin: string;
    nineZoneGrid: ZoneTelemetry[];
  };
  swatches: ColorSwatch[];
}
