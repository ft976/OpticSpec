'use client';

import React, {useState, useRef, useEffect, useMemo, useCallback} from 'react';
import {
  Upload,
  Copy,
  Check,
  RefreshCw,
  AlertCircle,
  Trash2,
  ScanEye,
} from 'lucide-react';
import {extractImageTelemetryAndCompress} from '@/lib/pixel-telemetry';

function extractSectionCMasterPrompt(fullOutput: string): string | null {
  const match = fullOutput.match(
    /(?:###?\s*C\.?\s*Final Master Prompt|\*\*C\.?\s*Final Master Prompt\*\*)([\s\S]*?)(?=(?:###?\s*D\.?\s*Negative Prompt|\*\*D\.?\s*Negative Prompt\*\*|$))/i
  );
  if (!match || !match[1]) return null;
  const cleaned = match[1].trim();
  return cleaned.length > 30 ? cleaned : null;
}

export default function OpticSpecStudio() {
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [promptOutput, setPromptOutput] = useState<string>('');
  const [isAnalyzing, setIsAnalyzing] = useState<boolean>(false);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);
  const [copiedFull, setCopiedFull] = useState<boolean>(false);
  const [copiedMasterOnly, setCopiedMasterOnly] = useState<boolean>(false);
  const [isDragging, setIsDragging] = useState<boolean>(false);

  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const abortControllerRef = useRef<AbortController | null>(null);
  const objectUrlRef = useRef<string | null>(null);

  const masterPromptOnly = useMemo(
    () => extractSectionCMasterPrompt(promptOutput),
    [promptOutput]
  );

  const analyzeImage = useCallback(
    async (sourceUrl: string, forceFresh = false) => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
      const controller = new AbortController();
      abortControllerRef.current = controller;

      setErrorMsg(null);
      setIsAnalyzing(true);
      setPromptOutput('');

      try {
        const {compressedDataUrl, telemetry} =
          await extractImageTelemetryAndCompress(sourceUrl, 768);

        const response = await fetch('/api/analyze', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          signal: controller.signal,
          body: JSON.stringify({
            imageDataUrl: compressedDataUrl,
            telemetry,
            forceFresh,
          }),
        });

        if (!response.ok) {
          const errData = await response.json().catch(() => ({}));
          throw new Error(errData.error || 'Failed to analyze image.');
        }

        if (!response.body) {
          const text = await response.text();
          setPromptOutput(text.trim());
          return;
        }

        const reader = response.body.getReader();
        const decoder = new TextDecoder();
        let accumulated = '';

        while (true) {
          const {done, value} = await reader.read();
          if (done) break;
          if (value) {
            accumulated += decoder.decode(value, {stream: true});
            setPromptOutput(accumulated);
          }
        }
      } catch (err) {
        if (err instanceof Error && err.name === 'AbortError') {
          return;
        }
        setErrorMsg(
          err instanceof Error ? err.message : 'Failed to analyze image.'
        );
      } finally {
        if (abortControllerRef.current === controller) {
          setIsAnalyzing(false);
        }
      }
    },
    []
  );

  const handleFile = useCallback(
    (file: File | undefined) => {
      if (!file) return;
      if (!file.type.startsWith('image/')) {
        setErrorMsg('Please upload a valid image file (JPG, PNG, WEBP).');
        return;
      }

      if (objectUrlRef.current) {
        URL.revokeObjectURL(objectUrlRef.current);
      }
      const previewUrl = URL.createObjectURL(file);
      objectUrlRef.current = previewUrl;
      setImagePreview(previewUrl);
      void analyzeImage(previewUrl, false);
    },
    [analyzeImage]
  );

  // Support Ctrl+V paste anywhere on the page
  useEffect(() => {
    const onPaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items;
      if (!items) return;
      for (let i = 0; i < items.length; i++) {
        if (items[i].type.startsWith('image/')) {
          const blob = items[i].getAsFile();
          if (blob) {
            handleFile(blob);
            break;
          }
        }
      }
    };
    window.addEventListener('paste', onPaste);
    return () => window.removeEventListener('paste', onPaste);
  }, [handleFile]);

  const handleCopyFull = () => {
    if (!promptOutput) return;
    navigator.clipboard.writeText(promptOutput);
    setCopiedFull(true);
    setTimeout(() => setCopiedFull(false), 2000);
  };

  const handleCopyMasterOnly = () => {
    if (!masterPromptOnly) return;
    navigator.clipboard.writeText(masterPromptOnly);
    setCopiedMasterOnly(true);
    setTimeout(() => setCopiedMasterOnly(false), 2000);
  };

  const handleReset = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }
    if (objectUrlRef.current) {
      URL.revokeObjectURL(objectUrlRef.current);
      objectUrlRef.current = null;
    }
    setIsAnalyzing(false);
    setImagePreview(null);
    setPromptOutput('');
    setErrorMsg(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="min-h-screen bg-[#0B0F17] text-[#F3F4F6] flex flex-col">
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={(e) => handleFile(e.target.files?.[0])}
        className="hidden"
      />

      {/* Header with Logo & Title */}
      <header className="border-b border-neutral-800/80 bg-[#0B0F17]">
        <div className="max-w-3xl mx-auto px-5 py-4 flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-[#111723] border border-neutral-800 flex items-center justify-center shrink-0">
            <ScanEye className="w-4 h-4 text-[#76B900]" />
          </div>
          <h1
            className="text-lg font-bold tracking-tight text-white whitespace-nowrap"
            style={{fontFamily: 'var(--font-display), sans-serif'}}
          >
            OpticSpec{' '}
            <span className="text-neutral-400 font-normal text-sm ml-1">
              Image to Prompt
            </span>
          </h1>
        </div>
      </header>

      {/* Simple Centered Main Content */}
      <main className="flex-1 max-w-3xl w-full mx-auto px-5 py-8 space-y-8">
        {/* 1. UPLOAD IMAGE AREA */}
        {!imagePreview ? (
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(e) => {
              e.preventDefault();
              setIsDragging(false);
              handleFile(e.dataTransfer.files?.[0]);
            }}
            onClick={() => fileInputRef.current?.click()}
            className={`cursor-pointer border-2 border-dashed rounded-2xl p-12 text-center transition-colors flex flex-col items-center justify-center min-h-[280px] ${
              isDragging
                ? 'border-[#76B900] bg-[#76B900]/5'
                : 'border-neutral-800 bg-[#111723] hover:border-neutral-600'
            }`}
          >
            <div className="w-12 h-12 rounded-full bg-neutral-900 border border-neutral-800 flex items-center justify-center mb-4">
              <Upload className="w-5 h-5 text-[#76B900]" />
            </div>
            <p className="text-base font-semibold text-white mb-1">
              Click to upload an image
            </p>
            <p className="text-xs text-neutral-400">
              or drag & drop / press{' '}
              <span className="font-mono text-neutral-300">Ctrl+V</span> to
              paste
            </p>
          </div>
        ) : (
          <div className="space-y-3">
            <div className="relative max-h-[380px] w-full flex items-center justify-center overflow-hidden rounded-xl">
              <img
                src={imagePreview}
                alt="Uploaded source"
                referrerPolicy="no-referrer"
                className="max-h-[380px] w-auto object-contain select-none rounded-xl"
              />
            </div>

            <div className="flex items-center justify-between gap-3">
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className="px-3.5 py-1.5 text-xs font-medium text-white bg-neutral-800 hover:bg-neutral-700 rounded-lg transition-colors flex items-center gap-1.5"
              >
                <Upload className="w-3.5 h-3.5" />
                <span>Upload Different Image</span>
              </button>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => analyzeImage(imagePreview, true)}
                  className="px-3.5 py-1.5 text-xs font-medium text-neutral-200 bg-neutral-900 border border-neutral-800 hover:bg-neutral-800 rounded-lg transition-colors flex items-center gap-1.5"
                >
                  <RefreshCw
                    className={`w-3.5 h-3.5 ${
                      isAnalyzing ? 'animate-spin text-[#76B900]' : ''
                    }`}
                  />
                  <span>Re-analyze</span>
                </button>

                <button
                  type="button"
                  onClick={handleReset}
                  title="Remove image"
                  className="p-1.5 text-neutral-400 hover:text-red-400 bg-neutral-900 border border-neutral-800 rounded-lg transition-colors"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ERROR ALERT */}
        {errorMsg && (
          <div className="flex items-center justify-between gap-4 text-red-300 text-sm">
            <div className="flex items-center gap-2">
              <AlertCircle className="w-4 h-4 text-red-400 shrink-0" />
              <span>{errorMsg}</span>
            </div>
            {imagePreview && (
              <button
                type="button"
                onClick={() => analyzeImage(imagePreview, true)}
                className="px-3 py-1.5 text-xs font-medium bg-red-900/60 hover:bg-red-800 text-white rounded-lg shrink-0"
              >
                Retry
              </button>
            )}
          </div>
        )}

        {/* 2. DIRECT UNBOXED PROMPT & ANALYSIS OUTPUT */}
        {(isAnalyzing || promptOutput) && (
          <div className="space-y-4 pt-2">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex items-center gap-2">
                {isAnalyzing && (
                  <RefreshCw className="w-4 h-4 text-[#76B900] animate-spin" />
                )}
                <span className="text-xs font-medium text-neutral-400">
                  {isAnalyzing && !promptOutput
                    ? 'Running 7-Pass Master Analysis (A. Quick Summary → B. Detailed Breakdown → C. Final Master Prompt → D. Negative Prompt → E. Short Version → F. Uncertainty Notes)...'
                    : 'Analysis & Prompt'}
                </span>
              </div>

              {promptOutput && (
                <div className="flex items-center gap-2">
                  {masterPromptOnly && (
                    <button
                      type="button"
                      onClick={handleCopyMasterOnly}
                      className="px-3.5 py-2 text-xs font-semibold text-white bg-neutral-800 hover:bg-neutral-700 rounded-lg transition-colors flex items-center gap-1.5 whitespace-nowrap"
                    >
                      {copiedMasterOnly ? (
                        <>
                          <Check className="w-3.5 h-3.5 text-[#76B900]" />
                          <span>Master Prompt Copied!</span>
                        </>
                      ) : (
                        <>
                          <Copy className="w-3.5 h-3.5 text-[#76B900]" />
                          <span>Copy Master Prompt (C)</span>
                        </>
                      )}
                    </button>
                  )}

                  <button
                    type="button"
                    onClick={handleCopyFull}
                    className="px-4 py-2 text-xs font-semibold text-[#0B0F17] bg-[#76B900] hover:bg-[#86D100] rounded-lg transition-colors flex items-center gap-1.5 whitespace-nowrap"
                  >
                    {copiedFull ? (
                      <>
                        <Check className="w-3.5 h-3.5" />
                        <span>Copied!</span>
                      </>
                    ) : (
                      <>
                        <Copy className="w-3.5 h-3.5" />
                        <span>Copy Full Output</span>
                      </>
                    )}
                  </button>
                </div>
              )}
            </div>

            {promptOutput && (
              <p className="text-sm sm:text-base text-neutral-100 leading-relaxed select-all whitespace-pre-wrap">
                {promptOutput}
              </p>
            )}
          </div>
        )}
      </main>

      {/* Footer Attribution */}
      <footer className="max-w-3xl w-full mx-auto px-5 py-6 text-center">
        <p className="text-xs sm:text-sm font-medium text-neutral-400 tracking-wide">
          Developed by Rehan...🌻
        </p>
      </footer>
    </div>
  );
}
