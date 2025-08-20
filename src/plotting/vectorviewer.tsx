// import React from "react};
import React, { useRef, useState, useEffect, useCallback, useMemo } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";

export default function VectorViewer({ backendUrl = "/backend", config }: { backendUrl?: string; config?: any }) {
  const [hasRendered, setHasRendered] = useState(false);
  const [directory, setDirectory] = useState<string>("C:/Users/ees1u24/Desktop/PIVTools/PlottingPlayground");
  const [index, setIndex] = useState<number>(1);
  const [type, setType] = useState<string>("ux");
  const [run, setRun] = useState<number>(1);
  const [lower, setLower] = useState<string>("");
  const [upper, setUpper] = useState<string>("");
  const [cmap, setCmap] = useState<string>("default");
  const [imageSrc, setImageSrc] = useState<string | null>(null);
  const [meta, setMeta] = useState<{ run: number; var: string; width?: number; height?: number; axes_bbox?: { left: number; top: number; width: number; height: number; png_width: number; png_height: number } } | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const dirInputRef = useRef<HTMLInputElement | null>(null);
  // Base paths from localStorage + selected index
  const [basePaths, setBasePaths] = useState<string[]>(() => {
    try { return JSON.parse(typeof window !== "undefined" ? localStorage.getItem("piv_base_paths") || "[]" : "[]"); } catch { return []; }
  });
  const [basePathIdx, setBasePathIdx] = useState<number>(0);
  // derive camera options from config if provided (same logic as Masking.tsx)
  const cameraOptions: string[] = (() => {
    const nFromPaths = config?.paths?.camera_numbers?.length ? Number(config.paths.camera_numbers[0]) : undefined;
    const nFromIm = config?.imProperties?.cameraCount ? Number(config.imProperties.cameraCount) : undefined;
    const n = (Number.isFinite(nFromPaths as number) && (nFromPaths as number) > 0)
      ? (nFromPaths as number)
      : (Number.isFinite(nFromIm as number) && (nFromIm as number) > 0) ? (nFromIm as number) : 1;
    const count = Number.isFinite(n) ? n : 1;
    return Array.from({ length: count }, (_, i) => `Cam${i + 1}`);
  })();

  // ensure camera state reflects available options
  const [camera, setCamera] = useState<string>(() => cameraOptions.length > 0 ? cameraOptions[0] : "Cam1");
  useEffect(() => {
    if (cameraOptions.length === 0) return;
    if (!cameraOptions.includes(camera)) setCamera(cameraOptions[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraOptions.length, cameraOptions[0]]);
  const [merged, setMerged] = useState<boolean>(false);
  const [playing, setPlaying] = useState(false);
  // pending value while dragging slider to avoid firing requests for every tick
  const [pendingIndex, setPendingIndex] = useState<number>(index);
  const [pointerDown, setPointerDown] = useState<boolean>(false);
  const commitTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const [limitsLoading, setLimitsLoading] = useState(false);

  // New: mean statistics mode
  const [meanMode, setMeanMode] = useState<boolean>(false);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState<string | null>(null);

  // New: stat-vars list from stats file
  const [statVars, setStatVars] = useState<string[] | null>(null);
  const [statVarsLoading, setStatVarsLoading] = useState(false);
  const [statVarsError, setStatVarsError] = useState<string | null>(null);

  // New: per-frame vars (from check_vars)
  const [frameVars, setFrameVars] = useState<string[] | null>(null);
  const [frameVarsLoading, setFrameVarsLoading] = useState(false);
  const [frameVarsError, setFrameVarsError] = useState<string | null>(null);

  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === "piv_base_paths") {
        try { setBasePaths(JSON.parse(e.newValue || "[]")); } catch {}
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  // Effective directory: prefer selected base path if available
  const effectiveDir = useMemo(() => {
    if (basePaths.length > 0 && basePathIdx >= 0 && basePathIdx < basePaths.length) {
      return basePaths[basePathIdx];
    }
    return directory;
  }, [basePaths, basePathIdx, directory]);

  // Auto-generate file paths from effective directory
  const matFile = useMemo(() => `${effectiveDir}/${String(index).padStart(5, "0")}.mat`, [effectiveDir, index]);
  const coordsFile = useMemo(() => `${effectiveDir}/coordinates.mat`, [effectiveDir]);

  // Folder browse (prefer Tauri; fallback to webkitdirectory)
  const handleBrowse = () => {
    try {
      const tauri = (window as any).__TAURI__;
      if (tauri?.dialog?.open) {
        tauri.dialog.open({ directory: true, multiple: false }).then((selected: any) => {
          if (typeof selected === "string") setDirectory(selected);
        }).catch(() => {});
        return;
      }
    } catch {
      // ignore and fallback
    }
    dirInputRef.current?.click();
  };

  const onDirPicked = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const anyFile: any = files[0];
    const rel: string = anyFile.webkitRelativePath || "";
    const root = rel.split("/")[0] || "";
    let folderPath = root;
    if (anyFile.path && rel) {
      const abs = String(anyFile.path);
      folderPath = abs.substring(0, abs.length - rel.length) + root;
    }
    setDirectory(folderPath);
    e.currentTarget.value = "";
  };

  const fetchImage = useCallback(async () => {
    setLoading(true);
    setError(null);
    // Do not clear imageSrc/meta until new image is loaded
    try {
      // Only send the selected base path, not the file path
      const basePath = effectiveDir;
      if (!basePath) throw new Error("Please provide a base path");
      const params = new URLSearchParams();
      params.set("base_path", basePath);
      params.set("frame", String(index));
      params.set("var", type);
      params.set("cmap", cmap);
      if (run && run > 0) params.set("run", String(run));
      if (lower.trim() !== "") params.set("lower_limit", String(Number(lower)));
      if (upper.trim() !== "") params.set("upper_limit", String(Number(upper)));
      params.set("camera", camera);
      params.set("merged", merged ? "1" : "0");
      const url = `${backendUrl}/plot/plot_vector?${params.toString()}`;
      const res = await fetch(url);
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Failed to fetch vector plot");
      setImageSrc(json.image ?? null);
      setMeta(json.meta ?? null);
      return true; // Indicate success
    } catch (e: any) {
      setError(e.message || "Unknown error");
      return false; // Indicate failure
    } finally {
      setLoading(false);
    }
  }, [effectiveDir, index, type, run, lower, upper, cmap, backendUrl, camera, merged]);

  // New: fetch list of variables available in the mean/stats file
  const fetchStatVars = useCallback(async () => {
    setStatVarsLoading(true);
    setStatVarsError(null);
    try {
      const basePath = effectiveDir;
      if (!basePath) throw new Error("Please provide a base path");
      const params = new URLSearchParams();
      params.set("base_path", basePath);
      params.set("camera", camera);
      params.set("merged", merged ? "1" : "0");
      const url = `${backendUrl}/plot/check_stat_vars?${params.toString()}`;
      const res = await fetch(url);
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || `Failed to fetch stat vars (${res.status})`);
      const vars = Array.isArray(json.vars) ? json.vars.map(String) : [];
      setStatVars(vars);
      // if current type not in vars, pick first available
      if (vars.length > 0) setType(prev => vars.includes(prev) ? prev : vars[0]);
    } catch (e: any) {
      setStatVarsError(e?.message ?? "Unknown error");
      setStatVars(null);
    } finally {
      setStatVarsLoading(false);
    }
  }, [effectiveDir, camera, merged, backendUrl]);

  // New: fetch variables available in a single frame file
  const fetchFrameVars = useCallback(async () => {
    setFrameVarsLoading(true);
    setFrameVarsError(null);
    try {
      const basePath = effectiveDir;
      if (!basePath) throw new Error("Please provide a base path");
      const params = new URLSearchParams();
      params.set("base_path", basePath);
      params.set("frame", String(index));
      params.set("camera", camera);
      params.set("merged", merged ? "1" : "0");
      const url = `${backendUrl}/plot/check_vars?${params.toString()}`;
      const res = await fetch(url);
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || `Failed to fetch frame vars (${res.status})`);
      const vars = Array.isArray(json.vars) ? json.vars.map(String) : [];
      setFrameVars(vars);
      if (vars.length > 0) setType(prev => (vars.includes(prev) ? prev : vars[0]));
    } catch (e: any) {
      setFrameVarsError(e?.message ?? "Unknown error");
      setFrameVars(null);
    } finally {
      setFrameVarsLoading(false);
    }
  }, [effectiveDir, index, camera, merged, backendUrl]);

  // New: fetch min/max limits for the selected variable from /plot/check_limits
  const fetchLimits = useCallback(async () => {
    setLimitsLoading(true);
    try {
      const basePath = effectiveDir;
      if (!basePath) throw new Error("Please provide a base path");
      const params = new URLSearchParams();
      params.set("base_path", basePath);
      params.set("camera", camera);
      params.set("merged", merged ? "1" : "0");
      params.set("var", type);
      const url = `${backendUrl}/plot/check_limits?${params.toString()}`;
      const res = await fetch(url);
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || `Failed to fetch limits (${res.status})`);
      // Expecting { min: number, max: number }
      if (typeof json.min === "number" && typeof json.max === "number") {
        setLower(String(json.min));
        setUpper(String(json.max));
      } else {
        console.warn("check_limits returned unexpected payload", json);
      }
    } catch (err) {
      console.error("Error fetching limits:", err);
      // keep existing lower/upper values on error
    } finally {
      setLimitsLoading(false);
    }
  }, [effectiveDir, camera, merged, type, backendUrl, setLower, setUpper]);

  // Toggle play: when starting playback, fetch limits first and then play
  const handlePlayToggle = () => {
    setPlaying(p => !p);
  };

  // New: fetch mean statistics image from /plot_stats (same params as fetchImage)
  const fetchStatsImage = useCallback(async () => {
    setStatsLoading(true);
    setStatsError(null);
    try {
      const basePath = effectiveDir;
      if (!basePath) throw new Error("Please provide a base path");
      const params = new URLSearchParams();
      params.set("base_path", basePath);
      params.set("frame", String(index)); // kept for compatibility, backend ignores frame for stats
      params.set("var", type);
      params.set("cmap", cmap);
      if (run && run > 0) params.set("run", String(run));
      if (lower.trim() !== "") params.set("lower_limit", String(Number(lower)));
      if (upper.trim() !== "") params.set("upper_limit", String(Number(upper)));
      params.set("camera", camera);
      params.set("merged", merged ? "1" : "0");

      const url = `${backendUrl}/plot/plot_stats?${params.toString()}`;
      const res = await fetch(url);
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || `Failed to fetch stats plot (${res.status})`);
      setImageSrc(json.image ?? null);
      setMeta(json.meta ?? null);
      setHasRendered(true);
    } catch (e: any) {
      setStatsError(e?.message ?? "Unknown error");
    } finally {
      setStatsLoading(false);
    }
  }, [effectiveDir, index, type, run, lower, upper, cmap, backendUrl, camera, merged]);

  // Render handler: when meanMode off, fetch frame vars first then image
  const handleRender = useCallback(async () => {
    setHasRendered(true);
    if (meanMode) {
      await fetchStatsImage();
      return;
    }
    // for frame plotting: fetch available variables for the frame first
    await fetchFrameVars();
    await fetchImage();
  }, [meanMode, fetchFrameVars, fetchImage, fetchStatsImage]);

  // New: fetch stat-vars first, then stats image
  const toggleMeanMode = async () => {
    const newVal = !meanMode;
    setMeanMode(newVal);
    if (newVal) {
      // clear any manual limits when switching to mean statistics
      setLower("");
      setUpper("");
      setStatsError(null);
      await fetchStatVars();
      await fetchStatsImage();
    } else {
      setStatsError(null);
      setStatVars(null);
      setStatVarsError(null);
    }
  };

  // Automatically render when index or other relevant parameters change
  useEffect(() => {
    if (!hasRendered || !(effectiveDir || basePaths.length > 0)) return;
    // When meanMode is active, always use stats endpoint; otherwise use normal plot
    if (meanMode) {
      void fetchStatsImage();
    } else {
      void fetchImage();
    }
  // include all relevant dependencies so changes to type/run/limits/camera/merged trigger correct endpoint
  }, [
    hasRendered,
    effectiveDir,
    index,
    type,
    run,
    lower,
    upper,
    cmap,
    camera,
    merged,
    basePathIdx,
    meanMode,
    fetchImage,
    fetchStatsImage,
  ]);

  // Play/pause effect
  useEffect(() => {
    if (playing) {
      playIntervalRef.current = setInterval(() => {
        setIndex(i => i + 1);
      }, 300);
    } else if (playIntervalRef.current) {
      clearInterval(playIntervalRef.current);
      playIntervalRef.current = null;
    }
    return () => {
      if (playIntervalRef.current) {
        clearInterval(playIntervalRef.current);
        playIntervalRef.current = null;
      }
    };
  }, [playing]);

  // Effect to pause playback if an image fails to load (e.g., end of sequence)
  useEffect(() => {
    if (error && playing) {
      setPlaying(false);
    }
  }, [error, playing]);

  // --- Tooltip / hover (value at cursor) additions ---
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [hoverData, setHoverData] = useState<{
    x: number; y: number; ux: number | null; uy: number | null; value: number | null;
    i: number; j: number; clientX: number; clientY: number;
  } | null>(null);
  const hoverDebounceRef = useRef<number | null>(null);
  const lastQueryRef = useRef<{ px: number; py: number; frame: number; varName: string; mean: boolean } | null>(null);
  const pendingFetchRef = useRef<boolean>(false);

  const clearHover = useCallback(() => {
    if (hoverDebounceRef.current) {
      window.clearTimeout(hoverDebounceRef.current);
      hoverDebounceRef.current = null;
    }
    pendingFetchRef.current = false;
    setHoverData(null);
  }, []);

  // Reset hover on image/meta change
  useEffect(() => { clearHover(); }, [imageSrc, meta, type, index, meanMode, camera, merged, clearHover]);

  const fetchValueAt = useCallback((xPercent: number, yPercent: number) => {
    if (pendingFetchRef.current) return;
    pendingFetchRef.current = true;
    const endpoint = meanMode ? "get_stats_value_at_position" : "get_vector_at_position";
    const params = new URLSearchParams();
    params.set("base_path", effectiveDir);
    params.set("camera", camera.replace(/[^\d]/g, "") || "1");
    params.set("frame", String(index));
    params.set("var", type);
    params.set("run", String(run));
    params.set("merged", merged ? "1" : "0");
    params.set("x_percent", xPercent.toString());
    params.set("y_percent", yPercent.toString());
    const url = `${backendUrl}/plot/${endpoint}?${params.toString()}`;
    fetch(url)
      .then(r => r.json().then(j => ({ ok: r.ok, json: j })))
      .then(({ ok, json }) => {
        pendingFetchRef.current = false;
        if (!ok || json.error) return;
        setHoverData(h => h ? { ...h, ...json } : null);
      })
      .catch(() => { pendingFetchRef.current = false; });
  }, [backendUrl, effectiveDir, camera, index, type, run, merged, meanMode]);

  // Restore: Tooltip only shows when mouse is inside axes_bbox region
  const onMouseMove = useCallback((e: React.MouseEvent) => {
    const bbox = meta?.axes_bbox;
    const imgEl = imgRef.current;
    if (!imgEl || !bbox) return;
    const rect = imgEl.getBoundingClientRect();
    const dispX = e.clientX - rect.left;
    const dispY = e.clientY - rect.top;
    if (dispX < 0 || dispY < 0 || dispX > rect.width || dispY > rect.height) {
      clearHover();
      return;
    }
    // Scale to PNG pixel coordinates
    const scaleX = bbox.png_width / rect.width;
    const scaleY = bbox.png_height / rect.height;
    const px = dispX * scaleX;
    const py = dispY * scaleY;
    // Is inside axes data region?
    if (px < bbox.left || px > bbox.left + bbox.width || py < bbox.top || py > bbox.top + bbox.height) {
      clearHover();
      return;
    }
    const xPercent = (px - bbox.left) / bbox.width;
    const yPercent = (py - bbox.top) / bbox.height;
    setHoverData({
      x: NaN, y: NaN, ux: null, uy: null, value: null,
      i: -1, j: -1, clientX: e.clientX, clientY: e.clientY
    });
    // Throttle calls
    const last = lastQueryRef.current;
    const keyChanged = !last ||
      last.px !== Math.round(px) ||
      last.py !== Math.round(py) ||
      last.frame !== index ||
      last.varName !== type ||
      last.mean !== meanMode;
    if (!keyChanged) return;
    lastQueryRef.current = { px: Math.round(px), py: Math.round(py), frame: index, varName: type, mean: meanMode };
    if (hoverDebounceRef.current) window.clearTimeout(hoverDebounceRef.current);
    hoverDebounceRef.current = window.setTimeout(() => {
      fetchValueAt(xPercent, yPercent);
    }, 120);
  }, [meta, fetchValueAt, index, type, meanMode, clearHover]);

  const onMouseLeave = useCallback(() => { clearHover(); }, [clearHover]);
  // --- End tooltip additions ---

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4 mb-4">
            <div className="flex flex-col gap-4 mb-4">
              {/* Base path selection */}
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium">Base Path:</label>
                {basePaths.length > 0 ? (
                  <>
                    <select
                      value={String(basePathIdx)}
                      onChange={e => setBasePathIdx(Number(e.target.value))}
                      className="border rounded px-2 py-1"
                    >
                      {basePaths.map((p, i) => {
                        // Show last two segments of the path
                        const norm = p.replace(/\\/g, "/").replace(/\/+$/, "");
                        const parts = norm.split("/").filter(Boolean);
                        const lastTwo = parts.length >= 2 ? parts.slice(-2).join("/") : norm;
                        return <option key={i} value={i}>{`${i}: /${lastTwo}`}</option>;
                      })}
                    </select>
                  </>
                ) : (
                  <>
                    <Input
                      type="text"
                      value={directory}
                      onChange={e => setDirectory(e.target.value)}
                      placeholder="Select directory"
                      className="w-full"
                    />
                    {/* Hidden directory input for web fallback */}
                    {/* @ts-ignore */}
                    <input
                      type="file"
                      style={{ display: "none" }}
                      ref={dirInputRef}
                      onChange={onDirPicked}
                      multiple
                      // @ts-ignore
                      webkitdirectory="true"
                      // @ts-ignore
                      directory="true"
                    />
                    <Button variant="outline" onClick={handleBrowse}>
                      Browse
                    </Button>
                  </>
                )}
              </div>

              {/* Camera and merged controls */}
              <div className="flex items-center gap-4">
                <label htmlFor="camera" className="text-sm font-medium">Camera:</label>
                <select
                  id="camera"
                  value={camera}
                  onChange={e => setCamera(e.target.value)}
                  className="border rounded px-2 py-1"
                >
                  {cameraOptions.map(c => (
                    <option key={c} value={c}>{c}</option>
                  ))}
                </select>

                {/* Merged Data and Mean Statistics (checkbox) */}
                <label className="flex items-center gap-2 text-sm font-medium">
                  <input
                    type="checkbox"
                    checked={merged}
                    onChange={e => setMerged(e.target.checked)}
                    className="accent-soton-blue w-4 h-4 rounded border-gray-300"
                  />
                  Merged Data
                </label>

                {/* Mean statistics checkbox placed beside Merged Data */}
                <label className="flex items-center gap-2 text-sm font-medium">
                  <input
                    type="checkbox"
                    checked={meanMode}
                    onChange={() => { void toggleMeanMode(); }}
                    className="accent-soton-blue w-4 h-4 rounded border-gray-300"
                  />
                  Mean Statistics
                  {statVarsLoading && <span className="ml-2 text-xs text-gray-500">Loading vars...</span>}
                  {meanMode && statsLoading && <span className="ml-2 text-xs text-gray-500">Computing...</span>}
                </label>
              </div>

              <div className={`flex items-center gap-2 transition-opacity duration-200 ${meanMode ? "opacity-40 pointer-events-none" : ""}`}>
                {/* File Index controls - faded/disabled when meanMode is active */}
                <label htmlFor="index" className="text-sm font-medium">File Index:</label>
                <Button size="sm" variant="outline" onClick={() => setIndex(i => Math.max(1, i - 1))}>-</Button>
                <Input id="index" type="number" min={1} value={index} onChange={e => setIndex(Math.max(1, Number(e.target.value)))} className="w-24" />
                <Button size="sm" variant="outline" onClick={() => setIndex(i => i + 1)}>+</Button>
              </div>

              <div className={`flex items-center gap-4 mb-4 transition-opacity duration-200 ${meanMode ? "opacity-40 pointer-events-none" : ""}`}>
                {/* Frame slider - faded/disabled when meanMode is active */}
                <label htmlFor="frame-slider" className="text-sm font-medium">Frame:</label>
                <input
                  id="frame-slider"
                  type="range"
                  min={1}
                  max={9999} // Use a high, arbitrary max
                  value={index}
                  onChange={e => setIndex(Number(e.target.value))}
                  className="w-64"
                />
                <span className="text-xs text-gray-500">{index}</span>
                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    variant={playing ? "default" : "outline"}
                    onClick={() => { handlePlayToggle(); }}
                    className="flex items-center gap-1"
                  >
                    {playing ? <span>&#10073;&#10073; Pause</span> : <span>&#9654; Play</span>}
                  </Button>
                </div>
              </div>

              <div className="flex items-center gap-3 flex-wrap">
                <label htmlFor="type" className="text-sm font-medium">Type:</label>
                <select id="type" value={type} onChange={e => setType(e.target.value)} className="border rounded px-2 py-1">
                  {meanMode ? (
                    statVarsLoading ? (
                      <option>Loading...</option>
                    ) : statVars && statVars.length > 0 ? (
                      statVars.map(v => <option key={v} value={v}>{v}</option>)
                    ) : (
                      <option disabled>No vars</option>
                    )
                  ) : (
                    frameVarsLoading ? (
                      <option>Loading...</option>
                    ) : frameVars && frameVars.length > 0 ? (
                      frameVars.map(v => <option key={v} value={v}>{v}</option>)
                    ) : (
                      <>
                        <option value="ux">ux</option>
                        <option value="uy">uy</option>
                      </>
                    )
                  )}
                </select>
                <label htmlFor="cmap" className="text-sm font-medium">Colormap:</label>
                <select id="cmap" value={cmap} onChange={e => setCmap(e.target.value)} className="border rounded px-2 py-1">
                  <option value="default">default</option>
                  <option value="viridis">viridis</option>
                  <option value="plasma">plasma</option>
                  <option value="inferno">inferno</option>
                  <option value="magma">magma</option>
                  <option value="cividis">cividis</option>
                  <option value="jet">jet</option>
                  <option value="gray">gray</option>
                  <option value="bone">bone</option>
                  <option value="copper">copper</option>
                  <option value="pink">pink</option>
                  <option value="spring">spring</option>
                  <option value="summer">summer</option>
                  <option value="autumn">autumn</option>
                  <option value="winter">winter</option>
                  <option value="hot">hot</option>
                  <option value="cool">cool</option>
                  <option value="Wistia">Wistia</option>
                  <option value="twilight">twilight</option>
                  <option value="hsv">hsv</option>
                </select>
                <label htmlFor="run" className="text-sm font-medium">Run:</label>
                <Input id="run" type="number" min={1} value={run} onChange={e => setRun(Math.max(1, Number(e.target.value)))} className="w-24" />
                <label className="text-sm font-medium">Lower:</label>
                <Input type="number" value={lower} onChange={e => setLower(e.target.value)} placeholder="auto" className="w-28" />
                <label className="text-sm font-medium">Upper:</label>
                <Input type="number" value={upper} onChange={e => setUpper(e.target.value)} placeholder="auto" className="w-28" />
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => { void fetchLimits(); }}
                  disabled={limitsLoading || meanMode} // disabled when mean statistics is active
                  className="ml-2"
                >
                  {limitsLoading ? "Getting..." : "Get Limits"}
                </Button>

                {/* Render button: uses stats endpoint when meanMode active */}
                <Button
                  className="bg-soton-blue"
                  onClick={() => { void handleRender(); }}
                  disabled={loading || statsLoading || frameVarsLoading}
                >
                  {(loading || statsLoading || frameVarsLoading) ? "Loading..." : "Render"}
                </Button>
              </div>

              {statsError && meanMode && (
                <div className="w-full p-3 mb-3 rounded border border-red-200 bg-red-50 text-red-700 text-sm">
                  {statsError}
                </div>
              )}

              {/* Note removed as requested */}
            </div>
          </div>

          {/* Slider and play button above image */}
          <div className="flex items-center gap-4 mb-4">
            {/* <label htmlFor="frame-slider" className="text-sm font-medium">Frame:</label>
            <input
              id="frame-slider"
              type="range"
              min={1}
              max={999}
              value={index}
              onChange={e => setIndex(Number(e.target.value))}
              className="w-64"
            />
            <span className="text-xs text-gray-500">{index}</span>
            <Button
              size="sm"
              variant={playing ? "default" : "outline"}
              onClick={() => setPlaying(p => !p)}
              className="flex items-center gap-1"
            >
              {playing ? (
                <span>&#10073;&#10073; Pause</span>
              ) : (
                <span>&#9654; Play</span>
              )}
            </Button> */}
          </div>

          {/* Image viewer placeholder, similar to ImagePairViewer */}
          <div className="mt-6">
            {error && (
              <div className="w-full p-3 mb-3 rounded border border-red-200 bg-red-50 text-red-700 text-sm">{error}</div>
            )}
            {imageSrc && !error && (
              <div
                className="flex flex-col items-center relative"
                onMouseMove={onMouseMove}
                onMouseLeave={onMouseLeave}
              >
                <img
                  ref={imgRef}
                  src={`data:image/png;base64,${imageSrc}`}
                  alt="Vector Result"
                  className="rounded border w-full max-w-3xl select-none pointer-events-auto"
                  draggable={false}
                />
                {/* Only show rendering overlay if not playing */}
                {loading && !playing && (
                  <div className="absolute inset-0 flex items-center justify-center bg-white bg-opacity-60">
                    <span className="text-gray-500">Rendering...</span>
                  </div>
                )}
                {meta && (
                  <div className="text-xs text-gray-500 mt-2">
                    Run: {meta.run} • Var: {meta.var}{meta.width && meta.height ? ` • ${meta.width}×${meta.height}` : ""}
                  </div>
                )}
                {/* Tooltip (only real tooltip, no debug/test) */}
                {hoverData && (
                  <div
                    className="pointer-events-none fixed z-50 px-2 py-1 text-xs rounded bg-black text-white shadow"
                    style={{ top: hoverData.clientY + 12, left: hoverData.clientX + 12 }}
                  >
                    {isNaN(hoverData.x) || hoverData.i < 0
                      ? "Loading..."
                      : (() => {
                          // Determine value for the currently requested variable
                          let varVal: number | null = null;
                          if (type === "ux" && hoverData.ux != null) varVal = hoverData.ux;
                          else if (type === "uy" && hoverData.uy != null) varVal = hoverData.uy;
                          else if (hoverData.value != null) varVal = hoverData.value;
                          return (
                            <>
                              <div>x: {hoverData.x.toFixed(3)}, y: {hoverData.y.toFixed(3)}</div>
                              {varVal != null && <div>{type}: {varVal.toFixed(3)}</div>}
                            </>
                          );
                        })()
                    }
                  </div>
                )}
              </div>
            )}
            {(!imageSrc || error) && (
              <div className="w-full h-64 flex items-center justify-center bg-gray-100 border rounded">
                <span className="text-gray-500">No image loaded</span>
              </div>
            )}
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
