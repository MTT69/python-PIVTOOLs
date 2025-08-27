"use client";
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

  // New: datum setting mode and offset values
  const [datumMode, setDatumMode] = useState<boolean>(false);
  const [xOffset, setXOffset] = useState<string>("0");
  const [yOffset, setYOffset] = useState<string>("0");
  const [cornerCoordinates, setCornerCoordinates] = useState<{
    topLeft: {x: number, y: number},
    topRight: {x: number, y: number},
    bottomLeft: {x: number, y: number},
    bottomRight: {x: number, y: number}
  } | null>(null);
  const [showCorners, setShowCorners] = useState<boolean>(false);

  // --- POD mode state ---
  const [podMode, setPodMode] = useState<boolean>(false);
  const [podRun, setPodRun] = useState<number>(1);
  const [podModeIndex, setPodModeIndex] = useState<number>(1);
  const [podMaxMode, setPodMaxMode] = useState<number>(1);
  const [podStacked, setPodStacked] = useState<boolean>(true);
  const [podComponent, setPodComponent] = useState<"ux" | "uy">("ux");
  const [podAlgorithm, setPodAlgorithm] = useState<string>("exact");
  const [podAvailable, setPodAvailable] = useState<{ randomised: boolean; exact: boolean }>({ randomised: false, exact: false });
  const [podEnergy, setPodEnergy] = useState<any>(null);
  const [podEnergyLoading, setPodEnergyLoading] = useState(false);
  const [podEnergyError, setPodEnergyError] = useState<string | null>(null);

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

  // New: handle setting datum when image is clicked in datum mode
  const handleImageClick = async (e: React.MouseEvent) => {
    if (!datumMode || !meta?.axes_bbox || !imgRef.current) return;

    const rect = imgRef.current.getBoundingClientRect();
    const dispX = e.clientX - rect.left;
    const dispY = e.clientY - rect.top;

    // Convert click position to image coordinates
    const bbox = meta.axes_bbox;
    const scaleX = bbox.png_width / rect.width;
    const scaleY = bbox.png_height / rect.height;
    const px = dispX * scaleX;
    const py = dispY * scaleY;

    // Check if click is inside the axes region
    if (px < bbox.left || px > bbox.left + bbox.width ||
        py < bbox.top || py > bbox.top + bbox.height) {
      return;
    }

    // Calculate normalized position (0-1) within the vector field
    const xPercent = (px - bbox.left) / bbox.width;
    const yPercent = (py - bbox.top) / bbox.height;

    // Query backend for physical coordinates at this point (like tooltip)
    try {
      const params = new URLSearchParams();
      params.set("base_path", effectiveDir);
      params.set("camera", camera.replace(/[^\d]/g, "") || "1");
      params.set("frame", String(index));
      params.set("var", type);
      params.set("run", String(run));
      params.set("merged", merged ? "1" : "0");
      params.set("x_percent", xPercent.toString());
      params.set("y_percent", yPercent.toString());
      const url = `${backendUrl}/plot/get_vector_at_position?${params.toString()}`;
      const res = await fetch(url);
      const json = await res.json();
      if (!res.ok || json.error) throw new Error(json.error || "Failed to get vector value");
      // Show the physical coordinates to the user
      alert(`New datum set at physical position: x=${json.x?.toFixed(4)}, y=${json.y?.toFixed(4)}`);
      // Send to backend and reload plot after
      await sendDatumToBackend(json.x, json.y);
      // Reload plot after datum is set
      void fetchImage();
    } catch (e: any) {
      setError(`Failed to set datum: ${e.message}`);
    }

    // Exit datum mode after setting
    setDatumMode(false);
  };

  // Function to update offsets only (no datum set)
  const updateOffsets = async () => {
    try {
      const body = {
        base_path_idx: basePathIdx,
        camera: camera.replace(/[^\d]/g, "") || "1",
        run: run,
        x_offset: Number(xOffset) || 0,
        y_offset: Number(yOffset) || 0,
        merged: merged ? 1 : 0,
      };
      const url = `${backendUrl}/calibration/set_datum`;
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Failed to update offsets");
      // Optionally show confirmation
      // alert("Offsets updated!");
      // Reload plot after updating offsets
      void fetchImage();
    } catch (e: any) {
      setError(`Failed to update offsets: ${e.message}`);
    }
  };

  // Function to send datum information to backend (now uses /set_datum)
  const sendDatumToBackend = async (x: number, y: number) => {
    try {
      const body = {
        base_path_idx: basePathIdx,
        camera: camera.replace(/[^\d]/g, "") || "1",
        run: run,
        x: x,
        y: y,
        x_offset: Number(xOffset) || 0,
        y_offset: Number(yOffset) || 0,
        merged: merged ? 1 : 0,
      };
      const url = `${backendUrl}/calibration/set_datum`;
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Failed to set datum");
      // Optionally show confirmation
      // alert("Datum successfully set!");
    } catch (e: any) {
      setError(`Failed to set datum: ${e.message}`);
    }
  };

  // Function to fetch corner coordinates from backend
  const fetchCornerCoordinates = async () => {
    setShowCorners(false);
    setCornerCoordinates(null);
    setError(null);
    try {
      const basePath = effectiveDir;
      if (!basePath) throw new Error("Please provide a base path");
      // Helper to fetch a single corner
      const fetchCorner = async (xPercent: number, yPercent: number) => {
        const params = new URLSearchParams();
        params.set("base_path", basePath);
        params.set("camera", camera.replace(/[^\d]/g, "") || "1");
        params.set("frame", String(index));
        params.set("var", type);
        params.set("run", String(run));
        params.set("merged", merged ? "1" : "0");
        params.set("x_percent", xPercent.toString());
        params.set("y_percent", yPercent.toString());
        const url = `${backendUrl}/plot/get_vector_at_position?${params.toString()}`;
        const res = await fetch(url);
        const json = await res.json();
        if (!res.ok || json.error) throw new Error(json.error || "Failed to get corner coordinate");
        // Expect json.x and json.y
        return { x: json.x, y: json.y };
      };
      // Four corners: (0,0), (1,0), (0,1), (1,1)
      const [topLeft, topRight, bottomLeft, bottomRight] = await Promise.all([
        fetchCorner(0, 0),
        fetchCorner(1, 0),
        fetchCorner(0, 1),
        fetchCorner(1, 1),
      ]);
      setCornerCoordinates({ topLeft, topRight, bottomLeft, bottomRight });
      setShowCorners(true);
    } catch (e: any) {
      setError(`Failed to get corner coordinates: ${e.message}`);
    }
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
      // Add offset values if provided
      if (xOffset.trim() !== "") params.set("x_offset", xOffset);
      if (yOffset.trim() !== "") params.set("y_offset", yOffset);

      const url = `${backendUrl}/plot/plot_vector?${params.toString()}`;
      const res = await fetch(url);
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Failed to fetch vector plot");
      setImageSrc(json.image ?? null);
      setMeta(json.meta ?? null);
      // If backend returned meta.run, update local run state to keep UI in sync
      if (json.meta && json.meta.run != null) {
        const parsed = Number(json.meta.run);
        if (Number.isFinite(parsed) && parsed > 0) setRun(parsed);
      }
      // If backend returned a var name, update the type selector as well
      if (json.meta && json.meta.var != null && typeof json.meta.var === "string") {
        if (json.meta.var !== type) setType(json.meta.var);
      }
      return true; // Indicate success
    } catch (e: any) {
      setError(e.message || "Unknown error");
      return false; // Indicate failure
    } finally {
      setLoading(false);
    }
  }, [effectiveDir, index, type, run, lower, upper, cmap, backendUrl, camera, merged, xOffset, yOffset]);

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
      if (json.meta && json.meta.run != null) {
        const parsed = Number(json.meta.run);
        if (Number.isFinite(parsed) && parsed > 0) setRun(parsed);
      }
      if (json.meta && json.meta.var != null && typeof json.meta.var === "string") {
        if (json.meta.var !== type) setType(json.meta.var);
      }
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

  // --- POD mode state and handlers ---
  const fetchPodImage = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      // Compose params for backend
      const basePath = effectiveDir;
      if (!basePath) throw new Error("Please provide a base path");
      const params = new URLSearchParams();
      params.set("base_path", basePath);
      params.set("camera", camera.replace(/[^\d]/g, "") || "1");
      params.set("run", String(podRun || run));
      params.set("mode", String(podModeIndex));
      params.set("component", podComponent);
      params.set("algorithm", podAlgorithm);
      params.set("cmap", cmap);
      if (lower.trim() !== "") params.set("lower_limit", String(Number(lower)));
      if (upper.trim() !== "") params.set("upper_limit", String(Number(upper)));
      params.set("merged", merged ? "1" : "0");

      const url = `${backendUrl}/plot_pod_mode?${params.toString()}`;
      const res = await fetch(url);
      const json = await res.json();
      if (!res.ok || json.error) throw new Error(json.error || "Failed to fetch POD mode image");
      setImageSrc(json.image_base64 ?? null);
      setMeta(json.meta ?? null);
      return true;
    } catch (e: any) {
      setError(e.message || "Unknown error");
      setImageSrc(null);
      setMeta(null);
      return false;
    } finally {
      setLoading(false);
    }
  }, [
    effectiveDir,
    camera,
    podRun,
    run,
    podModeIndex,
    podComponent,
    podAlgorithm,
    cmap,
    lower,
    upper,
    merged,
    backendUrl,
  ]);

  // Add togglePodMode
  const togglePodMode = useCallback(async () => {
    setPodMode(v => !v);
    // Optionally: fetch POD data when enabling
    // TODO: Add logic if needed
  }, []);

  // Add handlePodRender
  const handlePodRender = useCallback(async () => {
    await fetchPodImage();
  }, [fetchPodImage]);

  // Automatically render when index or other relevant parameters change
  useEffect(() => {
    if (!hasRendered || !(effectiveDir || basePaths.length > 0)) return;
    // When POD mode is active, use POD endpoint; when meanMode is active, use stats endpoint; otherwise use normal plot
    if (podMode) {
      void fetchPodImage();
    } else if (meanMode) {
      void fetchStatsImage();
    } else {
      void fetchImage();
    }
  // include all relevant dependencies
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
    podMode,
    fetchImage,
    fetchStatsImage,
    fetchPodImage,
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

  // --- Magnifier additions ---
  const magnifierRef = useRef<HTMLCanvasElement | null>(null);
  const [magVisible, setMagVisible] = useState(false);
  const [magPos, setMagPos] = useState({ left: 0, top: 0 });
  const [magnifierEnabled, setMagnifierEnabled] = useState(false);
  const MAG_SIZE = 180;
  const MAG_FACTOR = 2.5;
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;

  const handleMagnifierMove = (e: React.MouseEvent) => {
    if (!magnifierEnabled) {
      setMagVisible(false);
      return;
    }
    const img = imgRef.current;
    const mag = magnifierRef.current;
    if (!img || !mag) return;
    const rect = img.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    // Show only if inside image
    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
      setMagVisible(false);
      return;
    }
    setMagVisible(true);
    // Position magnifier near cursor, but not off image
    let left = x + 24;
    let top = y + 24;
    if (left + MAG_SIZE > rect.width) left = x - MAG_SIZE - 24;
    if (top + MAG_SIZE > rect.height) top = y - MAG_SIZE - 24;
    setMagPos({ left, top });
    // Draw zoomed region
    const ctx = mag.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, MAG_SIZE * dpr, MAG_SIZE * dpr);
    ctx.save();
    ctx.beginPath();
    ctx.arc((MAG_SIZE * dpr) / 2, (MAG_SIZE * dpr) / 2, (MAG_SIZE * dpr) / 2, 0, Math.PI * 2);
    ctx.clip();
    // Draw zoomed image
    const sx = (x * img.naturalWidth / rect.width) - (MAG_SIZE / (2 * MAG_FACTOR));
    const sy = (y * img.naturalHeight / rect.height) - (MAG_SIZE / (2 * MAG_FACTOR));
    ctx.drawImage(
      img,
      sx, sy,
      MAG_SIZE / MAG_FACTOR, MAG_SIZE / MAG_FACTOR,
      0, 0,
      MAG_SIZE * dpr, MAG_SIZE * dpr
    );
    // Draw crosshairs
    const cx = (MAG_SIZE * dpr) / 2;
    const cy = (MAG_SIZE * dpr) / 2;
    const lineLen = MAG_SIZE * dpr * 0.4;
    ctx.save();
    ctx.beginPath();
    ctx.lineWidth = Math.max(1, dpr);
    ctx.strokeStyle = 'rgba(0,0,0,0.7)';
    ctx.moveTo(cx - lineLen, cy);
    ctx.lineTo(cx + lineLen, cy);
    ctx.moveTo(cx, cy - lineLen);
    ctx.lineTo(cx, cy + lineLen);
    ctx.stroke();
    ctx.beginPath();
    ctx.lineWidth = Math.max(1, Math.ceil(dpr / 1.5));
    ctx.strokeStyle = 'rgba(255,255,255,0.9)';
    ctx.moveTo(cx - lineLen, cy);
    ctx.lineTo(cx + lineLen, cy);
    ctx.moveTo(cx, cy - lineLen);
    ctx.lineTo(cx, cy + lineLen);
    ctx.stroke();
    ctx.restore();
    ctx.restore();
    // Draw border
    ctx.beginPath();
    ctx.arc(cx, cy, (MAG_SIZE * dpr) / 2 - 2 * dpr, 0, Math.PI * 2);
    ctx.lineWidth = 2 * dpr;
    ctx.strokeStyle = '#333';
    ctx.stroke();
  };

  const handleMagnifierLeave = () => {
    setMagVisible(false);
  };

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

  // Determine max frame count from backend config only
  const [maxFrameCount, setMaxFrameCount] = useState<number>(9999);

  // Fetch config from backend to get accurate image count
  useEffect(() => {
    async function fetchConfig() {
      try {
        const res = await fetch(`${backendUrl}/config`);
        if (!res.ok) return;

        const json = await res.json();

        // Extract image count directly from backend config
        const backendNumImages = json.images?.num_images;

        // Only update if a valid number of images is returned
        if (Number.isFinite(backendNumImages) && backendNumImages > 0) {
          setMaxFrameCount(backendNumImages);
        }
      } catch (err) {
        console.error("Error fetching config for frame count:", err);
        // Keep existing value on error
      }
    }

    fetchConfig();
  }, [backendUrl]);

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Results</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4 mb-4">
            <div className="flex flex-col gap-4 mb-4">
              {/* Internal: last response meta is available in the compact status below */}
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
                    disabled={podMode} // Disable when POD mode is active
                  />
                  Mean Statistics
                  {statVarsLoading && <span className="ml-2 text-xs text-gray-500">Loading vars...</span>}
                  {meanMode && statsLoading && <span className="ml-2 text-xs text-gray-500">Computing...</span>}
                </label>

                {/* New: POD Mode toggle */}
                <label className="flex items-center gap-2 text-sm font-medium">
                  <input
                    type="checkbox"
                    checked={podMode}
                    onChange={() => { void togglePodMode(); }}
                    className="accent-soton-blue w-4 h-4 rounded border-gray-300"
                    disabled={meanMode} // Disable when mean mode is active
                  />
                  POD Mode
                </label>
              </div>

              {/* New: POD Mode Controls (show only when podMode is active) */}
              {podMode && (
                <div className="p-3 bg-gray-50 border rounded-md">
                  <h4 className="font-medium mb-2">POD Mode Settings</h4>
                  <div className="flex flex-wrap gap-3 items-center">
                    <label className="text-sm font-medium">Algorithm:</label>
                    <select
                      value={podAlgorithm}
                      onChange={e => setPodAlgorithm(e.target.value)}
                      className="border rounded px-2 py-1"
                    >
                      <option value="exact" disabled={!podAvailable.exact}>Exact</option>
                      <option value="randomised" disabled={!podAvailable.randomised}>Randomised</option>
                    </select>

                    <label className="text-sm font-medium">Run:</label>
                    <Input
                      type="number"
                      min={1}
                      value={podRun}
                      onChange={e => setPodRun(Math.max(1, Number(e.target.value)))}
                      className="w-20"
                    />

                    <label className="text-sm font-medium">Mode:</label>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setPodModeIndex(i => Math.max(1, i - 1))}
                      disabled={podModeIndex <= 1}
                      className={`transition-opacity ${podModeIndex <= 1 ? 'opacity-40' : 'opacity-100'}`}
                    >
                      -
                    </Button>
                    <Input
                      type="number"
                      min={1}
                      max={podMaxMode}
                      value={podModeIndex}
                      onChange={e => setPodModeIndex(Math.max(1, Math.min(podMaxMode, Number(e.target.value))))}
                      className="w-20"
                    />
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => setPodModeIndex(i => Math.min(podMaxMode, i + 1))}
                      disabled={podModeIndex >= podMaxMode}
                      className={`transition-opacity ${podModeIndex >= podMaxMode ? 'opacity-40' : 'opacity-100'}`}
                    >
                      +
                    </Button>
                    <span className="text-xs text-gray-500">{podModeIndex} / {podMaxMode}</span>

                    <label className="text-sm font-medium">Component:</label>
                    <select
                      value={podComponent}
                      onChange={e => setPodComponent(e.target.value as "ux" | "uy")}
                      className="border rounded px-2 py-1"
                    >
                      <option value="ux">ux</option>
                      <option value="uy">uy</option>
                    </select>

                    <Button
                      className="bg-soton-blue ml-auto"
                      onClick={handlePodRender}
                      disabled={loading}
                    >
                      {loading ? "Loading..." : "Render POD Mode"}
                    </Button>
                  </div>

                  {/* POD Energy Plot */}
                  {podEnergy && !podEnergyError && (
                    <PODModalEnergyPlot
                      energy={podEnergy}
                      selectedMode={podModeIndex}
                      onSelectMode={setPodModeIndex}
                      component={podComponent}
                      onComponentChange={setPodComponent}
                    />
                  )}

                  {podEnergyError && (
                    <div className="mt-2 p-2 bg-red-50 text-red-700 text-sm rounded">
                      {podEnergyError}
                    </div>
                  )}
                </div>
              )}

              {/* New: X and Y Offset inputs */}
              <div className="flex items-center gap-4">
                <label htmlFor="x-offset" className="text-sm font-medium">X Offset:</label>
                <Input
                  id="x-offset"
                  type="number"
                  value={xOffset}
                  onChange={e => setXOffset(e.target.value)}
                  className="w-24"
                  placeholder="0"
                />

                <label htmlFor="y-offset" className="text-sm font-medium">Y Offset:</label>
                <Input
                  id="y-offset"
                  type="number"
                  value={yOffset}
                  onChange={e => setYOffset(e.target.value)}
                  className="w-24"
                  placeholder="0"
                />

                {/* Update Offsets Button */}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={updateOffsets}
                >
                  Update Offsets
                </Button>

                {/* New: Set datum button */}
                <Button
                  size="sm"
                  variant={datumMode ? "default" : "outline"}
                  onClick={() => setDatumMode(!datumMode)}
                  className={`${datumMode ? "bg-yellow-500 hover:bg-yellow-600" : ""}`}
                >
                  {datumMode ? "Cancel Set Datum" : "Set New Datum"}
                </Button>

                {/* New: Show corner coordinates button */}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => fetchCornerCoordinates()}
                >
                  Show Corner Coordinates
                </Button>
              </div>

              {/* Display corner coordinates when available */}
              {showCorners && cornerCoordinates && (
                <div className="p-3 bg-gray-50 border rounded-md">
                  <h4 className="font-medium mb-2">Vector Field Corner Coordinates:</h4>
                  <div className="grid grid-cols-2 gap-2 text-sm">
                    <div className="flex gap-1">
                      <span className="font-medium">Top Left:</span>
                      <span>({cornerCoordinates.topLeft.x.toFixed(2)}, {cornerCoordinates.topLeft.y.toFixed(2)})</span>
                    </div>
                    <div className="flex gap-1">
                      <span className="font-medium">Top Right:</span>
                      <span>({cornerCoordinates.topRight.x.toFixed(2)}, {cornerCoordinates.topRight.y.toFixed(2)})</span>
                    </div>
                    <div className="flex gap-1">
                      <span className="font-medium">Bottom Left:</span>
                      <span>({cornerCoordinates.bottomLeft.x.toFixed(2)}, {cornerCoordinates.bottomLeft.y.toFixed(2)})</span>
                    </div>
                    <div className="flex gap-1">
                      <span className="font-medium">Bottom Right:</span>
                      <span>({cornerCoordinates.bottomRight.x.toFixed(2)}, {cornerCoordinates.bottomRight.y.toFixed(2)})</span>
                    </div>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => setShowCorners(false)}
                    className="mt-2"
                  >
                    Hide
                  </Button>
                </div>
              )}

              <div className={`flex items-center gap-2 transition-opacity duration-200 ${meanMode || podMode ? "opacity-40 pointer-events-none" : ""}`}>
                {/* File Index controls - faded/disabled when meanMode or podMode is active */}
                <label htmlFor="index" className="text-sm font-medium">File Index:</label>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setIndex(i => Math.max(1, i - 1))}
                  disabled={index <= 1}
                  className={`transition-opacity ${index <= 1 ? 'opacity-40' : 'opacity-100'}`}
                >
                  -
                </Button>
                <Input id="index" type="number" min={1} value={index} onChange={e => setIndex(Math.max(1, Number(e.target.value)))} className="w-24" />
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setIndex(i => i + 1)}
                  disabled={index >= maxFrameCount}
                  className={`transition-opacity ${index >= maxFrameCount ? 'opacity-40' : 'opacity-100'}`}
                >
                  +
                </Button>
              </div>

              <div className={`flex items-center gap-4 mb-4 transition-opacity duration-200 ${meanMode || podMode ? "opacity-40 pointer-events-none" : ""}`}>
                {/* Frame slider - faded/disabled when meanMode or podMode is active */}
                <label htmlFor="frame-slider" className="text-sm font-medium">Frame:</label>
                <input
                  id="frame-slider"
                  type="range"
                  min={1}
                  max={maxFrameCount}
                  value={index}
                  onChange={e => setIndex(Number(e.target.value))}
                  className="w-64"
                />
                <span className="text-xs text-gray-500">{index} / {maxFrameCount}</span>
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

          {/* Image viewer placeholder, similar to ImagePairViewer */}
          <div className="mt-6">
            {error && (
              <div className="w-full p-3 mb-3 rounded border border-red-200 bg-red-50 text-red-700 text-sm">{error}</div>
            )}
            {datumMode && (
              <div className="w-full p-3 mb-3 rounded border border-yellow-200 bg-yellow-50 text-yellow-700 text-sm">
                <strong>Set Datum Mode Active:</strong> Click on the image to set a new coordinate system origin.
              </div>
            )}
            {imageSrc && !error && (
              <div
                className="flex flex-col items-center relative"
                style={{
                  width: '100%',
                  maxWidth: '1100px',
                  margin: '0 auto',
                  cursor: datumMode ? 'crosshair' : 'default'
                }}
                onMouseMove={e => { onMouseMove(e); handleMagnifierMove(e); }}
                onMouseLeave={e => { onMouseLeave(); handleMagnifierLeave(); }}
                onClick={e => { if (datumMode) handleImageClick(e); }}
              >
                {/* Magnifier toggle - styled button with emoji, like Masking */}
                <div style={{ position: 'absolute', top: 8, right: 8, zIndex: 70 }}>
                  <button
                    type="button"
                    onClick={() => setMagnifierEnabled(v => !v)}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 6,
                      background: magnifierEnabled ? '#005fa3' : 'rgba(255,255,255,0.92)',
                      color: magnifierEnabled ? '#fff' : '#222',
                      border: magnifierEnabled ? '2px solid #005fa3' : '2px solid #bbb',
                      borderRadius: 8,
                      padding: '4px 14px 4px 10px',
                      fontSize: 16,
                      fontWeight: 500,
                      boxShadow: magnifierEnabled ? '0 2px 8px rgba(0,95,163,0.10)' : '0 1px 4px rgba(0,0,0,0.07)',
                      cursor: 'pointer',
                      transition: 'all 0.15s',
                      outline: magnifierEnabled ? '2px solid #005fa3' : 'none',
                    }}
                    aria-pressed={magnifierEnabled}
                  >
                    <span style={{ fontSize: 20, marginRight: 6 }}>{magnifierEnabled ? '🔎' : '🔍'}</span>
                    <span style={{ fontSize: 14, fontWeight: 500 }}>{magnifierEnabled ? 'Magnifier On' : 'Magnifier'}</span>
                  </button>
                </div>
                <img
                  ref={imgRef}
                  src={`data:image/png;base64,${imageSrc}`}
                  alt="Vector Result"
                  className="rounded border w-full max-w-5xl select-none pointer-events-auto"
                  style={{ width: '100%', maxWidth: '1000px', height: 'auto', display: 'block' }}
                  draggable={false}
                />
                {/* Magnifier Canvas */}
                <canvas
                  ref={magnifierRef}
                  style={{
                    display: magVisible ? 'block' : 'none',
                    position: 'absolute',
                    pointerEvents: 'none',
                    zIndex: 61,
                    width: MAG_SIZE,
                    height: MAG_SIZE,
                    left: magPos.left,
                    top: magPos.top,
                    borderRadius: '50%',
                    boxShadow: '0 2px 8px rgba(0,0,0,0.25)',
                    border: '2px solid #333',
                  }}
                  width={MAG_SIZE * dpr}
                  height={MAG_SIZE * dpr}
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
                {/* Tooltip (always visible, above magnifier) */}
                {hoverData && (
                  <div
                    className="pointer-events-none fixed px-2 py-1 text-xs rounded bg-black text-white shadow"
                    style={{ top: hoverData.clientY + 12, left: hoverData.clientX + 12, zIndex: 100 }}
                  >
                    {isNaN(hoverData.x) || hoverData.i < 0
                      ? "Loading..."
                      : (() => {
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

// --- PODModalEnergyPlot component ---
function PODModalEnergyPlot({
  energy,
  selectedMode,
  onSelectMode,
  component,
  onComponentChange,
}: {
  energy: any;
  selectedMode: number;
  onSelectMode: (k: number) => void;
  component: "ux" | "uy";
  onComponentChange: (c: "ux" | "uy") => void;
}) {
  // ...existing code for energy plot...
  let ef: number[] = [];
  let ec: number[] = [];
  if (energy.stacked) {
    ef = energy.energy_fraction || [];
    ec = energy.energy_cumulative || [];
  } else if (component === "ux") {
    ef = energy.energy_fraction_ux || [];
    ec = energy.energy_cumulative_ux || [];
  } else {
    ef = energy.energy_fraction_uy || [];
    ec = energy.energy_cumulative_uy || [];
  }
  const width = 340, height = 120, margin = 32;
  const barW = Math.max(6, (width - 2 * margin) / Math.max(1, ef.length));
  return (
    <div className="my-4">
      <div className="flex items-center gap-4 mb-2">
        <span className="font-medium text-sm">Cumulative Energy Breakdown</span>
        {!energy.stacked && (
          <select value={component} onChange={e => onComponentChange(e.target.value as "ux" | "uy")} className="border rounded px-2 py-1 text-xs">
            <option value="ux">ux</option>
            <option value="uy">uy</option>
          </select>
        )}
      </div>
      <svg width={width} height={height} style={{ background: "#f9fafb", borderRadius: 8 }}>
        {ef.map((v, i) => (
          <rect
            key={i}
            x={margin + i * barW}
            y={height - margin - v * (height - 2 * margin)}
            width={barW * 0.7}
            height={v * (height - 2 * margin)}
            fill="#005fa3"
            opacity={0.5}
          />
        ))}
        <polyline
          fill="none"
          stroke="#e67e22"
          strokeWidth={2}
          points={ec.map((v, i) => {
            const x = margin + i * barW + barW * 0.35;
            const y = height - margin - v * (height - 2 * margin);
            return `${x},${y}`;
          }).join(" ")}
        />
        <line x1={margin} y1={height - margin} x2={width - margin} y2={height - margin} stroke="#bbb" />
        <line x1={margin} y1={margin} x2={margin} y2={height - margin} stroke="#bbb" />
        {[0, 0.25, 0.5, 0.75, 1].map((v, idx) => {
          const y = height - margin - v * (height - 2 * margin);
          return (
            <g key={idx}>
              <line x1={margin - 4} y1={y} x2={margin} y2={y} stroke="#888" />
              <text x={margin - 8} y={y + 4} fontSize={10} textAnchor="end" fill="#444">{Math.round(v * 100)}%</text>
            </g>
          );
        })}
        {ef.map((_, i) => {
          const x = margin + i * barW + barW * 0.35;
          return (
            <g key={i}>
              <line x1={x} y1={height - margin} x2={x} y2={height - margin + 4} stroke="#888" />
              {(i + 1) % Math.ceil(ef.length / 10) === 0 || i === 0 || i === ef.length - 1 ? (
                <text x={x} y={height - margin + 16} fontSize={10} textAnchor="middle" fill="#444">{i + 1}</text>
              ) : null}
            </g>
          );
        })}
        <text x={width / 2} y={height - 4} fontSize={12} textAnchor="middle" fill="#333">Mode</text>
        <text x={margin - 24} y={margin - 8} fontSize={12} textAnchor="start" fill="#333" transform={`rotate(-90,${margin - 24},${margin - 8})`}>Energy</text>
      </svg>
      <div className="text-xs text-gray-500 mt-1">
        <span>Mode 1: {(ef[0] * 100).toFixed(1)}% | Mode {ef.length}: {(ec[ef.length - 1] * 100).toFixed(1)}% total</span>
      </div>
    </div>
  );
}
