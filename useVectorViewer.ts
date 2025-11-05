import { useRef, useState, useEffect, useCallback, useMemo } from "react";

interface UseVectorViewerProps {
  backendUrl: string;
  config?: any;
}

interface HoverData {
  x: number;
  y: number;
  ux: number | null;
  uy: number | null;
  value: number | null;
  i: number;
  j: number;
  clientX: number;
  clientY: number;
}

interface CornerCoordinates {
  topLeft: { x: number; y: number };
  topRight: { x: number; y: number };
  bottomLeft: { x: number; y: number };
  bottomRight: { x: number; y: number };
}

export const useVectorViewer = ({ backendUrl, config }: UseVectorViewerProps) => {
  // State variables
  const [basePaths, setBasePaths] = useState<string[]>(() => config?.paths?.base_paths || []);
  const [basePathIdx, setBasePathIdx] = useState<number>(0);
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
  const [hasRendered, setHasRendered] = useState<boolean>(false);
  const cameraOptions: number[] = useMemo(() => {
    return config?.paths?.camera_numbers || [];
  }, [config]);
  const [camera, setCamera] = useState<number>(() => cameraOptions.length > 0 ? cameraOptions[0] : 1);
  useEffect(() => {
    if (cameraOptions.length === 0) return;
    if (!cameraOptions.includes(camera)) setCamera(cameraOptions[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraOptions.length, cameraOptions[0]]);
  const [merged, setMerged] = useState<boolean>(false);
  const [playing, setPlaying] = useState(false);
  const [pendingIndex, setPendingIndex] = useState<number>(index);
  const [pointerDown, setPointerDown] = useState<boolean>(false);
  const commitTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const playIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const [limitsLoading, setLimitsLoading] = useState(false);
  const [meanMode, setMeanMode] = useState<boolean>(false);
  const [statsLoading, setStatsLoading] = useState(false);
  const [statsError, setStatsError] = useState<string | null>(null);
  const [statVars, setStatVars] = useState<string[] | null>(null);
  const [statVarsLoading, setStatVarsLoading] = useState(false);
  const [statVarsError, setStatVarsError] = useState<string | null>(null);
  const [frameVars, setFrameVars] = useState<string[] | null>(null);
  const [frameVarsLoading, setFrameVarsLoading] = useState(false);
  const [frameVarsError, setFrameVarsError] = useState<string | null>(null);
  const [datumMode, setDatumMode] = useState<boolean>(false);
  const [xOffset, setXOffset] = useState<string>("0");
  const [yOffset, setYOffset] = useState<string>("0");
  const [cornerCoordinates, setCornerCoordinates] = useState<CornerCoordinates | null>(null);
  const [showCorners, setShowCorners] = useState<boolean>(false);
  const effectiveDir = useMemo(() => {
    if (basePaths.length > 0 && basePathIdx >= 0 && basePathIdx < basePaths.length) {
      return basePaths[basePathIdx];
    }
    return "";
  }, [basePaths, basePathIdx]);
  const matFile = useMemo(() => `${effectiveDir}/${String(index).padStart(5, "0")}.mat`, [effectiveDir, index]);
  const coordsFile = useMemo(() => `${effectiveDir}/coordinates.mat`, [effectiveDir]);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const [hoverData, setHoverData] = useState<HoverData | null>(null);
  const hoverDebounceRef = useRef<number | null>(null);
  const lastQueryRef = useRef<{ px: number; py: number; frame: number; varName: string; mean: boolean } | null>(null);
  const pendingFetchRef = useRef<boolean>(false);
  const magnifierRef = useRef<HTMLCanvasElement | null>(null);
  const [magVisible, setMagVisible] = useState(false);
  const [magPos, setMagPos] = useState({ left: 0, top: 0 });
  const MAG_SIZE = 180;
  const MAG_FACTOR = 2.5;
  const dpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  const [maxFrameCount, setMaxFrameCount] = useState<number>(9999);
  const [appliedTransforms, setAppliedTransforms] = useState<string[]>([]);

  // Functions
  const fetchImage = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
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
      params.set("camera", String(camera));
      params.set("merged", merged ? "1" : "0");
      if (xOffset.trim() !== "") params.set("x_offset", xOffset);
      if (yOffset.trim() !== "") params.set("y_offset", yOffset);
      
      const url = `${backendUrl}/plot/plot_vector?${params.toString()}`;
      const res = await fetch(url);
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || "Failed to fetch vector plot");
      setImageSrc(json.image ?? null);
      setMeta(json.meta ?? null);
      if (json.meta && json.meta.run != null) {
        const parsed = Number(json.meta.run);
        if (Number.isFinite(parsed) && parsed > 0) setRun(parsed);
      }
      if (json.meta && json.meta.var != null && typeof json.meta.var === "string") {
        if (json.meta.var !== type) setType(json.meta.var);
      }
      return true;
    } catch (e: any) {
      setError(e.message || "Unknown error");
      return false;
    } finally {
      setLoading(false);
    }
  }, [effectiveDir, index, type, run, lower, upper, cmap, backendUrl, camera, merged, xOffset, yOffset]);

  const fetchStatVars = useCallback(async () => {
    setStatVarsLoading(true);
    setStatVarsError(null);
    try {
      const basePath = effectiveDir;
      if (!basePath) throw new Error("Please provide a base path");
      const params = new URLSearchParams();
      params.set("base_path", basePath);
      params.set("camera", String(camera));
      params.set("merged", merged ? "1" : "0");
      const url = `${backendUrl}/plot/check_stat_vars?${params.toString()}`;
      const res = await fetch(url);
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || `Failed to fetch stat vars (${res.status})`);
      const vars = Array.isArray(json.vars) ? json.vars.map(String) : [];
      setStatVars(vars);
      if (vars.length > 0) setType(prev => vars.includes(prev) ? prev : vars[0]);
    } catch (e: any) {
      setStatVarsError(e?.message ?? "Unknown error");
      setStatVars(null);
    } finally {
      setStatVarsLoading(false);
    }
  }, [effectiveDir, camera, merged, backendUrl]);

  const fetchFrameVars = useCallback(async () => {
    setFrameVarsLoading(true);
    setFrameVarsError(null);
    try {
      const basePath = effectiveDir;
      if (!basePath) throw new Error("Please provide a base path");
      const params = new URLSearchParams();
      params.set("base_path", basePath);
      params.set("frame", String(index));
      params.set("camera", String(camera));
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

  const fetchLimits = useCallback(async () => {
    setLimitsLoading(true);
    try {
      const basePath = effectiveDir;
      if (!basePath) throw new Error("Please provide a base path");
      const params = new URLSearchParams();
      params.set("base_path", basePath);
      params.set("camera", String(camera));
      params.set("merged", merged ? "1" : "0");
      params.set("var", type);
      const url = `${backendUrl}/plot/check_limits?${params.toString()}`;
      const res = await fetch(url);
      const json = await res.json();
      if (!res.ok) throw new Error(json.error || `Failed to fetch limits (${res.status})`);
      if (typeof json.min === "number" && typeof json.max === "number") {
        setLower(String(json.min));
        setUpper(String(json.max));
      } else {
        console.warn("check_limits returned unexpected payload", json);
      }
    } catch (err) {
      console.error("Error fetching limits:", err);
    } finally {
      setLimitsLoading(false);
    }
  }, [effectiveDir, camera, merged, type, backendUrl, setLower, setUpper]);

  const fetchAvailableRuns = useCallback(async () => {
    try {
      const basePath = effectiveDir;
      if (!basePath) return [];
      const params = new URLSearchParams();
      params.set("base_path", basePath);
      params.set("camera", String(camera));
      params.set("merged", merged ? "1" : "0");
      const url = `${backendUrl}/plot/check_runs?${params.toString()}`;
      const res = await fetch(url);
      const json = await res.json();
      if (!res.ok) return [];
      const runs = Array.isArray(json.runs) ? json.runs.map(Number).filter((n: number) => Number.isFinite(n) && n > 0) : [];
      return runs;
    } catch (e) {
      return [];
    }
  }, [effectiveDir, camera, merged, backendUrl]);

  const handlePlayToggle = () => {
    setPlaying(p => !p);
  };

  const fetchStatsImage = useCallback(async () => {
    setStatsLoading(true);
    setStatsError(null);
    try {
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
      params.set("camera", String(camera));
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

  const handleRender = useCallback(async () => {
    setHasRendered(true);
    if (meanMode) {
      await fetchStatsImage();
      return;
    }
    await fetchFrameVars();
    await fetchImage();
  }, [meanMode, fetchFrameVars, fetchImage, fetchStatsImage]);

  const toggleMeanMode = async () => {
    const newVal = !meanMode;
    setMeanMode(newVal);
    if (newVal) {
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

  const handleImageClick = async (e: React.MouseEvent) => {
    if (!datumMode || !meta?.axes_bbox || !imgRef.current) return;
    const rect = imgRef.current.getBoundingClientRect();
    const dispX = e.clientX - rect.left;
    const dispY = e.clientY - rect.top;
    const bbox = meta.axes_bbox;
    const scaleX = bbox.png_width / rect.width;
    const scaleY = bbox.png_height / rect.height;
    const px = dispX * scaleX;
    const py = dispY * scaleY;
    if (px < bbox.left || px > bbox.left + bbox.width ||
        py < bbox.top || py > bbox.top + bbox.height) {
      return;
    }
    const xPercent = (px - bbox.left) / bbox.width;
    const yPercent = (py - bbox.top) / bbox.height;
    try {
      const params = new URLSearchParams();
      params.set("base_path", effectiveDir);
      params.set("camera", String(camera));
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
      alert(`New datum set at physical position: x=${json.x?.toFixed(4)}, y=${json.y?.toFixed(4)}`);
      await sendDatumToBackend(json.x, json.y);
      void fetchImage();
    } catch (e: any) {
      setError(`Failed to set datum: ${e.message}`);
    }
    setDatumMode(false);
  };

  const updateOffsets = async () => {
    try {
      const body = {
        base_path_idx: basePathIdx,
        camera: camera,
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
      void fetchImage();
    } catch (e: any) {
      setError(`Failed to update offsets: ${e.message}`);
    }
  };

  const sendDatumToBackend = async (x: number, y: number) => {
    try {
      const body = {
        base_path_idx: basePathIdx,
        camera: camera,
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
    } catch (e: any) {
      setError(`Failed to set datum: ${e.message}`);
    }
  };

  const fetchCornerCoordinates = async () => {
    setShowCorners(false);
    setCornerCoordinates(null);
    setError(null);
    try {
      const basePath = effectiveDir;
      if (!basePath) throw new Error("Please provide a base path");
      const fetchCorner = async (xPercent: number, yPercent: number) => {
        const params = new URLSearchParams();
        params.set("base_path", basePath);
        params.set("camera", String(camera));
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
        return { x: json.x, y: json.y };
      };
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

  const clearHover = useCallback(() => {
    if (hoverDebounceRef.current) {
      window.clearTimeout(hoverDebounceRef.current);
      hoverDebounceRef.current = null;
    }
    pendingFetchRef.current = false;
    setHoverData(null);
  }, []);

  useEffect(() => { clearHover(); }, [imageSrc, meta, type, index, meanMode, camera, merged, clearHover]);

  const fetchValueAt = useCallback((xPercent: number, yPercent: number) => {
    if (pendingFetchRef.current) return;
    pendingFetchRef.current = true;
    const endpoint = meanMode ? "get_stats_value_at_position" : "get_vector_at_position";
    const params = new URLSearchParams();
    params.set("base_path", effectiveDir);
    params.set("camera", String(camera));
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
    const scaleX = bbox.png_width / rect.width;
    const scaleY = bbox.png_height / rect.height;
    const px = dispX * scaleX;
    const py = dispY * scaleY;
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

  const handleMagnifierMove = (e: React.MouseEvent) => {
    const img = imgRef.current;
    const mag = magnifierRef.current;
    if (!img || !mag) {
      setMagVisible(false);
      return;
    }
    const rect = img.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const y = e.clientY - rect.top;
    if (x < 0 || y < 0 || x > rect.width || y > rect.height) {
      setMagVisible(false);
      return;
    }
    setMagVisible(true);
    // Position magnifier so its center is exactly at the cursor position (using clientX/Y for fixed positioning)
    const left = e.clientX - (MAG_SIZE / 2);
    const top = e.clientY - (MAG_SIZE / 2);
    setMagPos({ left, top });
    const ctx = mag.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, MAG_SIZE * dpr, MAG_SIZE * dpr);
    ctx.save();
    ctx.beginPath();
    ctx.arc((MAG_SIZE * dpr) / 2, (MAG_SIZE * dpr) / 2, (MAG_SIZE * dpr) / 2, 0, Math.PI * 2);
    ctx.clip();
    // Calculate source position - sample from the exact pixel under cursor
    const srcCenterX = (x / rect.width) * img.naturalWidth;
    const srcCenterY = (y / rect.height) * img.naturalHeight;
    const srcSize = MAG_SIZE / MAG_FACTOR;
    const sx = srcCenterX - (srcSize / 2);
    const sy = srcCenterY - (srcSize / 2);
    ctx.drawImage(
      img,
      sx, sy,
      srcSize, srcSize,
      0, 0,
      MAG_SIZE * dpr, MAG_SIZE * dpr
    );
    const cx = (MAG_SIZE * dpr) / 2;
    const cy = (MAG_SIZE * dpr) / 2;
    const lineLen = MAG_SIZE * dpr * 0.3;
    ctx.save();
    ctx.beginPath();
    ctx.lineWidth = Math.max(2, dpr * 1.5);
    ctx.strokeStyle = 'rgba(0,0,0,0.8)';
    ctx.moveTo(cx - lineLen, cy);
    ctx.lineTo(cx + lineLen, cy);
    ctx.moveTo(cx, cy - lineLen);
    ctx.lineTo(cx, cy + lineLen);
    ctx.stroke();
    ctx.beginPath();
    ctx.lineWidth = Math.max(1, dpr);
    ctx.strokeStyle = 'rgba(255,255,255,0.95)';
    ctx.moveTo(cx - lineLen, cy);
    ctx.lineTo(cx + lineLen, cy);
    ctx.moveTo(cx, cy - lineLen);
    ctx.lineTo(cx, cy + lineLen);
    ctx.stroke();
    ctx.restore();
    ctx.restore();
    ctx.beginPath();
    ctx.arc(cx, cy, (MAG_SIZE * dpr) / 2 - 2 * dpr, 0, Math.PI * 2);
    ctx.lineWidth = 3 * dpr;
    ctx.strokeStyle = '#005fa3';
    ctx.stroke();
  };

  const handleMagnifierLeave = () => {
    setMagVisible(false);
  };

  useEffect(() => {
    async function fetchConfig() {
      try {
        const res = await fetch(`${backendUrl}/config`);
        if (!res.ok) return;
        const json = await res.json();
        const backendNumImages = json.images?.num_images;
        if (Number.isFinite(backendNumImages) && backendNumImages > 0) {
          setMaxFrameCount(backendNumImages);
        }
      } catch (err) {
        console.error("Error fetching config for frame count:", err);
      }
    }
    fetchConfig();
  }, [backendUrl]);

  useEffect(() => {
    if (!effectiveDir || meanMode) return;
    fetchAvailableRuns().then(runs => {
      if (runs.length > 0) {
        const maxRun = Math.max(...runs);
        setRun(maxRun);
        setHasRendered(true);
      }
    }).catch(() => {});
  }, [effectiveDir, meanMode, fetchAvailableRuns]);

  useEffect(() => {
    if (!hasRendered || !(effectiveDir || basePaths.length > 0)) return;
    if (meanMode) {
      void fetchStatsImage();
    } else {
      void fetchImage();
    }
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

  useEffect(() => {
    if (error && playing) {
      setPlaying(false);
    }
  }, [error, playing]);

  const basename = (p: string) => {
    if (!p) return "";
    const parts = p.replace(/\\/g, "/").split("/");
    return parts.filter(Boolean).pop() || p;
  };

  const base64ToBlob = useCallback((base64: string, mime = "image/png") => {
    const byteChars = atob(base64);
    const byteArrays: BlobPart[] = [];
    for (let offset = 0; offset < byteChars.length; offset += 512) {
      const slice = byteChars.slice(offset, offset + 512);
      const byteNumbers = new Array(slice.length);
      for (let i = 0; i < slice.length; i++) byteNumbers[i] = slice.charCodeAt(i);
      byteArrays.push(new Uint8Array(byteNumbers));
    }
    return new Blob(byteArrays, { type: mime });
  }, []);

  const downloadCurrentView = useCallback(() => {
    if (!imageSrc) return;
    const fileName = `vector_${meanMode ? "mean" : `frame_${index}`}_${type}.png`;
    const link = document.createElement("a");
    link.href = `data:image/png;base64,${imageSrc}`;
    link.download = fileName;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }, [imageSrc, meanMode, index, type]);

  const copyCurrentView = useCallback(async () => {
    if (!imageSrc) return;
    try {
      const blob = base64ToBlob(imageSrc, "image/png");
      const ClipboardItemCtor: any = (window as any).ClipboardItem;
      if (navigator.clipboard && "write" in navigator.clipboard && ClipboardItemCtor) {
        await navigator.clipboard.write([new ClipboardItemCtor({ "image/png": blob })]);
      } else {
        const url = URL.createObjectURL(blob);
        window.open(url, "_blank", "noopener,noreferrer");
        setTimeout(() => URL.revokeObjectURL(url), 10_000);
      }
    } catch (e: any) {
      setError(`Failed to copy image: ${e?.message ?? "Unknown error"}`);
    }
  }, [imageSrc, base64ToBlob]);

  const applyTransformation = useCallback(async (transformation: string) => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${backendUrl}/plot/transform_frame`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          base_path: effectiveDir,
          camera: camera,
          frame: index,
          transformation,
          merged: merged,
          type_name: "instantaneous", // or get from somewhere
        }),
      });
      const result = await response.json();
      if (result.success) {
        setAppliedTransforms(prev => [...prev, transformation]);
        // Reload the image
        await handleRender();
      } else {
        setError(result.error || "Transformation failed");
      }
    } catch (e: any) {
      setError(`Transformation failed: ${e?.message ?? "Unknown error"}`);
    } finally {
      setLoading(false);
    }
  }, [backendUrl, effectiveDir, camera, index, merged, handleRender]);

  const clearTransforms = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await fetch(`${backendUrl}/plot/clear_transform`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          base_path: effectiveDir,
          camera: camera,
          frame: index,
          merged: merged,
          type_name: "instantaneous",
        }),
      });
      const result = await response.json();
      if (result.success) {
        setAppliedTransforms([]);
        // Reload the image
        await handleRender();
      } else {
        setError(result.error || "Clear transforms failed");
      }
    } catch (e: any) {
      setError(`Clear transforms failed: ${e?.message ?? "Unknown error"}`);
    } finally {
      setLoading(false);
    }
  }, [backendUrl, effectiveDir, camera, index, merged, handleRender]);

  const [transformationJob, setTransformationJob] = useState<{
    job_id: string;
    status: string;
    progress: number;
    processed_frames: number;
    total_frames: number;
    elapsed_time?: number;
    estimated_remaining?: number;
    error?: string;
  } | null>(null);

  const applyTransformationToAllFrames = useCallback(async (transformations: string[]) => {
    try {
      setLoading(true);
      setError(null);
      setTransformationJob(null);

      const response = await fetch(`${backendUrl}/plot/transform_all_frames`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          base_path: effectiveDir,
          camera: camera,
          transformations,
          merged: merged,
          type_name: "instantaneous",
          image_count: maxFrameCount,
        }),
      });
      const result = await response.json();
      if (result.job_id) {
        setTransformationJob({
          job_id: result.job_id,
          status: result.status,
          progress: 0,
          processed_frames: 0,
          total_frames: result.total_frames,
        });
        // Start polling for status
        pollTransformationStatus(result.job_id);
      } else {
        setError(result.error || "Failed to start transformation job");
      }
    } catch (e: any) {
      setError(`Failed to start transformation: ${e?.message ?? "Unknown error"}`);
    } finally {
      setLoading(false);
    }
  }, [backendUrl, effectiveDir, camera, merged, maxFrameCount]);

  const pollTransformationStatus = useCallback(async (jobId: string) => {
    try {
      const response = await fetch(`${backendUrl}/plot/transform_all_frames/status/${jobId}`);
      const status = await response.json();

      setTransformationJob(status);

      if (status.status === "running" || status.status === "starting") {
        // Continue polling
        setTimeout(() => pollTransformationStatus(jobId), 1000);
      } else if (status.status === "completed") {
        // Reload current frame to show changes
        await handleRender();
      }
    } catch (e: any) {
      setError(`Failed to check transformation status: ${e?.message ?? "Unknown error"}`);
    }
  }, [backendUrl, handleRender]);

  return {
    basePaths,
    basePathIdx,
    setBasePathIdx,
    index,
    setIndex,
    type,
    setType,
    run,
    setRun,
    lower,
    setLower,
    upper,
    setUpper,
    cmap,
    setCmap,
    imageSrc,
    meta,
    loading,
    error,
    cameraOptions,
    camera,
    setCamera,
    merged,
    setMerged,
    playing,
    setPlaying,
    limitsLoading,
    meanMode,
    setMeanMode,
    statsLoading,
    statsError,
    statVars,
    statVarsLoading,
    frameVars,
    frameVarsLoading,
    datumMode,
    setDatumMode,
    xOffset,
    setXOffset,
    yOffset,
    setYOffset,
    cornerCoordinates,
    showCorners,
    setShowCorners,
    imgRef,
    hoverData,
    magnifierRef,
    magVisible,
    magPos,
    maxFrameCount,
    MAG_SIZE,
    dpr,
    handlePlayToggle,
    handleRender,
    toggleMeanMode,
    handleImageClick,
    updateOffsets,
    fetchCornerCoordinates,
    fetchLimits,
    onMouseMove,
    onMouseLeave,
    handleMagnifierMove,
    handleMagnifierLeave,
    basename,
    downloadCurrentView,
    copyCurrentView,
    applyTransformation,
    applyTransformationToAllFrames,
    transformationJob,
    appliedTransforms,
    setAppliedTransforms,
    clearTransforms,
    effectiveDir,  // Added: Export for component use
  };
};
