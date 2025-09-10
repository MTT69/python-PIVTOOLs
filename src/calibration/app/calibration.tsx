"use client";
import React, { useState, useEffect } from "react";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import StereoCalibration from "@/components/setup/StereoCalibration";

type CalibrationMethod = "scale_factor" | "pinhole" | "stereo";

interface Config {
  images: { num_images?: number };
  paths: { camera_numbers?: number[] };
  calibration?: {
    active?: CalibrationMethod;
    scale_factor?: any;
    pinhole?: any;
    stereo?: any;
    [key: string]: any;
  };
}

function useConfig(): [Config, (path: string[], value: any) => void] {
  const [config, setConfig] = useState<Config>({ images: {}, paths: {} });
  const debounceTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);
  const pendingUpdate = React.useRef<{ path: string[]; value: any } | null>(null);

  useEffect(() => {
    fetch("/backend/config")
      .then(r => r.json())
      .then(setConfig)
      .catch(() => {});
  }, []);

  function updateConfig(path: string[], value: any) {
    // Only update if value is different
    let current = config;
    for (let i = 0; i < path.length; i++) {
      current = current?.[path[i]];
    }
    if (deepEqual(current, value)) return; // Prevent unnecessary updates
    pendingUpdate.current = { path, value };
    if (debounceTimer.current) clearTimeout(debounceTimer.current);
    debounceTimer.current = setTimeout(() => {
      if (pendingUpdate.current) {
        fetch("/backend/update_config", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(
            pendingUpdate.current.path.length === 1
              ? { [pendingUpdate.current.path[0]]: pendingUpdate.current.value }
              : { [pendingUpdate.current.path[0]]: { [pendingUpdate.current.path[1]]: pendingUpdate.current.value } }
          ),
        }).then(() => {
          fetch("/backend/config")
            .then(r => r.json())
            .then(setConfig)
            .catch(() => {});
        });
        pendingUpdate.current = null;
      }
    }, 500);
  }

  return [config, updateConfig];
}

// Helper to get camera options from config (same as Masking/VectorViewer)
function getCameraOptions(config: any): number[] {
  const camNums = config?.paths?.camera_numbers;
  const imCount = config?.imProperties?.cameraCount;
  let count = 1;
  if (Array.isArray(camNums) && camNums.length > 0) {
    // If array, use max value or length
    const maxCam = Math.max(...camNums.map(Number));
    count = Math.max(camNums.length, maxCam);
  } else if (typeof camNums === "number" && camNums > 0) {
    count = camNums;
  } else if (typeof imCount === "number" && imCount > 0) {
    count = imCount;
  }
  return Array.from({ length: count }, (_, i) => i + 1);
}

// Custom hook to check calibration status
const useCalibrationStatus = (sourcePathIdx: number, camera: number) => {
  const [status, setStatus] = useState<string>("not_started");
  const [details, setDetails] = useState<any>(null);

  useEffect(() => {
    let active = true;
    const fetchStatus = async () => {
      try {
        const res = await fetch(`/backend/calibration/status?source_path_idx=${sourcePathIdx}&camera=${camera}`);
        const data = await res.json();
        if (active) {
          setStatus(data.status || "not_started");
          setDetails(data);
        }
      } catch {
        if (active) setStatus("not_started");
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [sourcePathIdx, camera]);
  return { status, details };
};

// Vector calibration job status hook
const useVectorCalibrationStatus = (jobId: string | null) => {
  const [status, setStatus] = useState<string>("not_started");
  const [details, setDetails] = useState<any>(null);

  useEffect(() => {
    if (!jobId) return;
    let active = true;
    const fetchStatus = async () => {
      try {
        const res = await fetch(`/backend/calibration/vectors/status/${jobId}`);
        const data = await res.json();
        if (active) {
          setStatus(data.status || "not_started");
          setDetails(data);
        }
      } catch {
        if (active) setStatus("not_started");
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [jobId]);
  return { status, details };
};

// Scale Factor status hook
const useScaleFactorStatus = (sourcePathIdx: number, camera: number) => {
  const [status, setStatus] = useState<string>("not_started");
  useEffect(() => {
    let active = true;
    const fetchStatus = async () => {
      try {
        const res = await fetch(`/backend/calibration/status?source_path_idx=${sourcePathIdx}&camera=${camera}&type=scale_factor`);
        const data = await res.json();
        if (active) setStatus(data.status || "not_started");
      } catch {
        if (active) setStatus("not_started");
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, [sourcePathIdx, camera]);
  return status;
};

// Add hook for stereo vectors status
const useStereoVectorStatus = (sourcePathIdx: number, cam1: number) => {
  const [status, setStatus] = React.useState<string>("not_started");
  useEffect(() => {
    let active = true;
    const fetchStatus = async () => {
      try {
        const res = await fetch(`/backend/calibration/status?source_path_idx=${sourcePathIdx}&camera=${cam1}&type=stereo_vectors`);
        const data = await res.json();
        if (active) setStatus(data.status || 'not_started');
      } catch {
        if (active) setStatus('not_started');
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 2000);
    return () => { active = false; clearInterval(interval); };
  }, [sourcePathIdx, cam1]);
  return status;
};

// Helper to show just the last segment of a path
const basename = (p: string) => {
  if (!p) return "";
  const parts = p.replace(/\\/g, "/").split("/");
  return parts.filter(Boolean).pop() || p;
};

// Helper: deep equality check for config updates
function deepEqual(a: any, b: any): boolean {
  if (a === b) return true;
  if (typeof a !== typeof b) return false;
  if (typeof a !== "object" || a === null || b === null) return false;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) {
      if (!deepEqual(a[i], b[i])) return false;
    }
    return true;
  }
  const keysA = Object.keys(a);
  const keysB = Object.keys(b);
  if (keysA.length !== keysB.length) return false;
  for (const k of keysA) {
    if (!deepEqual(a[k], b[k])) return false;
  }
  return true;
}

// Helper to load source paths from localStorage
function useSourcePaths() {
  const [sourcePaths, setSourcePaths] = useState<string[]>(() => {
    try {
      return JSON.parse(typeof window !== "undefined" ? localStorage.getItem("piv_source_paths") || "[]" : "[]");
    } catch {
      return [];
    }
  });
  useEffect(() => {
    const onStorage = (e: StorageEvent) => {
      if (e.key === "piv_source_paths") {
        try { setSourcePaths(JSON.parse(e.newValue || "[]")); } catch {}
      }
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);
  return [sourcePaths, setSourcePaths] as const;
}

// --- Scale Factor Calibration UI ---
const ScaleFactorCalibration: React.FC<{ config: Config; updateConfig: (path: string[], value: any) => void; setActive: () => void; isActive: boolean }> = ({ config, updateConfig, setActive, isActive }) => {
  // Use unified camera logic
  const cameraOptions = getCameraOptions(config);
  const numCameras = cameraOptions.length;
  const calibrationBlock = config.calibration ?? {};
  const calib = calibrationBlock.scale_factor ?? {};
  // Use string state for all number inputs
  const [dt, setDt] = useState<string>(calib.dt !== undefined ? String(calib.dt) : "");
  const [pxPerMm, setPxPerMm] = useState<string>(calib.px_per_mm !== undefined ? String(calib.px_per_mm) : "");
  const [xOffsets, setXOffsets] = useState<string[]>(Array.isArray(calib.x_offset) ? calib.x_offset.map(String) : Array(numCameras).fill("0"));
  const [yOffsets, setYOffsets] = useState<string[]>(Array.isArray(calib.y_offset) ? calib.y_offset.map(String) : Array(numCameras).fill("0"));
  const [sourcePaths] = useSourcePaths();
  const [sourcePathIdx, setSourcePathIdx] = useState<number>(0);
  const [camera, setCamera] = useState(1);
  const [calibrating, setCalibrating] = useState(false);
  const [scaleFactorJobId, setScaleFactorJobId] = useState<string | null>(null);
  const status = useScaleFactorStatus(sourcePathIdx, camera);

  // Add scale factor job status hook
  const useScaleFactorJobStatus = (jobId: string | null) => {
    const [status, setStatus] = useState<string>("not_started");
    const [details, setDetails] = useState<any>(null);

    useEffect(() => {
      if (!jobId) return;
      let active = true;
      const fetchStatus = async () => {
        try {
          const res = await fetch(`/backend/calibration/scale_factor/status/${jobId}`);
          const data = await res.json();
          if (active) {
            setStatus(data.status || "not_started");
            setDetails(data);
          }
        } catch {
          if (active) setStatus("not_started");
        }
      };
      fetchStatus();
      const interval = setInterval(fetchStatus, 2000);
      return () => {
        active = false;
        clearInterval(interval);
      };
    }, [jobId]);
    return { status, details };
  };

  const { status: scaleFactorJobStatus, details: scaleFactorJobDetails } = useScaleFactorJobStatus(scaleFactorJobId);

  useEffect(() => {
    setDt(calib.dt !== undefined ? String(calib.dt) : "");
    setPxPerMm(calib.px_per_mm !== undefined ? String(calib.px_per_mm) : "");
    setXOffsets(Array.isArray(calib.x_offset) ? calib.x_offset.map(String) : Array(numCameras).fill("0"));
    setYOffsets(Array.isArray(calib.y_offset) ? calib.y_offset.map(String) : Array(numCameras).fill("0"));
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [numCameras, config.calibration?.scale_factor]);

  // Update offsets if number of cameras changes
  useEffect(() => {
    setXOffsets(prev => {
      const arr = [...prev];
      while (arr.length < numCameras) arr.push("0");
      return arr.slice(0, numCameras);
    });
    setYOffsets(prev => {
      const arr = [...prev];
      while (arr.length < numCameras) arr.push("0");
      return arr.slice(0, numCameras);
    });
  }, [numCameras]);

  // Debounced auto-save (only update config if valid)
  const debounceTimer = React.useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    if (debounceTimer.current) clearTimeout(debounceTimer.current);
    debounceTimer.current = setTimeout(() => {
      // Only update if all fields are valid numbers
      const dtNum = Number(dt);
      const pxPerMmNum = Number(pxPerMm);
      const xOffsetNums = xOffsets.map(v => Number(v));
      const yOffsetNums = yOffsets.map(v => Number(v));
      const valid =
        !isNaN(dtNum) && dt !== "" &&
        !isNaN(pxPerMmNum) && pxPerMm !== "" &&
        xOffsetNums.every((n, i) => xOffsets[i] !== "" && !isNaN(n)) &&
        yOffsetNums.every((n, i) => yOffsets[i] !== "" && !isNaN(n));
      if (valid) {
        updateConfig(["calibration", "scale_factor"], {
          dt: dtNum,
          px_per_mm: pxPerMmNum,
          x_offset: xOffsetNums,
          y_offset: yOffsetNums,
        });
      }
    }, 500);
    return () => {
      if (debounceTimer.current) clearTimeout(debounceTimer.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dt, pxPerMm, xOffsets, yOffsets]);

  const calibrateVectors = async () => {
    setCalibrating(true);
    try {
      const response = await fetch('/backend/calibration/scale_factor/calibrate_vectors', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_path_idx: sourcePathIdx,
          camera: camera,
          dt: Number(dt),
          px_per_mm: Number(pxPerMm),
          image_count: config.images?.num_images ?? 1000,
          x_offset: xOffsets.map(Number),
          y_offset: yOffsets.map(Number),
          type_name: "instantaneous"
        })
      });
      const result = await response.json();
      if (response.ok) {
        console.log(`Scale factor calibration started! Job ID: ${result.job_id}`);
        setScaleFactorJobId(result.job_id);
      } else {
        throw new Error(result.error || "Failed to start scale factor calibration");
      }
    } catch (e: any) {
      console.error(`Error starting scale factor calibration: ${e.message}`);
    } finally {
      setCalibrating(false);
    }
  };

  return (
    <Card>
      {/* Add progress display if job is running */}
      {scaleFactorJobId && scaleFactorJobDetails && (
        <div className="mb-4 p-4 border rounded bg-blue-50">
          <div className="flex items-center gap-2 text-sm mb-2">
            <strong>Scale Factor Calibration Progress:</strong>
            <span className="font-medium">{scaleFactorJobStatus}</span>
          </div>
          {(scaleFactorJobStatus === 'running' || scaleFactorJobStatus === 'starting') && (
            <div className="flex items-center gap-2 text-green-600 text-sm">
              <span className="animate-spin inline-block w-4 h-4 border-2 border-green-600 border-t-transparent rounded-full"></span>
              Processing files...
            </div>
          )}
          <div className="w-full bg-gray-200 h-2 rounded overflow-hidden">
            <div className={`h-2 bg-green-600`} style={{ width: `${scaleFactorJobDetails.progress || 0}%` }}></div>
          </div>
          <div className="text-xs text-muted-foreground mt-1">
            Progress: {scaleFactorJobDetails.progress || 0}%
            {scaleFactorJobDetails.processed_files !== undefined && scaleFactorJobDetails.total_files !== undefined &&
              ` (Files: ${scaleFactorJobDetails.processed_files}/${scaleFactorJobDetails.total_files})`}
          </div>
          {scaleFactorJobStatus === 'completed' && (
            <div className="mt-2 text-xs text-green-600">
              Scale factor calibration completed! Processed {scaleFactorJobDetails.processed_files} files.
            </div>
          )}
          {scaleFactorJobStatus === 'failed' && scaleFactorJobDetails.error && (
            <div className="mt-2 text-xs text-red-600">
              Error: {scaleFactorJobDetails.error}
            </div>
          )}
        </div>
      )}

      <CardHeader>
        <CardTitle>Scale Factor Calibration</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="grid md:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs font-medium">Source Path</label>
            <Select value={String(sourcePathIdx)} onValueChange={v => setSourcePathIdx(Number(v))}>
              <SelectTrigger id="srcpath"><SelectValue placeholder="Pick source path" /></SelectTrigger>
              <SelectContent>
                {sourcePaths.map((p, i) => (
                  <SelectItem key={i} value={String(i)}>{basename(p)}</SelectItem>
                ))}
              </SelectContent>
            </Select>
            <p className="text-xs text-muted-foreground mt-1">Configured in Settings → Directories.</p>
          </div>
          <div>
            <label className="block text-xs font-medium">Camera</label>
            <Select value={String(camera)} onValueChange={v => setCamera(Number(v))}>
              <SelectTrigger id="camera"><SelectValue placeholder="Select camera" /></SelectTrigger>
              <SelectContent>
                {cameraOptions.map((c, i) => (
                  <SelectItem key={i} value={String(c)}>{`Camera ${c}`}</SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>
        </div>
        <div>
          <label className="block text-xs font-medium">Δt (seconds)</label>
          <Input
            type="text"
            inputMode="decimal"
            pattern="[0-9]*\.?[0-9]*"
            value={dt}
            onChange={e => setDt(e.target.value)}
            onBlur={e => {
              // Only update config if valid number
              if (e.target.value !== "" && !isNaN(Number(e.target.value))) {
                setDt(e.target.value);
              }
            }}
            placeholder="1.0"
          />
        </div>
        <div>
          <label className="block text-xs font-medium">Pixels per mm</label>
          <Input
            type="text"
            inputMode="decimal"
            pattern="[0-9]*\.?[0-9]*"
            value={pxPerMm}
            onChange={e => setPxPerMm(e.target.value)}
            onBlur={e => {
              if (e.target.value !== "" && !isNaN(Number(e.target.value))) {
                setPxPerMm(e.target.value);
              }
            }}
            placeholder="1.0"
          />
        </div>
        {/* Table/grid for X/Y offsets per camera */}
        <div>
          <label className="block text-xs font-medium mb-1">Camera Offsets (px) - Set to 0,0 for auto bottom-left origin</label>
          <div className="overflow-x-auto">
            <table className="min-w-[320px] border text-xs">
              <thead>
                <tr>
                  <th className="px-2 py-1 border">Camera</th>
                  <th className="px-2 py-1 border">X Offset</th>
                  <th className="px-2 py-1 border">Y Offset</th>
                </tr>
              </thead>
              <tbody>
                {Array.from({length: numCameras}).map((_,i)=>(
                  <tr key={i}>
                    <td className="px-2 py-1 border text-center">{i+1}</td>
                    <td className="px-2 py-1 border">
                      <Input
                        type="text"
                        inputMode="decimal"
                        pattern="[0-9]*\.?[0-9]*"
                        value={xOffsets[i]||""}
                        onChange={e=>{
                          const next = [...xOffsets]; next[i]=e.target.value; setXOffsets(next);
                        }}
                        onBlur={e=>{
                          if (e.target.value !== "" && !isNaN(Number(e.target.value))) {
                            const next = [...xOffsets]; next[i]=e.target.value; setXOffsets(next);
                          }
                        }}
                        className="w-24"
                        placeholder="0"
                      />
                    </td>
                    <td className="px-2 py-1 border">
                      <Input
                        type="text"
                        inputMode="decimal"
                        pattern="[0-9]*\.?[0-9]*"
                        value={yOffsets[i]||""}
                        onChange={e=>{
                          const next = [...yOffsets]; next[i]=e.target.value; setYOffsets(next);
                        }}
                        onBlur={e=>{
                          if (e.target.value !== "" && !isNaN(Number(e.target.value))) {
                            const next = [...yOffsets]; next[i]=e.target.value; setYOffsets(next);
                          }
                        }}
                        className="w-24"
                        placeholder="0"
                      />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Status indicator (same as pinhole) */}
        <div className="mb-2">
          {status === "running" && (
            <div className="flex items-center gap-2 text-blue-600 text-sm">
              <span className="animate-spin inline-block w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></span>
              Calibration is running...
            </div>
          )}
          {status === "completed" && (
            <div className="flex items-center gap-2 text-green-600 text-sm">
              <span className="inline-block w-3 h-3 bg-green-600 rounded-full"></span>
              Calibration completed!
            </div>
          )}
          {status === "error" && (
            <div className="flex items-center gap-2 text-red-600 text-sm">
              <span className="inline-block w-3 h-3 bg-red-600 rounded-full"></span>
              Calibration error!
            </div>
          )}
          {status === "not_started" && (
            <div className="flex items-center gap-2 text-gray-400 text-sm">
              <span className="inline-block w-3 h-3 bg-gray-400 rounded-full"></span>
              Calibration not started.
            </div>
          )}
        </div>

        <div className="flex gap-2">
          <Button
            onClick={calibrateVectors}
            disabled={false}
            className="bg-green-600 hover:bg-green-700 text-white px-6 py-3"
          >
            {calibrating ? "Calibrating..." : "Calibrate All Vectors"}
          </Button>
          {!isActive && <Button variant="outline" onClick={setActive}>Set as Active</Button>}
          {isActive && <span className="text-green-600 text-xs font-semibold ml-2">Active</span>}
        </div>
        <div className="text-xs text-gray-500 mt-2">
          This method calibrates all vectors using scale factor conversion: (pixels - offset) / px_per_mm / dt.<br />
          Set offsets to 0,0 to automatically place bottom-left corner at origin (0,0).<br />
          Updates the calibration.scale_factor block in config.yaml.
        </div>
      </CardContent>
    </Card>
  );
};

// --- Pinhole Calibration UI (Production CV2) ---
const PinholeCalibration: React.FC<{ config: Config; updateConfig: (path: string[], value: any) => void; setActive: () => void; isActive: boolean }> = ({ config, updateConfig, setActive, isActive }) => {
  // Read calibration parameters from config
  const calibrationBlock = config.calibration ?? {};
  const calib = calibrationBlock.pinhole ?? {};
  // Use string state for all number inputs
  const [sourcePaths] = useSourcePaths();
  const [sourcePathIdx, setSourcePathIdx] = useState<number>(0);
  const [camera, setCamera] = useState<number>(calib.camera ?? 1);
  const [imageIndex, setImageIndex] = useState<string>(calib.image_index !== undefined ? String(calib.image_index) : "0");
  const [filePattern, setFilePattern] = useState(calib.file_pattern ?? "calib%05d.tif");
  const [patternCols, setPatternCols] = useState<string>(calib.pattern_cols !== undefined ? String(calib.pattern_cols) : "10");
  const [patternRows, setPatternRows] = useState<string>(calib.pattern_rows !== undefined ? String(calib.pattern_rows) : "10");
  const [dotSpacingMm, setDotSpacingMm] = useState<string>(calib.dot_spacing_mm !== undefined ? String(calib.dot_spacing_mm) : "28.89");
  const [enhanceDots, setEnhanceDots] = useState(calib.enhance_dots ?? true);
  const [asymmetric, setAsymmetric] = useState(calib.asymmetric ?? false);
  const [dt, setDt] = useState<string>(calib.dt !== undefined ? String(calib.dt) : "1.0");

  // Sync UI state with config when config changes
  useEffect(() => {
    setSourcePathIdx(typeof calib.source_path_idx === "number" ? calib.source_path_idx : 0);
    setCamera(calib.camera ?? 1);
    setImageIndex(calib.image_index !== undefined ? String(calib.image_index) : "0");
    setFilePattern(calib.file_pattern ?? "calib%05d.tif");
    setPatternCols(calib.pattern_cols !== undefined ? String(calib.pattern_cols) : "10");
    setPatternRows(calib.pattern_rows !== undefined ? String(calib.pattern_rows) : "10");
    setDotSpacingMm(calib.dot_spacing_mm !== undefined ? String(calib.dot_spacing_mm) : "28.89");
    setEnhanceDots(calib.enhance_dots ?? true);
    setAsymmetric(calib.asymmetric ?? false);
    setDt(calib.dt !== undefined ? String(calib.dt) : "1.0");
  }, [config.calibration?.pinhole]); // Only depend on config.calibration?.pinhole

  // Only update config if all fields are valid numbers
  useEffect(() => {
    const valid = Number.isFinite(sourcePathIdx) &&
      imageIndex !== "" && !isNaN(Number(imageIndex)) &&
      patternCols !== "" && !isNaN(Number(patternCols)) &&
      patternRows !== "" && !isNaN(Number(patternRows)) &&
      dotSpacingMm !== "" && !isNaN(Number(dotSpacingMm)) &&
      dt !== "" && !isNaN(Number(dt));
    const newConfig = {
      source_path_idx: sourcePathIdx,
      camera,
      image_index: Number(imageIndex),
      file_pattern: filePattern,
      pattern_cols: Number(patternCols),
      pattern_rows: Number(patternRows),
      dot_spacing_mm: Number(dotSpacingMm),
      enhance_dots: enhanceDots,
      asymmetric: asymmetric,
      dt: Number(dt),
    };
    if (valid && !deepEqual(config.calibration?.pinhole, newConfig)) {
      updateConfig(["calibration", "pinhole"], newConfig);
    }
  }, [sourcePathIdx, camera, imageIndex, filePattern, patternCols, patternRows, dotSpacingMm, enhanceDots, asymmetric, dt]); // Only depend on UI state

  // States for display
  const [imageB64, setImageB64] = useState<string | null>(null);
  const [totalImages, setTotalImages] = useState(0);
  const [gridPoints, setGridPoints] = useState<[number, number][]>([]);
  const [showIndices, setShowIndices] = useState(true);
  const [dewarpedB64, setDewarpedB64] = useState<string | null>(null);
  const [cameraModel, setCameraModel] = useState<any>(null);
  const [gridData, setGridData] = useState<any>(null);
  const [nativeSize, setNativeSize] = useState<{ w: number; h: number }>({ w: 1024, h: 1024 });
  const [generating, setGenerating] = useState(false);
  const [vectorJobId, setVectorJobId] = useState<string | null>(null);
  const [planarJobId, setPlanarJobId] = useState<string | null>(null);
  const [loadingResults, setLoadingResults] = useState(false);

  // Use the vector calibration status hook
  const { status: vectorStatus, details: vectorDetails } = useVectorCalibrationStatus(vectorJobId);

  const generateCameraModel = async () => {
    setGenerating(true);
    setLoadingResults(true); // Show spinner immediately
    try {
      // First load and process the current image to show results
      const imageResponse = await fetch(`/backend/calibration/planar/get_image?source_path_idx=${sourcePathIdx}&camera=${camera}&image_index=${imageIndex}&file_pattern=${encodeURIComponent(filePattern)}`);
      if (imageResponse.status === 404) {
        alert('Calibration image not found. (File or folder does not exist)');
        setGenerating(false);
        return;
      }
      const imageData = await imageResponse.json();
      if (imageResponse.ok) {
        setImageB64(imageData.image);
        setNativeSize({ w: imageData.width, h: imageData.height });
        setTotalImages(imageData.total_images);
        // Start batch processing
        const response = await fetch('/backend/calibration/planar/calibrate_all', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            source_path_idx: sourcePathIdx,
            camera: camera,
            file_pattern: filePattern,
            pattern_cols: patternCols,
            pattern_rows: patternRows,
            dot_spacing_mm: dotSpacingMm,
            enhance_dots: enhanceDots,
            asymmetric: asymmetric,
            dt: dt
          })
        });
        const result = await response.json();
        if (response.ok && result.job_id) {
          setPlanarJobId(result.job_id);
          // Poll job status until completed, then load results
          const pollForCompletion = () => {
            const interval = setInterval(async () => {
              try {
                const statusResponse = await fetch(`/backend/calibration/planar/calibrate_all/status/${result.job_id}`);
                const statusData = await statusResponse.json();
                if (statusData.status === 'completed') {
                  clearInterval(interval);
                  setTimeout(() => {
                    loadResultsForCurrentImage();
                  }, 1000);
                } else if (statusData.status === 'failed' || statusData.status === 'error') {
                  clearInterval(interval);
                  console.error('Calibration failed:', statusData.error);
                }
              } catch (e) {
                console.log('Error polling planar calibration job:', e);
              }
            }, 2000);
          };
          pollForCompletion();
        } else {
          throw new Error(result.error || 'Failed to start camera model generation');
        }
      } else {
        throw new Error(imageData.error || 'Failed to load image');
      }
    } catch (e: any) {
      console.error(`Error starting camera model generation: ${e.message}`);
    } finally {
      setGenerating(false);
      // Do not setLoadingResults(false) here; let loadResultsForCurrentImage handle it
    }
  };

  // Helper function to load results for current image
  const loadResultsForCurrentImage = async () => {
    setLoadingResults(true);
    try {
      console.log(`Loading calibration results for image index ${imageIndex}...`);
      const compResponse = await fetch('/backend/calibration/planar/compute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_path_idx: sourcePathIdx,
          camera: camera,
          image_index: imageIndex,
          file_pattern: filePattern,
          pattern_cols: patternCols,
          pattern_rows: patternRows,
          dot_spacing_mm: dotSpacingMm,
          enhance_dots: enhanceDots,
          asymmetric: asymmetric,
          dt: dt
        })
      });

      const compData = await compResponse.json();

      if (compResponse.ok) {
        console.log('Raw response data:', compData);
        if (compData.results?.grid_data) {
          console.log('Setting grid data:', compData.results.grid_data);
          setGridData(compData.results.grid_data);
          setGridPoints(compData.results.grid_data.grid_points || []);

          // Check if grid PNG is available
          if (compData.results.grid_data.grid_png) {
            console.log('Grid PNG found in response');
          } else {
            console.log('No grid PNG in response');
          }
        }
        if (compData.results?.camera_model) {
          console.log('Setting camera model:', compData.results.camera_model);
          setCameraModel(compData.results.camera_model);
        }
        if (compData.results?.dewarped_image) {
          setDewarpedB64(compData.results.dewarped_image);
        }
        console.log('Calibration results loaded successfully');
      } else {
        console.error('Error in response:', compData);
      }
    } catch (e: any) {
      console.error(`Error loading results: ${e.message}`);
    } finally {
      setLoadingResults(false);
    }
  };

  const calibrateVectors = async () => {
    try {
      const response = await fetch('/backend/calibration/vectors/calibrate_all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_path_idx: sourcePathIdx,
          camera: camera,
          model_index: imageIndex,
          dt: dt,
          image_count: config.images?.num_images ?? 1000,
          vector_pattern: "%05d.mat",
          type_name: "instantaneous"
        })
      });

      const result = await response.json();

      if (response.ok) {
        console.log(`Vector calibration started using model ${result.model_used}!`);
        setVectorJobId(result.job_id);
      } else {
        throw new Error(result.error || 'Failed to start vector calibration');
      }
    } catch (e: any) {
      console.error(`Error starting vector calibration: ${e.message}`);
    }
  };

  // Camera dropdown options - derive from config like other components
  const cameraOptions = getCameraOptions(config);
  // Ensure valid camera selection when cameraOptions change
  useEffect(() => {
    if (cameraOptions.length > 0 && !cameraOptions.includes(camera)) {
      setCamera(cameraOptions[0]);
    }
  }, [cameraOptions, camera]);

  // Status polling
  const { status: calibrationStatus, details: calibrationDetails } = useCalibrationStatus(Number(sourcePathIdx), camera);
  const stereoVectorStatus = useStereoVectorStatus(sourcePathIdx, camera);

  return (
    <div className="space-y-6">
      {/* Vector Calibration Status - moved above main card */}
      {vectorJobId && vectorDetails && (
        <Card className="mb-4">
          <CardHeader><CardTitle>Vector Calibration Status</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            <div className="flex items-center gap-2 text-sm">Status: <span className="font-medium">{vectorStatus}</span></div>
            {(vectorStatus === 'running' || vectorStatus === 'starting') && (
              <div className="flex items-center gap-2 text-green-600 text-sm">
                <span className="animate-spin inline-block w-4 h-4 border-2 border-green-600 border-t-transparent rounded-full"></span>
                Vector calibration is running...
              </div>
            )}
            <div className="w-full bg-gray-200 h-2 rounded overflow-hidden">
              <div className={`h-2 bg-green-600`} style={{ width: `${vectorDetails.progress || 0}%` }}></div>
            </div>
            <div className="text-xs text-muted-foreground">Progress: {vectorDetails.progress || 0}% {vectorDetails.processed_frames !== undefined && vectorDetails.total_frames !== undefined && `(Frames: ${vectorDetails.processed_frames}/${vectorDetails.total_frames})`}</div>
            {vectorStatus === 'completed' && (
              <div className="mt-2 text-xs text-green-600">
                Vector calibration completed successfully! All runs with valid data were processed.
              </div>
            )}
            {vectorStatus === 'failed' && vectorDetails.error && (
              <div className="mt-2 text-xs text-red-600">
                Error: {vectorDetails.error}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      <Card>
        <CardHeader>
          <CardTitle>Pinhole Calibration (Planar)</CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium">Source Path</label>
              <Select value={String(sourcePathIdx)} onValueChange={v => setSourcePathIdx(Number(v))}>
                <SelectTrigger id="srcpath"><SelectValue placeholder="Pick source path" /></SelectTrigger>
                <SelectContent>
                  {sourcePaths.map((p, i) => (
                    <SelectItem key={i} value={String(i)}>{basename(p)}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
              <p className="text-xs text-muted-foreground mt-1">Configured in Settings → Directories.</p>
            </div>
            <div>
              <label className="text-sm font-medium">Camera</label>
              <Select value={String(camera)} onValueChange={v => setCamera(Number(v))}>
                <SelectTrigger id="camera"><SelectValue placeholder="Select camera" /></SelectTrigger>
                <SelectContent>
                  {cameraOptions.map((c, i) => (
                    <SelectItem key={i} value={String(c)}>{`Camera ${c}`}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div>
              <label className="text-sm font-medium">Model Index:</label>
              <Input
                type="text"
                inputMode="numeric"
                value={imageIndex}
                onChange={e => setImageIndex(e.target.value)}
                onBlur={e => {
                  if (e.target.value !== "" && !isNaN(Number(e.target.value))) {
                    setImageIndex(e.target.value);
                  }
                }}
                min="0"
              />
            </div>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium">File Pattern:</label>
              <Input
                value={filePattern}
                onChange={e => setFilePattern(e.target.value)}
                placeholder="calib%05d.tif"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Pattern Cols:</label>
              <Input
                type="text"
                inputMode="numeric"
                value={patternCols}
                onChange={e => setPatternCols(e.target.value)}
                onBlur={e => {
                  if (e.target.value !== "" && !isNaN(Number(e.target.value))) {
                    setPatternCols(e.target.value);
                  }
                }}
                min="1"
              />
            </div>
            <div>
              <label className="text-sm font-medium">Pattern Rows:</label>
              <Input
                type="text"
                inputMode="numeric"
                value={patternRows}
                onChange={e => setPatternRows(e.target.value)}
                onBlur={e => {
                  if (e.target.value !== "" && !isNaN(Number(e.target.value))) {
                    setPatternRows(e.target.value);
                  }
                }}
                min="1"
              />
            </div>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium">Dot Spacing (mm):</label>
              <Input
                type="text"
                inputMode="decimal"
                pattern="[0-9]*\.?[0-9]*"
                value={dotSpacingMm}
                onChange={e => setDotSpacingMm(e.target.value)}
                onBlur={e => {
                  if (e.target.value !== "" && !isNaN(Number(e.target.value))) {
                    setDotSpacingMm(e.target.value);
                  }
                }}
              />
            </div>
            <div>
              <label className="text-sm font-medium">Δt (seconds):</label>
              <Input
                type="text"
                inputMode="decimal"
                pattern="[0-9]*\.?[0-9]*"
                value={dt}
                onChange={e => setDt(e.target.value)}
                onBlur={e => {
                  if (e.target.value !== "" && !isNaN(Number(e.target.value))) {
                    setDt(e.target.value);
                  }
                }}
                min="0.001"
              />
            </div>
            <div>
              <label className="block text-xs font-medium">
                <input
                  type="checkbox"
                  checked={enhanceDots}
                  onChange={e => setEnhanceDots(e.target.checked)}
                  className="mr-2"
                />
                Enhance Dots
              </label>
            </div>
          </div>
          {/* Calibration Status Indicator */}
          <div className="mb-2">
            {calibrationStatus === "running" && (
              <div className="flex items-center gap-2 text-blue-600 text-sm">
                <span className="animate-spin inline-block w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full"></span>
                Calibration is running...
              </div>
            )}
            {calibrationStatus === "completed" && (
              <div className="flex items-center gap-2 text-green-600 text-sm">
                <span className="inline-block w-3 h-3 bg-green-600 rounded-full"></span>
                Calibration completed!
              </div>
            )}
            {calibrationStatus === "error" && (
              <div className="flex items-center gap-2 text-red-600 text-sm">
                <span className="inline-block w-3 h-3 bg-red-600 rounded-full"></span>
                Calibration error: {calibrationDetails?.error}
              </div>
            )}
            {calibrationStatus === "not_started" && (
              <div className="flex items-center gap-2 text-gray-400 text-sm">
                <span className="inline-block w-3 h-3 bg-gray-400 rounded-full"></span>
                Calibration not started.
              </div>
            )}
          </div>

          {/* TWO MAIN BUTTONS */}
          <div className="border-t pt-4">
            <div className="flex gap-4 items-center">
              <Button
                onClick={generateCameraModel}
                disabled={generating || vectorStatus === "running" || calibrationStatus === "running"}
                className="bg-blue-600 hover:bg-blue-700 text-white px-6 py-3"
              >
                {generating ? 'Generating...' : 'Generate Camera Model'}
              </Button>
              <Button
                onClick={calibrateVectors}
                disabled={vectorStatus === "running" || vectorStatus === "starting"}
                className="bg-green-600 hover:bg-green-700 text-white px-6 py-3"
              >
                {vectorStatus === "running" || vectorStatus === "starting" ? 'Calibrating...' : 'Calibrate Vectors'}
              </Button>
              {/* Removed explicit Load Results button (auto handled) */}
              {!isActive && <Button variant="outline" onClick={setActive}>Set as Active</Button>}
              {isActive && <span className="text-green-600 text-xs font-semibold ml-2">Active</span>}
            </div>
            <div className="text-xs text-gray-500 mt-2">
              <p><strong>Generate Camera Model:</strong> Process all calibration images to create camera models.</p>
              <p><strong>Calibrate Vectors:</strong> Use camera model index {imageIndex} to calibrate PIV vectors.</p>
              <p><strong>Load Results:</strong> Load existing calibration results for the current image index.</p>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Image Display Section - Show grid visualization if available */}
      {(loadingResults) && (
        <div className="flex items-center justify-center py-8">
          <span className="animate-spin inline-block w-8 h-8 border-4 border-blue-600 border-t-transparent rounded-full"></span>
          <span className="ml-3 text-blue-600 text-sm">Loading calibration results...</span>
        </div>
      )}
      {gridData && gridData.grid_png && !loadingResults && (
        <div className="grid lg:grid-cols-2 gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Grid Visualization (Detected Indices)</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="relative border rounded inline-block">
                <img
                  src={`data:image/png;base64,${gridData.grid_png}`}
                  alt="Grid visualization"
                  style={{ maxWidth: "512px", width: "100%" }}
                />
                <div className="absolute top-2 right-2 bg-black bg-opacity-70 text-white px-2 py-1 rounded text-xs">
                  Model {imageIndex} ({totalImages} images found)
                </div>
              </div>
              {gridPoints.length > 0 && (
                <div className="text-xs text-gray-600 mt-2">
                  Grid points detected: {gridPoints.length}
                </div>
              )}
            </CardContent>
          </Card>
          {/* Consolidated Camera Metrics Panel */}
          <Card>
            <CardHeader>
              <CardTitle>Camera Metrics</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="text-xs space-y-2">
                {gridData && (
                  <>
                    <div><b>Reprojection Error (overall):</b> {gridData.reprojection_error?.toFixed(3)} px</div>
                    <div><b>Mean |x| Error:</b> {gridData.reprojection_error_x_mean?.toFixed(3)} px</div>
                    <div><b>Mean |y| Error:</b> {gridData.reprojection_error_y_mean?.toFixed(3)} px</div>
                    <div><b>Pattern Size:</b> {gridData.pattern_size?.join(' x ')}</div>
                    <div><b>Dot Spacing:</b> {gridData.dot_spacing_mm} mm</div>
                    <div><b>Estimated Pixels per mm:</b> {gridData.pixels_per_mm ? gridData.pixels_per_mm.toFixed(3) : 'N/A'}</div>
                    <div><b>Image Name:</b> {gridData.original_filename}</div>
                    <div><b>Timestamp:</b> {gridData.timestamp}</div>
                  </>
                )}
                {cameraModel && (
                  <>
                    <div><b>Camera Matrix:</b></div>
                    {cameraModel.camera_matrix && cameraModel.camera_matrix.map((row: number[], i: number) => (
                      <div key={i}>[{row.map(v => v.toFixed(3)).join(', ')}]</div>
                    ))}
                    <div><b>Focal Length:</b> fx={cameraModel.focal_length?.[0]?.toFixed(1)}, fy={cameraModel.focal_length?.[1]?.toFixed(1)}</div>
                    <div><b>Principal Point:</b> cx={cameraModel.principal_point?.[0]?.toFixed(1)}, cy={cameraModel.principal_point?.[1]?.toFixed(1)}</div>
                    <div><b>Distortion Coeffs:</b> [{cameraModel.dist_coeffs?.map((d: number) => d.toFixed(4)).join(', ')}]</div>
                    <div><b>Reprojection Error:</b> {cameraModel.reprojection_error?.toFixed(3)} px</div>
                  </>
                )}
              </div>
            </CardContent>
          </Card>
        </div>
      )}



      <div className="flex gap-2">
        {!isActive && <Button variant="outline" onClick={setActive}>Set as Active</Button>}
        {isActive && <span className="text-green-600 text-xs font-semibold ml-2">Active</span>}
      </div>
    </div>
  );
}


// --- Main Calibration Page ---
const Calibration: React.FC = () => {
  const [method, setMethod] = useState<CalibrationMethod>("pinhole");
  const [config, updateConfig] = useConfig();
  const active = config.calibration?.active || "pinhole";

  // Only change active method, do not overwrite configs
  function setActiveMethod(m: CalibrationMethod) {
    updateConfig(["calibration", "active"], m);
  }

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Calibration</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4 items-center">
            <label className="text-sm font-medium">Method:</label>
            <Select value={method} onValueChange={v => setMethod(v as CalibrationMethod)}>
              <SelectTrigger id="method"><SelectValue placeholder="Select method" /></SelectTrigger>
              <SelectContent>
                <SelectItem value="scale_factor">Scale Factor</SelectItem>
                <SelectItem value="pinhole">Pinhole (Planar)</SelectItem>
                <SelectItem value="stereo">Stereo Calibration</SelectItem>
              </SelectContent>
            </Select>
            <span className="ml-4 text-xs text-gray-500">Active: <b>{active}</b></span>
          </div>
        </CardContent>
      </Card>
      {method === "scale_factor" && (
        <ScaleFactorCalibration
          config={config}
          updateConfig={updateConfig}
          setActive={() => setActiveMethod("scale_factor")}
          isActive={active === "scale_factor"}
        />
      )}
      {method === "pinhole" && (
        <PinholeCalibration
          config={config}
          updateConfig={updateConfig}
          setActive={() => setActiveMethod("pinhole")}
          isActive={active === "pinhole"}
        />
      )}
      {method === "stereo" && (
        <StereoCalibration
          config={config}
          updateConfig={updateConfig}
          setActive={() => setActiveMethod("stereo")}
          isActive={active === "stereo"}
        />
      )}
      {/* Stereo method can be added here in the future */}
    </div>
  );
};

export default Calibration;
