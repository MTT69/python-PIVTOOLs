"use client";

import React, { useState, useEffect, useRef } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import { Loader2 } from 'lucide-react';

// Helper functions
function getCameraOptions(config: any): number[] {
  if (!config) return [1, 2];
  const dirs = config.directories || {};
  const cameraCounts = dirs.camera_counts || [];
  if (cameraCounts.length === 0) return [1, 2];

  const maxCameras = Math.max(...cameraCounts);
  return Array.from({ length: maxCameras }, (_, i) => i + 1);
}

const basename = (p: string) => {
  if (!p) return "";
  const parts = p.replace(/\\/g, "/").split("/");
  // Show last two segments if possible
  if (parts.length >= 2) return parts.slice(-2).join("/");
  return parts.filter(Boolean).pop() || p;
};

function useSourcePaths() {
  const [sourcePaths, setSourcePaths] = useState<string[]>([]);

  useEffect(() => {
    const stored = localStorage.getItem('config');
    if (stored) {
      try {
        const config = JSON.parse(stored);
        setSourcePaths(config.directories?.source_paths || []);
      } catch (e) {
        console.error('Error loading source paths:', e);
      }
    }
  }, []);

  return [sourcePaths];
}

// Deep equality check for config updates
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

interface Config {
  calibration?: {
    stereo?: {
      source_path_idx?: number;
      camera_pair?: [number, number];
      file_pattern?: string;
      pattern_cols?: number;
      pattern_rows?: number;
      dot_spacing_mm?: number;
      enhance_dots?: boolean;
      asymmetric?: boolean;
    };
  };
  directories?: {
    source_paths?: string[];
    camera_counts?: number[];
  };
  images?: {
    num_images?: number;
  };
}

// --- Stereo Calibration Status Hook ---
const useStereoCalibrationStatus = (jobId: string | null) => {
  const [status, setStatus] = React.useState<string>("not_started");
  const [details, setDetails] = React.useState<any>(null);
  React.useEffect(() => {
    if (!jobId) return;
    let active = true;
    const fetchStatus = async () => {
      try {
        const res = await fetch(`/backend/stereo/calibration/status/${jobId}`);
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
    return () => { active = false; clearInterval(interval); };
  }, [jobId]);
  return { status, details };
};

const StereoCalibration: React.FC<{
  config?: Config;
  updateConfig: (path: string[], value: any) => void;
  setActive: () => void;
  isActive: boolean;
}> = ({ config = {}, updateConfig, setActive, isActive }) => {

  // Configuration state
  const [sourcePaths] = useSourcePaths();
  const calibrationBlock = (config as Config).calibration ?? {};
  const calib = calibrationBlock.stereo ?? {};

  const [sourcePathIdx, setSourcePathIdx] = useState<number>(calib.source_path_idx ?? 0);
  const [cameraPair, setCameraPair] = useState<[number, number]>(calib.camera_pair ?? [1, 2]);
  const [filePattern, setFilePattern] = useState(calib.file_pattern ?? "planar_calibration_plate_*.tif");
  const [patternCols, setPatternCols] = useState<string>(String(calib.pattern_cols ?? 10));
  const [patternRows, setPatternRows] = useState<string>(String(calib.pattern_rows ?? 10));
  const [dotSpacingMm, setDotSpacingMm] = useState<string>(String(calib.dot_spacing_mm ?? 28.89));
  const [enhanceDots, setEnhanceDots] = useState(calib.enhance_dots ?? true);
  const [asymmetric, setAsymmetric] = useState(calib.asymmetric ?? false);

  // Job management state
  const [jobId, setJobId] = useState<string | null>(null);
  const { status: calibrationStatus, details: calibrationDetails } = useStereoCalibrationStatus(jobId);
  const [calibrationResults, setCalibrationResults] = useState<any>(null);
  const [vectorJob, setVectorJob] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [gridImages, setGridImages] = useState<any>(null);
  const [currentGridIndex, setCurrentGridIndex] = useState(1);
  const [vectorPollingActive, setVectorPollingActive] = useState(false);
  const [vectorJobId, setVectorJobId] = useState<string | null>(null);
  const [showCompletionMessage, setShowCompletionMessage] = useState(false);

  const cameraOptions = getCameraOptions(config);
  const [camera1, camera2] = cameraPair;

  // Debounced config updates - only update when values actually change
  const debounceTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const prevConfigRef = useRef<any>(null);

  useEffect(() => {
    const newConfig = {
      source_path_idx: sourcePathIdx,
      camera_pair: cameraPair,
      file_pattern: filePattern,
      pattern_cols: Number(patternCols),
      pattern_rows: Number(patternRows),
      dot_spacing_mm: Number(dotSpacingMm),
      enhance_dots: enhanceDots,
      asymmetric: asymmetric,
    };

    // Only update if config actually changed using deep equality
    if (!prevConfigRef.current || !deepEqual(prevConfigRef.current, newConfig)) {
      prevConfigRef.current = newConfig;

      // Clear existing timer
      if (debounceTimer.current) clearTimeout(debounceTimer.current);

      // Debounce the update - only send after 500ms of inactivity
      debounceTimer.current = setTimeout(() => {
        console.log('Updating stereo config after debounce');
        updateConfig(["calibration", "stereo"], newConfig);
      }, 500);
    }

    return () => {
      if (debounceTimer.current) clearTimeout(debounceTimer.current);
    };
  }, [sourcePathIdx, cameraPair, filePattern, patternCols, patternRows, dotSpacingMm, enhanceDots, asymmetric, updateConfig]);

  // Auto-adjust camera pair when options change
  useEffect(() => {
    if (cameraOptions.length >= 2) {
      const needsUpdate =
        cameraPair[0] === cameraPair[1] ||
        !cameraOptions.includes(cameraPair[0]) ||
        !cameraOptions.includes(cameraPair[1]);

      if (needsUpdate) {
        setCameraPair([cameraOptions[0], cameraOptions[1]]);
      }
    }
  }, [cameraOptions, cameraPair]);

  // Load grid images when calibration results are available
  useEffect(() => {
    if (calibrationResults) {
      console.log('Calibration results changed, loading grid images...');
      loadGridImages();
    }
  }, [calibrationResults]);

  // Reload grid images when index changes
  useEffect(() => {
    if (calibrationResults && currentGridIndex) {
      console.log('Grid index changed to:', currentGridIndex, 'reloading grid images...');
      loadGridImages();
    }
  }, [currentGridIndex]);

  // Handle job completion effects - more robust completion detection
  useEffect(() => {
    if (calibrationStatus === 'completed') {
      if (!showCompletionMessage) {
        setShowCompletionMessage(true);
        setTimeout(() => setShowCompletionMessage(false), 5000);
      }

      // Try to load results from the job data if available
      if (calibrationDetails?.results) {
        const pairKey = `cam${camera1}_cam${camera2}`;
        console.log('Job completed, checking for results in job data with key:', pairKey);
        console.log('Available job result keys:', Object.keys(calibrationDetails.results));

        if (calibrationDetails.results[pairKey] && !calibrationDetails.results[pairKey].error) {
          console.log('Found results in job data, setting calibrationResults');
          setCalibrationResults(calibrationDetails.results[pairKey]);
        }
      }

      // Always try to load existing results as fallback
      if (!calibrationResults) {
        console.log('No calibration results set, trying to load existing results as fallback');
        setTimeout(() => {
          loadExistingResults();
        }, 1000);
      }
    }
  }, [calibrationStatus, calibrationDetails, showCompletionMessage, calibrationResults, camera1, camera2]);

  // Additional effect to handle grid loading after calibration results are set
  useEffect(() => {
    if (calibrationResults && !gridImages) {
      setTimeout(() => {
        console.log('Loading grid images after calibration results were set...');
        loadGridImages();
      }, 500);
    }
  }, [calibrationResults, gridImages]);

  // API functions using the backend endpoints
  const loadExistingResults = async () => {
    console.log('Loading existing results for camera pair:', camera1, camera2);
    try {
      setIsLoading(true);
      const response = await fetch(
        `/backend/stereo/calibration/load_results?source_path_idx=${sourcePathIdx}&cam1=${camera1}&cam2=${camera2}`
      );

      if (response.ok) {
        const data = await response.json();
        console.log('Load results response:', data);
        if (data.exists) {
          console.log('Setting calibration results from existing:', data.results);
          setCalibrationResults(data.results);
          // Also load grid images after setting results
          setTimeout(() => {
            loadGridImages();
          }, 100);
        } else {
          console.log('No existing calibration found for this camera pair');
        }
      } else {
        console.error('Error response:', response.status, response.statusText);
      }
    } catch (e) {
      console.error('Error loading existing results:', e);
    } finally {
      setIsLoading(false);
    }
  };

  // NEW: explicit button triggered load
  const handleLoadExistingClick = () => {
    loadExistingResults();
  };

  const loadGridImages = async () => {
    console.log('Loading grid images for index:', currentGridIndex);
    try {
      const response = await fetch(
        `/backend/stereo/calibration/get_grid_images?source_path_idx=${sourcePathIdx}&cam1=${camera1}&cam2=${camera2}&image_index=${currentGridIndex}`
      );

      if (response.ok) {
        const data = await response.json();
        console.log('Grid images response:', data);
        setGridImages(data);
      } else {
        console.error('Error loading grid images:', response.status, response.statusText);
      }
    } catch (e) {
      console.error('Error loading grid images:', e);
    }
  };

  const runCalibration = async () => {
    try {
      const response = await fetch('/backend/stereo/calibration/run_enhanced', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_path_idx: sourcePathIdx,
          camera_pairs: [cameraPair],
          file_pattern: filePattern,
          pattern_cols: Number(patternCols),
          pattern_rows: Number(patternRows),
          dot_spacing_mm: Number(dotSpacingMm),
          asymmetric,
          enhance_dots: enhanceDots
        })
      });

      if (response.ok) {
        const data = await response.json();
        setJobId(data.job_id);
      }
    } catch (e) {
      console.error('Error starting calibration:', e);
    }
  };

  const runVectorCalibration = async () => {
    if (!calibrationResults) return;
    try {
      setIsLoading(true);
      const response = await fetch('/backend/stereo/vectors/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          source_path_idx: sourcePathIdx,
          camera_pairs: [cameraPair],
          image_count: (config as any)?.images?.num_images ?? 1000,
          vector_pattern: '%05d.mat',
          type_name: 'instantaneous',
          runs_to_process: [6]
        })
      });

      if (response.ok) {
        const data = await response.json();
        setVectorJobId(data.job_id);
        setVectorPollingActive(true);
        setVectorJob({ job_id: data.job_id, status: 'starting', progress: 0 });
      }
    } catch (e) {
      console.error('Error starting vector calibration:', e);
    } finally {
      setIsLoading(false);
    }
  };

  // Poll vector job status
  useEffect(() => {
    if (!vectorPollingActive || !vectorJobId) return;
    let active = true;
    const fetchStatus = async () => {
      try {
        const res = await fetch(`/backend/stereo/vectors/status/${vectorJobId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (!active) return;
        setVectorJob(data);
        if (data.status === 'completed' || data.status === 'failed' || data.status === 'cancelled') {
          setVectorPollingActive(false);
        }
      } catch (e) {
        console.error('Vector status error:', e);
      }
    };
    fetchStatus();
    const id = setInterval(fetchStatus, 2000);
    return () => { active = false; clearInterval(id); };
  }, [vectorPollingActive, vectorJobId]);

  // When calibration results appear ensure grid images loaded
  useEffect(() => {
    if (calibrationResults && !gridImages) {
      loadGridImages();
    }
  }, [calibrationResults, gridImages]);

  // Update grid images if index changes
  useEffect(() => {
    if (calibrationResults) {
      loadGridImages();
    }
  }, [currentGridIndex]);

  const nextGrid = () => {
    if (!gridImages?.available_indices) return;
    const idxList = gridImages.available_indices;
    const currentPos = idxList.indexOf(currentGridIndex);
    if (currentPos >= 0 && currentPos < idxList.length - 1) setCurrentGridIndex(idxList[currentPos + 1]);
  };
  const prevGrid = () => {
    if (!gridImages?.available_indices) return;
    const idxList = gridImages.available_indices;
    const currentPos = idxList.indexOf(currentGridIndex);
    if (currentPos > 0) setCurrentGridIndex(idxList[currentPos - 1]);
  };

  const getStatusBadge = (status: string) => {
    const statusColors = {
      'starting': 'bg-yellow-100 text-yellow-800',
      'running': 'bg-blue-100 text-blue-800',
      'completed': 'bg-green-100 text-green-800',
      'failed': 'bg-red-100 text-red-800',
      'cancelled': 'bg-gray-100 text-gray-800'
    };

    return (
      <span className={`px-2 py-1 rounded-full text-xs font-medium ${statusColors[status as keyof typeof statusColors] || 'bg-gray-100 text-gray-800'}`}>
        {status}
      </span>
    );
  };

  return (
    <div className="space-y-6">
      {/* Vector Reconstruction Status - moved above calibration block */}
      {vectorJob && (
        <Card className="mb-4">
          <CardHeader><CardTitle>Vector Reconstruction Status</CardTitle></CardHeader>
          <CardContent className="space-y-2">
            <div className="flex items-center gap-2 text-sm">Status: <span className="font-medium">{vectorJob.status}</span></div>
            {(vectorJob.status === 'running' || vectorJob.status === 'starting') && (
              <div className="flex items-center gap-2 text-purple-600 text-sm">
                <span className="animate-spin inline-block w-4 h-4 border-2 border-purple-600 border-t-transparent rounded-full"></span>
                Vector reconstruction is running...
              </div>
            )}
            <div className="w-full bg-gray-200 h-2 rounded overflow-hidden">
              <div className={`h-2 bg-purple-600`} style={{ width: `${vectorJob.progress || 0}%` }}></div>
            </div>
            <div className="text-xs text-muted-foreground">Progress: {vectorJob.progress || 0}% {vectorJob.processed_frames !== undefined && vectorJob.total_frames !== undefined && `(Frames: ${vectorJob.processed_frames}/${vectorJob.total_frames})`}</div>
            {vectorJob.status === 'completed' && vectorJob.results && (
              <div className="mt-2 text-xs">
                {Object.keys(vectorJob.results).map(k => {
                  const r = vectorJob.results[k];
                  return (
                    <div key={k} className="mb-1">Pair {r.camera_pair?.join('-')} Success Rate: {r.success_rate?.toFixed ? r.success_rate.toFixed(1) : r.success_rate}%</div>
                  );
                })}
              </div>
            )}
          </CardContent>
        </Card>
      )}
      <Card>
        <CardHeader>
          <CardTitle>Stereo Calibration - Production Backend Integration</CardTitle>
        </CardHeader>
        <CardContent className="space-y-6">
          {/* Actions Row */}
          <div className="flex flex-wrap gap-3 mb-2">
            <Button onClick={runCalibration} disabled={calibrationStatus === 'running'} className="bg-blue-600 hover:bg-blue-700 text-white">{calibrationStatus === 'running' ? 'Running...' : 'Run Stereo Calibration'}</Button>
            <Button variant="outline" onClick={handleLoadExistingClick}>Load Existing Calibration</Button>
            <Button onClick={runVectorCalibration} disabled={!calibrationResults || (vectorJob && ['running','starting'].includes(vectorJob.status))} className="bg-purple-600 hover:bg-purple-700 text-white">{vectorJob && ['running','starting'].includes(vectorJob.status) ? 'Reconstructing...' : 'Run Vector Reconstruction'}</Button>
          </div>
          {/* Configuration Section */}
          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium">Source Path</label>
              <Select value={String(sourcePathIdx)} onValueChange={v => setSourcePathIdx(Number(v))}>
                <SelectTrigger>
                  <SelectValue>
                    {sourcePaths[sourcePathIdx] ? basename(sourcePaths[sourcePathIdx]) : "Pick source path"}
                  </SelectValue>
                </SelectTrigger>
                <SelectContent>
                  {sourcePaths.map((p, i) => (
                    <SelectItem key={i} value={String(i)}>{basename(p)}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium">Camera 1</label>
              <Select value={String(camera1)} onValueChange={v => setCameraPair([Number(v), camera2])}>
                <SelectTrigger>
                  <SelectValue placeholder="Select camera 1" />
                </SelectTrigger>
                <SelectContent>
                  {cameraOptions.filter(c => c !== camera2).map(c => (
                    <SelectItem key={c} value={String(c)}>Camera {c}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>

            <div>
              <label className="text-sm font-medium">Camera 2</label>
              <Select value={String(camera2)} onValueChange={v => setCameraPair([camera1, Number(v)])}>
                <SelectTrigger>
                  <SelectValue placeholder="Select camera 2" />
                </SelectTrigger>
                <SelectContent>
                  {cameraOptions.filter(c => c !== camera1).map(c => (
                    <SelectItem key={c} value={String(c)}>Camera {c}</SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium">File Pattern</label>
              <Input
                value={filePattern}
                onChange={e => setFilePattern(e.target.value)}
                placeholder="planar_calibration_plate_*.tif"
              />
            </div>

            <div>
              <label className="text-sm font-medium">Pattern Cols</label>
              <Input
                type="number"
                value={patternCols}
                onChange={e => setPatternCols(e.target.value)}
                min="3"
              />
            </div>

            <div>
              <label className="text-sm font-medium">Pattern Rows</label>
              <Input
                type="number"
                value={patternRows}
                onChange={e => setPatternRows(e.target.value)}
                min="3"
              />
            </div>
          </div>

          <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
            <div>
              <label className="text-sm font-medium">Dot Spacing (mm)</label>
              <Input
                type="number"
                step="0.01"
                value={dotSpacingMm}
                onChange={e => setDotSpacingMm(e.target.value)}
              />
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={enhanceDots}
                onChange={e => setEnhanceDots(e.target.checked)}
              />
              <label className="text-sm font-medium">Enhance Dots</label>
            </div>

            <div className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={asymmetric}
                onChange={e => setAsymmetric(e.target.checked)}
              />
              <label className="text-sm font-medium">Asymmetric Grid</label>
            </div>
          </div>

          {/* Status indicator */}
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
            {calibrationStatus === "failed" && (
              <div className="flex items-center gap-2 text-red-600 text-sm">
                <span className="inline-block w-3 h-3 bg-red-600 rounded-full"></span>
                Calibration error!
              </div>
            )}
            {calibrationStatus === "not_started" && (
              <div className="flex items-center gap-2 text-gray-400 text-sm">
                <span className="inline-block w-3 h-3 bg-gray-400 rounded-full"></span>
                Calibration not started.
              </div>
            )}
          </div>

          {/* Calibration Results */}
          {calibrationResults && (
            <Card className="mt-4">
              <CardHeader>
                <CardTitle>Calibration Results</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 text-center">
                  <div>
                    <div className="text-sm text-muted-foreground">Reprojection Error</div>
                    <div className="text-xl font-bold">
                      {calibrationResults.calibration_quality?.stereo_reprojection_error?.toFixed(3) || 'N/A'} px
                    </div>
                  </div>

                  <div>
                    <div className="text-sm text-muted-foreground">Relative Angle</div>
                    <div className="text-xl font-bold">
                      {calibrationResults.calibration_quality?.relative_angle_deg?.toFixed(1) || 'N/A'}°
                    </div>
                  </div>

                  <div>
                    <div className="text-sm text-muted-foreground">Image Pairs</div>
                    <div className="text-xl font-bold">
                      {calibrationResults.calibration_quality?.num_image_pairs || 'N/A'}
                    </div>
                  </div>

                  <div>
                    <div className="text-sm text-muted-foreground">Baseline</div>
                    <div className="text-xl font-bold">
                      {calibrationResults.calibration_quality?.baseline_distance?.toFixed(1) || 'N/A'} mm
                    </div>
                  </div>
                </div>

                {calibrationResults.quality_warning && (
                  <div className="mt-4 text-sm text-yellow-600 bg-yellow-50 p-2 rounded">
                    Warning: {calibrationResults.quality_warning}
                  </div>
                )}

                {calibrationResults.quality_status && (
                  <div className="mt-4 text-sm text-green-600 bg-green-50 p-2 rounded">
                    Status: {calibrationResults.quality_status}
                  </div>
                )}
              </CardContent>
            </Card>
          )}

          {/* Grid Detection Viewer */}
          {calibrationResults && (
            <Card className="mt-4">
              <CardHeader><CardTitle>Grid Detections</CardTitle></CardHeader>
              <CardContent className="space-y-4">
                <div className="flex items-center gap-2">
                  <Button variant="outline" onClick={prevGrid} disabled={!gridImages || !gridImages.available_indices || gridImages.available_indices.indexOf(currentGridIndex) <= 0}>Prev</Button>
                  <div className="text-sm">Index: {currentGridIndex}</div>
                  <Button variant="outline" onClick={nextGrid} disabled={!gridImages || !gridImages.available_indices || gridImages.available_indices.indexOf(currentGridIndex) === gridImages.available_indices.length - 1}>Next</Button>
                  {gridImages && <div className="text-xs text-muted-foreground">Available: {gridImages.available_indices?.join(', ') || 'None'}</div>}
                </div>
                <div className="grid md:grid-cols-2 gap-4">
                  {([camera1, camera2] as number[]).map(cam => {
                    const camKey = `cam${cam}`;
                    const data = gridImages?.results?.[camKey];
                    return (
                      <div key={cam} className="border rounded p-2">
                        <div className="text-sm font-medium mb-1">Camera {cam}</div>
                        {data?.grid_image ? (
                          <img src={`data:image/png;base64,${data.grid_image}`} alt={`Grid Cam${cam}`} className="w-full border" />
                        ) : <div className="text-xs text-muted-foreground">No image</div>}
                        <div className="mt-2 grid grid-cols-2 gap-2 text-xs">
                          <div>Reproj Err: {data?.reprojection_error?.toFixed ? data.reprojection_error.toFixed(3) : '—'}</div>
                          <div>Points: {data?.grid_points ? data.grid_points.length : '—'}</div>
                        </div>
                      </div>
                    );
                  })}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Vector Reconstruction Status */}
          {vectorJob && (
            <Card className="mt-4">
              <CardHeader><CardTitle>Vector Reconstruction Status</CardTitle></CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-center gap-2 text-sm">Status: <span className="font-medium">{vectorJob.status}</span></div>
                {(vectorJob.status === 'running' || vectorJob.status === 'starting') && (
                  <div className="flex items-center gap-2 text-purple-600 text-sm">
                    <span className="animate-spin inline-block w-4 h-4 border-2 border-purple-600 border-t-transparent rounded-full"></span>
                    Vector reconstruction is running...
                  </div>
                )}
                <div className="w-full bg-gray-200 h-2 rounded overflow-hidden">
                  <div className={`h-2 bg-purple-600`} style={{ width: `${vectorJob.progress || 0}%` }}></div>
                </div>
                <div className="text-xs text-muted-foreground">Progress: {vectorJob.progress || 0}% {vectorJob.processed_frames !== undefined && vectorJob.total_frames !== undefined && `(Frames: ${vectorJob.processed_frames}/${vectorJob.total_frames})`}</div>
                {vectorJob.status === 'completed' && vectorJob.results && (
                  <div className="mt-2 text-xs">
                    {Object.keys(vectorJob.results).map(k => {
                      const r = vectorJob.results[k];
                      return (
                        <div key={k} className="mb-1">Pair {r.camera_pair?.join('-')} Success Rate: {r.success_rate?.toFixed ? r.success_rate.toFixed(1) : r.success_rate}%</div>
                      );
                    })}
                  </div>
                )}
              </CardContent>
            </Card>
          )}
        </CardContent>
      </Card>

      <div className="flex gap-2 items-center">
        {!isActive && <Button variant="outline" onClick={setActive}>Set as Active</Button>}
        {isActive && (
          <span className="px-2 py-1 bg-green-100 text-green-800 rounded-full text-xs font-medium">
            Active
          </span>
        )}
      </div>
    </div>
  );
};

export default StereoCalibration;
