"use client";
import React, { useEffect, useMemo, useState, useCallback } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useVectorViewer } from "@/hooks/useVectorViewer";
import { useStatisticsCalculation } from "@/hooks/useStatisticsCalculation";

// Inline Vector Merging Hook
const useVectorMerging = (backendUrl: string, basePathIdx: number, cameraOptions: number[], maxFrameCount: number) => {
  const [selectedCameras, setSelectedCameras] = useState<number[]>([]);
  const [merging, setMerging] = useState(false);
  const [mergingJobId, setMergingJobId] = useState<string | null>(null);
  const [showDialog, setShowDialog] = useState(false);
  const [jobStatus, setJobStatus] = useState<string>("not_started");
  const [jobDetails, setJobDetails] = useState<any>(null);

  // Initialize selected cameras when options change
  useEffect(() => {
    if (cameraOptions.length >= 2 && selectedCameras.length === 0) {
      setSelectedCameras([cameraOptions[0], cameraOptions[1]]);
    }
  }, [cameraOptions, selectedCameras.length]);

  const mergeVectors = useCallback(async (frameIdx?: number) => {
    if (selectedCameras.length < 2) {
      alert("Please select at least 2 cameras to merge");
      return;
    }

    setMerging(true);
    setJobStatus("starting");

    try {
      const endpoint = frameIdx !== undefined ? "/merge_vectors/merge_one" : "/merge_vectors/merge_all";
      const body: any = {
        base_path_idx: basePathIdx,
        cameras: selectedCameras,
        type_name: "instantaneous",
        endpoint: "",
        image_count: maxFrameCount,
      };

      if (frameIdx !== undefined) {
        body.frame_idx = frameIdx;
      }

      const response = await fetch(`${backendUrl}${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Failed to start merging");
      }

      if (frameIdx !== undefined) {
        // Single frame merge - completed immediately
        setJobStatus("completed");
        setJobDetails({ progress: 100, message: data.message });
        setMerging(false);
      } else {
        // Batch merge - start polling
        setMergingJobId(data.job_id);
        setJobStatus("running");
      }
    } catch (error: any) {
      console.error("Error starting merge:", error);
      setJobStatus("failed");
      setJobDetails({ error: error.message });
      setMerging(false);
    }
  }, [backendUrl, basePathIdx, selectedCameras, maxFrameCount]);

  // Poll for job status
  useEffect(() => {
    if (!mergingJobId || jobStatus === "completed" || jobStatus === "failed") {
      return;
    }

    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch(`${backendUrl}/merge_vectors/status/${mergingJobId}`);
        const data = await response.json();

        setJobStatus(data.status);
        setJobDetails(data);

        if (data.status === "completed" || data.status === "failed") {
          setMerging(false);
          clearInterval(pollInterval);
        }
      } catch (error) {
        console.error("Error polling merge status:", error);
      }
    }, 1000);

    return () => clearInterval(pollInterval);
  }, [mergingJobId, jobStatus, backendUrl]);

  const resetMerging = useCallback(() => {
    setMerging(false);
    setMergingJobId(null);
    setJobStatus("not_started");
    setJobDetails(null);
  }, []);

  return {
    selectedCameras,
    setSelectedCameras,
    merging,
    mergingJobId,
    showDialog,
    setShowDialog,
    jobStatus,
    jobDetails,
    mergeVectors,
    resetMerging,
  };
};

export default function VectorViewer({ backendUrl = "/backend", config }: { backendUrl?: string; config?: any }) {
  const {
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
    cameraOptions: hookCameraOptions,
    camera,
    setCamera,
    merged,
    setMerged,
    playing,
    limitsLoading,
    meanMode,
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
    handlePlayToggle,
    handleRender,
    toggleMeanMode,
    handleImageClick,
    updateOffsets,
    fetchCornerCoordinates,
    onMouseMove,
    onMouseLeave,
    handleMagnifierMove,
    handleMagnifierLeave,
    basename,
    downloadCurrentView,
    copyCurrentView,
    fetchLimits,  // Added: Now available from the hook
    applyTransformation,
    applyTransformationToAllFrames,
    transformationJob,
    appliedTransforms,
    setAppliedTransforms,
    clearTransforms,
    MAG_SIZE,
    dpr,
    effectiveDir,  // Added: Import effectiveDir from hook
  } = useVectorViewer({ backendUrl, config });
  
  // Override camera options with config if available
  const cameraOptions = useMemo(() => {
    const cameras = config?.paths?.camera_numbers || [];
    return cameras.length > 0 ? cameras : hookCameraOptions;
  }, [config?.paths?.camera_numbers, hookCameraOptions]);

  // Initialize camera from first available option when config loads
  useEffect(() => {
    if (cameraOptions.length > 0 && !camera) {
      setCamera(cameraOptions[0]);
    }
  }, [cameraOptions, camera, setCamera]);

  // Remember last valid hover values so we don't show a loading spinner
  // while backend updates arrive. Only update when hoverData contains valid coords.
  const [lastValidHover, setLastValidHover] = useState<any | null>(null);
  useEffect(() => {
    if (hoverData && typeof hoverData.x === "number" && !isNaN(hoverData.x) && hoverData.i >= 0) {
      setLastValidHover(hoverData);
    }
  }, [hoverData]);

  // Use statistics calculation hook
  const {
    selectedCameras,
    includeMerged,
    calculating,
    statisticsJobId,
    showDialog: showStatisticsDialog,
    setSelectedCameras,
    setIncludeMerged,
    setShowDialog: setShowStatisticsDialog,
    jobStatus: statisticsStatus,
    jobDetails: statisticsDetails,
    calculateStatistics,
    resetStatistics,
  } = useStatisticsCalculation(
    backendUrl,
    basePathIdx,
    cameraOptions,
    maxFrameCount  // Use maxFrameCount from useVectorViewer, which is fetched from backend config
  );

  // Derived values from job details
  const statisticsProgress = statisticsDetails?.overall_progress || statisticsDetails?.progress || 0;
  const statisticsError = statisticsDetails?.error || null;

  // Use vector merging hook
  const {
    selectedCameras: selectedMergeCameras,
    merging,
    mergingJobId,
    showDialog: showMergingDialog,
    setSelectedCameras: setSelectedMergeCameras,
    setShowDialog: setShowMergingDialog,
    jobStatus: mergingStatus,
    jobDetails: mergingDetails,
    mergeVectors,
    resetMerging,
  } = useVectorMerging(
    backendUrl,
    basePathIdx,
    cameraOptions,
    maxFrameCount
  );

  // Derived values from merging job details
  const mergingProgress = mergingDetails?.progress || 0;
  const mergingError = mergingDetails?.error || null;
;

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Results Viewer</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4 mb-4">
            <div className="flex flex-col gap-4 mb-4">
              {/* Base path selection */}
              <div className="flex items-center gap-2">
                <label className="text-sm font-medium">Base Path:</label>
                <Select value={String(basePathIdx)} onValueChange={v => setBasePathIdx(Number(v))}>
                  <SelectTrigger id="basepath"><SelectValue placeholder="Pick base path" /></SelectTrigger>
                  <SelectContent>
                    {basePaths.map((p, i) => (
                      <SelectItem key={i} value={String(i)}>{basename(p)}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Camera and data source controls - Improved styling */}
              <div className="p-3 bg-gray-50 border border-gray-200 rounded-lg">
                <div className="flex flex-wrap items-center gap-4">
                  <div className="flex items-center gap-2">
                    <label htmlFor="camera" className="text-sm font-medium text-gray-700">Camera:</label>
                    <Select value={String(camera)} onValueChange={v => setCamera(Number(v))} disabled={cameraOptions.length === 0}>
                      <SelectTrigger id="camera" className="w-28">
                        <SelectValue placeholder={cameraOptions.length === 0 ? "No cameras" : "Select"} />
                      </SelectTrigger>
                      <SelectContent>
                        {cameraOptions.length === 0 ? (
                          <SelectItem value="none" disabled>No cameras available</SelectItem>
                        ) : (
                          cameraOptions.map((c: number, i: number) => (
                            <SelectItem key={i} value={String(c)}>{c}</SelectItem>
                          ))
                        )}
                      </SelectContent>
                    </Select>
                  </div>

                  {/* Improved toggle switches */}
                  <div className="flex items-center gap-4 ml-4">
                    <label className="inline-flex items-center gap-2 px-3 py-1.5 bg-white border border-gray-300 rounded-md hover:bg-gray-50 cursor-pointer transition-colors">
                      <input
                        type="checkbox"
                        checked={merged}
                        onChange={e => setMerged(e.target.checked)}
                        className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                      />
                      <span className="text-sm font-medium text-gray-700">Use Merged Data</span>
                    </label>

                    <label className="inline-flex items-center gap-2 px-3 py-1.5 bg-white border border-gray-300 rounded-md hover:bg-gray-50 cursor-pointer transition-colors">
                      <input
                        type="checkbox"
                        checked={meanMode}
                        onChange={() => { void toggleMeanMode(); }}
                        className="w-4 h-4 text-blue-600 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                      />
                      <span className="text-sm font-medium text-gray-700">Show Mean Statistics</span>
                      {statVarsLoading && <span className="text-xs text-gray-500">(loading...)</span>}
                    </label>
                  </div>
                </div>
              </div>
              
              {/* Better error message when mean stats don't exist */}
              {meanMode && statsError && (
                <div className="p-4 bg-amber-50 border-l-4 border-amber-400 rounded-r-lg">
                  <div className="flex items-start gap-3">
                    <svg className="w-5 h-5 text-amber-600 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                      <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                    </svg>
                    <div className="flex-1">
                      <h4 className="text-sm font-semibold text-amber-800 mb-1">Mean Statistics Not Available</h4>
                      <p className="text-sm text-amber-700">{statsError}</p>
                      <p className="text-sm text-amber-600 mt-2">
                        You need to calculate statistics first. Use the "Statistics Calculation" section below to get started.
                      </p>
                    </div>
                  </div>
                </div>
              )}

              {/* X and Y Offset inputs - Hidden in mean mode since they only apply to individual frames */}
              {!meanMode && (
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
              )}

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

              {/* Vector Merging Section - Only show if multiple cameras */}
              {cameraOptions.length > 1 && (
                <div className="p-3 bg-gradient-to-r from-green-50 to-emerald-50 border border-green-200 rounded-lg">
                  <button
                    onClick={() => setShowMergingDialog(!showMergingDialog)}
                    className="w-full flex items-center justify-between text-left"
                  >
                    <div className="flex items-center gap-2">
                      <h4 className="text-sm font-semibold text-green-900">Merge Vectors</h4>
                      <span className="text-xs text-green-700 bg-green-100 px-2 py-0.5 rounded">
                        Combine camera views
                      </span>
                    </div>
                    <svg 
                      className={`w-5 h-5 text-green-700 transition-transform ${showMergingDialog ? 'rotate-180' : ''}`}
                      fill="none" 
                      stroke="currentColor" 
                      viewBox="0 0 24 24"
                    >
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                    </svg>
                  </button>

                  {showMergingDialog && (
                    <div className="mt-3 space-y-3">
                      <p className="text-xs text-green-800">
                        Merge vector fields from multiple cameras into a single combined field.
                      </p>

                      {!mergingJobId && (
                        <div className="space-y-3">
                          <div>
                            <label className="text-xs font-medium text-gray-700 mb-1.5 block">Select Cameras to Merge:</label>
                            <div className="flex flex-wrap gap-2">
                              {cameraOptions.map((cam: number) => (
                                <label key={cam} className="inline-flex items-center gap-1.5 px-2.5 py-1.5 bg-white border border-gray-300 rounded-md hover:bg-gray-50 cursor-pointer transition-colors text-sm">
                                  <input
                                    type="checkbox"
                                    checked={selectedMergeCameras.includes(cam)}
                                    onChange={e => {
                                      if (e.target.checked) {
                                        setSelectedMergeCameras([...selectedMergeCameras, cam]);
                                      } else {
                                        setSelectedMergeCameras(selectedMergeCameras.filter(c => c !== cam));
                                      }
                                    }}
                                    className="w-3.5 h-3.5 text-green-600 border-gray-300 rounded focus:ring-green-500"
                                  />
                                  <span className="text-xs font-medium">Cam {cam}</span>
                                </label>
                              ))}
                            </div>
                          </div>

                          <div className="flex gap-2">
                            <Button
                              onClick={() => mergeVectors(index)}
                              disabled={merging || selectedMergeCameras.length < 2}
                              className="flex-1 bg-green-600 hover:bg-green-700 text-white text-xs py-2"
                            >
                              {merging ? "Merging..." : `Merge Frame ${index}`}
                            </Button>
                            <Button
                              onClick={() => mergeVectors()}
                              disabled={merging || selectedMergeCameras.length < 2}
                              className="flex-1 bg-green-700 hover:bg-green-800 text-white text-xs py-2"
                            >
                              {merging ? "Merging..." : `Merge All (${maxFrameCount})`}
                            </Button>
                          </div>
                          <p className="text-xs text-gray-600 italic">
                            Use "Merge Frame" to test on current frame, or "Merge All" to process all frames with multiprocessing.
                          </p>
                        </div>
                      )}

                      {mergingJobId && (
                        <div className="space-y-2">
                          <div className="flex items-center gap-2">
                            <div className="flex-1">
                              <div className="text-xs font-medium text-gray-700 mb-1">
                                {mergingStatus === "completed" ? "Merge Complete!" : 
                                 mergingStatus === "failed" ? "Merge Failed" :
                                 mergingStatus === "running" ? "Merging vectors..." : "Starting..."}
                              </div>
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div 
                                  className="bg-green-600 h-2 rounded-full transition-all duration-300" 
                                  style={{ width: `${mergingProgress}%` }}
                                />
                              </div>
                              <div className="text-xs text-gray-600 mt-1">
                                {mergingDetails?.processed_frames || 0} / {mergingDetails?.total_frames || 0} frames
                              </div>
                            </div>
                          </div>

                          {mergingStatus === "completed" && (
                            <div className="p-3 bg-green-50 border-l-4 border-green-500 rounded-r">
                              <div className="flex items-start gap-2">
                                <svg className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                  <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                                </svg>
                                <div>
                                  <h5 className="text-xs font-semibold text-green-800">Merge Successful</h5>
                                  <p className="text-xs text-green-700 mt-0.5">
                                    Vectors merged successfully. Enable "Use Merged Data" to view the results.
                                  </p>
                                </div>
                              </div>
                            </div>
                          )}

                          {mergingStatus === "failed" && mergingError && (
                            <div className="space-y-2">
                              <div className="p-3 bg-red-50 border-l-4 border-red-500 rounded-r">
                                <div className="flex items-start gap-2">
                                  <svg className="w-4 h-4 text-red-600 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                                  </svg>
                                  <div>
                                    <h5 className="text-xs font-semibold text-red-800">Merge Failed</h5>
                                    <p className="text-xs text-red-700 mt-0.5">{mergingError}</p>
                                  </div>
                                </div>
                              </div>
                              <Button
                                size="sm"
                                onClick={resetMerging}
                                variant="outline"
                                className="w-full text-xs"
                              >
                                Try Again
                              </Button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Statistics Calculation Section - Collapsible */}
              <div className="p-3 bg-gradient-to-r from-blue-50 to-indigo-50 border border-blue-200 rounded-lg">
                <button
                  onClick={() => setShowStatisticsDialog(!showStatisticsDialog)}
                  className="w-full flex items-center justify-between text-left"
                >
                  <div className="flex items-center gap-2">
                    <h4 className="text-sm font-semibold text-blue-900">Statistics Calculation</h4>
                    <span className="text-xs text-blue-700 bg-blue-100 px-2 py-0.5 rounded">
                      Required for mean statistics
                    </span>
                  </div>
                  <svg 
                    className={`w-5 h-5 text-blue-700 transition-transform ${showStatisticsDialog ? 'rotate-180' : ''}`}
                    fill="none" 
                    stroke="currentColor" 
                    viewBox="0 0 24 24"
                  >
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>

                {showStatisticsDialog && (
                  <div className="mt-3 space-y-3">
                    <p className="text-xs text-blue-800">
                      Calculate mean velocities and Reynolds stresses across all frames.
                    </p>

                    {!statisticsJobId && (
                      <div className="space-y-3">
                        <div>
                          <label className="text-xs font-medium text-gray-700 mb-1.5 block">Select Cameras:</label>
                          <div className="flex flex-wrap gap-2">
                            {cameraOptions.map((cam: string) => (
                              <label key={cam} className="inline-flex items-center gap-1.5 px-2.5 py-1.5 bg-white border border-gray-300 rounded-md hover:bg-gray-50 cursor-pointer transition-colors text-sm">
                                <input
                                  type="checkbox"
                                  checked={selectedCameras.includes(cam)}
                                  onChange={e => {
                                    if (e.target.checked) {
                                      setSelectedCameras([...selectedCameras, cam]);
                                    } else {
                                      setSelectedCameras(selectedCameras.filter(c => c !== cam));
                                    }
                                  }}
                                  className="w-3.5 h-3.5 text-blue-600 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                                />
                                <span className="font-medium">{cam}</span>
                              </label>
                            ))}
                          </div>
                        </div>

                        <label className="inline-flex items-center gap-2 px-2.5 py-1.5 bg-white border border-gray-300 rounded-md hover:bg-gray-50 cursor-pointer transition-colors text-sm">
                          <input
                            type="checkbox"
                            checked={includeMerged}
                            onChange={e => setIncludeMerged(e.target.checked)}
                            className="w-3.5 h-3.5 text-blue-600 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                          />
                          <span className="font-medium">Include Merged Data</span>
                          <span className="text-xs text-gray-500">(if available)</span>
                        </label>

                        <Button
                          onClick={calculateStatistics}
                          disabled={calculating || (selectedCameras.length === 0 && !includeMerged)}
                          className="w-full bg-green-600 hover:bg-green-700 text-white"
                          size="sm"
                        >
                          {calculating ? "Starting Calculation..." : "Calculate Statistics"}
                        </Button>
                      </div>
                    )}

                    {/* Progress display */}
                    {statisticsJobId && statisticsStatus !== "completed" && statisticsStatus !== "failed" && (
                      <div className="space-y-2">
                        <div className="flex items-center justify-between text-xs">
                          <span className="font-medium text-gray-700">Status:</span>
                          <span className={`font-medium ${statisticsStatus === "running" ? "text-blue-600" : "text-gray-600"}`}>
                            {statisticsStatus === "running" ? "Processing..." : statisticsStatus}
                          </span>
                        </div>
                        
                        <div className="space-y-1">
                          <div className="w-full bg-gray-200 h-2 rounded-full overflow-hidden">
                            <div 
                              className="h-2 bg-gradient-to-r from-blue-500 to-blue-600 transition-all duration-300" 
                              style={{ width: `${statisticsProgress}%` }}
                            ></div>
                          </div>
                          <p className="text-xs text-gray-600 text-right">
                            {statisticsProgress.toFixed(1)}% complete
                          </p>
                        </div>

                        {statisticsDetails?.camera && (
                          <p className="text-xs text-gray-600">
                            Processing: <span className="font-medium">{statisticsDetails.camera}</span>
                          </p>
                        )}
                      </div>
                    )}

                    {statisticsStatus === "completed" && (
                      <div className="space-y-2">
                        <div className="p-3 bg-green-50 border-l-4 border-green-500 rounded-r">
                          <div className="flex items-start gap-2">
                            <svg className="w-4 h-4 text-green-600 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z" clipRule="evenodd" />
                            </svg>
                            <div>
                              <h5 className="text-xs font-semibold text-green-800">Statistics Calculated!</h5>
                              <p className="text-xs text-green-700 mt-0.5">
                                Enable "Show Mean Statistics" above to view them.
                              </p>
                            </div>
                          </div>
                        </div>
                        <Button
                          size="sm"
                          onClick={resetStatistics}
                          variant="outline"
                          className="w-full text-xs"
                        >
                          Calculate New Statistics
                        </Button>
                      </div>
                    )}

                    {statisticsStatus === "failed" && statisticsError && (
                      <div className="space-y-2">
                        <div className="p-3 bg-red-50 border-l-4 border-red-500 rounded-r">
                          <div className="flex items-start gap-2">
                            <svg className="w-4 h-4 text-red-600 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" clipRule="evenodd" />
                            </svg>
                            <div>
                              <h5 className="text-xs font-semibold text-red-800">Calculation Failed</h5>
                              <p className="text-xs text-red-700 mt-0.5">{statisticsError}</p>
                            </div>
                          </div>
                        </div>
                        <Button
                          size="sm"
                          onClick={resetStatistics}
                          variant="outline"
                          className="w-full text-xs"
                        >
                          Try Again
                        </Button>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Visualization Controls - Now works for both regular and mean statistics */}
              <div className="p-3 bg-white border border-gray-200 rounded-lg">
                <h4 className="text-sm font-semibold text-gray-700 mb-3">Visualization Settings</h4>
                <div className="flex items-center gap-4 mb-3 flex-wrap">
                  <div className="flex items-center gap-2">
                    <label htmlFor="type" className="text-sm font-medium text-gray-700">Variable:</label>
                    <Select value={type} onValueChange={v => setType(v)}>
                      <SelectTrigger id="type" className="w-32">
                        <SelectValue placeholder="Select" />
                      </SelectTrigger>
                      <SelectContent>
                        {meanMode ? (
                          statVarsLoading ? (
                            <SelectItem value="loading">Loading...</SelectItem>
                          ) : statVars && statVars.length > 0 ? (
                            statVars.map(v => <SelectItem key={v} value={v}>{v}</SelectItem>)
                          ) : (
                            <SelectItem value="none" disabled>No variables</SelectItem>
                          )
                        ) : (
                          frameVarsLoading ? (
                            <SelectItem value="loading">Loading...</SelectItem>
                          ) : frameVars && frameVars.length > 0 ? (
                            frameVars.map(v => <SelectItem key={v} value={v}>{v}</SelectItem>)
                          ) : (
                            <>
                              <SelectItem value="ux">ux</SelectItem>
                              <SelectItem value="uy">uy</SelectItem>
                            </>
                          )
                        )}
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex items-center gap-2">
                    <label htmlFor="cmap" className="text-sm font-medium text-gray-700">Colormap:</label>
                    <Select value={cmap} onValueChange={v => setCmap(v)}>
                      <SelectTrigger id="cmap" className="w-32">
                        <SelectValue placeholder="Select" />
                      </SelectTrigger>
                      <SelectContent>
                        <SelectItem value="default">default</SelectItem>
                        <SelectItem value="viridis">viridis</SelectItem>
                        <SelectItem value="plasma">plasma</SelectItem>
                        <SelectItem value="inferno">inferno</SelectItem>
                        <SelectItem value="magma">magma</SelectItem>
                        <SelectItem value="cividis">cividis</SelectItem>
                        <SelectItem value="jet">jet</SelectItem>
                        <SelectItem value="gray">gray</SelectItem>
                        <SelectItem value="bone">bone</SelectItem>
                        <SelectItem value="copper">copper</SelectItem>
                        <SelectItem value="pink">pink</SelectItem>
                        <SelectItem value="spring">spring</SelectItem>
                        <SelectItem value="summer">summer</SelectItem>
                        <SelectItem value="autumn">autumn</SelectItem>
                        <SelectItem value="winter">winter</SelectItem>
                        <SelectItem value="hot">hot</SelectItem>
                        <SelectItem value="cool">cool</SelectItem>
                        <SelectItem value="Wistia">Wistia</SelectItem>
                        <SelectItem value="twilight">twilight</SelectItem>
                        <SelectItem value="hsv">hsv</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="flex items-center gap-2">
                    <label htmlFor="run" className="text-sm font-medium text-gray-700">Run:</label>
                    <Input id="run" type="number" min={1} value={run} onChange={e => setRun(Math.max(1, Number(e.target.value)))} className="w-20" />
                  </div>
                </div>

                <div className="flex items-center gap-4 flex-wrap">
                  <div className="flex items-center gap-2">
                    <label className="text-sm font-medium text-gray-700">Lower:</label>
                    <Input type="number" value={lower} onChange={e => setLower(e.target.value)} placeholder="auto" className="w-24" />
                  </div>
                  
                  <div className="flex items-center gap-2">
                    <label className="text-sm font-medium text-gray-700">Upper:</label>
                    <Input type="number" value={upper} onChange={e => setUpper(e.target.value)} placeholder="auto" className="w-24" />
                  </div>

                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => { void fetchLimits(); }}
                    disabled={limitsLoading}
                    className="text-xs"
                  >
                    {limitsLoading ? "Getting..." : "Auto-Calculate Limits"}
                  </Button>

                  <div className="ml-auto flex items-center gap-2 flex-wrap">
                    <Button
                      className="bg-blue-600 hover:bg-blue-700 text-white"
                      onClick={() => { void handleRender(); }}
                      disabled={loading || statsLoading || frameVarsLoading}
                    >
                      {(loading || statsLoading || frameVarsLoading) ? "Loading..." : "Render"}
                    </Button>

                    <Button
                      size="sm"
                      variant="outline"
                      onClick={downloadCurrentView}
                      disabled={!imageSrc || loading || statsLoading}
                    >
                      Download PNG
                    </Button>
                    
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => { void copyCurrentView(); }}
                      disabled={!imageSrc || loading || statsLoading}
                    >
                      Copy PNG
                    </Button>
                  </div>
                </div>
              </div>
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

            {/* Image Operations Row */}
            {imageSrc && !error && (
              <div className="p-4 bg-gradient-to-br from-blue-50 to-indigo-50 border-2 border-blue-200 rounded-lg mb-4 shadow-sm">
                <div className="flex flex-col gap-4">
                  {/* Prominent action buttons */}
                  <div className="flex items-center gap-3">
                    <Button
                      size="default"
                      onClick={() => applyTransformationToAllFrames(appliedTransforms)}
                      disabled={loading || appliedTransforms.length === 0}
                      className="flex items-center gap-2 bg-blue-600 hover:bg-blue-700 text-white font-semibold px-6 py-2 shadow-md"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-5 h-5" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 14h-5a2 2 0 0 1-2-2V7" />
                        <path d="M14 2L7 9l7 7" />
                      </svg>
                      Apply to All Frames
                    </Button>
                    <Button
                      size="default"
                      variant="destructive"
                      onClick={clearTransforms}
                      disabled={loading}
                      className="font-semibold px-6 py-2 shadow-md"
                    >
                      Clear Transforms
                    </Button>
                    {appliedTransforms.length > 0 && (
                      <span className="text-sm font-medium text-blue-700 bg-blue-100 px-3 py-1 rounded-full">
                        {appliedTransforms.length} transform{appliedTransforms.length !== 1 ? 's' : ''} applied
                      </span>
                    )}
                  </div>
                  
                  {/* Individual transform buttons */}
                  <div className="flex flex-wrap items-center gap-2 pt-3 border-t border-blue-200">
                    <label className="text-sm font-semibold text-gray-700 mr-2">Apply individually to current frame:</label>
                    <Button size="sm" variant="outline" onClick={() => applyTransformation('rotate_90_ccw')}>Rotate Left</Button>
                    <Button size="sm" variant="outline" onClick={() => applyTransformation('rotate_90_cw')}>Rotate Right</Button>
                    <Button size="sm" variant="outline" onClick={() => applyTransformation('flip_lr')}>Flip Horizontal</Button>
                    <Button size="sm" variant="outline" onClick={() => applyTransformation('flip_ud')}>Flip Vertical</Button>
                    <Button size="sm" variant="outline" onClick={() => applyTransformation('swap_ux_uy')}>Swap UX/UY</Button>
                    <Button size="sm" variant="outline" onClick={() => applyTransformation('invert_ux_uy')}>Invert UX/UY</Button>
                  </div>
                </div>
              </div>
            )}
            
            {/* Cursor position display - now under transforms */}
            {imageSrc && !error && (
              <div className="mb-4 p-3 bg-gradient-to-r from-gray-50 to-gray-100 border rounded-lg shadow-sm"
                   style={{ minHeight: 56, height: 56, display: 'flex', alignItems: 'center' }}>
                <div className="flex items-center justify-between w-full h-full">
                  <div className="flex items-center gap-6 h-full">
                    <div className="text-sm font-medium text-gray-700">
                      Cursor Position:
                    </div>
                    {/*
                      Show the current hover values if valid, otherwise fall back to the
                      last valid values (lastValidHover). If neither exists, show the hint.
                    */}
                    {(() => {
                      const isHoverValid = hoverData && typeof hoverData.x === "number" && !isNaN(hoverData.x) && hoverData.i >= 0;
                      const display = isHoverValid ? hoverData : lastValidHover;
                      if (!display) {
                        return (
                          <div className="text-sm text-gray-500 italic h-full flex items-center">
                            Hover over the plot area to see coordinates
                          </div>
                        );
                      }

                      let varVal: number | null = null;
                      if (type === "ux" && display.ux != null) varVal = display.ux;
                      else if (type === "uy" && display.uy != null) varVal = display.uy;
                      else if (display.value != null) varVal = display.value;

                      return (
                        <div className="flex items-center gap-6 h-full">
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-medium text-gray-600 uppercase tracking-wide">X:</span>
                            <span className="font-mono text-sm font-semibold text-soton-blue bg-white px-2 py-1 rounded border"
                                  style={{ minWidth: 70, textAlign: 'center', display: 'inline-block' }}>
                              {display.x.toFixed(3)}
                            </span>
                          </div>
                          <div className="flex items-center gap-2">
                            <span className="text-xs font-medium text-gray-600 uppercase tracking-wide">Y:</span>
                            <span className="font-mono text-sm font-semibold text-soton-blue bg-white px-2 py-1 rounded border"
                                  style={{ minWidth: 70, textAlign: 'center', display: 'inline-block' }}>
                              {display.y.toFixed(3)}
                            </span>
                          </div>
                          {varVal != null ? (
                            <div className="flex items-center gap-2">
                              <span className="text-xs font-medium text-gray-600 uppercase tracking-wide">
                                {type}:
                              </span>
                              <span className="font-mono text-sm font-semibold text-white bg-soton-blue px-2 py-1 rounded border"
                                    style={{ minWidth: 70, textAlign: 'center', display: 'inline-block' }}>
                                {varVal.toFixed(3)}
                              </span>
                            </div>
                          ) : null}
                        </div>
                      );
                    })()}
                  </div>
                  <div className="text-xs text-gray-500">
                    {meanMode ? "Mean Statistics Mode" : `Frame ${index}`}
                  </div>
                </div>
              </div>
            )}
              {transformationJob && (
                <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <div className="flex items-center justify-between mb-2">
                    <span className="text-sm font-medium text-blue-900">
                      Applying transformations to all frames...
                    </span>
                    <span className="text-sm text-blue-700">
                      {transformationJob.status === 'completed' ? 'Complete' :
                       transformationJob.status === 'failed' ? 'Failed' :
                       `${transformationJob.progress}%`}
                    </span>
                  </div>
                  <div className="w-full bg-blue-200 rounded-full h-2">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                      style={{ width: `${transformationJob.progress}%` }}
                    ></div>
                  </div>
                  <div className="mt-2 text-xs text-blue-700">
                    {transformationJob.processed_frames} / {transformationJob.total_frames} frames processed
                    {transformationJob.elapsed_time && (
                      <span className="ml-2">
                        ({Math.round(transformationJob.elapsed_time)}s elapsed)
                      </span>
                    )}
                    {transformationJob.estimated_remaining && transformationJob.status === 'running' && (
                      <span className="ml-2">
                        (~{Math.round(transformationJob.estimated_remaining)}s remaining)
                      </span>
                    )}
                  </div>
                  {transformationJob.error && (
                    <div className="mt-2 text-xs text-red-600">
                      Error: {transformationJob.error}
                    </div>
                  )}
                </div>
              )}


            {imageSrc && !error && (
              <div
                className="flex flex-col items-center relative"
                style={{
                  width: '100%',
                  maxWidth: '1100px',
                  margin: '0 auto',
                  cursor: datumMode ? 'crosshair' : magVisible ? 'none' : 'default'
                }}
                onMouseMove={e => { onMouseMove(e); handleMagnifierMove(e); }}
                onMouseLeave={e => { onMouseLeave(); handleMagnifierLeave(); }}
                onClick={e => { if (datumMode) handleImageClick(e); }}
              >
                {/* Left / Right frame increment arrows */}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setIndex(Math.max(1, index - 1))}
                  disabled={meanMode || index <= 1}
                  title="Previous frame"
                  aria-label="Previous frame"
                  className="absolute left-3 top-1/2 -translate-y-1/2 z-50 rounded-full p-2 bg-white bg-opacity-90 text-gray-700 border border-gray-200 hover:bg-gray-100"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
                    <path d="M12 4L6 10l6 6" />
                  </svg>
                </Button>

                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setIndex(Math.min(maxFrameCount, index + 1))}
                  disabled={meanMode || index >= maxFrameCount}
                  title="Next frame"
                  aria-label="Next frame"
                  className="absolute right-3 top-1/2 -translate-y-1/2 z-50 rounded-full p-2 bg-white bg-opacity-90 text-gray-700 border border-gray-200 hover:bg-gray-100"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true" focusable="false">
                    <path d="M8 4l6 6-6 6" />
                  </svg>
                </Button>

                {/* Remove magnifier toggle button */}
                {/* <div style={{ position: 'absolute', top: 8, right: 8, zIndex: 70 }}>
                  <button ...> ... </button>
                </div> */}
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
                    position: 'fixed',
                    pointerEvents: 'none',
                    zIndex: 9999,
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
              </div>
            )}

            {/* Frame slider and play button - Hidden in mean mode since it doesn't apply */}
            {maxFrameCount > 0 && !meanMode && (
              <div className="flex flex-col md:flex-row items-center justify-center gap-6 mt-6 p-4 bg-gray-50 border rounded-lg">
                {/* Camera selector */}
                <div className="flex items-center gap-2">
                  <label htmlFor="frame-camera" className="text-sm font-medium whitespace-nowrap">Camera:</label>
                  <Select value={String(camera)} onValueChange={v => setCamera(Number(v))} disabled={cameraOptions.length === 0}>
                    <SelectTrigger id="frame-camera" className="w-28">
                      <SelectValue placeholder={cameraOptions.length === 0 ? "No cameras" : "Select"} />
                    </SelectTrigger>
                    <SelectContent>
                      {cameraOptions.length === 0 ? (
                        <SelectItem value="none" disabled>No cameras available</SelectItem>
                      ) : (
                        cameraOptions.map((c: number, i: number) => (
                          <SelectItem key={i} value={String(c)}>{c}</SelectItem>
                        ))
                      )}
                    </SelectContent>
                  </Select>
                </div>

                {/* Frame slider + numeric selector */}
                <div className="flex items-center gap-3 flex-1">
                  <label htmlFor="frame-slider" className="text-sm font-medium whitespace-nowrap">Frame:</label>
                  <input
                    id="frame-slider"
                    type="range"
                    min={1}
                    max={maxFrameCount}
                    value={index}
                    onChange={e => setIndex(Number(e.target.value))}
                    className="flex-1 min-w-[200px]"
                  />
                  {/* Numeric input to directly set current frame */}
                  <Input
                    id="frame-input"
                    type="number"
                    min={1}
                    max={maxFrameCount}
                    value={index}
                    onChange={e => setIndex(Math.max(1, Math.min(maxFrameCount, Number(e.target.value || 1))))}
                    className="w-24"
                  />
                  <span className="text-xs text-gray-500 whitespace-nowrap">{index} / {maxFrameCount}</span>
                </div>

                {/* Play button */}
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
