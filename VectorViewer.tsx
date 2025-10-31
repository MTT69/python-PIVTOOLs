"use client";
import React, { useEffect, useState } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { useVectorViewer } from "@/hooks/useVectorViewer";

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
    cameraOptions,
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
    MAG_SIZE,
    dpr,
    rotation,
    setRotation,
    flipHorizontal,
    setFlipHorizontal,
    flipVertical,
    setFlipVertical,
    transpose,
    setTranspose,
    effectiveDir,  // Added: Import effectiveDir from hook
  } = useVectorViewer({ backendUrl, config });
  
  // Add cache-busting key that changes when transformations change
  const transformKey = `${rotation}-${flipHorizontal}-${flipVertical}-${transpose}`;
  
  // Add separate loading state for batch operations
  const [batchLoading, setBatchLoading] = useState(false);
  const [showBatchModal, setShowBatchModal] = useState(false);
  const [batchResult, setBatchResult] = useState<{
    success: boolean;
    processed?: number;
    failed?: number;
    output?: string;
    coordsTransformed?: boolean;
    error?: string;
  } | null>(null);
  
  // Handlers for image operations
  const handleRotateLeft = () => {
    setRotation((prev) => (prev + 3) % 4); // Counter-clockwise: add 3 (same as -1 mod 4)
  };

  const handleRotateRight = () => {
    setRotation((prev) => (prev + 1) % 4); // Clockwise
  };

  const handleFlipHorizontal = () => {
    setFlipHorizontal((prev) => !prev);
  };

  const handleFlipVertical = () => {
    setFlipVertical((prev) => !prev);
  };

  const handleTranspose = () => {
    setTranspose((prev) => !prev);
  };

  const resetTransformations = () => {
    setRotation(0);
    setFlipHorizontal(false);
    setFlipVertical(false);
    setTranspose(false);
  };

  // Handler for applying transformations to all frames
  const handleApplyToAll = async () => {
    if (!effectiveDir) {
      return;
    }

    setShowBatchModal(true);
    setBatchResult(null);
    setBatchLoading(true);

    try {
      const body = {
        base_path: effectiveDir,
        camera: parseInt(camera.replace(/[^\d]/g, "") || "1"),
        frame_start: 1,
        frame_end: maxFrameCount,
        rotation: rotation,
        flip_horizontal: flipHorizontal,
        flip_vertical: flipVertical,
        transpose: transpose,
        output_mode: "new_dir",
        type_name: "instantaneous",
        merged: merged,
        use_uncalibrated: false
      };

      const url = `${backendUrl}/plot/apply_transformations_batch`;
      const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      const json = await res.json();
      
      if (!res.ok) {
        throw new Error(json.error || "Failed to apply transformations");
      }

      setBatchResult({
        success: true,
        processed: json.processed_count,
        failed: json.failed_count,
        output: json.output_directory,
        coordsTransformed: json.coordinates_transformed,
      });
      
    } catch (e: any) {
      setBatchResult({
        success: false,
        error: e.message,
      });
    } finally {
      setBatchLoading(false);
    }
  };

  const closeBatchModal = () => {
    setShowBatchModal(false);
    setBatchResult(null);
  };

  // Auto re-render when transformations change (if already rendered)
  useEffect(() => {
    if (imageSrc && (rotation !== 0 || flipHorizontal || flipVertical || transpose)) {
      void handleRender();
    }
  }, [rotation, flipHorizontal, flipVertical, transpose]);

  // Remember last valid hover values so we don't show a loading spinner
  // while backend updates arrive. Only update when hoverData contains valid coords.
  const [lastValidHover, setLastValidHover] = useState<any | null>(null);
  useEffect(() => {
    if (hoverData && typeof hoverData.x === "number" && !isNaN(hoverData.x) && hoverData.i >= 0) {
      setLastValidHover(hoverData);
    }
  }, [hoverData]);

  return (
    <div className="space-y-6">
      {/* Batch Transformation Modal */}
      {showBatchModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-[100]">
          <div className="bg-white rounded-lg shadow-2xl p-6 max-w-md w-full mx-4">
            {batchLoading ? (
              <>
                <div className="flex flex-col items-center gap-4">
                  <div className="relative">
                    <div className="w-16 h-16 border-4 border-gray-200 rounded-full"></div>
                    <div className="w-16 h-16 border-4 border-soton-blue border-t-transparent rounded-full animate-spin absolute top-0 left-0"></div>
                  </div>
                  <h3 className="text-lg font-semibold text-gray-900">
                    Applying Transformations
                  </h3>
                  <div className="text-center text-sm text-gray-600">
                    <p>Processing {maxFrameCount} frames...</p>
                    <p className="mt-2">
                      Rotation: {rotation * 90}°
                      {flipHorizontal && " • Flip H"}
                      {flipVertical && " • Flip V"}
                      {transpose && " • Transpose"}
                    </p>
                  </div>
                </div>
              </>
            ) : batchResult ? (
              <>
                {batchResult.success ? (
                  <div className="flex flex-col items-center gap-4">
                    <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-8 h-8 text-green-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <polyline points="20 6 9 17 4 12" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      Transformation Complete!
                    </h3>
                    <div className="w-full space-y-2 text-sm">
                      <div className="flex justify-between py-2 border-b">
                        <span className="text-gray-600">Processed:</span>
                        <span className="font-semibold text-green-600">{batchResult.processed} frames</span>
                      </div>
                      {batchResult.failed !== undefined && batchResult.failed > 0 && (
                        <div className="flex justify-between py-2 border-b">
                          <span className="text-gray-600">Failed:</span>
                          <span className="font-semibold text-red-600">{batchResult.failed} frames</span>
                        </div>
                      )}
                      {batchResult.output && (
                        <div className="py-2 border-b">
                          <span className="text-gray-600 block mb-1">Output Directory:</span>
                          <span className="font-mono text-xs bg-gray-100 px-2 py-1 rounded block break-all">
                            {batchResult.output}
                          </span>
                        </div>
                      )}
                      <div className="flex justify-between py-2">
                        <span className="text-gray-600">Coordinates Transformed:</span>
                        <span className="font-semibold">{batchResult.coordsTransformed ? "Yes" : "No"}</span>
                      </div>
                    </div>
                    <Button
                      onClick={closeBatchModal}
                      className="w-full bg-soton-blue hover:bg-soton-blue/90 mt-4"
                    >
                      Close
                    </Button>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-4">
                    <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center">
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-8 h-8 text-red-600" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="10" />
                        <line x1="15" y1="9" x2="9" y2="15" />
                        <line x1="9" y1="9" x2="15" y2="15" />
                      </svg>
                    </div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      Transformation Failed
                    </h3>
                    <p className="text-sm text-gray-600 text-center">
                      {batchResult.error}
                    </p>
                    <Button
                      onClick={closeBatchModal}
                      variant="outline"
                      className="w-full mt-4"
                    >
                      Close
                    </Button>
                  </div>
                )}
              </>
            ) : null}
          </div>
        </div>
      )}

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
                <Select value={String(basePathIdx)} onValueChange={v => setBasePathIdx(Number(v))}>
                  <SelectTrigger id="basepath"><SelectValue placeholder="Pick base path" /></SelectTrigger>
                  <SelectContent>
                    {basePaths.map((p, i) => (
                      <SelectItem key={i} value={String(i)}>{basename(p)}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              {/* Camera and merged controls */}
              <div className="flex items-center gap-4">
                <label htmlFor="camera" className="text-sm font-medium">Camera:</label>
                <Select value={camera} onValueChange={v => setCamera(v)}>
                  <SelectTrigger id="camera"><SelectValue placeholder="Select camera" /></SelectTrigger>
                  <SelectContent>
                    {cameraOptions.map((c, i) => (
                      <SelectItem key={i} value={c}>{c}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>

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

              <div className={`flex items-center gap-4 mb-4 flex-wrap transition-opacity duration-200 ${meanMode ? "opacity-40 pointer-events-none" : ""}`}>
                {/* Type/colormap/run/limits/render/export controls */}
                <label htmlFor="type" className="text-sm font-medium">Type:</label>
                <Select value={type} onValueChange={v => setType(v)}>
                  <SelectTrigger id="type"><SelectValue placeholder="Select type" /></SelectTrigger>
                  <SelectContent>
                    {meanMode ? (
                      statVarsLoading ? (
                        <SelectItem value="loading">Loading...</SelectItem>
                      ) : statVars && statVars.length > 0 ? (
                        statVars.map(v => <SelectItem key={v} value={v}>{v}</SelectItem>)
                      ) : (
                        <SelectItem value="none" disabled>No vars</SelectItem>
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
                <label htmlFor="cmap" className="text-sm font-medium">Colormap:</label>
                <Select value={cmap} onValueChange={v => setCmap(v)}>
                  <SelectTrigger id="cmap"><SelectValue placeholder="Select colormap" /></SelectTrigger>
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
                  disabled={limitsLoading || meanMode}
                  className="ml-2"
                >
                  {limitsLoading ? "Getting..." : "Get Limits"}
                </Button>

                {/* Group render + export buttons so they don't overflow; allow wrapping on small screens */}
                <div className="flex items-center gap-2 flex-wrap ml-2">
                  <Button
                    className="bg-soton-blue flex-shrink-0"
                    onClick={() => { void handleRender(); }}
                    disabled={loading || statsLoading || frameVarsLoading}
                  >
                    {(loading || statsLoading || frameVarsLoading) ? "Loading..." : "Render"}
                  </Button>

                  <div className="flex items-center gap-2 flex-wrap">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={downloadCurrentView}
                      disabled={!imageSrc || loading || statsLoading}
                      className="flex-shrink-0"
                    >
                      Download PNG
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => { void copyCurrentView(); }}
                      disabled={!imageSrc || loading || statsLoading}
                      className="flex-shrink-0"
                    >
                      Copy PNG
                    </Button>
                  </div>
                </div>
              </div>

              {statsError && meanMode && (
                <div className="w-full p-3 mb-3 rounded border border-red-200 bg-red-50 text-red-700 text-sm">
                  {statsError}
                </div>
              )}
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
            
            {/* Integrated status bar for coordinate information */}
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

            {/* Image Operations Row */}
            {imageSrc && !error && (
              <div className="flex items-center gap-2 p-3 bg-gray-50 border rounded-md mb-4">
                <label className="text-sm font-medium mr-2">Image Operations:</label>
                
                {/* Rotate Left */}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleRotateLeft}
                  title="Rotate 90° Counter-Clockwise"
                  className="flex items-center gap-1.5"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M2.5 2v6h6M2.66 15.57a10 10 0 1 0 .57-8.38" />
                  </svg>
                  <span>Rotate Left</span>
                </Button>

                {/* Rotate Right */}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleRotateRight}
                  title="Rotate 90° Clockwise"
                  className="flex items-center gap-1.5"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38" />
                  </svg>
                  <span>Rotate Right</span>
                </Button>

                {/* Flip Horizontal */}
                <Button
                  size="sm"
                  variant={flipHorizontal ? "default" : "outline"}
                  onClick={handleFlipHorizontal}
                  title="Flip Horizontally"
                  className="flex items-center gap-1.5"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M8 3H5a2 2 0 0 0-2 2v14c0 1.1.9 2 2 2h3" />
                    <path d="M16 3h3a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2h-3" />
                    <path d="M12 20v2" />
                    <path d="M12 14v2" />
                    <path d="M12 8v2" />
                    <path d="M12 2v2" />
                  </svg>
                  <span>Flip Horizontal</span>
                </Button>

                {/* Flip Vertical */}
                <Button
                  size="sm"
                  variant={flipVertical ? "default" : "outline"}
                  onClick={handleFlipVertical}
                  title="Flip Vertically"
                  className="flex items-center gap-1.5"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 8V5a2 2 0 0 0-2-2H5a2 2 0 0 0-2 2v3" />
                    <path d="M21 16v3a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-3" />
                    <path d="M4 12H2" />
                    <path d="M10 12H8" />
                    <path d="M16 12h-2" />
                    <path d="M22 12h-2" />
                  </svg>
                  <span>Flip Vertical</span>
                </Button>

                {/* Transpose */}
                <Button
                  size="sm"
                  variant={transpose ? "default" : "outline"}
                  onClick={handleTranspose}
                  title="Transpose (Swap X and Y axes)"
                  className="flex items-center gap-1.5"
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M21 14h-5a2 2 0 0 1-2-2V7" />
                    <path d="M14 2L7 9l7 7" />
                  </svg>
                  <span>Transpose</span>
                </Button>

                {/* Reset transformations */}
                {(rotation !== 0 || flipHorizontal || flipVertical || transpose) && (
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={resetTransformations}
                    title="Reset all transformations"
                    className="flex items-center gap-1.5 ml-2 text-red-600 hover:text-red-700 hover:bg-red-50"
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8" />
                      <path d="M3 3v5h5" />
                    </svg>
                    <span>Reset</span>
                  </Button>
                )}

                {/* Spacer to push Apply to All to the right */}
                <div className="flex-grow"></div>

                {/* Apply to All button */}
                <Button
                  size="sm"
                  variant="outline"
                  onClick={handleApplyToAll}
                  title="Apply current transformations to all frames"
                  className="flex items-center gap-1.5 ml-auto"
                  disabled={(rotation === 0 && !flipHorizontal && !flipVertical && !transpose) || batchLoading || loading}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M3 7V5a2 2 0 0 1 2-2h2" />
                    <path d="M17 3h2a2 2 0 0 1 2 2v2" />
                    <path d="M21 17v2a2 2 0 0 1-2 2h-2" />
                    <path d="M7 21H5a2 2 0 0 1-2-2v-2" />
                    <rect x="7" y="7" width="10" height="10" rx="1" />
                  </svg>
                  <span>Apply to All</span>
                </Button>
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
                  key={transformKey} // Force re-render when transformations change
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
              </div>
            )}

            {/* Frame slider and play button: keep visible whenever we know maxFrameCount */}
            {maxFrameCount > 0 && (
              <div className={`flex flex-col md:flex-row items-center justify-center gap-4 mt-6 transition-opacity duration-200 ${meanMode ? "opacity-40 pointer-events-none" : ""}`}>
                {/* Frame slider + numeric selector */}
                <label htmlFor="frame-slider" className="text-sm font-medium">Frame:</label>
                <div className="flex items-center gap-3">
                  <input
                    id="frame-slider"
                    type="range"
                    min={1}
                    max={maxFrameCount}
                    value={index}
                    onChange={e => setIndex(Number(e.target.value))}
                    className="w-64"
                    disabled={meanMode}
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
                    disabled={meanMode}
                  />
                  <span className="text-xs text-gray-500">{index} / {maxFrameCount}</span>
                </div>

                <div className="flex items-center gap-2">
                  <Button
                    size="sm"
                    variant={playing ? "default" : "outline"}
                    onClick={() => { handlePlayToggle(); }}
                    className="flex items-center gap-1"
                    disabled={meanMode}
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
