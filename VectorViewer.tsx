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
  
  // Remember last valid hover values so we don't show a loading spinner
  // while backend updates arrive. Only update when hoverData contains valid coords.
  const [lastValidHover, setLastValidHover] = useState<any | null>(null);
  useEffect(() => {
    if (hoverData && typeof hoverData.x === "number" && !isNaN(hoverData.x) && hoverData.i >= 0) {
      setLastValidHover(hoverData);
    }
  }, [hoverData]);
;

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
              <div className="p-3 bg-gray-50 border rounded-md mb-4">
                <div className="flex flex-col gap-3">
                  <div className="flex items-center gap-2">
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => applyTransformationToAllFrames(appliedTransforms)}
                      disabled={loading || appliedTransforms.length === 0}
                      className="flex items-center gap-2"
                    >
                      <svg xmlns="http://www.w3.org/2000/svg" className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 14h-5a2 2 0 0 1-2-2V7" />
                        <path d="M14 2L7 9l7 7" />
                      </svg>
                      Apply to All Frames
                    </Button>
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={clearTransforms}
                      disabled={loading}
                    >
                      Clear Transforms
                    </Button>
                  </div>
                  <div className="flex flex-wrap items-center gap-2 pt-2 border-t">
                    <label className="text-sm font-medium mr-2">Apply individually to current frame:</label>
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
