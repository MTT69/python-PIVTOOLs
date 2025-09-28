"use client";

import React, { useEffect, useMemo, useRef, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Slider } from "@/components/ui/slider";

export default function VideoMaker({ backendUrl = '/backend', config }: { backendUrl?: string; config?: any }) {
  // Directory / base paths
  const [directory, setDirectory] = useState<string>('');
  const dirInputRef = useRef<HTMLInputElement | null>(null);
  const [basePaths, setBasePaths] = useState<string[]>(() => {
    try {
      return JSON.parse(typeof window !== 'undefined' ? localStorage.getItem('piv_base_paths') || '[]' : '[]');
    } catch {
      return [];
    }
  });
  const [basePathIdx, setBasePathIdx] = useState<number>(0);

  // Camera options derived from config (same logic as VectorViewer)
  const cameraOptions: string[] = useMemo(() => {
    const nFromPaths = config?.paths?.camera_numbers?.length ? Number(config.paths.camera_numbers[0]) : undefined;
    const nFromIm = config?.imProperties?.cameraCount ? Number(config.imProperties.cameraCount) : undefined;
    const n = (Number.isFinite(nFromPaths as number) && (nFromPaths as number) > 0)
      ? (nFromPaths as number)
      : (Number.isFinite(nFromIm as number) && (nFromIm as number) > 0) ? (nFromIm as number) : 1;
    const count = Number.isFinite(n) ? n : 1;
    return Array.from({ length: count }, (_, i) => `Cam${i + 1}`);
  }, [config]);

  const [camera, setCamera] = useState<string>(() => cameraOptions.length > 0 ? cameraOptions[0] : 'Cam1');
  useEffect(() => {
    if (cameraOptions.length === 0) return;
    if (!cameraOptions.includes(camera)) setCamera(cameraOptions[0]);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraOptions.length, cameraOptions[0]]);

  // Other selection state
  const [type, setType] = useState<string>('ux');
  const [cmap, setCmap] = useState<string>('default');
  const [run, setRun] = useState<number>(1);
  const [lower, setLower] = useState<string>('');
  const [upper, setUpper] = useState<string>('');
  const [merged, setMerged] = useState<boolean>(false);

  // Resolution settings (updated)
  const [resolution, setResolution] = useState<string>('1080p');
  const resolutionOptions = [
    { label: '1080p (1920x1080)', value: '1080p' },
    { label: '4K (3840x2160)', value: '4k' }
  ];

  // Video browser state
  const [activeTab, setActiveTab] = useState<string>('create');
  const [availableVideos, setAvailableVideos] = useState<string[]>([]);
  const [selectedVideo, setSelectedVideo] = useState<string>('');
  const [loadingVideos, setLoadingVideos] = useState<boolean>(false);

  // Add new state for handling the video creation process
  const [creating, setCreating] = useState<boolean>(false);
  const [videoResult, setVideoResult] = useState<{ success?: boolean; message?: string; out_path?: string } | null>(null);
  const [videoStatus, setVideoStatus] = useState<{
    processing: boolean;
    progress: number;
    status: string;
    message?: string;
    out_path?: string;
    computed_limits?: {
      lower: number;
      upper: number;
      actual_min: number;
      actual_max: number;
      percentile_based: boolean;
    };
  } | null>(null);

  // Add a delay before showing the video after completion
  const [videoReady, setVideoReady] = useState<boolean>(false);
  const [videoError, setVideoError] = useState<boolean>(false);

  // Effective directory: prefer selected base path if available
  const effectiveDir = useMemo(() => {
    if (basePaths.length > 0 && basePathIdx >= 0 && basePathIdx < basePaths.length) {
      return basePaths[basePathIdx];
    }
    return directory;
  }, [basePaths, basePathIdx, directory]);

  // Keep local directory text input in sync when basePaths change
  useEffect(() => {
    if (effectiveDir) setDirectory(effectiveDir);
  }, [effectiveDir]);

  // Directory picker (web fallback)
  const handleBrowse = () => {
    dirInputRef.current?.click();
  };
  const onDirPicked = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;
    const anyFile: any = files[0];
    const rel: string = anyFile.webkitRelativePath || '';
    const root = rel.split('/')[0] || '';
    let folderPath = root;
    if (anyFile.path && rel) {
      // In some electron/tauri contexts file.path contains full paths
      folderPath = anyFile.path.replace(/\\/g, '/').split('/' + rel)[0] || root;
    }
    setDirectory(folderPath);
    e.currentTarget.value = '';
  };

  // Fetch available videos when tab changes to browse
  useEffect(() => {
    if (activeTab === 'browse') {
      fetchAvailableVideos();
    }
  }, [activeTab, effectiveDir]);

  const fetchAvailableVideos = async () => {
    if (!effectiveDir) return;

    setLoadingVideos(true);
    setAvailableVideos([]);
    setSelectedVideo('');

    try {
      console.log(`Fetching videos from: ${effectiveDir}`);
      const response = await fetch(`${backendUrl}/video/list_videos?base_path=${encodeURIComponent(effectiveDir)}`);

      if (!response.ok) {
        throw new Error(`Server returned ${response.status}`);
      }

      const data = await response.json();
      console.log(`Found ${data.videos?.length || 0} videos`);

      if (data.videos?.length) {
        setAvailableVideos(data.videos);
        setSelectedVideo(data.videos[0]);
      } else {
        setAvailableVideos([]);
      }
    } catch (error) {
      console.error('Error fetching videos:', error);
      setAvailableVideos([]);
    } finally {
      setLoadingVideos(false);
    }
  };

  // Assemble params (but do not send anything yet)
  const buildParams = () => {
    const params: any = {
      base_path: effectiveDir,
      camera: camera.replace(/[^\d]/g, '') || '1',
      var: type,
      run: String(run),
      merged: merged ? '1' : '0',
      cmap,
      lower,
      upper,
      num_images: config?.images?.num_images || 0,
      resolution,
    };
    return params;
  };

  // Function to handle video creation
  const handleCreateVideo = async (isTest: boolean = false) => {
    setCreating(true);
    setVideoResult(null);
    setVideoStatus(null);

    try {
      const params = buildParams();
      if (isTest) {
        params.test_mode = true;
        params.test_frames = 50;
      }
      const url = `${backendUrl}/video/start_video`;

      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(params),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || 'Failed to start video creation');
      }

      // Start polling status immediately
      const pollStatus = () => {
        fetch(`${backendUrl}/video/video_status`)
          .then(res => res.json())
          .then(status => {
            setVideoStatus(status);
            if (status.processing) {
              setTimeout(pollStatus, 500); // Poll every 500ms for smoother progress
            } else {
              setCreating(false);
              if (status.error) {
                setVideoResult({
                  success: false,
                  message: status.error
                });
              } else {
                setVideoResult({
                  success: true,
                  message: isTest ? 'Test video created successfully!' : 'Video creation completed!',
                  out_path: status.out_path
                });
                // Refresh video list if we're in browse tab
                if (activeTab === 'browse') {
                  fetchAvailableVideos();
                }
              }
            }
          })
          .catch(err => {
            console.error('Polling error', err);
            setCreating(false);
            setVideoResult({
              success: false,
              message: 'Error polling status.'
            });
          });
      };
      pollStatus();
    } catch (error: any) {
      setVideoResult({
        success: false,
        message: `Error: ${error.message}`
      });
      setCreating(false);
    }
  };

  // Function to handle video cancellation
  const handleCancelVideo = async () => {
    try {
      const response = await fetch(`${backendUrl}/video/cancel_video`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });
      if (!response.ok) {
        throw new Error('Failed to cancel video');
      }
      setVideoStatus({ processing: false, progress: 0, status: 'canceled' });
      setVideoResult({
        success: false,
        message: 'Video creation canceled.'
      });
      setCreating(false);
    } catch (error: any) {
      console.error('Cancel error', error);
      setVideoResult({
        success: false,
        message: `Error canceling: ${error.message}`
      });
    }
  };

  // Helper to show just the last segment of a path
  const basename = (p: string) => {
    if (!p) return "";
    const parts = p.replace(/\\/g, "/").split("/");
    return parts.filter(Boolean).pop() || p;
  };

  // Function to safely create video URLs with proper encoding
  const createVideoUrl = (path: string) => {
    if (!path) return '';
    try {
      // Double-encode to handle special characters properly
      const encodedPath = encodeURIComponent(path);
      return `${backendUrl}/video/download?path=${encodedPath}`;
    } catch (error) {
      console.error('Error creating video URL:', error);
      return '';
    }
  };

  useEffect(() => {
    if (videoResult?.out_path) {
      setVideoReady(false);
      setVideoError(false);
      // Wait 2 seconds before showing video to allow backend to finish writing
      const timer = setTimeout(() => setVideoReady(true), 2000);
      return () => clearTimeout(timer);
    }
  }, [videoResult?.out_path]);

  const handleVideoError = () => {
    setVideoError(true);
  };

  const handleRetryVideo = () => {
    setVideoError(false);
    setVideoReady(false);
    setTimeout(() => setVideoReady(true), 1500); // shorter retry delay
  };

  return (
    <div className="space-y-6">
      <Card>
        <CardHeader>
          <CardTitle>Video Creation & Browsing</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex flex-col gap-4 mb-4">
            {/* Base path selection */}
            <div className="flex items-center gap-2">
              <label className="text-sm font-medium">Base Path:</label>
              {basePaths.length > 0 ? (
                <Select value={String(basePathIdx)} onValueChange={v => setBasePathIdx(Number(v))}>
                  <SelectTrigger id="basepath"><SelectValue placeholder="Pick base path" /></SelectTrigger>
                  <SelectContent>
                    {basePaths.map((p, i) => (
                      <SelectItem key={i} value={String(i)}>{basename(p)}</SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              ) : (
                <>
                  <Input
                    type="text"
                    value={directory}
                    onChange={(e) => setDirectory(e.target.value)}
                    placeholder="Select directory"
                    className="w-full"
                  />
                  <input
                    ref={dirInputRef}
                    type="file"
                    style={{ display: 'none' }}
                    onChange={onDirPicked}
                  />
                  <Button variant="outline" onClick={handleBrowse}>
                    Browse
                  </Button>
                </>
              )}
            </div>

            <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
              <TabsList className="w-full">
                <TabsTrigger value="create" className="flex-1">Create Video</TabsTrigger>
                <TabsTrigger value="browse" className="flex-1">Browse Videos</TabsTrigger>
              </TabsList>

              <TabsContent value="create" className="pt-4">
                {/* Camera selection and merged checkbox */}
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
                  {/* Merged Data checkbox */}
                  <label className="flex items-center gap-2 text-sm font-medium">
                    <input
                      type="checkbox"
                      checked={merged}
                      onChange={e => setMerged(e.target.checked)}
                      className="accent-soton-blue w-4 h-4 rounded border-gray-300"
                    />
                    Merged Data
                  </label>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Type (variable)</label>
                    <Select value={type} onValueChange={v => setType(v)}>
                      <SelectTrigger id="type"><SelectValue placeholder="Select type" /></SelectTrigger>
                      <SelectContent>
                        <SelectItem value="ux">ux</SelectItem>
                        <SelectItem value="uy">uy</SelectItem>
                        <SelectItem value="mag">mag</SelectItem>
                      </SelectContent>
                    </Select>
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Colormap</label>
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
                  </div>

                  <div className="space-y-2">
                    <label className="text-sm font-medium">Run</label>
                    <Input
                      type="number"
                      min={1}
                      value={run}
                      onChange={(e) => setRun(Number(e.target.value || 1))}
                    />
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Lower limit</label>
                    <Input
                      value={lower}
                      onChange={(e) => setLower(e.target.value)}
                      placeholder="auto"
                    />
                  </div>
                  <div className="space-y-2">
                    <label className="text-sm font-medium">Upper limit</label>
                    <Input
                      value={upper}
                      onChange={(e) => setUpper(e.target.value)}
                      placeholder="auto"
                    />
                  </div>
                </div>

                {/* Resolution settings */}
                <div className="space-y-2 mt-4">
                  <label className="text-sm font-medium">Resolution</label>
                  <Select value={resolution} onValueChange={setResolution}>
                    <SelectTrigger id="resolution">
                      <SelectValue placeholder="Select resolution" />
                    </SelectTrigger>
                    <SelectContent>
                      {resolutionOptions.map(option => (
                        <SelectItem key={option.value} value={option.value}>
                          {option.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>

                {/* Display result message */}
                {videoResult && (
                  <div className={`w-full p-3 mt-4 rounded border ${
                    videoResult.success
                      ? 'border-green-200 bg-green-50 text-green-700'
                      : 'border-red-200 bg-red-50 text-red-700'
                  } text-sm`}>
                    {videoResult.message}

                    {/* Display computed limits if available */}
                    {videoStatus?.computed_limits && (
                      <div className="mt-2 p-2 bg-white bg-opacity-50 rounded text-xs">
                        <div className="font-medium mb-1">Video Limits:</div>
                        <div>Lower: {videoStatus.computed_limits.lower.toFixed(3)} | Upper: {videoStatus.computed_limits.upper.toFixed(3)}</div>
                        <div>Data range: {videoStatus.computed_limits.actual_min.toFixed(3)} to {videoStatus.computed_limits.actual_max.toFixed(3)}</div>
                        {videoStatus.computed_limits.percentile_based && (
                          <div className="text-gray-600 italic">Limits auto-computed from 5th-95th percentiles</div>
                        )}
                      </div>
                    )}

                    {videoResult.out_path && videoReady && (
                      <div className="mt-2">
                        <video
                          controls
                          className="w-full rounded border"
                          style={{ maxHeight: 512 }}
                          src={createVideoUrl(videoResult.out_path)}
                          onError={handleVideoError}
                        >
                          Your browser does not support the video tag.
                        </video>
                        {videoError && (
                          <div className="p-2 bg-red-50 text-red-600 text-sm mt-1 rounded flex items-center gap-2">
                            Error loading video. The file might not be ready yet.
                            <Button variant="outline" size="sm" onClick={handleRetryVideo}>
                              Retry
                            </Button>
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                )}

                {/* Progress bar and status */}
                {videoStatus && videoStatus.processing && (
                  <div className="w-full space-y-2 mt-4">
                    <div className="flex items-center justify-between text-sm">
                      <span>Processing video...</span>
                      <span>{videoStatus.progress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className="bg-soton-blue h-2 rounded-full transition-all duration-300"
                        style={{ width: `${videoStatus.progress}%` }}
                      />
                    </div>
                    {videoStatus.message && (
                      <div className="text-xs text-gray-600">{videoStatus.message}</div>
                    )}
                  </div>
                )}

                <div className="pt-4 flex flex-col gap-2">
                  <div className="flex justify-end gap-2">
                    <Button
                      variant="outline"
                      onClick={handleCancelVideo}
                      disabled={!videoStatus?.processing}
                    >
                      Cancel
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => handleCreateVideo(true)}
                      disabled={creating || videoStatus?.processing}
                    >
                      Test Video (50 frames)
                    </Button>
                    <Button
                      className="bg-soton-blue"
                      onClick={() => handleCreateVideo(false)}
                      disabled={creating || videoStatus?.processing}
                    >
                      {creating ? "Starting..." : videoStatus?.processing ? "Processing..." : "Create Full Video"}
                    </Button>
                  </div>
                </div>
              </TabsContent>

              <TabsContent value="browse" className="pt-4">
                {loadingVideos ? (
                  <div className="text-center p-4">Loading available videos...</div>
                ) : availableVideos.length > 0 ? (
                  <div className="space-y-4">
                    <div className="space-y-2">
                      <label className="text-sm font-medium">Select Video</label>
                      <Select value={selectedVideo} onValueChange={setSelectedVideo}>
                        <SelectTrigger id="videoSelect">
                          <SelectValue placeholder="Select a video" />
                        </SelectTrigger>
                        <SelectContent>
                          {availableVideos.map((video, i) => (
                            <SelectItem key={i} value={video}>
                              {basename(video)}
                            </SelectItem>
                          ))}
                        </SelectContent>
                      </Select>
                    </div>

                    {selectedVideo && (
                      <div className="mt-4">
                        <div className="mb-2 text-xs text-gray-500 break-all">
                          Path: {selectedVideo}
                        </div>
                        <video
                          controls
                          className="w-full rounded border"
                          style={{ maxHeight: 512 }}
                          src={createVideoUrl(selectedVideo)}
                          onError={(e) => {
                            console.error('Video playback error:', e);
                            // Show error message when video fails to load
                            const target = e.target as HTMLVideoElement;
                            target.insertAdjacentHTML('afterend',
                              `<div class="p-2 bg-red-50 text-red-600 text-sm mt-1 rounded">
                                Error loading video. Please check the file path.
                              </div>`
                            );
                          }}
                        >
                          Your browser does not support the video tag.
                        </video>
                      </div>
                    )}

                    <div className="flex justify-end">
                      <Button
                        variant="outline"
                        onClick={fetchAvailableVideos}
                        className="mt-2"
                      >
                        Refresh Videos
                      </Button>
                    </div>
                  </div>
                ) : (
                  <div className="text-center p-4">
                    No videos found. Create a video first or select a different base path.
                  </div>
                )}
              </TabsContent>
            </Tabs>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
