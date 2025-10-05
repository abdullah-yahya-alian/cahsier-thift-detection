
import React, { useState, useRef, useEffect, useCallback } from 'react';
import type { PoseLandmarkerResult } from '../types';
import { CameraIcon, UploadIcon, ExportIcon, StopIcon } from './Icons';

type Mode = 'live' | 'video';
type FacingMode = 'user' | 'environment';

const PoseAnalyzer: React.FC = () => {
    const [mode, setMode] = useState<Mode>('live');
    const [isLoadingModel, setIsLoadingModel] = useState(true);
    const [isWebcamOn, setIsWebcamOn] = useState(false);
    const [isVideoLoaded, setIsVideoLoaded] = useState(false);
    const [isExporting, setIsExporting] = useState(false);
    const [statusText, setStatusText] = useState("Loading Pose Landmarker model...");
    const [cameraFacing, setCameraFacing] = useState<FacingMode>('environment');

    const videoRef = useRef<HTMLVideoElement>(null);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const poseLandmarkerRef = useRef<any>(null); // Using 'any' for PoseLandmarker from CDN
    const animationFrameIdRef = useRef<number | null>(null);
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const recordedChunksRef = useRef<Blob[]>([]);

    const drawingUtilsRef = useRef<any>(null);


    useEffect(() => {
        const initPoseLandmarker = async () => {
            try {
                const vision = await window.mp.tasks.vision.FilesetResolver.forVisionTasks(
                    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"
                );
                poseLandmarkerRef.current = await window.mp.tasks.vision.PoseLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task`,
                        delegate: "GPU",
                    },
                    runningMode: "VIDEO",
                    numPoses: 2,
                    minPoseDetectionConfidence: 0.5,
                    minPosePresenceConfidence: 0.5,
                    minTrackingConfidence: 0.5,
                });
                drawingUtilsRef.current = new window.mp.tasks.vision.DrawingUtils(canvasRef.current?.getContext("2d")!);
                setIsLoadingModel(false);
                setStatusText("Model loaded. Select a mode to start.");
            } catch (error) {
                console.error("Error loading Pose Landmarker model:", error);
                setStatusText("Failed to load model. Please refresh the page.");
            }
        };
        
        // Poll for the MediaPipe library to be available on the window object.
        const checkInterval = setInterval(() => {
            if (window.mp && window.mp.tasks && window.mp.tasks.vision) {
                clearInterval(checkInterval);
                initPoseLandmarker();
            }
        }, 100);

        // Fail after 20 seconds if the library isn't loaded (allows for slow networks/CDNs).
        const timeout = setTimeout(() => {
             clearInterval(checkInterval);
             if (!poseLandmarkerRef.current) {
                setStatusText("Failed to load MediaPipe library. Please refresh.");
                console.error("MediaPipe library not found on window object after timeout.");
             }
        }, 20000);

        return () => {
            clearInterval(checkInterval);
            clearTimeout(timeout);
            if (animationFrameIdRef.current) {
                cancelAnimationFrame(animationFrameIdRef.current);
            }
            if (poseLandmarkerRef.current) {
                poseLandmarkerRef.current.close();
            }
        };
    }, []);

    const predictLoop = useCallback(() => {
        if (!poseLandmarkerRef.current || !videoRef.current || !canvasRef.current || !window.mp?.tasks?.vision) return;

        const video = videoRef.current;
        if (video.paused || video.ended) {
            if (isExporting && mediaRecorderRef.current?.state === "recording") {
                mediaRecorderRef.current.stop();
            }
            return;
        }

        const canvas = canvasRef.current;
        const canvasCtx = canvas.getContext('2d');
        if (!canvasCtx) return;

        if (video.videoWidth > 0 && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }

        let startTimeMs = performance.now();
        const results: PoseLandmarkerResult = poseLandmarkerRef.current.detectForVideo(video, startTimeMs);

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);

        if (isExporting) {
            canvasCtx.drawImage(video, 0, 0, canvas.width, canvas.height);
        }

        if (results.landmarks) {
            for (const landmarks of results.landmarks) {
                drawingUtilsRef.current.drawLandmarks(landmarks, {
                    radius: (data: any) => window.mp.tasks.vision.DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1),
                     color: '#4ade80', // green-400
                });
                drawingUtilsRef.current.drawConnectors(landmarks, window.mp.tasks.vision.PoseLandmarker.POSE_CONNECTIONS, {
                    color: '#22d3ee', // cyan-400
                });
            }
        }
        canvasCtx.restore();

        animationFrameIdRef.current = requestAnimationFrame(predictLoop);
    }, [isExporting]);


    const startWebcamWithFacing = async (facing: FacingMode) => {
        if (!poseLandmarkerRef.current) return;
        setIsVideoLoaded(false);
        try {
            const stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    width: 1280,
                    height: 720,
                    facingMode: { ideal: facing },
                },
                audio: false,
            });
            videoRef.current!.srcObject = stream;
            videoRef.current!.addEventListener('loadeddata', () => {
                videoRef.current?.play();
                setIsWebcamOn(true);
                setStatusText(`Analyzing live webcam feed (${facing === 'user' ? 'front' : 'back'})...`);
                predictLoop();
            }, { once: true });
        } catch (err) {
            console.error("Error accessing webcam:", err);
            setStatusText("Could not access webcam. Please check permissions.");
        }
    };

    const stopWebcam = () => {
        if (animationFrameIdRef.current) cancelAnimationFrame(animationFrameIdRef.current);
        const stream = videoRef.current?.srcObject as MediaStream;
        stream?.getTracks().forEach(track => track.stop());
        if (videoRef.current) {
            videoRef.current.srcObject = null;
        }
        setIsWebcamOn(false);
        setStatusText("Webcam stopped.");
    };

    const toggleWebcam = async () => {
        if (isWebcamOn) {
            stopWebcam();
        } else {
            await startWebcamWithFacing(cameraFacing);
        }
    };

    const switchCameraFacing = async (nextFacing: FacingMode) => {
        setCameraFacing(nextFacing);
        if (isWebcamOn) {
            stopWebcam();
            await startWebcamWithFacing(nextFacing);
        }
    };
    
    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            if (isWebcamOn) {
                const stream = videoRef.current?.srcObject as MediaStream;
                stream?.getTracks().forEach(track => track.stop());
                videoRef.current!.srcObject = null;
                setIsWebcamOn(false);
            }
             if (animationFrameIdRef.current) {
                cancelAnimationFrame(animationFrameIdRef.current);
            }
            const url = URL.createObjectURL(file);
            videoRef.current!.src = url;
            videoRef.current!.addEventListener('loadeddata', () => {
                setIsVideoLoaded(true);
                videoRef.current?.play();
                setStatusText("Analyzing uploaded video...");
                predictLoop();
            });
        }
    };

    const handleExport = () => {
        if (!isVideoLoaded && !isWebcamOn) {
            setStatusText("Please start webcam or upload a video to export.");
            return;
        }
        if (isExporting) return;
        
        setIsExporting(true);
        setStatusText("Exporting video... please wait.");
        recordedChunksRef.current = [];
        
        if (videoRef.current) {
            videoRef.current.currentTime = 0;
            videoRef.current.play();
        }

        const stream = canvasRef.current!.captureStream(30);
        mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'video/webm; codecs=vp9' });

        mediaRecorderRef.current.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunksRef.current.push(event.data);
            }
        };

        mediaRecorderRef.current.onstop = () => {
            const blob = new Blob(recordedChunksRef.current, { type: 'video/webm' });
            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'pose_analysis.webm';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
            setIsExporting(false);
            setStatusText("Export complete!");
        };

        mediaRecorderRef.current.start();
    };


    const renderContent = () => {
        const videoVisible = isWebcamOn || isVideoLoaded;
        return (
            <div className="relative w-full max-w-4xl mx-auto aspect-video bg-black rounded-lg overflow-hidden shadow-2xl shadow-cyan-500/10 border-2 border-gray-700">
                 {!videoVisible && (
                    <div className="absolute inset-0 flex items-center justify-center">
                        <div className="text-center text-gray-400">
                            <CameraIcon className="w-16 h-16 mx-auto mb-4 text-gray-600" />
                            <p>Start your webcam or upload a video to begin analysis.</p>
                        </div>
                    </div>
                 )}
                 <video
                    ref={videoRef}
                    className={`absolute top-0 left-0 w-full h-full object-contain transition-opacity duration-500 ${isExporting ? 'opacity-0' : 'opacity-100'}`}
                    playsInline
                    autoPlay={mode === 'live'}
                    muted={mode === 'live'}
                    onEnded={() => { if(isVideoLoaded && !isExporting) setStatusText("Video finished.")}}
                ></video>
                <canvas
                    ref={canvasRef}
                    className="absolute top-0 left-0 w-full h-full object-contain"
                ></canvas>
            </div>
        );
    }

    return (
        <div className="w-full flex flex-col items-center gap-6">
            <div className="w-full max-w-4xl p-4 bg-gray-800/50 rounded-lg border border-gray-700 shadow-lg">
                <div className="flex flex-col sm:flex-row justify-between items-center gap-4">
                     <div className="flex items-center gap-2 text-sm text-cyan-300">
                        <div className={`w-3 h-3 rounded-full ${isLoadingModel ? 'bg-yellow-500 animate-pulse' : 'bg-green-500'}`}></div>
                        <span>{statusText}</span>
                    </div>

                    <div className="grid grid-cols-2 sm:flex gap-2">
                        <button
                            onClick={() => setMode('live')}
                            disabled={isLoadingModel || isExporting}
                            className={`px-4 py-2 text-sm font-medium rounded-md flex items-center gap-2 transition-all duration-200 ${mode === 'live' ? 'bg-cyan-500 text-white shadow-md' : 'bg-gray-700 hover:bg-gray-600'} disabled:opacity-50 disabled:cursor-not-allowed`}
                        >
                            <CameraIcon className="w-5 h-5" /> Live Camera
                        </button>
                        <button
                            onClick={() => setMode('video')}
                            disabled={isLoadingModel || isExporting}
                            className={`px-4 py-2 text-sm font-medium rounded-md flex items-center gap-2 transition-all duration-200 ${mode === 'video' ? 'bg-cyan-500 text-white shadow-md' : 'bg-gray-700 hover:bg-gray-600'} disabled:opacity-50 disabled:cursor-not-allowed`}
                        >
                             <UploadIcon className="w-5 h-5" /> Upload Video
                        </button>
                    </div>
                </div>
            </div>

            {renderContent()}

            <div className="w-full max-w-4xl p-4 bg-gray-800/50 rounded-lg border border-gray-700 flex flex-col sm:flex-row justify-center items-center gap-4">
                 {mode === 'live' && (
                    <div className="flex items-center gap-3">
                        <button onClick={toggleWebcam} disabled={isLoadingModel || isExporting} className={`px-5 py-3 font-semibold rounded-lg flex items-center gap-3 transition-colors duration-200 ${isWebcamOn ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'} text-white shadow-lg disabled:opacity-50 disabled:cursor-not-allowed`}>
                            {isWebcamOn ? <><StopIcon /> Stop Webcam</> : <><CameraIcon /> Start Webcam</>}
                        </button>
                        <div className="flex items-center gap-2 bg-gray-700/60 rounded-lg p-1">
                            <button
                                onClick={() => switchCameraFacing('user')}
                                disabled={isLoadingModel || isExporting}
                                className={`px-3 py-2 text-sm rounded-md ${cameraFacing === 'user' ? 'bg-cyan-600 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
                                title="Front camera"
                            >Front</button>
                            <button
                                onClick={() => switchCameraFacing('environment')}
                                disabled={isLoadingModel || isExporting}
                                className={`px-3 py-2 text-sm rounded-md ${cameraFacing === 'environment' ? 'bg-cyan-600 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
                                title="Back camera"
                            >Back</button>
                        </div>
                    </div>
                 )}
                 {mode === 'video' && (
                     <label className={`px-5 py-3 font-semibold rounded-lg flex items-center gap-3 transition-colors duration-200 bg-indigo-600 hover:bg-indigo-700 text-white shadow-lg ${isLoadingModel || isExporting ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'}`}>
                         <UploadIcon /> Upload Video
                         <input type="file" accept="video/mp4" onChange={handleFileChange} className="hidden" disabled={isLoadingModel || isExporting}/>
                     </label>
                 )}
                 <button onClick={handleExport} disabled={isLoadingModel || isExporting || (!isWebcamOn && !isVideoLoaded)} className="px-5 py-3 font-semibold rounded-lg flex items-center gap-3 transition-colors duration-200 bg-cyan-600 hover:bg-cyan-700 text-white shadow-lg disabled:opacity-50 disabled:cursor-not-allowed">
                     {isExporting ? (
                         <>
                         <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                        </svg>
                         Exporting...
                         </>
                     ) : (
                         <><ExportIcon /> Export Video</>
                     )}
                 </button>
            </div>
        </div>
    );
};

export default PoseAnalyzer;
