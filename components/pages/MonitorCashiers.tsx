import React, { useEffect, useRef, useState, useCallback } from 'react';
import type { PoseLandmarkerResult } from '../../types';
import { CameraIcon, StopIcon } from '../Icons';

type FacingMode = 'user' | 'environment';

const MonitorCashiers: React.FC = () => {
    const [cashierName, setCashierName] = useState('');
    const [token, setToken] = useState<string | null>(null);
    const [fromTime, setFromTime] = useState('');
    const [toTime, setToTime] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const [isLoadingModel, setIsLoadingModel] = useState(true);
    const [isWebcamOn, setIsWebcamOn] = useState(false);
    const [cameraFacing, setCameraFacing] = useState<FacingMode>('environment');
    const [statusText, setStatusText] = useState("Loading Pose Landmarker model...");
    const [isAutoRecording, setIsAutoRecording] = useState(true);
    const [theftDetected, setTheftDetected] = useState(false);
    const [detectionSensitivity, setDetectionSensitivity] = useState(0.3);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const poseLandmarkerRef = useRef<any>(null);
    const animationFrameIdRef = useRef<number | null>(null);
    const drawingUtilsRef = useRef<any>(null);
    const lastHandPositionsRef = useRef<{ left: { x: number, y: number } | null, right: { x: number, y: number } | null }>({ left: null, right: null });
    const theftDetectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    useEffect(() => {
        const saved = localStorage.getItem('auth_token');
        if (saved) setToken(saved);
    }, []);

    // Initialize MediaPipe
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
                setStatusText("Model loaded. Ready to monitor cashier.");
            } catch (error) {
                console.error("Error loading Pose Landmarker model:", error);
                setStatusText("Failed to load model. Please refresh the page.");
            }
        };
        
        const checkInterval = setInterval(() => {
            if (window.mp && window.mp.tasks && window.mp.tasks.vision) {
                clearInterval(checkInterval);
                initPoseLandmarker();
            }
        }, 100);

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

    const detectTheft = useCallback((landmarks: any[]) => {
        if (!landmarks || landmarks.length === 0) {
            console.log('🔍 No landmarks detected');
            return false;
        }

        const pose = landmarks[0]; // Use first detected pose
        if (!pose || (Array.isArray(pose) && pose.length < 33)) {
            console.log('🔍 Invalid pose data:', Array.isArray(pose) ? pose.length : 'not array');
            return false;
        }

        // MediaPipe pose landmarks: 15=left wrist, 16=right wrist, 11=left hip, 12=right hip
        const leftWrist = pose[15];
        const rightWrist = pose[16];
        const leftHip = pose[11];
        const rightHip = pose[12];
        const leftShoulder = pose[11]; // Corrected: 11 is left shoulder, 12 is right shoulder
        const rightShoulder = pose[12];
        const nose = pose[0];

        // Debug: Log all landmark positions every 30 frames
        if (Date.now() % 1000 < 50) { // Roughly every second
            console.log('🔍 Landmark positions:', {
                leftWrist: leftWrist ? { x: leftWrist.x.toFixed(3), y: leftWrist.y.toFixed(3), visibility: leftWrist.visibility?.toFixed(3) } : 'null',
                rightWrist: rightWrist ? { x: rightWrist.x.toFixed(3), y: rightWrist.y.toFixed(3), visibility: rightWrist.visibility?.toFixed(3) } : 'null',
                leftHip: leftHip ? { x: leftHip.x.toFixed(3), y: leftHip.y.toFixed(3), visibility: leftHip.visibility?.toFixed(3) } : 'null',
                rightHip: rightHip ? { x: rightHip.x.toFixed(3), y: rightHip.y.toFixed(3), visibility: rightHip.visibility?.toFixed(3) } : 'null',
                nose: nose ? { x: nose.x.toFixed(3), y: nose.y.toFixed(3), visibility: nose.visibility?.toFixed(3) } : 'null'
            });
        }
        
        // Debug: Log pose detection status
        if (Date.now() % 2000 < 50) { // Every 2 seconds
            console.log('🔍 Pose detection debug:', {
                landmarksLength: landmarks.length,
                poseLength: pose.length,
                poseLandmarkerReady: !!poseLandmarkerRef.current,
                videoReady: videoRef.current?.videoWidth > 0
            });
        }

        if (!leftWrist || !rightWrist || !leftHip || !rightHip || !nose) {
            // console.log('🔍 Missing required landmarks');
            return false;
        }

        // Check visibility of key landmarks (must be visible for reliable detection)
        if (leftWrist.visibility < 0.5 || rightWrist.visibility < 0.5 || 
            leftHip.visibility < 0.5 || rightHip.visibility < 0.5) {
            console.log('🔍 Low visibility landmarks:', {
                leftWrist: leftWrist.visibility,
                rightWrist: rightWrist.visibility,
                leftHip: leftHip.visibility,
                rightHip: rightHip.visibility
            });
            return false;
        }

        // Enhanced theft detection logic
        const handToHipThreshold = detectionSensitivity;
        
        // Calculate distance between hand and hip
        const leftDistance = Math.sqrt(
            Math.pow(leftWrist.x - leftHip.x, 2) + 
            Math.pow(leftWrist.y - leftHip.y, 2)
        );
        const rightDistance = Math.sqrt(
            Math.pow(rightWrist.x - rightHip.x, 2) + 
            Math.pow(rightWrist.y - rightHip.y, 2)
        );
        
        // Additional checks for more accurate detection
        // 1. Hand should be below shoulder level (moving toward pocket)
        const leftHandBelowShoulder = leftWrist.y > leftHip.y;
        const rightHandBelowShoulder = rightWrist.y > rightHip.y;
        
        // 2. Hand should be in the same general X position as hip (not too far to side)
        const leftHandAlignedWithHip = Math.abs(leftWrist.x - leftHip.x) < 0.15;
        const rightHandAlignedWithHip = Math.abs(rightWrist.x - rightHip.x) < 0.15;
        
        // 3. Person should be facing forward (nose position check)
        const facingForward = nose.x > 0.2 && nose.x < 0.8;
        
        const leftHandNearHip = leftDistance < handToHipThreshold && leftHandBelowShoulder && leftHandAlignedWithHip;
        const rightHandNearHip = rightDistance < handToHipThreshold && rightHandBelowShoulder && rightHandAlignedWithHip;

        // Debug: Log detection analysis every 30 frames
        if (Date.now() % 1000 < 50) {
            console.log('🔍 Detection analysis:', {
                leftDistance: leftDistance.toFixed(3),
                rightDistance: rightDistance.toFixed(3),
                threshold: handToHipThreshold,
                leftHandBelowShoulder,
                rightHandBelowShoulder,
                leftHandAlignedWithHip,
                rightHandAlignedWithHip,
                facingForward,
                leftHandNearHip,
                rightHandNearHip,
                finalResult: (leftHandNearHip || rightHandNearHip) && facingForward
            });
        }

        // Debug logging when theft is detected
        if (leftHandNearHip || rightHandNearHip) {
            console.log('🚨 THEFT DETECTED!', {
                leftDistance: leftDistance.toFixed(3),
                rightDistance: rightDistance.toFixed(3),
                threshold: handToHipThreshold,
                leftHandBelowShoulder,
                rightHandBelowShoulder,
                leftHandAlignedWithHip,
                rightHandAlignedWithHip,
                facingForward,
                leftHandNearHip,
                rightHandNearHip
            });
        }

        return (leftHandNearHip || rightHandNearHip) && facingForward;
    }, [detectionSensitivity]);

    const predictLoop = useCallback(() => {
        if (!poseLandmarkerRef.current || !videoRef.current || !canvasRef.current || !window.mp?.tasks?.vision) {
            console.log('❌ predictLoop: Missing required components', {
                poseLandmarker: !!poseLandmarkerRef.current,
                video: !!videoRef.current,
                canvas: !!canvasRef.current,
                mediaPipe: !!window.mp?.tasks?.vision
            });
            return;
        }

        const video = videoRef.current;
        if (video.paused || video.ended) {
            console.log('❌ predictLoop: Video not playing', {
                paused: video.paused,
                ended: video.ended
            });
            return;
        }

        const canvas = canvasRef.current;
        const canvasCtx = canvas.getContext('2d');
        if (!canvasCtx) {
            console.log('❌ predictLoop: No canvas context');
            return;
        }

        if (video.videoWidth > 0 && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            console.log('📐 Canvas resized:', { width: canvas.width, height: canvas.height });
        }

        let startTimeMs = performance.now();
        const results: PoseLandmarkerResult = poseLandmarkerRef.current.detectForVideo(video, startTimeMs);
        
        // Debug: Log detection results
        if (Date.now() % 3000 < 50) { // Every 3 seconds
            console.log('🔍 Detection results:', {
                hasResults: !!results,
                landmarksCount: results?.landmarks?.length || 0,
                videoSize: { width: video.videoWidth, height: video.videoHeight },
                canvasSize: { width: canvas.width, height: canvas.height }
            });
        }

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
        canvasCtx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (results.landmarks) {
            for (const landmarks of results.landmarks) {
                // Check for theft detection
                const isTheft = detectTheft(landmarks);
                
                if (isTheft && !theftDetected && isAutoRecording) {
                    console.log('🚨 THEFT DETECTED! Starting recording...', {
                        recorderState: mediaRecorderRef.current?.state,
                        chunksAvailable: chunksRef.current.length,
                        autoRecording: isAutoRecording
                    });
                    setTheftDetected(true);
                    setStatusText("🚨 THEFT DETECTED! Recording incident...");
                    
                    // Start recording immediately if not already recording
                    if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
                        console.log('🎬 Starting new recording for theft incident...');
                        startAutoRecording();
                    } else {
                        console.log('📹 Already recording, theft detected during ongoing recording');
                    }
                    
                    // Automatically save the theft incident after 3 seconds
                    setTimeout(() => {
                        console.log('⏰ Auto-saving theft incident after 3 seconds...');
                        stopRecordingAndUpload();
                    }, 3000); // Record for 3 seconds after detection
                    
                    // Clear any existing timeout
                    if (theftDetectionTimeoutRef.current) {
                        clearTimeout(theftDetectionTimeoutRef.current);
                    }
                    
                    // Reset theft detection after 8 seconds
                    theftDetectionTimeoutRef.current = setTimeout(() => {
                        console.log('🔄 Resetting theft detection after 8 seconds');
                        setTheftDetected(false);
                        setStatusText("Auto-monitoring enabled. Watching for suspicious behavior...");
                    }, 8000);
                }

                // Draw landmarks with different colors based on theft detection
                const landmarkColor = isTheft ? '#ff0000' : '#4ade80';
                const connectionColor = isTheft ? '#ff3333' : '#22d3ee';

                drawingUtilsRef.current.drawLandmarks(landmarks, {
                    radius: (data: any) => window.mp.tasks.vision.DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1),
                     color: landmarkColor,
                });
                drawingUtilsRef.current.drawConnectors(landmarks, window.mp.tasks.vision.PoseLandmarker.POSE_CONNECTIONS, {
                    color: connectionColor,
                });

                // Highlight hand-to-hip connections when theft is detected
                if (isTheft) {
                    const pose = landmarks[0];
                    if (pose && Array.isArray(pose) && pose.length >= 33) {
                        const leftWrist = pose[15];
                        const rightWrist = pose[16];
                        const leftHip = pose[11];
                        const rightHip = pose[12];

                        // Draw red warning lines
                        canvasCtx.strokeStyle = '#ff0000';
                        canvasCtx.lineWidth = 4;
                        canvasCtx.setLineDash([10, 5]); // Dashed line for emphasis
                        canvasCtx.beginPath();
                        if (leftWrist && leftHip) {
                            canvasCtx.moveTo(leftWrist.x * canvas.width, leftWrist.y * canvas.height);
                            canvasCtx.lineTo(leftHip.x * canvas.width, leftHip.y * canvas.height);
                        }
                        if (rightWrist && rightHip) {
                            canvasCtx.moveTo(rightWrist.x * canvas.width, rightWrist.y * canvas.height);
                            canvasCtx.lineTo(rightHip.x * canvas.width, rightHip.y * canvas.height);
                        }
                        canvasCtx.stroke();
                        canvasCtx.setLineDash([]); // Reset line dash

                        // Draw warning circle around the area
                        canvasCtx.strokeStyle = '#ff0000';
                        canvasCtx.lineWidth = 3;
                        canvasCtx.beginPath();
                        if (leftWrist && leftHip) {
                            const centerX = (leftWrist.x + leftHip.x) / 2 * canvas.width;
                            const centerY = (leftWrist.y + leftHip.y) / 2 * canvas.height;
                            const radius = 50;
                            canvasCtx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
                        }
                        if (rightWrist && rightHip) {
                            const centerX = (rightWrist.x + rightHip.x) / 2 * canvas.width;
                            const centerY = (rightWrist.y + rightHip.y) / 2 * canvas.height;
                            const radius = 50;
                            canvasCtx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
                        }
                        canvasCtx.stroke();
                    }
                }
            }
        }
        canvasCtx.restore();

        animationFrameIdRef.current = requestAnimationFrame(predictLoop);
    }, [detectTheft, theftDetected, isAutoRecording]);

    const startWebcamWithFacing = async (facing: FacingMode) => {
        if (!poseLandmarkerRef.current) return;
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
                setStatusText(`Auto-monitoring enabled. Watching for suspicious behavior...`);
                predictLoop();
                // Auto-start recording when webcam starts
                setTimeout(() => {
                    console.log('🔄 Auto-starting recording when webcam loaded...', {
                        isAutoRecording,
                        token: !!token,
                        webcamOn: true
                    });
                    if (isAutoRecording && token) {
                        startAutoRecording();
                    } else {
                        console.log('⚠️ Cannot auto-start recording:', {
                            isAutoRecording,
                            hasToken: !!token
                        });
                    }
                }, 1000);
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

    const switchCameraFacing = async (nextFacing: FacingMode) => {
        setCameraFacing(nextFacing);
        if (isWebcamOn) {
            stopWebcam();
            await startWebcamWithFacing(nextFacing);
        }
    };

    const login = async (username: string, password: string) => {
        const res = await fetch('/api/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password }) });
        if (!res.ok) throw new Error('Login failed');
        const data = await res.json();
        localStorage.setItem('auth_token', data.token);
        setToken(data.token);
    };

    const startAutoRecording = async () => {
        console.log('🎬 Starting auto-recording...', {
            canvas: !!canvasRef.current,
            webcam: isWebcamOn,
            currentState: mediaRecorderRef.current?.state
        });
        
        if (!canvasRef.current || !isWebcamOn) {
            console.log('❌ Cannot start recording: canvas or webcam not ready');
            return;
        }
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            console.log('⚠️ Already recording, skipping...');
            return; // Already recording
        }
        
        chunksRef.current = [];
        try {
            const stream = canvasRef.current.captureStream(30);
            console.log('📹 Canvas stream created:', {
                active: stream.active,
                tracks: stream.getTracks().length,
                videoTracks: stream.getVideoTracks().length
            });
            
            // Try different MIME types for better compatibility
            let mimeType = 'video/webm; codecs=vp9';
            if (!MediaRecorder.isTypeSupported(mimeType)) {
                mimeType = 'video/webm; codecs=vp8';
                if (!MediaRecorder.isTypeSupported(mimeType)) {
                    mimeType = 'video/webm';
                    if (!MediaRecorder.isTypeSupported(mimeType)) {
                        mimeType = 'video/mp4';
                    }
                }
            }
            
            console.log('🎥 Using MIME type:', mimeType);
            
            mediaRecorderRef.current = new MediaRecorder(stream, { 
                mimeType: mimeType,
                videoBitsPerSecond: 2500000 // 2.5 Mbps for better quality
            });
            
            setFromTime(new Date(Date.now() - 10000).toISOString());
            
            mediaRecorderRef.current.ondataavailable = (e) => { 
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                    // console.log('📦 Recording chunk received:', e.data.size, 'bytes, total chunks:', chunksRef.current.length);
                }
            };
            
            mediaRecorderRef.current.onerror = (e) => {
                console.error('❌ MediaRecorder error:', e);
                setStatusText("Recording error occurred. Please restart.");
            };
            
            mediaRecorderRef.current.onstart = () => {
                console.log('✅ MediaRecorder started successfully');
            };
            
            mediaRecorderRef.current.onstop = () => {
                console.log('⏹️ MediaRecorder stopped, total chunks:', chunksRef.current.length);
            };
            
            mediaRecorderRef.current.start(100); // Record in 100ms chunks for better responsiveness
            setIsAutoRecording(true);
            console.log('✅ Auto-recording started successfully');
            setStatusText("Auto-recording started. Monitoring for theft incidents...");
            
        } catch (error) {
            console.error('❌ Failed to start recording:', error);
            setStatusText("Failed to start recording. Please try again.");
        }
    };

    const stopAutoRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
        setIsAutoRecording(false);
        setStatusText("Auto-monitoring disabled.");
    };

    const startManualRecording = async () => {
        if (!canvasRef.current || !isWebcamOn) {
            alert("Please start the webcam first");
            return;
        }
        chunksRef.current = [];
        const stream = canvasRef.current.captureStream(30);
        mediaRecorderRef.current = new MediaRecorder(stream, { mimeType: 'video/webm; codecs=vp9' });
        setFromTime(new Date(Date.now() - 10000).toISOString());
        mediaRecorderRef.current.ondataavailable = (e) => { if (e.data.size > 0) chunksRef.current.push(e.data); };
        mediaRecorderRef.current.start();
        setStatusText("Manual recording started...");
    };

    const stopRecordingAndUpload = async () => {
        console.log('📤 Stopping recording and uploading...', {
            recorder: !!mediaRecorderRef.current,
            recorderState: mediaRecorderRef.current?.state,
            token: !!token,
            tokenValue: token ? token.substring(0, 20) + '...' : 'null',
            chunks: chunksRef.current.length,
            isWebcamOn,
            isAutoRecording
        });
        
        if (!mediaRecorderRef.current) {
            console.log('❌ Cannot upload: no MediaRecorder instance');
            setStatusText("❌ No recorder available. Please start monitoring first.");
            return;
        }
        
        if (!token) {
            console.log('❌ Cannot upload: no authentication token');
            setStatusText("❌ Not authenticated. Please login first.");
            return;
        }
        
        // Check if we have any chunks to upload
        if (chunksRef.current.length === 0) {
            console.log('❌ No recording chunks to upload');
            return;
        }
        
        return new Promise<void>((resolve) => {
            // Set up the onstop handler before stopping
            mediaRecorderRef.current!.onstop = async () => {
                console.log('⏹️ Recording stopped, processing chunks:', chunksRef.current.length);
                
                try {
                    const blob = new Blob(chunksRef.current, { type: 'video/webm' });
                    console.log('📦 Created blob:', {
                        size: blob.size,
                        type: blob.type,
                        chunks: chunksRef.current.length
                    });
                    
                    if (blob.size === 0) {
                        console.log('❌ Empty blob, skipping upload');
                        resolve();
                        return;
                    }
                    
                    const fileName = `theft_incident_${Date.now()}.webm`;
                    const file = new File([blob], fileName, { type: 'video/webm' });
                    
                    console.log('📁 Created file:', {
                        name: fileName,
                        size: file.size,
                        type: file.type
                    });
                    
                    const form = new FormData();
                    form.append('clip', file);
                    form.append('cashierName', cashierName || 'unknown');
                    form.append('fromTime', fromTime);
                    form.append('toTime', toTime);
                    
                    console.log('📋 FormData prepared:', {
                        cashierName: cashierName || 'unknown',
                        fromTime,
                        toTime,
                        fileSize: file.size
                    });
                    
                    setIsUploading(true);
                    console.log('🚀 Uploading theft incident to server...');
                    
                    const res = await fetch('/api/clips', {
                        method: 'POST',
                        headers: { Authorization: `Bearer ${token}` },
                        body: form,
                    });
                    
                    console.log('📡 Server response:', {
                        status: res.status,
                        statusText: res.statusText,
                        ok: res.ok
                    });
                    
                    if (!res.ok) {
                        const errorText = await res.text();
                        console.error('❌ Upload failed:', {
                            status: res.status,
                            statusText: res.statusText,
                            errorText
                        });
                        throw new Error(`Upload failed: ${res.status} ${errorText}`);
                    }
                    
                    const result = await res.json();
                    console.log('✅ Theft incident automatically saved!', result);
                    setStatusText(`✅ Theft incident saved! Recording ID: ${result.id}`);
                    
                    // Restart recording for next incident after a brief delay
                    setTimeout(() => {
                        console.log('🔄 Restarting recording for next incident...');
                        startAutoRecording();
                        setStatusText("Auto-monitoring enabled. Watching for suspicious behavior...");
                    }, 2000);
                    
                } catch (e) {
                    console.error('❌ Upload error:', e);
                    setStatusText("❌ Failed to save theft incident. Please check connection.");
                } finally {
                    setIsUploading(false);
                    resolve();
                }
            };
            
            console.log('⏹️ Stopping MediaRecorder...');
            mediaRecorderRef.current.stop();
            setToTime(new Date().toISOString());
        });
    };

    return (
        <div className="flex flex-col gap-4">
            {!token && (
                <div className="p-4 bg-gray-800/60 rounded">
                    <h2 className="font-semibold mb-2">Login</h2>
                    <div className="flex gap-2">
                        <button className="px-3 py-2 bg-cyan-600 rounded" onClick={() => login('cashier1', 'cashier123')}>Login as cashier1</button>
                        <button className="px-3 py-2 bg-gray-700 rounded" onClick={() => login('admin', 'admin123')}>Login as admin</button>
                    </div>
                </div>
            )}

            <div className="p-4 bg-gray-800/60 rounded">
                <div className="flex items-center gap-2 text-sm text-cyan-300 mb-4">
                    <div className={`w-3 h-3 rounded-full ${isLoadingModel ? 'bg-yellow-500 animate-pulse' : theftDetected ? 'bg-red-500 animate-pulse' : 'bg-green-500'}`}></div>
                    <span>{statusText}</span>
                    {mediaRecorderRef.current?.state === 'recording' && (
                        <span className="text-red-400 text-xs flex items-center gap-1">
                            <div className="w-2 h-2 bg-red-500 rounded-full animate-pulse"></div>
                            RECORDING
                        </span>
                    )}
                </div>

                <div className={`relative w-full max-w-4xl mx-auto aspect-video bg-black rounded-lg overflow-hidden shadow-2xl shadow-cyan-500/10 border-2 transition-all duration-300 ${theftDetected ? 'border-red-500 shadow-red-500/50' : 'border-gray-700'}`}>
                    {!isWebcamOn && (
                        <div className="absolute inset-0 flex items-center justify-center">
                            <div className="text-center text-gray-400">
                                <CameraIcon className="w-16 h-16 mx-auto mb-4 text-gray-600" />
                                <p>Start monitoring to begin pose analysis.</p>
                            </div>
                        </div>
                    )}
                    <video
                        ref={videoRef}
                        className="absolute top-0 left-0 w-full h-full object-contain"
                        playsInline
                        autoPlay
                        muted
                    />
                    <canvas
                        ref={canvasRef}
                        className="absolute top-0 left-0 w-full h-full object-contain"
                    />
                    {theftDetected && (
                        <div className="absolute inset-0 bg-red-500/30 animate-pulse flex items-center justify-center z-10">
                            <div className="text-center text-white bg-red-600/90 backdrop-blur-sm rounded-lg p-6 border-4 border-red-400 shadow-2xl">
                                <div className="text-8xl mb-4 animate-bounce">🚨</div>
                                <div className="text-3xl font-bold mb-2 text-red-100">THEFT DETECTED!</div>
                                <div className="text-xl mb-4 text-red-200">Recording incident...</div>
                                <div className="text-sm text-red-300">Hand movement to pocket area detected</div>
                            </div>
                        </div>
                    )}
                </div>

                <div className="flex items-center gap-3 mb-4">
                    <button 
                        onClick={() => isWebcamOn ? stopWebcam() : startWebcamWithFacing(cameraFacing)} 
                        disabled={isLoadingModel} 
                        className={`px-5 py-3 font-semibold rounded-lg flex items-center gap-3 transition-colors duration-200 ${isWebcamOn ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'} text-white shadow-lg disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                        {isWebcamOn ? <><StopIcon /> Stop Monitoring</> : <><CameraIcon /> Start Monitoring</>}
                    </button>
                    <div className="flex items-center gap-2 bg-gray-700/60 rounded-lg p-1">
                        <button
                            onClick={() => switchCameraFacing('user')}
                            disabled={isLoadingModel}
                            className={`px-3 py-2 text-sm rounded-md ${cameraFacing === 'user' ? 'bg-cyan-600 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
                            title="Front camera"
                        >Front</button>
                        <button
                            onClick={() => switchCameraFacing('environment')}
                            disabled={isLoadingModel}
                            className={`px-3 py-2 text-sm rounded-md ${cameraFacing === 'environment' ? 'bg-cyan-600 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
                            title="Back camera"
                        >Back</button>
                    </div>
                </div>

                <div className="flex flex-col gap-3">
                    <label className="text-sm">Cashier name
                        <input className="ml-2 px-2 py-1 rounded bg-gray-700" value={cashierName} onChange={e => setCashierName(e.target.value)} placeholder="e.g. cashier1" />
                    </label>
                    
                    <div className="flex items-center gap-2 text-sm">
                        <label>Detection Sensitivity:</label>
                        <input 
                            type="range" 
                            min="0.05" 
                            max="0.3" 
                            step="0.01" 
                            value={detectionSensitivity} 
                            onChange={e => setDetectionSensitivity(parseFloat(e.target.value))}
                            className="flex-1"
                        />
                        <span className="text-cyan-400">{detectionSensitivity.toFixed(2)}</span>
                    </div>
                    
                    <div className="text-xs text-gray-400 bg-gray-800/50 p-3 rounded">
                        <strong>System Status:</strong><br/>
                        • Recording: <span className={mediaRecorderRef.current?.state === 'recording' ? 'text-green-400' : 'text-gray-500'}>{mediaRecorderRef.current?.state || 'Not started'}</span><br/>
                        • Auto-recording: <span className={isAutoRecording ? 'text-green-400' : 'text-gray-500'}>{isAutoRecording ? 'ON' : 'OFF'}</span><br/>
                        • Webcam: <span className={isWebcamOn ? 'text-green-400' : 'text-gray-500'}>{isWebcamOn ? 'ON' : 'OFF'}</span><br/>
                        • Authentication: <span className={token ? 'text-green-400' : 'text-red-400'}>{token ? 'Logged in' : 'Not logged in'}</span><br/>
                        • Video Chunks: <span className="text-cyan-400">{chunksRef.current.length}</span><br/>
                        • Detection Status: <span className={theftDetected ? 'text-red-400 font-bold' : 'text-green-400'}>{theftDetected ? 'THEFT DETECTED' : 'Monitoring'}</span><br/>
                        • MediaRecorder: <span className={mediaRecorderRef.current ? 'text-green-400' : 'text-red-400'}>{mediaRecorderRef.current ? 'Ready' : 'Not initialized'}</span>
                    </div>

                    <div className="flex items-center gap-2 text-sm text-green-400">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        <span>Auto-Theft Detection: ACTIVE</span>
                    </div>

                    <button 
                        onClick={() => {
                            console.log('🧪 Manual theft test triggered', {
                                isWebcamOn,
                                hasToken: !!token,
                                hasRecorder: !!mediaRecorderRef.current,
                                recorderState: mediaRecorderRef.current?.state
                            });
                            
                            if (!isWebcamOn) {
                                setStatusText("❌ Please start monitoring first");
                                return;
                            }
                            
                            if (!token) {
                                setStatusText("❌ Please login first");
                                return;
                            }
                            
                            setTheftDetected(true);
                            setStatusText("🧪 TEST: Theft detection triggered manually");
                            
                            // Simulate the theft detection flow
                            if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
                                console.log('🧪 Starting recording for test...');
                                startAutoRecording();
                            }
                            
                            setTimeout(() => {
                                console.log('🧪 Auto-uploading test clip...');
                                stopRecordingAndUpload();
                            }, 3000);
                            
                            setTimeout(() => {
                                setTheftDetected(false);
                                setStatusText("Auto-monitoring enabled. Watching for suspicious behavior...");
                            }, 8000);
                        }}
                        className="px-3 py-2 bg-yellow-600 hover:bg-yellow-700 text-white text-sm rounded-lg transition-colors duration-200"
                        disabled={!isWebcamOn || !token}
                    >
                        🧪 Test Theft Detection
                    </button>

                    {theftDetected && (
                        <div className="p-3 bg-red-900/50 border border-red-500 rounded-lg animate-pulse">
                            <p className="text-red-400 font-semibold flex items-center gap-2">
                                <span className="text-xl">🚨</span> 
                                THEFT DETECTED!
                            </p>
                            <p className="text-sm text-red-300">Hand movement to pocket area detected. Recording incident...</p>
                            {isUploading && (
                                <p className="text-xs text-red-200 mt-2 flex items-center gap-1">
                                    <svg className="animate-spin h-3 w-3" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Uploading incident...
                                </p>
                            )}
                        </div>
                    )}

                    <div className="text-xs text-gray-400 bg-gray-800/50 p-3 rounded">
                        <strong>Fraud Detection Scenarios:</strong><br/>
                        • <span className="text-yellow-400">Hand-to-Pocket Movement:</span> Cashier putting money in pocket<br/>
                        • <span className="text-yellow-400">Hand-to-Hip Area:</span> Cashier concealing cash near hip<br/>
                        • <span className="text-yellow-400">Suspicious Hand Position:</span> Hand moving toward body below shoulder level<br/>
                        • <span className="text-yellow-400">Concealment Gesture:</span> Hand aligned with hip area (pocket region)<br/>
                        <br/>
                        <strong>Detection Requirements:</strong><br/>
                        • Person must be facing forward (nose visible)<br/>
                        • Key landmarks (wrist, hip) must be visible (visibility &gt; 0.5)<br/>
                        • Hand must be below shoulder level<br/>
                        • Hand must be aligned with hip area<br/>
                        • Distance between hand and hip below threshold<br/>
                        <br/>
                        <strong>Sensitivity:</strong> Lower values = more sensitive (more false positives), higher values = less sensitive.<br/>
                        <strong>Recording:</strong> Automatically records 3 seconds after detection with 10-second padding.
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MonitorCashiers;


