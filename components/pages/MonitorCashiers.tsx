import React, { useEffect, useRef, useState, useCallback } from 'react';
import type { PoseLandmarkerResult } from '../../types';
import { CameraIcon, StopIcon } from '../Icons';

type FacingMode = 'user' | 'environment';
type StreamSource = 'webcam' | 'rtsp';

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
    const [streamSource, setStreamSource] = useState<StreamSource>('webcam');
    const [rtspUrl, setRtspUrl] = useState('rtsp://admin:11111111a@192.168.1.120:554/Streaming/Channels/201');
    const [streamQuality, setStreamQuality] = useState<'normal' | 'lowlatency'>('lowlatency');
    const [loggedInUser, setLoggedInUser] = useState<string | null>(null);

    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const chunksRef = useRef<Blob[]>([]);
    const canvasRef = useRef<HTMLCanvasElement>(null);
    const videoRef = useRef<HTMLVideoElement>(null);
    const poseLandmarkerRef = useRef<any>(null);
    const animationFrameIdRef = useRef<number | null>(null);
    const drawingUtilsRef = useRef<any>(null);
    const lastHandPositionsRef = useRef<{ left: { x: number, y: number } | null, right: { x: number, y: number } | null }>({ left: null, right: null });
    const theftDetectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);
    
    // Time-based buffer for theft clips
    const timeBufferRef = useRef<{ blob: Blob, timestamp: number }[]>([]);
    const bufferDurationRef = useRef<number>(6000); // 6 seconds buffer
    const isRecordingBufferRef = useRef<boolean>(false);
    const maxBufferSizeRef = useRef<number>(30); // Max 30 chunks to prevent memory issues

    const IDX = { NOSE: 0, LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12, LEFT_WRIST: 15, RIGHT_WRIST: 16, LEFT_HIP: 23, RIGHT_HIP: 24, };

    const suspiciousFramesRef = useRef(0); // temporal smoothing


    useEffect(() => {
        const saved = localStorage.getItem('auth_token');
        console.log('🔄 Checking for saved token on component mount:', {
            tokenFound: !!saved,
            tokenLength: saved ? saved.length : 0,
            tokenPreview: saved ? saved.substring(0, 20) + '...' : 'null'
        });
        if (saved) {
            setToken(saved);
            // Try to decode the token to get username (simple JWT decode)
            try {
                const payload = JSON.parse(atob(saved.split('.')[1]));
                setLoggedInUser(payload.username);
                console.log('✅ Token loaded and user set:', payload.username);
            } catch (e) {
                console.log('❌ Could not decode token:', e);
            }
        } else {
            console.log('ℹ️ No saved token found in localStorage');
        }
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


    const detectTheft = useCallback((pose: any[]) => {
        if (!pose || pose.length < 33) return false;

        const lS = pose[IDX.LEFT_SHOULDER];
        const rS = pose[IDX.RIGHT_SHOULDER];
        const lW = pose[IDX.LEFT_WRIST];
        const rW = pose[IDX.RIGHT_WRIST];
        const lH = pose[IDX.LEFT_HIP];
        const rH = pose[IDX.RIGHT_HIP];

        const visOk = (p: any) => (p?.visibility ?? 1) >= 0.4; if (![lS, rS, lW, rW, lH, rH].every(visOk)) return false;

        const dist = (a: any, b: any) => Math.hypot(a.x - b.x, a.y - b.y);

        // Use body width to scale thresholds (more robust to distance from camera) 
        const shoulderWidth = dist(lS, rS);
        const hipWidth = dist(lH, rH);
        const bodyScale = Math.max(shoulderWidth, hipWidth) || 0.25;
        if (bodyScale < 0.05) return false; // person too small

        // detectionSensitivity interpreted as fraction of body width (suggest 0.5–0.7)
        const nearHipDist = Math.min(Math.max(detectionSensitivity ?? 0.6, 0.2), 1.0) * bodyScale;
        const xAlignThresh = 0.35 * bodyScale;

        const leftNear =
            dist(lW, lH) < nearHipDist &&
            lW.y > lS.y && // below shoulder (y increases downward)
            Math.abs(lW.x - lH.x) < xAlignThresh;

        const rightNear =
            dist(rW, rH) < nearHipDist &&
            rW.y > rS.y &&
            Math.abs(rW.x - rH.x) < xAlignThresh;

        // Temporal smoothing (require multiple frames) 
        if (leftNear || rightNear) {
            suspiciousFramesRef.current = Math.min(suspiciousFramesRef.current + 1, 30);
        } else {
            suspiciousFramesRef.current = Math.max(suspiciousFramesRef.current - 1, 0);
        }
        const isTheft = suspiciousFramesRef.current >= 6; // ~0.2s at 30 fps

        // Optional debug 
        if (Date.now() % 1000 < 50) {
            console.log('Detection analysis', {
                bodyScale: bodyScale.toFixed(3),
                nearHipDist: nearHipDist.toFixed(3),
                xAlignThresh: xAlignThresh.toFixed(3),
                leftNear, rightNear,
                smoothedFrames: suspiciousFramesRef.current,
                isTheft
            });
        }

        return isTheft;
    }, [detectionSensitivity]);


    const detectTheftOld = useCallback((landmarks: any[]) => {
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

        // if (results.landmarks) {//old block
        //     for (const landmarks of results.landmarks) {
        //         // Check for theft detection
        //         const isTheft = detectTheft(landmarks);

        //         if (isTheft && !theftDetected && isAutoRecording) {
        //             console.log('🚨 THEFT DETECTED! Starting recording...', {
        //                 recorderState: mediaRecorderRef.current?.state,
        //                 chunksAvailable: chunksRef.current.length,
        //                 autoRecording: isAutoRecording
        //             });
        //             setTheftDetected(true);
        //             setStatusText("🚨 THEFT DETECTED! Recording incident...");

        //             // Start recording immediately if not already recording
        //             if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
        //                 console.log('🎬 Starting new recording for theft incident...');
        //                 startAutoRecording();
        //             } else {
        //                 console.log('📹 Already recording, theft detected during ongoing recording');
        //             }

        //             // Automatically save the theft incident after 6 seconds (5 seconds after + 1 second for processing)
        //             setTimeout(() => {
        //                 console.log('⏰ Auto-saving theft incident after 6 seconds...');
        //                 stopRecordingAndUpload();
        //             }, 6000); // Record for 6 seconds after detection (total: 5s before + 6s after = 11s minimum)

        //             // Clear any existing timeout
        //             if (theftDetectionTimeoutRef.current) {
        //                 clearTimeout(theftDetectionTimeoutRef.current);
        //             }

        //             // Reset theft detection after 10 seconds
        //             theftDetectionTimeoutRef.current = setTimeout(() => {
        //                 console.log('🔄 Resetting theft detection after 10 seconds');
        //                 setTheftDetected(false);
        //                 setStatusText("Auto-monitoring enabled. Watching for suspicious behavior...");
        //             }, 10000);
        //         }

        //         // Draw landmarks with different colors based on theft detection
        //         const landmarkColor = isTheft ? '#ff0000' : '#4ade80';
        //         const connectionColor = isTheft ? '#ff3333' : '#22d3ee';

        //         drawingUtilsRef.current.drawLandmarks(landmarks, {
        //             radius: (data: any) => window.mp.tasks.vision.DrawingUtils.lerp(data.from!.z, -0.15, 0.1, 5, 1),
        //             color: landmarkColor,
        //         });
        //         drawingUtilsRef.current.drawConnectors(landmarks, window.mp.tasks.vision.PoseLandmarker.POSE_CONNECTIONS, {
        //             color: connectionColor,
        //         });

        //         // Highlight hand-to-hip connections when theft is detected
        //         if (isTheft) {
        //             const pose = landmarks[0];
        //             if (pose && Array.isArray(pose) && pose.length >= 33) {
        //                 const leftWrist = pose[15];
        //                 const rightWrist = pose[16];
        //                 const leftHip = pose[11];
        //                 const rightHip = pose[12];

        //                 // Draw red warning lines
        //                 canvasCtx.strokeStyle = '#ff0000';
        //                 canvasCtx.lineWidth = 4;
        //                 canvasCtx.setLineDash([10, 5]); // Dashed line for emphasis
        //                 canvasCtx.beginPath();
        //                 if (leftWrist && leftHip) {
        //                     canvasCtx.moveTo(leftWrist.x * canvas.width, leftWrist.y * canvas.height);
        //                     canvasCtx.lineTo(leftHip.x * canvas.width, leftHip.y * canvas.height);
        //                 }
        //                 if (rightWrist && rightHip) {
        //                     canvasCtx.moveTo(rightWrist.x * canvas.width, rightWrist.y * canvas.height);
        //                     canvasCtx.lineTo(rightHip.x * canvas.width, rightHip.y * canvas.height);
        //                 }
        //                 canvasCtx.stroke();
        //                 canvasCtx.setLineDash([]); // Reset line dash

        //                 // Draw warning circle around the area
        //                 canvasCtx.strokeStyle = '#ff0000';
        //                 canvasCtx.lineWidth = 3;
        //                 canvasCtx.beginPath();
        //                 if (leftWrist && leftHip) {
        //                     const centerX = (leftWrist.x + leftHip.x) / 2 * canvas.width;
        //                     const centerY = (leftWrist.y + leftHip.y) / 2 * canvas.height;
        //                     const radius = 50;
        //                     canvasCtx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        //                 }
        //                 if (rightWrist && rightHip) {
        //                     const centerX = (rightWrist.x + rightHip.x) / 2 * canvas.width;
        //                     const centerY = (rightWrist.y + rightHip.y) / 2 * canvas.height;
        //                     const radius = 50;
        //                     canvasCtx.arc(centerX, centerY, radius, 0, 2 * Math.PI);
        //                 }
        //                 canvasCtx.stroke();
        //             }
        //         }
        //     }
        // }


        if (results.landmarks) {
            for (const pose of results.landmarks) {
                const isTheft = detectTheft(pose);

                if (isTheft && !theftDetected && isAutoRecording) {
                    console.log('🚨 THEFT DETECTED! Starting short recording...', {
                        recorderState: mediaRecorderRef.current?.state,
                        chunksAvailable: chunksRef.current.length,
                        autoRecording: isAutoRecording
                    });
                    setTheftDetected(true);
                    setStatusText("🚨 THEFT DETECTED! Recording short incident clip...");

                    // Start recording immediately for theft incident
                    if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
                        console.log('🎬 Starting short recording for theft incident...');
                        startShortRecording();
                    } else {
                        console.log('📹 Already recording, theft detected during ongoing recording');
                    }

                    // Automatically save the theft incident after 8 seconds (short clip)
                    setTimeout(() => {
                        console.log('⏰ Auto-saving short theft incident after 8 seconds...');
                        stopRecordingAndUpload();
                    }, 8000); // Record for 8 seconds total

                    // Clear any existing timeout
                    if (theftDetectionTimeoutRef.current) {
                        clearTimeout(theftDetectionTimeoutRef.current);
                    }

                    // Reset theft detection after 12 seconds
                    theftDetectionTimeoutRef.current = setTimeout(() => {
                        console.log('🔄 Resetting theft detection after 12 seconds');
                        setTheftDetected(false);
                        setStatusText("Auto-monitoring enabled. Watching for suspicious behavior...");
                    }, 12000);
                }




                const landmarkColor = isTheft ? '#ff0000' : '#4ade80';
                const connectionColor = isTheft ? '#ff3333' : '#22d3ee';

                drawingUtilsRef.current.drawLandmarks(pose, {
                    radius: (data: any) =>
                        window.mp.tasks.vision.DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1),
                    color: landmarkColor,
                });
                drawingUtilsRef.current.drawConnectors(
                    pose,
                    window.mp.tasks.vision.PoseLandmarker.POSE_CONNECTIONS,
                    { color: connectionColor }
                );

                if (isTheft) {
                    const lW = pose[IDX.LEFT_WRIST];
                    const rW = pose[IDX.RIGHT_WRIST];
                    const lH = pose[IDX.LEFT_HIP];
                    const rH = pose[IDX.RIGHT_HIP];

                    canvasCtx.strokeStyle = '#ff0000';
                    canvasCtx.lineWidth = 4;
                    canvasCtx.setLineDash([10, 5]);
                    canvasCtx.beginPath();
                    if (lW && lH) {
                        canvasCtx.moveTo(lW.x * canvas.width, lW.y * canvas.height);
                        canvasCtx.lineTo(lH.x * canvas.width, lH.y * canvas.height);
                    }
                    if (rW && rH) {
                        canvasCtx.moveTo(rW.x * canvas.width, rW.y * canvas.height);
                        canvasCtx.lineTo(rH.x * canvas.width, rH.y * canvas.height);
                    }
                    canvasCtx.stroke();
                    canvasCtx.setLineDash([]);

                    const drawCircleBetween = (a: any, b: any) => {
                        const cx = (a.x + b.x) * 0.5 * canvas.width;
                        const cy = (a.y + b.y) * 0.5 * canvas.height;
                        canvasCtx.beginPath();
                        canvasCtx.arc(cx, cy, 50, 0, 2 * Math.PI);
                        canvasCtx.stroke();
                    };
                    canvasCtx.strokeStyle = '#ff0000';
                    canvasCtx.lineWidth = 3;
                    if (lW && lH) drawCircleBetween(lW, lH);
                    if (rW && rH) drawCircleBetween(rW, rH);
                }
            }
        }


        canvasCtx.restore();

        animationFrameIdRef.current = requestAnimationFrame(predictLoop);
    }, [detectTheft, theftDetected, isAutoRecording]);

    const startStream = async (facing?: FacingMode) => {
        if (!poseLandmarkerRef.current) return;

        // Defensive check to ensure streamSource is valid
        if (!streamSource || (streamSource !== 'webcam' && streamSource !== 'rtsp')) {
            console.error('❌ Invalid streamSource in startStream:', streamSource);
            setStatusText("❌ Invalid stream source. Please select webcam or RTSP.");
            return;
        }

        try {
            if (streamSource === 'webcam') {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: 1280,
                        height: 720,
                        facingMode: { ideal: facing || cameraFacing },
                    },
                    audio: false,
                });
                videoRef.current!.srcObject = stream;
            } else if (streamSource === 'rtsp') {
                // For RTSP streams, use the server proxy with quality selection
                const endpoint = streamQuality === 'lowlatency' ? '/api/stream-lowlatency' : '/api/stream';
                const proxyUrl = `${endpoint}?url=${encodeURIComponent(rtspUrl)}`;
                videoRef.current!.src = proxyUrl;
                videoRef.current!.crossOrigin = 'anonymous';
                console.log(`🎥 Using ${streamQuality} RTSP stream: ${proxyUrl}`);
                
                // RTSP stream setup complete
                console.log('🎥 RTSP stream configured');

                // Add timeout to detect if stream fails to load
                const loadTimeout = setTimeout(() => {
                    if (videoRef.current && videoRef.current.readyState < 2) {
                        console.log('⚠️ RTSP stream taking too long to load, trying fallback...');
                        setStatusText("⚠️ Stream loading slowly, trying alternative format...");

                        // Try the other quality setting as fallback
                        const fallbackEndpoint = streamQuality === 'lowlatency' ? '/api/stream' : '/api/stream-lowlatency';
                        const fallbackUrl = `${fallbackEndpoint}?url=${encodeURIComponent(rtspUrl)}`;
                        videoRef.current.src = fallbackUrl;
                        console.log(`🔄 Trying fallback stream: ${fallbackUrl}`);
                    }
                }, 10000); // 10 second timeout

                // Clear timeout when video loads successfully
                videoRef.current!.addEventListener('loadeddata', () => {
                    clearTimeout(loadTimeout);
                }, { once: true });
            }

            videoRef.current!.addEventListener('loadeddata', () => {
                videoRef.current?.play();
                setIsWebcamOn(true);
                
                // Stream loaded successfully - sync streamSource with actual video source
                const detectedStreamType = videoRef.current!.src?.includes('/api/stream') ? 'rtsp' : 'webcam';
                if (detectedStreamType !== streamSource) {
                    console.log('🔄 Auto-syncing streamSource:', {
                        previous: streamSource,
                        detected: detectedStreamType,
                        videoSrc: videoRef.current!.src
                    });
                    setStreamSource(detectedStreamType);
                }
                console.log('✅ Stream loaded and ready');
                
                setStatusText(`Auto-monitoring enabled. Watching for suspicious behavior...`);
                predictLoop();
                // Don't auto-start recording - only record when theft is detected
                console.log('🔄 Monitoring started - recording will begin only when theft is detected', {
                    isAutoRecording,
                    token: !!token,
                    streamOn: true,
                    streamSource,
                    videoSrc: videoRef.current?.src,
                    videoSrcObject: !!videoRef.current?.srcObject
                });
            }, { once: true });

            videoRef.current!.addEventListener('error', (e) => {
                console.error("Stream error:", e);
                console.error("Video error details:", {
                    error: videoRef.current?.error,
                    networkState: videoRef.current?.networkState,
                    readyState: videoRef.current?.readyState,
                    src: videoRef.current?.src,
                    streamSource,
                    rtspUrl
                });

                if (streamSource === 'rtsp') {
                    setStatusText(`❌ RTSP stream error. Check URL: ${rtspUrl}`);
                    // Try to get more specific error information
                    if (videoRef.current?.error) {
                        const error = videoRef.current.error;
                        console.error("Video error code:", error.code, "Message:", error.message);
                        switch (error.code) {
                            case 1:
                                setStatusText("❌ RTSP stream aborted. Check camera connection.");
                                break;
                            case 2:
                                setStatusText("❌ RTSP network error. Check URL and network.");
                                break;
                            case 3:
                                setStatusText("❌ RTSP decode error. Try different quality setting.");
                                break;
                            case 4:
                                setStatusText("❌ RTSP format not supported. Check stream format.");
                                break;
                            default:
                                setStatusText(`❌ RTSP error (${error.code}): ${error.message}`);
                        }
                    }
                } else {
                    setStatusText("Could not access webcam. Please check permissions.");
                }
            });

        } catch (err) {
            console.error("Error starting stream:", err);
            if (streamSource === 'rtsp') {
                setStatusText("Could not connect to RTSP stream. Please check URL and network.");
            } else {
                setStatusText("Could not access webcam. Please check permissions.");
            }
        }
    };

    const stopStream = () => {
        if (animationFrameIdRef.current) cancelAnimationFrame(animationFrameIdRef.current);

        if (streamSource === 'webcam') {
            const stream = videoRef.current?.srcObject as MediaStream;
            stream?.getTracks().forEach(track => track.stop());
            if (videoRef.current) {
                videoRef.current.srcObject = null;
            }
        } else if (streamSource === 'rtsp') {
            if (videoRef.current) {
                videoRef.current.src = '';
            }
        }

        setIsWebcamOn(false);
        setStatusText(`${streamSource === 'webcam' ? 'Webcam' : 'RTSP stream'} stopped.`);
    };

    const switchCameraFacing = async (nextFacing: FacingMode) => {
        setCameraFacing(nextFacing);
        if (isWebcamOn && streamSource === 'webcam') {
            stopStream();
            await startStream(nextFacing);
        }
    };

    const login = async (username: string, password: string) => {
        const res = await fetch('/api/login', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ username, password }) });
        if (!res.ok) throw new Error('Login failed');
        const data = await res.json();
        console.log('🔐 Login successful:', {
            username,
            tokenReceived: !!data.token,
            tokenLength: data.token ? data.token.length : 0,
            tokenPreview: data.token ? data.token.substring(0, 20) + '...' : 'null'
        });
        localStorage.setItem('auth_token', data.token);
        setToken(data.token);
        setLoggedInUser(username);
        console.log('✅ Token set in state and localStorage');
    };

    const logout = () => {
        localStorage.removeItem('auth_token');
        setToken(null);
        setLoggedInUser(null);
        setStatusText("Logged out successfully.");
        // Stop any ongoing streams/recordings
        if (isWebcamOn) {
            stopStream();
        }
    };

    const startTimeBuffer = async () => {
        // Defensive check to ensure streamSource is valid
        if (!streamSource || (streamSource !== 'webcam' && streamSource !== 'rtsp')) {
            console.error('❌ Invalid streamSource:', streamSource);
            setStatusText("❌ Invalid stream source. Please restart monitoring.");
            return;
        }

        console.log('🔄 Starting time-based buffer recording...', {
            canvas: !!canvasRef.current,
            webcam: isWebcamOn,
            streamSource,
            videoSrc: videoRef.current?.src,
            videoSrcObject: !!videoRef.current?.srcObject,
            rtspUrl
        });

        // Check if canvas is ready
        if (!canvasRef.current) {
            console.log('❌ Cannot start time buffer: canvas not ready');
            setStatusText("❌ Canvas not ready. Please restart monitoring.");
            return;
        }

        // Determine actual stream type based on video source
        const actualStreamType = videoRef.current?.src?.includes('/api/stream') ? 'rtsp' : 'webcam';
        console.log('🔍 Stream type detection for time buffer:', {
            declaredStreamSource: streamSource,
            actualStreamType,
            videoSrc: videoRef.current?.src,
            videoSrcObject: !!videoRef.current?.srcObject,
            isWebcamOn
        });

        // Check if stream is ready based on actual stream type
        if (actualStreamType === 'webcam') {
            if (!isWebcamOn || !videoRef.current?.srcObject) {
                console.log('❌ Cannot start time buffer: webcam stream not ready', {
                    isWebcamOn,
                    hasSrcObject: !!videoRef.current?.srcObject
                });
                setStatusText("❌ Webcam stream not ready. Please restart monitoring.");
                return;
            }
        } else if (actualStreamType === 'rtsp') {
            if (!videoRef.current?.src || !videoRef.current.src.includes('/api/stream')) {
                console.log('❌ Cannot start time buffer: RTSP stream not ready', {
                    videoSrc: videoRef.current?.src,
                    hasApiStream: videoRef.current?.src?.includes('/api/stream')
                });
                setStatusText("❌ RTSP stream not ready. Please restart monitoring.");
                return;
            }
        }

        if (isRecordingBufferRef.current) {
            console.log('⚠️ Time buffer already recording, skipping...');
            return;
        }

        // Clear previous buffer
        timeBufferRef.current = [];
        isRecordingBufferRef.current = true;

        try {
            const stream = canvasRef.current.captureStream(30);
            console.log('📹 Canvas stream created for time buffer:', {
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

            console.log('🎥 Using MIME type for time buffer:', mimeType);

            mediaRecorderRef.current = new MediaRecorder(stream, {
                mimeType: mimeType,
                videoBitsPerSecond: 1000000 // 1 Mbps for smaller files
            });

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    const currentTime = Date.now();
                    
                    // Add new chunk with timestamp
                    timeBufferRef.current.push({
                        blob: e.data,
                        timestamp: currentTime
                    });
                    
                    // Clean up old chunks (keep only last 6 seconds)
                    const cutoffTime = currentTime - bufferDurationRef.current;
                    timeBufferRef.current = timeBufferRef.current.filter(chunk => chunk.timestamp > cutoffTime);
                    
                    // Also limit by chunk count to prevent memory issues
                    if (timeBufferRef.current.length > maxBufferSizeRef.current) {
                        timeBufferRef.current = timeBufferRef.current.slice(-maxBufferSizeRef.current);
                    }
                    
                    console.log(`📦 Buffer updated: ${timeBufferRef.current.length} chunks, latest: ${new Date(currentTime).toLocaleTimeString()}`);
                }
            };

            mediaRecorderRef.current.onerror = (e) => {
                console.error('❌ MediaRecorder error for time buffer:', e);
                setStatusText("Time buffer recording error occurred. Please restart.");
            };

            mediaRecorderRef.current.onstart = () => {
                console.log('✅ Time buffer started successfully');
            };

            mediaRecorderRef.current.onstop = () => {
                console.log('⏹️ Time buffer stopped, total chunks:', timeBufferRef.current.length);
            };

            mediaRecorderRef.current.start(200); // Record in 200ms chunks for better balance
            setIsAutoRecording(true);
            console.log('✅ Time buffer recording started successfully');
            setStatusText("Time buffer active. Monitoring for theft incidents...");

        } catch (error) {
            console.error('❌ Failed to start time buffer recording:', error);
            setStatusText("Failed to start time buffer. Please try again.");
            isRecordingBufferRef.current = false;
        }
    };

    const startShortRecording = async () => {
        // Defensive check to ensure streamSource is valid
        if (!streamSource || (streamSource !== 'webcam' && streamSource !== 'rtsp')) {
            console.error('❌ Invalid streamSource:', streamSource);
            setStatusText("❌ Invalid stream source. Please restart monitoring.");
            return;
        }

        console.log('🎬 Starting short recording for theft incident...', {
            canvas: !!canvasRef.current,
            webcam: isWebcamOn,
            streamSource,
            videoSrc: videoRef.current?.src,
            videoSrcObject: !!videoRef.current?.srcObject,
            rtspUrl
        });

        // Check if canvas is ready
        if (!canvasRef.current) {
            console.log('❌ Cannot start short recording: canvas not ready');
            setStatusText("❌ Canvas not ready. Please restart monitoring.");
            return;
        }

        // Determine actual stream type based on video source
        const actualStreamType = videoRef.current?.src?.includes('/api/stream') ? 'rtsp' : 'webcam';
        console.log('🔍 Stream type detection for short recording:', {
            declaredStreamSource: streamSource,
            actualStreamType,
            videoSrc: videoRef.current?.src,
            videoSrcObject: !!videoRef.current?.srcObject,
            isWebcamOn
        });

        // Check if stream is ready based on actual stream type
        if (actualStreamType === 'webcam') {
            if (!isWebcamOn || !videoRef.current?.srcObject) {
                console.log('❌ Cannot start short recording: webcam stream not ready', {
                    isWebcamOn,
                    hasSrcObject: !!videoRef.current?.srcObject
                });
                setStatusText("❌ Webcam stream not ready. Please restart monitoring.");
                return;
            }
        } else if (actualStreamType === 'rtsp') {
            if (!videoRef.current?.src || !videoRef.current.src.includes('/api/stream')) {
                console.log('❌ Cannot start short recording: RTSP stream not ready', {
                    videoSrc: videoRef.current?.src,
                    hasApiStream: videoRef.current?.src?.includes('/api/stream')
                });
                setStatusText("❌ RTSP stream not ready. Please restart monitoring.");
                return;
            }
        }

        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            console.log('⚠️ Already recording, skipping...');
            return; // Already recording
        }

        // Clear previous chunks
        chunksRef.current = [];

        try {
            const stream = canvasRef.current.captureStream(30);
            console.log('📹 Canvas stream created for short recording:', {
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

            console.log('🎥 Using MIME type for short recording:', mimeType);

            mediaRecorderRef.current = new MediaRecorder(stream, {
                mimeType: mimeType,
                videoBitsPerSecond: 1000000 // 1 Mbps for smaller files
            });

            // Set the start time to current time for short recording
            setFromTime(new Date().toISOString());

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorderRef.current.onerror = (e) => {
                console.error('❌ MediaRecorder error for short recording:', e);
                setStatusText("Short recording error occurred. Please restart.");
            };

            mediaRecorderRef.current.onstart = () => {
                console.log('✅ Short recording started successfully');
            };

            mediaRecorderRef.current.onstop = () => {
                console.log('⏹️ Short recording stopped, total chunks:', chunksRef.current.length);
            };

            mediaRecorderRef.current.start(100); // Record in 100ms chunks for better responsiveness
            console.log('✅ Short recording started successfully for theft incident');

        } catch (error) {
            console.error('❌ Failed to start short recording:', error);
            setStatusText("Failed to start short recording. Please try again.");
        }
    };

    const startAutoRecording = async () => {
        // Defensive check to ensure streamSource is valid
        if (!streamSource || (streamSource !== 'webcam' && streamSource !== 'rtsp')) {
            console.error('❌ Invalid streamSource:', streamSource);
            setStatusText("❌ Invalid stream source. Please restart monitoring.");
            return;
        }

        console.log('🎬 Starting auto-recording...', {
            canvas: !!canvasRef.current,
            webcam: isWebcamOn,
            currentState: mediaRecorderRef.current?.state,
            streamSource,
            videoSrc: videoRef.current?.src,
            videoSrcObject: !!videoRef.current?.srcObject,
            rtspUrl
        });

        // Check if canvas is ready
        if (!canvasRef.current) {
            console.log('❌ Cannot start recording: canvas not ready');
            setStatusText("❌ Canvas not ready. Please restart monitoring.");
            return;
        }

        // Determine actual stream type based on video source
        const actualStreamType = videoRef.current?.src?.includes('/api/stream') ? 'rtsp' : 'webcam';
        console.log('🔍 Stream type detection:', {
            declaredStreamSource: streamSource,
            actualStreamType,
            videoSrc: videoRef.current?.src,
            videoSrcObject: !!videoRef.current?.srcObject,
            isWebcamOn
        });

        // Check if stream is ready based on actual stream type
        if (actualStreamType === 'webcam') {
            if (!isWebcamOn || !videoRef.current?.srcObject) {
                console.log('❌ Cannot start recording: webcam stream not ready', {
                    isWebcamOn,
                    hasSrcObject: !!videoRef.current?.srcObject
                });
                setStatusText("❌ Webcam stream not ready. Please restart monitoring.");
                return;
            }
        } else if (actualStreamType === 'rtsp') {
            if (!videoRef.current?.src || !videoRef.current.src.includes('/api/stream')) {
                console.log('❌ Cannot start recording: RTSP stream not ready', {
                    videoSrc: videoRef.current?.src,
                    hasApiStream: videoRef.current?.src?.includes('/api/stream')
                });
                setStatusText("❌ RTSP stream not ready. Please restart monitoring.");
                return;
            }
        }
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            console.log('⚠️ Already recording, skipping...');
            return; // Already recording
        }

        chunksRef.current = [];
        
        // Use actual stream type for recording logic
        console.log('🎬 Starting recording with actual stream type:', actualStreamType);
        
        // Different recording approaches based on actual stream type
        if (actualStreamType === 'webcam') {
            // Use MediaRecorder for webcam streams
            try {
                const stream = canvasRef.current.captureStream(30);
                console.log('📹 Canvas stream created for webcam:', {
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

                // Set the start time to 5 seconds before current time for padding
                setFromTime(new Date(Date.now() - 5000).toISOString());

                mediaRecorderRef.current.ondataavailable = (e) => {
                    if (e.data.size > 0) {
                        chunksRef.current.push(e.data);
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
                console.log('✅ Auto-recording started successfully for webcam');
                setStatusText("Auto-recording started. Monitoring for theft incidents...");

            } catch (error) {
                console.error('❌ Failed to start webcam recording:', error);
                setStatusText("Failed to start recording. Please try again.");
            }
        } else if (actualStreamType === 'rtsp') {
            // For RTSP streams, use canvas capture stream (same as webcam)
            try {
                const stream = canvasRef.current.captureStream(30);
                console.log('📹 Canvas stream created for RTSP:', {
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

                console.log('🎥 Using MIME type for RTSP:', mimeType);

                mediaRecorderRef.current = new MediaRecorder(stream, {
                    mimeType: mimeType,
                    videoBitsPerSecond: 2500000 // 2.5 Mbps for better quality
                });

                // Set the start time to 5 seconds before current time for padding
                setFromTime(new Date(Date.now() - 5000).toISOString());

                mediaRecorderRef.current.ondataavailable = (e) => {
                    if (e.data.size > 0) {
                        chunksRef.current.push(e.data);
                    }
                };

                mediaRecorderRef.current.onerror = (e) => {
                    console.error('❌ MediaRecorder error for RTSP:', e);
                    setStatusText("Recording error occurred. Please restart.");
                };

                mediaRecorderRef.current.onstart = () => {
                    console.log('✅ MediaRecorder started successfully for RTSP');
                };

                mediaRecorderRef.current.onstop = () => {
                    console.log('⏹️ MediaRecorder stopped for RTSP, total chunks:', chunksRef.current.length);
                };

                mediaRecorderRef.current.start(100); // Record in 100ms chunks for better responsiveness
                setIsAutoRecording(true);
                console.log('✅ Auto-recording started successfully for RTSP');
                setStatusText("Auto-recording started for RTSP stream. Monitoring for theft incidents...");

            } catch (error) {
                console.error('❌ Failed to start RTSP recording:', error);
                setStatusText("Failed to start RTSP recording. Please try again.");
            }
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

    const extractTheftClip = async (theftDetectionTime: number) => {
        console.log('🎬 Extracting theft clip from time buffer...', {
            theftDetectionTime,
            bufferChunks: timeBufferRef.current.length,
            currentTime: Date.now()
        });

        // Get fresh token from localStorage to avoid closure issues
        const currentToken = localStorage.getItem('auth_token');
        if (!currentToken) {
            console.log('❌ No authentication token found in localStorage');
            setStatusText("❌ Not authenticated. Please login first.");
            return;
        }

        // Calculate clip timing with padding
        const paddingBefore = 3000; // 3 seconds before
        const paddingAfter = 3000;  // 3 seconds after
        const clipStartTime = theftDetectionTime - paddingBefore;
        const clipEndTime = theftDetectionTime + paddingAfter;
        
        console.log('📐 Clip timing:', {
            theftDetectionTime,
            clipStartTime,
            clipEndTime,
            paddingBefore,
            paddingAfter,
            totalClipDuration: paddingBefore + paddingAfter
        });

        // Use ALL chunks from buffer to create a valid video (don't filter by time)
        // This ensures the video stream is not broken
        const allChunks = timeBufferRef.current.map(chunk => chunk.blob);
        
        console.log('🔍 Using all buffer chunks for valid video:', {
            totalChunks: timeBufferRef.current.length,
            bufferTimeRange: timeBufferRef.current.length > 0 ? {
                oldest: new Date(timeBufferRef.current[0].timestamp).toLocaleTimeString(),
                newest: new Date(timeBufferRef.current[timeBufferRef.current.length - 1].timestamp).toLocaleTimeString()
            } : 'No chunks'
        });
        
        if (allChunks.length === 0) {
            console.log('❌ No chunks available in time buffer');
            setStatusText("❌ No video data available for theft clip.");
            return;
        }

        try {
            // Create blob from ALL chunks to ensure valid video stream
            const blob = new Blob(allChunks, { type: 'video/webm' });
            console.log('📦 Created theft clip blob:', {
                size: blob.size,
                type: blob.type,
                chunks: allChunks.length,
                sizeInMB: (blob.size / (1024 * 1024)).toFixed(2)
            });

            if (blob.size === 0) {
                console.log('❌ Empty theft clip blob');
                setStatusText("❌ Empty theft clip. Please try again.");
                return;
            }

            // Create file with proper naming and metadata
            const timestamp = new Date(theftDetectionTime).toISOString().replace(/[:.]/g, '-');
            const fileName = `theft_incident_${timestamp}.webm`;
            const file = new File([blob], fileName, { 
                type: 'video/webm',
                lastModified: theftDetectionTime
            });

            console.log('📁 Created theft clip file:', {
                name: fileName,
                size: file.size,
                sizeInMB: (file.size / (1024 * 1024)).toFixed(2),
                type: file.type,
                lastModified: file.lastModified
            });

            // Prepare form data with proper timing
            const form = new FormData();
            form.append('clip', file);
            form.append('cashierName', cashierName || 'unknown');
            form.append('fromTime', new Date(clipStartTime).toISOString());
            form.append('toTime', new Date(clipEndTime).toISOString());
            form.append('theftDetectionTime', new Date(theftDetectionTime).toISOString());
            form.append('clipDuration', (paddingBefore + paddingAfter).toString());

            console.log('📋 Theft clip FormData prepared:', {
                cashierName: cashierName || 'unknown',
                fromTime: new Date(clipStartTime).toISOString(),
                toTime: new Date(clipEndTime).toISOString(),
                theftDetectionTime: new Date(theftDetectionTime).toISOString(),
                clipDuration: paddingBefore + paddingAfter,
                fileSize: file.size,
                fileSizeInMB: (file.size / (1024 * 1024)).toFixed(2),
                tokenAvailable: !!currentToken
            });

            setIsUploading(true);
            console.log('🚀 Uploading theft clip to server...');

            const res = await fetch('/api/clips', {
                method: 'POST',
                headers: { Authorization: `Bearer ${currentToken}` },
                body: form,
            });

            console.log('📡 Server response for theft clip:', {
                status: res.status,
                statusText: res.statusText,
                ok: res.ok
            });

            if (!res.ok) {
                const errorText = await res.text();
                console.error('❌ Theft clip upload failed:', {
                    status: res.status,
                    statusText: res.statusText,
                    errorText
                });
                throw new Error(`Upload failed: ${res.status} ${errorText}`);
            }

            const result = await res.json();
            console.log('✅ Theft clip saved successfully!', result);
            setStatusText(`✅ Theft clip saved! Size: ${(file.size / (1024 * 1024)).toFixed(2)}MB, ID: ${result.id}`);

            // Restart time buffer for next incident after a brief delay
            setTimeout(() => {
                console.log('🔄 Restarting time buffer for next incident...');
                startTimeBuffer();
                setStatusText("Time buffer active. Monitoring for theft incidents...");
            }, 2000);

        } catch (e) {
            console.error('❌ Theft clip extraction/upload error:', e);
            setStatusText("❌ Failed to save theft clip. Please check connection.");
        } finally {
            setIsUploading(false);
        }
    };

    const stopRecordingAndUpload = async () => {
        // Defensive check to ensure streamSource is valid
        if (!streamSource || (streamSource !== 'webcam' && streamSource !== 'rtsp')) {
            console.error('❌ Invalid streamSource in upload:', streamSource);
            setStatusText("❌ Invalid stream source. Cannot upload.");
            return;
        }

        console.log('📤 Stopping recording and uploading...', {
            recorder: !!mediaRecorderRef.current,
            recorderState: mediaRecorderRef.current?.state,
            token: !!token,
            tokenValue: token ? token.substring(0, 20) + '...' : 'null',
            chunks: chunksRef.current.length,
            isWebcamOn,
            isAutoRecording,
            streamSource
        });

        // Check token from both state and localStorage for debugging
        const currentToken = localStorage.getItem('auth_token');
        console.log('🔍 Token check:', {
            stateToken: !!token,
            localStorageToken: !!currentToken,
            stateTokenValue: token ? token.substring(0, 20) + '...' : 'null',
            localStorageTokenValue: currentToken ? currentToken.substring(0, 20) + '...' : 'null'
        });

        if (!token && !currentToken) {
            console.log('❌ Cannot upload: no authentication token in state or localStorage');
            setStatusText("❌ Not authenticated. Please login first.");
            return;
        }

        // Check MediaRecorder availability (both webcam and RTSP use same approach now)
        if (!mediaRecorderRef.current) {
            console.log('❌ Cannot upload: no MediaRecorder instance');
            setStatusText("❌ No recorder available. Please start monitoring first.");
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
                    // Get fresh token from localStorage to avoid closure issues
                    const currentToken = localStorage.getItem('auth_token');
                    if (!currentToken) {
                        console.log('❌ No authentication token found in localStorage');
                        setStatusText("❌ Not authenticated. Please login first.");
                        resolve();
                        return;
                    }

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
                        fileSize: file.size,
                        tokenAvailable: !!currentToken
                    });

                    setIsUploading(true);
                    console.log('🚀 Uploading theft incident to server...');

                    const res = await fetch('/api/clips', {
                        method: 'POST',
                        headers: { Authorization: `Bearer ${currentToken}` },
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
            // Set end time to 5 seconds after current time for padding
            setToTime(new Date(Date.now() + 5000).toISOString());
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

            {token && loggedInUser && (
                <div className="p-4 bg-green-800/60 rounded flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <span className="text-green-400">✅</span>
                        <span className="text-sm">Logged in as: <strong className="text-green-300">{loggedInUser}</strong></span>
                    </div>
                    <button
                        onClick={logout}
                        className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors"
                        title="Logout"
                    >
                        Logout
                    </button>
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
                        onClick={() => isWebcamOn ? stopStream() : startStream()}
                        disabled={isLoadingModel}
                        className={`px-5 py-3 font-semibold rounded-lg flex items-center gap-3 transition-colors duration-200 ${isWebcamOn ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'} text-white shadow-lg disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                        {isWebcamOn ? <><StopIcon /> Stop Monitoring</> : <><CameraIcon /> Start Monitoring</>}
                    </button>
                    {streamSource === 'webcam' && (
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
                    )}
                </div>

                <div className="flex flex-col gap-3">
                    <label className="text-sm">Cashier name
                        <input className="ml-2 px-2 py-1 rounded bg-gray-700" value={cashierName} onChange={e => setCashierName(e.target.value)} placeholder="e.g. cashier1" />
                    </label>

                    <div className="flex items-center gap-3">
                        <label className="text-sm">Stream Source:</label>
                        <div className="flex items-center gap-2 bg-gray-700/60 rounded-lg p-1">
                            <button
                                onClick={() => {
                                    console.log('🔄 Switching to webcam stream');
                                    setStreamSource('webcam');
                                }}
                                disabled={isWebcamOn && streamSource === 'webcam'}
                                className={`px-3 py-2 text-sm rounded-md ${streamSource === 'webcam' ? 'bg-cyan-600 text-white' : 'bg-gray-700 hover:bg-gray-600'} ${(isWebcamOn && streamSource === 'webcam') ? 'opacity-50 cursor-not-allowed' : ''}`}
                                title="Use webcam"
                            >Webcam</button>
                            <button
                                onClick={() => {
                                    console.log('🔄 Switching to RTSP stream');
                                    setStreamSource('rtsp');
                                }}
                                disabled={isWebcamOn && streamSource === 'rtsp'}
                                className={`px-3 py-2 text-sm rounded-md ${streamSource === 'rtsp' ? 'bg-cyan-600 text-white' : 'bg-gray-700 hover:bg-gray-600'} ${(isWebcamOn && streamSource === 'rtsp') ? 'opacity-50 cursor-not-allowed' : ''}`}
                                title="Use RTSP stream"
                            >RTSP</button>
                        </div>
                    </div>

                    {streamSource === 'rtsp' && (
                        <div className="space-y-2">
                            <label className="text-sm">RTSP URL
                                <input
                                    className="ml-2 px-2 py-1 rounded bg-gray-700 w-full mt-1"
                                    value={rtspUrl}
                                    onChange={e => setRtspUrl(e.target.value)}
                                    placeholder="rtsp://username:password@ip:port/path"
                                    disabled={isWebcamOn}
                                />
                            </label>

                            <div className="flex items-center gap-3">
                                <label className="text-sm">Stream Quality:</label>
                                <div className="flex items-center gap-2 bg-gray-700/60 rounded-lg p-1">
                                    <button
                                        onClick={() => setStreamQuality('lowlatency')}
                                        disabled={isWebcamOn}
                                        className={`px-3 py-2 text-sm rounded-md ${streamQuality === 'lowlatency' ? 'bg-green-600 text-white' : 'bg-gray-700 hover:bg-gray-600'} ${isWebcamOn ? 'opacity-50 cursor-not-allowed' : ''}`}
                                        title="Ultra-low latency (480x360, 10fps)"
                                    >Low Latency</button>
                                    <button
                                        onClick={() => setStreamQuality('normal')}
                                        disabled={isWebcamOn}
                                        className={`px-3 py-2 text-sm rounded-md ${streamQuality === 'normal' ? 'bg-cyan-600 text-white' : 'bg-gray-700 hover:bg-gray-600'} ${isWebcamOn ? 'opacity-50 cursor-not-allowed' : ''}`}
                                        title="Normal quality (640x480, 15fps)"
                                    >Normal</button>
                                </div>
                                <button
                                    onClick={async () => {
                                        if (!rtspUrl) {
                                            setStatusText("❌ Please enter RTSP URL first");
                                            return;
                                        }
                                        setStatusText("🧪 Testing RTSP connection...");
                                        try {
                                            const response = await fetch(`/api/test-rtsp?url=${encodeURIComponent(rtspUrl)}`);
                                            const result = await response.json();
                                            if (result.success) {
                                                setStatusText("✅ RTSP connection test successful!");
                                                console.log("RTSP test result:", result);
                                            } else {
                                                setStatusText(`❌ RTSP test failed: ${result.message}`);
                                                console.error("RTSP test failed:", result);
                                            }
                                        } catch (error) {
                                            setStatusText("❌ RTSP test error: " + error.message);
                                            console.error("RTSP test error:", error);
                                        }
                                    }}
                                    disabled={isWebcamOn || !rtspUrl}
                                    className="px-3 py-2 text-sm rounded-md bg-yellow-600 hover:bg-yellow-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                                    title="Test RTSP connection"
                                >Test Connection</button>
                            </div>

                            <div className="text-xs text-yellow-400 bg-yellow-900/20 p-2 rounded">
                                <strong>⚠️ RTSP Requirements:</strong><br />
                                • FFmpeg must be installed on the server<br />
                                • RTSP camera must be accessible from server<br />
                                • Stream will be converted to MP4 for browser compatibility<br />
                                <br />
                                <strong>Quality Options:</strong><br />
                                • <span className="text-green-400">Low Latency:</span> 480x360, 10fps, 200kbps (minimal delay)<br />
                                • <span className="text-cyan-400">Normal:</span> 640x480, 15fps, 500kbps (balanced quality)
                            </div>
                        </div>
                    )}

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
                        <strong>System Status:</strong><br />
                        • Recording: <span className={mediaRecorderRef.current?.state === 'recording' ? 'text-green-400' : 'text-gray-500'}>{mediaRecorderRef.current?.state || 'Not started'}</span><br />
                        • Auto-recording: <span className={isAutoRecording ? 'text-green-400' : 'text-gray-500'}>{isAutoRecording ? 'ON' : 'OFF'}</span><br />
                        • Stream Source: <span className="text-cyan-400">{streamSource.toUpperCase()}</span><br />
                        • Stream Status: <span className={isWebcamOn ? 'text-green-400' : 'text-gray-500'}>{isWebcamOn ? 'ON' : 'OFF'}</span><br />
                        • Authentication: <span className={token ? 'text-green-400' : 'text-red-400'}>{token ? 'Logged in' : 'Not logged in'}</span><br />
                        • Video Chunks: <span className="text-cyan-400">{chunksRef.current.length}</span><br />
                        • Detection Status: <span className={theftDetected ? 'text-red-400 font-bold' : 'text-green-400'}>{theftDetected ? 'THEFT DETECTED' : 'Monitoring'}</span><br />
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

                            // Simulate the theft detection flow with short recording
                            if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
                                console.log('🧪 Starting short recording for test...');
                                startShortRecording();
                            }

                            setTimeout(() => {
                                console.log('🧪 Auto-uploading short test clip...');
                                stopRecordingAndUpload();
                            }, 8000); // 8 seconds for test to match real detection

                            setTimeout(() => {
                                setTheftDetected(false);
                                setStatusText("Auto-monitoring enabled. Watching for suspicious behavior...");
                            }, 10000);
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
                        <strong>Fraud Detection Scenarios:</strong><br />
                        • <span className="text-yellow-400">Hand-to-Pocket Movement:</span> Cashier putting money in pocket<br />
                        • <span className="text-yellow-400">Hand-to-Hip Area:</span> Cashier concealing cash near hip<br />
                        • <span className="text-yellow-400">Suspicious Hand Position:</span> Hand moving toward body below shoulder level<br />
                        • <span className="text-yellow-400">Concealment Gesture:</span> Hand aligned with hip area (pocket region)<br />
                        <br />
                        <strong>Detection Requirements:</strong><br />
                        • Person must be facing forward (nose visible)<br />
                        • Key landmarks (wrist, hip) must be visible (visibility &gt; 0.5)<br />
                        • Hand must be below shoulder level<br />
                        • Hand must be aligned with hip area<br />
                        • Distance between hand and hip below threshold<br />
                        <br />
                        <strong>Sensitivity:</strong> Lower values = more sensitive (more false positives), higher values = less sensitive.<br />
                        <strong>Recording:</strong> Records 8 seconds only when theft is detected (8s total clip duration, no continuous recording).
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MonitorCashiers;


