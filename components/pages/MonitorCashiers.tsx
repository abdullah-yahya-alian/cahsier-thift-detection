import React, { useEffect, useRef, useState, useCallback } from 'react';
import type { PoseLandmarkerResult } from '../../types';
import { CameraIcon, StopIcon } from '../Icons';

type FacingMode = 'user' | 'environment';
type StreamSource = 'webcam' | 'rtsp';

type NamedROI = { x: number; y: number; w: number; h: number; name?: string };

const MonitorCashiers: React.FC = () => {
    const [cashierName, setCashierName] = useState('');
    const [token, setToken] = useState<string | null>(null);
    const [fromTime, setFromTime] = useState('');
    const [toTime, setToTime] = useState('');
    const [isUploading, setIsUploading] = useState(false);
    const [isLoadingModel, setIsLoadingModel] = useState(true);
    const [isWebcamOn, setIsWebcamOn] = useState(false);
    const [cameraFacing, setCameraFacing] = useState<FacingMode>('environment');
    const [statusText, setStatusText] = useState('Loading Pose Landmarker model...');
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
    const lastHandPositionsRef = useRef<{ left: { x: number; y: number } | null; right: { x: number; y: number } | null }>({ left: null, right: null });
    const theftDetectionTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    // Time-based buffer for theft clips
    const timeBufferRef = useRef<{ blob: Blob; timestamp: number }[]>([]);
    const bufferDurationRef = useRef<number>(6000);
    const isRecordingBufferRef = useRef<boolean>(false);
    const maxBufferSizeRef = useRef<number>(30);

    const IDX = { NOSE: 0, LEFT_SHOULDER: 11, RIGHT_SHOULDER: 12, LEFT_WRIST: 15, RIGHT_WRIST: 16, LEFT_HIP: 23, RIGHT_HIP: 24 };

    const suspiciousFramesRef = useRef(0);

    const handLandmarkerRef = useRef<any>(null);
    // Removed object detector (no generic object detection)
    const birdsEyePriorityRef = useRef<boolean>(true);

    // Dynamic device + drawer detection
    const deviceROIRef = useRef<NamedROI | null>(null);
    const drawerStateRef = useRef<'open' | 'closed' | 'unknown'>('unknown');
    const lastDeviceDetectTsRef = useRef<number>(0);
    const drawerOpenFramesRef = useRef<number>(0);
    const closedStableFramesRef = useRef<number>(0);

    // Offscreen small processing canvas for CPU-friendly operations
    const procCanvasRef = useRef<HTMLCanvasElement | null>(null);
    const PROC_W = 1500;
    const PROC_H = 1000;

    // Baseline (closed) drawer snapshot
    const drawerBaselineRef = useRef<ImageData | null>(null);
    const lastDrawerUpdateTsRef = useRef<number>(0);

    // helper to draw small name labels
    const drawLabel = (ctx: CanvasRenderingContext2D, x: number, y: number, text: string, bg = 'rgba(0,0,0,0.6)', fg = '#fff') => {
        ctx.save();
        ctx.font = '12px system-ui, sans-serif';
        const padX = 6,
            padY = 3;
        const metrics = ctx.measureText(text);
        const w = metrics.width + padX * 2;
        const h = 16 + padY * 2;
        ctx.fillStyle = bg;
        ctx.strokeStyle = 'rgba(255,255,255,0.25)';
        ctx.lineWidth = 1;
        const rx = x,
            ry = y - h;
        if ((ctx as any).roundRect) {
            ctx.beginPath();
            (ctx as any).roundRect(rx, ry, w, h, 6);
            ctx.fill();
            ctx.stroke();
        } else {
            ctx.fillRect(rx, ry, w, h);
            ctx.strokeRect(rx, ry, w, h);
        }
        ctx.fillStyle = fg;
        ctx.fillText(text, x + padX, y - padY);
        ctx.restore();
    };

    // check if any point (normalized) is inside a normalized ROI
    const anyPointInROI = (points: { x: number; y: number }[], roi: NamedROI) => {
        return points.some((p) => p.x >= roi.x && p.x <= roi.x + roi.w && p.y >= roi.y && p.y <= roi.y + roi.h);
    };

    useEffect(() => {
        const saved = localStorage.getItem('auth_token');
        if (saved) {
            setToken(saved);
            try {
                const payload = JSON.parse(atob(saved.split('.')[1]));
                setLoggedInUser(payload.username);
            } catch (e) { }
        }
    }, []);

    // Setup offscreen processing canvas
    useEffect(() => {
        const c = document.createElement('canvas');
        c.width = PROC_W;
        c.height = PROC_H;
        procCanvasRef.current = c;
    }, []);

    // Initialize MediaPipe
    useEffect(() => {
        const initPoseLandmarker = async () => {
            try {
                const vision = await (window as any).mp.tasks.vision.FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm');
                poseLandmarkerRef.current = await (window as any).mp.tasks.vision.PoseLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task`,
                        delegate: 'GPU'
                    },
                    runningMode: 'VIDEO',
                    numPoses: 2,
                    minPoseDetectionConfidence: 0.4,
                    minPosePresenceConfidence: 0.4,
                    minTrackingConfidence: 0.7
                });
                drawingUtilsRef.current = new (window as any).mp.tasks.vision.DrawingUtils(canvasRef.current?.getContext('2d')!);
                handLandmarkerRef.current = await (window as any).mp.tasks.vision.HandLandmarker.createFromOptions(vision, {
                    baseOptions: {
                        modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
                        delegate: 'GPU'
                    },
                    runningMode: 'VIDEO',
                    numHands: 2,
                    minHandDetectionConfidence: 0.35,
                    minHandPresenceConfidence: 0.5,
                    minTrackingConfidence: 0.7
                });

                setIsLoadingModel(false);
                setStatusText('Models loaded. Ready to monitor cashier (pose, hands).');
            } catch (error) {
                console.error('Error loading MediaPipe model:', error);
                setStatusText('Failed to load model. Please refresh the page.');
            }
        };

        const checkInterval = setInterval(() => {
            if ((window as any).mp?.tasks?.vision) {
                clearInterval(checkInterval);
                initPoseLandmarker();
            }
        }, 100);

        const timeout = setTimeout(() => {
            clearInterval(checkInterval);
            if (!poseLandmarkerRef.current) {
                setStatusText('Failed to load MediaPipe library. Please refresh.');
            }
        }, 20000);

        return () => {
            clearInterval(checkInterval);
            clearTimeout(timeout);
            if (animationFrameIdRef.current) {
                cancelAnimationFrame(animationFrameIdRef.current);
            }
            if (poseLandmarkerRef.current) poseLandmarkerRef.current.close();
            if (handLandmarkerRef.current) handLandmarkerRef.current.close();
        };
    }, []);

    // CPU helpers: downscale frame, compute gradients (Sobel), device detection and drawer state

    const drawToProc = (video: HTMLVideoElement) => {
        const pc = procCanvasRef.current!;
        const ctx = pc.getContext('2d')!;
        ctx.drawImage(video, 0, 0, PROC_W, PROC_H);
        return ctx.getImageData(0, 0, PROC_W, PROC_H);
    };

    const computeGrayscale = (img: ImageData) => {
        const { data, width, height } = img;
        const gray = new Uint8ClampedArray(width * height);
        for (let i = 0, g = 0; i < data.length; i += 4, g++) {
            const r = data[i],
                gC = data[i + 1],
                b = data[i + 2];
            gray[g] = (0.2126 * r + 0.7152 * gC + 0.0722 * b) | 0;
        }
        return gray;
    };

    const computeSobelMagnitude = (gray: Uint8ClampedArray, width: number, height: number) => {
        // Simple Sobel, ignore 1px border
        const mag = new Float32Array(width * height);
        const gxK = [-1, 0, 1, -2, 0, 2, -1, 0, 1];
        const gyK = [-1, -2, -1, 0, 0, 0, 1, 2, 1];
        for (let y = 1; y < height - 1; y++) {
            for (let x = 1; x < width - 1; x++) {
                let gx = 0,
                    gy = 0,
                    k = 0;
                for (let ky = -1; ky <= 1; ky++) {
                    for (let kx = -1; kx <= 1; kx++, k++) {
                        const v = gray[(y + ky) * width + (x + kx)];
                        gx += gxK[k] * v;
                        gy += gyK[k] * v;
                    }
                }
                const m = Math.hypot(gx, gy);
                mag[y * width + x] = m;
            }
        }
        return mag;
    };

    const sampleBorderVsInteriorScore = (mag: Float32Array, W: number, H: number, rx: number, ry: number, rw: number, rh: number) => {
        const x0 = Math.max(1, rx),
            y0 = Math.max(1, ry),
            x1 = Math.min(W - 2, rx + rw),
            y1 = Math.min(H - 2, ry + rh);
        if (x1 - x0 < 10 || y1 - y0 < 10) return -1;

        let borderSum = 0,
            borderCnt = 0;
        let interiorSum = 0,
            interiorCnt = 0;

        const step = 3;
        // top/bottom borders
        for (let x = x0; x <= x1; x += step) {
            borderSum += mag[y0 * W + x] + mag[y1 * W + x];
            borderCnt += 2;
        }
        // left/right borders
        for (let y = y0; y <= y1; y += step) {
            borderSum += mag[y * W + x0] + mag[y * W + x1];
            borderCnt += 2;
        }
        // interior samples (sparser)
        const stepIn = 6;
        for (let y = y0 + stepIn; y <= y1 - stepIn; y += stepIn) {
            for (let x = x0 + stepIn; x <= x1 - stepIn; x += stepIn) {
                interiorSum += mag[y * W + x];
                interiorCnt++;
            }
        }

        if (borderCnt === 0 || interiorCnt === 0) return -1;
        const borderAvg = borderSum / borderCnt;
        const interiorAvg = interiorSum / interiorCnt;

        // Prefer strong borders and relatively calmer interior
        return borderAvg - interiorAvg;
    };

    const detectDeviceROI_CPU = (video: HTMLVideoElement): NamedROI | null => {
        if (!procCanvasRef.current) return null;
        const img = drawToProc(video);
        const gray = computeGrayscale(img);
        const mag = computeSobelMagnitude(gray, PROC_W, PROC_H);

        // search candidates in lower 2/3 of the image
        const yStart = Math.floor(PROC_H * 0.35);
        const yEnd = Math.floor(PROC_H * 0.92);

        // device aspect around ~1.4-1.8; test few widths
        const widths = [0.28, 0.36, 0.44]; // relative to PROC_W
        const aspect = 1.6;
        const stepXY = 10; // px step

        // prefer near last known position
        const last = deviceROIRef.current;
        const preferX = last ? Math.round(last.x * PROC_W + (last.w * PROC_W) / 2) : PROC_W * 0.7;
        const preferY = last ? Math.round(last.y * PROC_H + (last.h * PROC_H) / 2) : PROC_H * 0.7;

        let bestScore = -1;
        let best: NamedROI | null = null;

        for (const wRel of widths) {
            const rw = Math.max(30, Math.floor(PROC_W * wRel));
            const rh = Math.max(20, Math.floor(rw / aspect));

            for (let y = yStart; y <= yEnd - rh; y += stepXY) {
                for (let x = 5; x <= PROC_W - rw - 5; x += stepXY) {
                    // small spatial bias toward previous position
                    const cx = x + rw / 2;
                    const cy = y + rh / 2;
                    const distBias = last ? -0.0007 * ((cx - preferX) ** 2 + (cy - preferY) ** 2) : 0;

                    const s = sampleBorderVsInteriorScore(mag, PROC_W, PROC_H, x, y, rw, rh) + distBias;
                    if (s > bestScore) {
                        bestScore = s;
                        best = { x: x / PROC_W, y: y / PROC_H, w: rw / PROC_W, h: rh / PROC_H, name: 'Cash Device' };
                    }
                }
            }
        }

        // Minimum quality bar
        if (bestScore < 2.0) return null;
        return best;
    };

    const getDrawerZoneFromDevice = (dev: NamedROI): NamedROI => {
        // a region below the device where drawer would protrude when opened
        const x = dev.x + dev.w * 0.05;
        const y = dev.y + dev.h * 0.9;
        const w = dev.w * 0.9;
        const h = dev.h * 0.5;
        return { x, y, w, h, name: 'Drawer' };
    };

    const updateDrawerState = (video: HTMLVideoElement) => {
        if (!deviceROIRef.current || !procCanvasRef.current) return;

        // downscale frame
        const pc = procCanvasRef.current;
        const ctx = pc.getContext('2d')!;
        ctx.drawImage(video, 0, 0, PROC_W, PROC_H);
        const cur = ctx.getImageData(0, 0, PROC_W, PROC_H);

        const drawerZone = getDrawerZoneFromDevice(deviceROIRef.current);
        // clamp & convert to int coords in proc space
        const rx = Math.max(0, Math.floor(drawerZone.x * PROC_W));
        const ry = Math.max(0, Math.floor(drawerZone.y * PROC_H));
        const rw = Math.min(PROC_W - rx, Math.floor(drawerZone.w * PROC_W));
        const rh = Math.min(PROC_H - ry, Math.floor(drawerZone.h * PROC_H));
        if (rw < 8 || rh < 8) return;

        const curPatch = ctx.getImageData(rx, ry, rw, rh);

        // collect baseline if missing or refreshed after stable closed
        const now = performance.now();
        if (!drawerBaselineRef.current) {
            drawerBaselineRef.current = curPatch;
            lastDrawerUpdateTsRef.current = now;
            drawerStateRef.current = 'closed';
            drawerOpenFramesRef.current = 0;
            return;
        }

        // compare luminance diff to baseline
        const base = drawerBaselineRef.current;
        if (base.width !== rw || base.height !== rh) {
            // size changed because device ROI changed — rebuild baseline
            drawerBaselineRef.current = curPatch;
            lastDrawerUpdateTsRef.current = now;
            drawerStateRef.current = 'closed';
            drawerOpenFramesRef.current = 0;
            closedStableFramesRef.current = 0;
            return;
        }

        let sumDiff = 0;
        const curD = curPatch.data;
        const baseD = base.data;
        for (let i = 0; i < curD.length; i += 4) {
            const cr = curD[i],
                cg = curD[i + 1],
                cb = curD[i + 2];
            const br = baseD[i],
                bg = baseD[i + 1],
                bb = baseD[i + 2];
            const cl = 0.2126 * cr + 0.7152 * cg + 0.0722 * cb;
            const bl = 0.2126 * br + 0.7152 * bg + 0.0722 * bb;
            sumDiff += Math.abs(cl - bl);
        }

        const pxCount = rw * rh;
        const avgDiff = sumDiff / pxCount;

        // Heuristic thresholds
        const OPEN_T = 12; // avg luminance delta to consider "open"
        if (avgDiff > OPEN_T) {
            drawerOpenFramesRef.current = Math.min(drawerOpenFramesRef.current + 1, 10);
            closedStableFramesRef.current = Math.max(closedStableFramesRef.current - 1, 0);
        } else {
            drawerOpenFramesRef.current = Math.max(drawerOpenFramesRef.current - 1, 0);
            closedStableFramesRef.current = Math.min(closedStableFramesRef.current + 1, 30);
        }

        if (drawerOpenFramesRef.current >= 3) {
            drawerStateRef.current = 'open';
        } else if (drawerOpenFramesRef.current <= 0) {
            drawerStateRef.current = 'closed';
            // refresh baseline if stably closed for a while (to adapt lighting)
            if (closedStableFramesRef.current >= 15 && now - lastDrawerUpdateTsRef.current > 4000) {
                drawerBaselineRef.current = curPatch;
                lastDrawerUpdateTsRef.current = now;
            }
        }
    };

    const detectTheft = useCallback(
        (pose: any[]) => {
            if (!pose || pose.length < 33) return false;

            const lS = pose[IDX.LEFT_SHOULDER];
            const rS = pose[IDX.RIGHT_SHOULDER];
            const lW = pose[IDX.LEFT_WRIST];
            const rW = pose[IDX.RIGHT_WRIST];
            const lH = pose[IDX.LEFT_HIP];
            const rH = pose[IDX.RIGHT_HIP];

            const visOk = (p: any) => (p?.visibility ?? 1) >= 0.4;
            if (![lS, rS, lW, rW, lH, rH].every(visOk)) return false;

            const dist = (a: any, b: any) => Math.hypot(a.x - b.x, a.y - b.y);
            const shoulderWidth = dist(lS, rS);
            const hipWidth = dist(lH, rH);
            const bodyScale = Math.max(shoulderWidth, hipWidth) || 0.25;
            if (bodyScale < 0.05) return false;

            const birdsEye = birdsEyePriorityRef.current === true;
            const nearHipDist = Math.min(Math.max(detectionSensitivity ?? 0.6, 0.2), 1.0) * bodyScale;
            const xAlignThresh = (birdsEye ? 0.5 : 0.35) * bodyScale;

            const leftNear = dist(lW, lH) < nearHipDist && (birdsEye ? true : lW.y > lS.y) && Math.abs(lW.x - lH.x) < xAlignThresh;

            const rightNear = dist(rW, rH) < nearHipDist && (birdsEye ? true : rW.y > rS.y) && Math.abs(rW.x - rH.x) < xAlignThresh;

            if (leftNear || rightNear) {
                suspiciousFramesRef.current = Math.min(suspiciousFramesRef.current + 1, 40);
            } else {
                suspiciousFramesRef.current = Math.max(suspiciousFramesRef.current - 1, 0);
            }

            if ((window as any).__handInTargetROI === true) {
                suspiciousFramesRef.current = Math.min(suspiciousFramesRef.current + 2, 40);
            }

            const isTheft = suspiciousFramesRef.current >= 6;
            return isTheft;
        },
        [detectionSensitivity]
    );

    const predictLoop = useCallback(() => {
        if (!poseLandmarkerRef.current || !videoRef.current || !canvasRef.current || !(window as any).mp?.tasks?.vision) {
            return;
        }

        const video = videoRef.current;
        if (video.paused || video.ended) {
            return;
        }

        const canvas = canvasRef.current;
        const canvasCtx = canvas.getContext('2d');
        if (!canvasCtx) {
            return;
        }

        if (video.videoWidth > 0 && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
        }

        // Periodic device detection (every 30s) with fallback
        const now = performance.now();
        if (now - lastDeviceDetectTsRef.current > 30000 || !deviceROIRef.current) {
            const roi = detectDeviceROI_CPU(video);
            if (roi) {
                deviceROIRef.current = roi;
                setStatusText('Cash device detected/updated');
            } else {
                // keep last position if available
                if (!deviceROIRef.current) {
                    setStatusText('Device not found yet, will retry in 30s...');
                } else {
                    setStatusText('Device detection failed, keeping last known position');
                }
            }
            lastDeviceDetectTsRef.current = now;
        }

        // Update drawer state every frame (cheap)
        updateDrawerState(video);

        let startTimeMs = performance.now();
        const results: PoseLandmarkerResult = poseLandmarkerRef.current.detectForVideo(video, startTimeMs);

        canvasCtx.save();
        canvasCtx.clearRect(0, 0, canvas.width, canvas.height);
        canvasCtx.drawImage(video, 0, 0, canvas.width, canvas.height);

        if (results.landmarks) {
            for (const pose of results.landmarks) {
                const isTheft = detectTheft(pose);

                if (isTheft && !theftDetected && isAutoRecording) {
                    setTheftDetected(true);
                    setStatusText('🚨 THEFT DETECTED! Recording short incident clip...');
                    if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
                        startShortRecording();
                    }
                    setTimeout(() => {
                        stopRecordingAndUpload();
                    }, 8000);

                    if (theftDetectionTimeoutRef.current) {
                        clearTimeout(theftDetectionTimeoutRef.current);
                    }
                    theftDetectionTimeoutRef.current = setTimeout(() => {
                        setTheftDetected(false);
                        setStatusText('Auto-monitoring enabled. Watching for suspicious behavior...');
                    }, 12000);
                }

                const landmarkColor = isTheft ? '#ff0000' : '#4ade80';
                const connectionColor = isTheft ? '#ff3333' : '#22d3ee';

                drawingUtilsRef.current.drawLandmarks(pose, {
                    radius: (data: any) => (window as any).mp.tasks.vision.DrawingUtils.lerp(data.from?.z ?? 0, -0.15, 0.1, 5, 1),
                    color: landmarkColor
                });
                drawingUtilsRef.current.drawConnectors(pose, (window as any).mp.tasks.vision.PoseLandmarker.POSE_CONNECTIONS, { color: connectionColor });

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

        // Hands detection for boost (target ROI = drawer if open, else device box)
        const ctx = canvas.getContext('2d')!;
        const W = canvas.width,
            H = canvas.height;
        (window as any).__handInTargetROI = false;

        const dev = deviceROIRef.current;
        if (dev) {
            // Draw device ROI and label (always)
            ctx.save();
            ctx.strokeStyle = '#f59e0b';
            ctx.lineWidth = 2;
            ctx.setLineDash([6, 4]);
            ctx.strokeRect(dev.x * W, dev.y * H, dev.w * W, dev.h * H);
            drawLabel(ctx, dev.x * W, dev.y * H, 'Cash Device', 'rgba(120, 53, 15, 0.6)', '#ffd38b');
            ctx.restore();

            // Drawer overlay and label
            const drawerZone = getDrawerZoneFromDevice(dev);
            if (drawerStateRef.current === 'open') {
                ctx.save();
                ctx.fillStyle = 'rgba(34,197,94,0.15)';
                ctx.fillRect(drawerZone.x * W, drawerZone.y * H, drawerZone.w * W, drawerZone.h * H);
                ctx.strokeStyle = '#22c55e';
                ctx.setLineDash([4, 3]);
                ctx.strokeRect(drawerZone.x * W, drawerZone.y * H, drawerZone.w * W, drawerZone.h * H);
                drawLabel(ctx, drawerZone.x * W, drawerZone.y * H, 'Drawer: OPEN', 'rgba(6,95,70,0.75)', '#a7f3d0');
                ctx.restore();
            } else {
                // closed/unknown: show label only near device
                drawLabel(ctx, dev.x * W, Math.max(12, dev.y * H - 6), `Drawer: ${drawerStateRef.current.toUpperCase()}`, 'rgba(30,41,59,0.7)', '#cbd5e1');
            }

            // Set target ROI for hand boost
            const targetROI = drawerStateRef.current === 'open' ? drawerZone : dev;

            if (handLandmarkerRef.current) {
                const handRes = handLandmarkerRef.current.detectForVideo(video, performance.now());
                if (handRes?.landmarks?.length) {
                    for (const handLm of handRes.landmarks) {
                        drawingUtilsRef.current.drawLandmarks(handLm, { color: '#fde047', radius: 2.5 });
                        drawingUtilsRef.current.drawConnectors(handLm, (window as any).mp.tasks.vision.HandLandmarker.HAND_CONNECTIONS, { color: '#facc15' });

                        const fingerTips = [8, 12, 16, 20].map((i) => ({ x: handLm[i].x, y: handLm[i].y }));
                        if (anyPointInROI(fingerTips, targetROI)) {
                            (window as any).__handInTargetROI = true;
                        }
                    }
                }
            }
        }

        canvasCtx.restore();
        animationFrameIdRef.current = requestAnimationFrame(predictLoop);
    }, [detectTheft, theftDetected, isAutoRecording]);

    const startStream = async (facing?: FacingMode) => {
        if (!poseLandmarkerRef.current) return;

        if (!streamSource || (streamSource !== 'webcam' && streamSource !== 'rtsp')) {
            setStatusText('❌ Invalid stream source. Please select webcam or RTSP.');
            return;
        }

        try {
            if (streamSource === 'webcam') {
                const stream = await navigator.mediaDevices.getUserMedia({
                    video: {
                        width: 1280,
                        height: 720,
                        facingMode: { ideal: facing || cameraFacing }
                    },
                    audio: false
                });
                videoRef.current!.srcObject = stream;
            } else if (streamSource === 'rtsp') {
                const endpoint = streamQuality === 'lowlatency' ? '/api/stream-lowlatency' : '/api/stream';
                const proxyUrl = `${endpoint}?url=${encodeURIComponent(rtspUrl)}`;
                videoRef.current!.src = proxyUrl;
                videoRef.current!.crossOrigin = 'anonymous';
                const loadTimeout = setTimeout(() => {
                    if (videoRef.current && videoRef.current.readyState < 2) {
                        setStatusText('⚠️ Stream loading slowly, trying alternative format...');
                        const fallbackEndpoint = streamQuality === 'lowlatency' ? '/api/stream' : '/api/stream-lowlatency';
                        const fallbackUrl = `${fallbackEndpoint}?url=${encodeURIComponent(rtspUrl)}`;
                        videoRef.current!.src = fallbackUrl;
                    }
                }, 10000);
                videoRef.current!.addEventListener(
                    'loadeddata',
                    () => {
                        clearTimeout(loadTimeout);
                    },
                    { once: true }
                );
            }

            videoRef.current!.addEventListener(
                'loadeddata',
                () => {
                    videoRef.current?.play();
                    setIsWebcamOn(true);

                    const detectedStreamType = videoRef.current!.src?.includes('/api/stream') ? 'rtsp' : 'webcam';
                    if (detectedStreamType !== streamSource) {
                        setStreamSource(detectedStreamType);
                    }
                    setStatusText(`Auto-monitoring enabled. Watching for suspicious behavior...`);

                    // Reset device/drawer state
                    deviceROIRef.current = null;
                    drawerBaselineRef.current = null;
                    drawerStateRef.current = 'unknown';
                    drawerOpenFramesRef.current = 0;
                    closedStableFramesRef.current = 0;
                    lastDeviceDetectTsRef.current = 0;

                    predictLoop();
                },
                { once: true }
            );

            videoRef.current!.addEventListener('error', (e) => {
                if (streamSource === 'rtsp') {
                    setStatusText(`❌ RTSP stream error. Check URL: ${rtspUrl}`);
                    if (videoRef.current?.error) {
                        const error = videoRef.current.error;
                        switch (error.code) {
                            case 1:
                                setStatusText('❌ RTSP stream aborted. Check camera connection.');
                                break;
                            case 2:
                                setStatusText('❌ RTSP network error. Check URL and network.');
                                break;
                            case 3:
                                setStatusText('❌ RTSP decode error. Try different quality setting.');
                                break;
                            case 4:
                                setStatusText('❌ RTSP format not supported. Check stream format.');
                                break;
                            default:
                                setStatusText(`❌ RTSP error (${error.code}): ${error.message}`);
                        }
                    }
                } else {
                    setStatusText('Could not access webcam. Please check permissions.');
                }
            });
        } catch (err) {
            if (streamSource === 'rtsp') {
                setStatusText('Could not connect to RTSP stream. Please check URL and network.');
            } else {
                setStatusText('Could not access webcam. Please check permissions.');
            }
        }
    };

    const stopStream = () => {
        if (animationFrameIdRef.current) cancelAnimationFrame(animationFrameIdRef.current);

        if (streamSource === 'webcam') {
            const stream = videoRef.current?.srcObject as MediaStream;
            stream?.getTracks().forEach((track) => track.stop());
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
        localStorage.setItem('auth_token', data.token);
        setToken(data.token);
        setLoggedInUser(username);
    };

    const logout = () => {
        localStorage.removeItem('auth_token');
        setToken(null);
        setLoggedInUser(null);
        setStatusText('Logged out successfully.');
        if (isWebcamOn) {
            stopStream();
        }
    };

    const startTimeBuffer = async () => {
        if (!streamSource || (streamSource !== 'webcam' && streamSource !== 'rtsp')) {
            setStatusText('❌ Invalid stream source. Please restart monitoring.');
            return;
        }

        if (!canvasRef.current) {
            setStatusText('❌ Canvas not ready. Please restart monitoring.');
            return;
        }

        const actualStreamType = videoRef.current?.src?.includes('/api/stream') ? 'rtsp' : 'webcam';

        if (actualStreamType === 'webcam') {
            if (!isWebcamOn || !videoRef.current?.srcObject) {
                setStatusText('❌ Webcam stream not ready. Please restart monitoring.');
                return;
            }
        } else if (actualStreamType === 'rtsp') {
            if (!videoRef.current?.src || !videoRef.current.src.includes('/api/stream')) {
                setStatusText('❌ RTSP stream not ready. Please restart monitoring.');
                return;
            }
        }

        if (isRecordingBufferRef.current) return;

        timeBufferRef.current = [];
        isRecordingBufferRef.current = true;

        try {
            const stream = canvasRef.current.captureStream(30);

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

            mediaRecorderRef.current = new MediaRecorder(stream, {
                mimeType: mimeType,
                videoBitsPerSecond: 1000000
            });

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    const currentTime = Date.now();
                    timeBufferRef.current.push({ blob: e.data, timestamp: currentTime });
                    const cutoffTime = currentTime - bufferDurationRef.current;
                    timeBufferRef.current = timeBufferRef.current.filter((chunk) => chunk.timestamp > cutoffTime);
                    if (timeBufferRef.current.length > maxBufferSizeRef.current) {
                        timeBufferRef.current = timeBufferRef.current.slice(-maxBufferSizeRef.current);
                    }
                }
            };

            mediaRecorderRef.current.onerror = () => {
                setStatusText('Time buffer recording error occurred. Please restart.');
            };

            mediaRecorderRef.current.start(200);
            setIsAutoRecording(true);
            setStatusText('Time buffer active. Monitoring for theft incidents...');
        } catch (error) {
            setStatusText('Failed to start time buffer. Please try again.');
            isRecordingBufferRef.current = false;
        }
    };

    const startShortRecording = async () => {
        if (!streamSource || (streamSource !== 'webcam' && streamSource !== 'rtsp')) {
            setStatusText('❌ Invalid stream source. Please restart monitoring.');
            return;
        }

        if (!canvasRef.current) {
            setStatusText('❌ Canvas not ready. Please restart monitoring.');
            return;
        }

        const actualStreamType = videoRef.current?.src?.includes('/api/stream') ? 'rtsp' : 'webcam';
        if (actualStreamType === 'webcam') {
            if (!isWebcamOn || !videoRef.current?.srcObject) {
                setStatusText('❌ Webcam stream not ready. Please restart monitoring.');
                return;
            }
        } else if (actualStreamType === 'rtsp') {
            if (!videoRef.current?.src || !videoRef.current.src.includes('/api/stream')) {
                setStatusText('❌ RTSP stream not ready. Please restart monitoring.');
                return;
            }
        }

        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            return;
        }

        chunksRef.current = [];

        try {
            const stream = canvasRef.current.captureStream(30);

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

            mediaRecorderRef.current = new MediaRecorder(stream, {
                mimeType: mimeType,
                videoBitsPerSecond: 1000000
            });

            setFromTime(new Date().toISOString());

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorderRef.current.start(100);
        } catch (error) {
            setStatusText('Failed to start short recording. Please try again.');
        }
    };

    const startAutoRecording = async () => {
        if (!streamSource || (streamSource !== 'webcam' && streamSource !== 'rtsp')) {
            setStatusText('❌ Invalid stream source. Please restart monitoring.');
            return;
        }

        if (!canvasRef.current) {
            setStatusText('❌ Canvas not ready. Please restart monitoring.');
            return;
        }

        const actualStreamType = videoRef.current?.src?.includes('/api/stream') ? 'rtsp' : 'webcam';
        if (actualStreamType === 'webcam') {
            if (!isWebcamOn || !videoRef.current?.srcObject) {
                setStatusText('❌ Webcam stream not ready. Please restart monitoring.');
                return;
            }
        } else if (actualStreamType === 'rtsp') {
            if (!videoRef.current?.src || !videoRef.current.src.includes('/api/stream')) {
                setStatusText('❌ RTSP stream not ready. Please restart monitoring.');
                return;
            }
        }
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') return;

        chunksRef.current = [];

        try {
            const stream = canvasRef.current.captureStream(30);

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

            mediaRecorderRef.current = new MediaRecorder(stream, {
                mimeType: mimeType,
                videoBitsPerSecond: 2500000
            });

            setFromTime(new Date(Date.now() - 5000).toISOString());

            mediaRecorderRef.current.ondataavailable = (e) => {
                if (e.data.size > 0) {
                    chunksRef.current.push(e.data);
                }
            };

            mediaRecorderRef.current.start(100);
            setIsAutoRecording(true);
            setStatusText('Auto-recording started. Monitoring for theft incidents...');
        } catch (error) {
            setStatusText('Failed to start recording. Please try again.');
        }
    };

    const stopAutoRecording = () => {
        if (mediaRecorderRef.current && mediaRecorderRef.current.state === 'recording') {
            mediaRecorderRef.current.stop();
        }
        setIsAutoRecording(false);
        setStatusText('Auto-monitoring disabled.');
    };

    const extractTheftClip = async (theftDetectionTime: number) => {
        const currentToken = localStorage.getItem('auth_token');
        if (!currentToken) {
            setStatusText('❌ Not authenticated. Please login first.');
            return;
        }

        const paddingBefore = 3000;
        const paddingAfter = 3000;
        const clipStartTime = theftDetectionTime - paddingBefore;
        const clipEndTime = theftDetectionTime + paddingAfter;

        const allChunks = timeBufferRef.current.map((chunk) => chunk.blob);
        if (allChunks.length === 0) {
            setStatusText('❌ No video data available for theft clip.');
            return;
        }

        try {
            const blob = new Blob(allChunks, { type: 'video/webm' });
            if (blob.size === 0) {
                setStatusText('❌ Empty theft clip. Please try again.');
                return;
            }

            const timestamp = new Date(theftDetectionTime).toISOString().replace(/[:.]/g, '-');
            const fileName = `theft_incident_${timestamp}.webm`;
            const file = new File([blob], fileName, {
                type: 'video/webm',
                lastModified: theftDetectionTime
            });

            const form = new FormData();
            form.append('clip', file);
            form.append('cashierName', cashierName || 'unknown');
            form.append('fromTime', new Date(clipStartTime).toISOString());
            form.append('toTime', new Date(clipEndTime).toISOString());
            form.append('theftDetectionTime', new Date(theftDetectionTime).toISOString());
            form.append('clipDuration', (paddingBefore + paddingAfter).toString());

            setIsUploading(true);
            const res = await fetch('/api/clips', {
                method: 'POST',
                headers: { Authorization: `Bearer ${currentToken}` },
                body: form
            });

            if (!res.ok) {
                const errorText = await res.text();
                throw new Error(`Upload failed: ${res.status} ${errorText}`);
            }

            const result = await res.json();
            setStatusText(`✅ Theft clip saved! Size: ${(file.size / (1024 * 1024)).toFixed(2)}MB, ID: ${result.id}`);

            setTimeout(() => {
                startTimeBuffer();
                setStatusText('Time buffer active. Monitoring for theft incidents...');
            }, 2000);
        } catch (e) {
            setStatusText('❌ Failed to save theft clip. Please check connection.');
        } finally {
            setIsUploading(false);
        }
    };

    const stopRecordingAndUpload = async () => {
        if (!streamSource || (streamSource !== 'webcam' && streamSource !== 'rtsp')) {
            setStatusText('❌ Invalid stream source. Cannot upload.');
            return;
        }

        const currentToken = localStorage.getItem('auth_token');
        if (!token && !currentToken) {
            setStatusText('❌ Not authenticated. Please login first.');
            return;
        }

        if (!mediaRecorderRef.current) {
            setStatusText('❌ No recorder available. Please start monitoring first.');
            return;
        }

        if (chunksRef.current.length === 0) {
            return;
        }

        return new Promise<void>((resolve) => {
            mediaRecorderRef.current!.onstop = async () => {
                try {
                    const currentToken = localStorage.getItem('auth_token');
                    if (!currentToken) {
                        setStatusText('❌ Not authenticated. Please login first.');
                        resolve();
                        return;
                    }

                    const blob = new Blob(chunksRef.current, { type: 'video/webm' });
                    if (blob.size === 0) {
                        resolve();
                        return;
                    }

                    const fileName = `theft_incident_${Date.now()}.webm`;
                    const file = new File([blob], fileName, { type: 'video/webm' });

                    const form = new FormData();
                    form.append('clip', file);
                    form.append('cashierName', cashierName || 'unknown');
                    form.append('fromTime', fromTime);
                    form.append('toTime', toTime);

                    setIsUploading(true);

                    const res = await fetch('/api/clips', {
                        method: 'POST',
                        headers: { Authorization: `Bearer ${currentToken}` },
                        body: form
                    });

                    if (!res.ok) {
                        const errorText = await res.text();
                        throw new Error(`Upload failed: ${res.status} ${errorText}`);
                    }

                    const result = await res.json();
                    setStatusText(`✅ Theft incident saved! Recording ID: ${result.id}`);

                    setTimeout(() => {
                        startAutoRecording();
                        setStatusText('Auto-monitoring enabled. Watching for suspicious behavior...');
                    }, 2000);
                } catch (e) {
                    setStatusText('❌ Failed to save theft incident. Please check connection.');
                } finally {
                    setIsUploading(false);
                    resolve();
                }
            };

            mediaRecorderRef.current!.stop();
            setToTime(new Date(Date.now() + 5000).toISOString());
        });
    };

    return (
        <div className="flex flex-col gap-4">
            {!token && (
                <div className="p-4 bg-gray-800/60 rounded">
                    <h2 className="font-semibold mb-2">Login</h2>
                    <div className="flex gap-2">
                        <button className="px-3 py-2 bg-cyan-600 rounded" onClick={() => login('cashier1', 'cashier123')}>
                            Login as cashier1
                        </button>
                        <button className="px-3 py-2 bg-gray-700 rounded" onClick={() => login('admin', 'admin123')}>
                            Login as admin
                        </button>
                    </div>
                </div>
            )}

            {token && loggedInUser && (
                <div className="p-4 bg-green-800/60 rounded flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <span className="text-green-400">✅</span>
                        <span className="text-sm">
                            Logged in as: <strong className="text-green-300">{loggedInUser}</strong>
                        </span>
                    </div>
                    <button onClick={logout} className="px-3 py-1 bg-red-600 hover:bg-red-700 text-white text-sm rounded transition-colors" title="Logout">
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
                    <video ref={videoRef} className="absolute top-0 left-0 w-full h-full object-contain" playsInline autoPlay muted />
                    <canvas ref={canvasRef} className="absolute top-0 left-0 w-full h-full object-contain" />
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
                        onClick={() => (isWebcamOn ? stopStream() : startStream())}
                        disabled={isLoadingModel}
                        className={`px-5 py-3 font-semibold rounded-lg flex items-center gap-3 transition-colors duration-200 ${isWebcamOn ? 'bg-red-600 hover:bg-red-700' : 'bg-green-600 hover:bg-green-700'} text-white shadow-lg disabled:opacity-50 disabled:cursor-not-allowed`}
                    >
                        {isWebcamOn ? (
                            <>
                                <StopIcon /> Stop Monitoring
                            </>
                        ) : (
                            <>
                                <CameraIcon /> Start Monitoring
                            </>
                        )}
                    </button>
                    {streamSource === 'webcam' && (
                        <div className="flex items-center gap-2 bg-gray-700/60 rounded-lg p-1">
                            <button
                                onClick={() => switchCameraFacing('user')}
                                disabled={isLoadingModel}
                                className={`px-3 py-2 text-sm rounded-md ${cameraFacing === 'user' ? 'bg-cyan-600 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
                                title="Front camera"
                            >
                                Front
                            </button>
                            <button
                                onClick={() => switchCameraFacing('environment')}
                                disabled={isLoadingModel}
                                className={`px-3 py-2 text-sm rounded-md ${cameraFacing === 'environment' ? 'bg-cyan-600 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
                                title="Back camera"
                            >
                                Back
                            </button>
                        </div>
                    )}
                </div>

                <div className="flex flex-col gap-3">
                    <label className="text-sm">
                        Cashier name
                        <input className="ml-2 px-2 py-1 rounded bg-gray-700" value={cashierName} onChange={(e) => setCashierName(e.target.value)} placeholder="e.g. cashier1" />
                    </label>

                    <div className="flex items-center gap-3">
                        <label className="text-sm">Stream Source:</label>
                        <div className="flex items-center gap-2 bg-gray-700/60 rounded-lg p-1">
                            <button
                                onClick={() => {
                                    setStreamSource('webcam');
                                }}
                                disabled={isWebcamOn && streamSource === 'webcam'}
                                className={`px-3 py-2 text-sm rounded-md ${streamSource === 'webcam' ? 'bg-cyan-600 text-white' : 'bg-gray-700 hover:bg-gray-600'} ${isWebcamOn && streamSource === 'webcam' ? 'opacity-50 cursor-not-allowed' : ''}`}
                                title="Use webcam"
                            >
                                Webcam
                            </button>
                            <button
                                onClick={() => {
                                    setStreamSource('rtsp');
                                }}
                                disabled={isWebcamOn && streamSource === 'rtsp'}
                                className={`px-3 py-2 text-sm rounded-md ${streamSource === 'rtsp' ? 'bg-cyan-600 text-white' : 'bg-gray-700 hover:bg-gray-600'} ${isWebcamOn && streamSource === 'rtsp' ? 'opacity-50 cursor-not-allowed' : ''}`}
                                title="Use RTSP stream"
                            >
                                RTSP
                            </button>
                        </div>
                    </div>

                    {streamSource === 'rtsp' && (
                        <div className="space-y-2">
                            <label className="text-sm">
                                RTSP URL
                                <input className="ml-2 px-2 py-1 rounded bg-gray-700 w-full mt-1" value={rtspUrl} onChange={(e) => setRtspUrl(e.target.value)} placeholder="rtsp://username:password@ip:port/path" disabled={isWebcamOn} />
                            </label>

                            <div className="flex items-center gap-3">
                                <label className="text-sm">Stream Quality:</label>
                                <div className="flex items-center gap-2 bg-gray-700/60 rounded-lg p-1">
                                    <button
                                        onClick={() => setStreamQuality('lowlatency')}
                                        disabled={isWebcamOn}
                                        className={`px-3 py-2 text-sm rounded-md ${streamQuality === 'lowlatency' ? 'bg-green-600 text-white' : 'bg-gray-700 hover:bg-gray-600'} ${isWebcamOn ? 'opacity-50 cursor-not-allowed' : ''}`}
                                        title="Ultra-low latency (480x360, 10fps)"
                                    >
                                        Low Latency
                                    </button>
                                    <button
                                        onClick={() => setStreamQuality('normal')}
                                        disabled={isWebcamOn}
                                        className={`px-3 py-2 text-sm rounded-md ${streamQuality === 'normal' ? 'bg-cyan-600 text-white' : 'bg-gray-700 hover:bg-gray-600'} ${isWebcamOn ? 'opacity-50 cursor-not-allowed' : ''}`}
                                        title="Normal quality (640x480, 15fps)"
                                    >
                                        Normal
                                    </button>
                                </div>
                                <button
                                    onClick={async () => {
                                        if (!rtspUrl) {
                                            setStatusText('❌ Please enter RTSP URL first');
                                            return;
                                        }
                                        setStatusText('🧪 Testing RTSP connection...');
                                        try {
                                            const response = await fetch(`/api/test-rtsp?url=${encodeURIComponent(rtspUrl)}`);
                                            const result = await response.json();
                                            if (result.success) {
                                                setStatusText('✅ RTSP connection test successful!');
                                            } else {
                                                setStatusText(`❌ RTSP test failed: ${result.message}`);
                                            }
                                        } catch (error: any) {
                                            setStatusText('❌ RTSP test error: ' + error.message);
                                        }
                                    }}
                                    disabled={isWebcamOn || !rtspUrl}
                                    className="px-3 py-2 text-sm rounded-md bg-yellow-600 hover:bg-yellow-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
                                    title="Test RTSP connection"
                                >
                                    Test Connection
                                </button>
                            </div>

                            <div className="text-xs text-yellow-400 bg-yellow-900/20 p-2 rounded">
                                <strong>⚠️ RTSP Requirements:</strong>
                                <br />• FFmpeg must be installed on the server
                                <br />• RTSP camera must be accessible from server
                                <br />• Stream will be converted to MP4 for browser compatibility
                                <br />
                                <br />
                                <strong>Quality Options:</strong>
                                <br />• <span className="text-green-400">Low Latency:</span> 480x360, 10fps, 200kbps (minimal delay)
                                <br />• <span className="text-cyan-400">Normal:</span> 640x480, 15fps, 500kbps (balanced quality)
                            </div>
                        </div>
                    )}

                    <div className="flex items-center gap-2 text-sm">
                        <label>Detection Sensitivity:</label>
                        <input type="range" min="0.05" max="0.3" step="0.01" value={detectionSensitivity} onChange={(e) => setDetectionSensitivity(parseFloat(e.target.value))} className="flex-1" />
                        <span className="text-cyan-400">{detectionSensitivity.toFixed(2)}</span>
                    </div>

                    <div className="text-xs text-gray-400 bg-gray-800/50 p-3 rounded">
                        <strong>System Status:</strong>
                        <br />• Recording: <span className={mediaRecorderRef.current?.state === 'recording' ? 'text-green-400' : 'text-gray-500'}>{mediaRecorderRef.current?.state || 'Not started'}</span>
                        <br />• Auto-recording: <span className={isAutoRecording ? 'text-green-400' : 'text-gray-500'}>{isAutoRecording ? 'ON' : 'OFF'}</span>
                        <br />• Stream Source: <span className="text-cyan-400">{streamSource.toUpperCase()}</span>
                        <br />• Stream Status: <span className={isWebcamOn ? 'text-green-400' : 'text-gray-500'}>{isWebcamOn ? 'ON' : 'OFF'}</span>
                        <br />• Authentication: <span className={token ? 'text-green-400' : 'text-red-400'}>{token ? 'Logged in' : 'Not logged in'}</span>
                        <br />• Video Chunks: <span className="text-cyan-400">{chunksRef.current.length}</span>
                        <br />• Detection Status: <span className={theftDetected ? 'text-red-400 font-bold' : 'text-green-400'}>{theftDetected ? 'THEFT DETECTED' : 'Monitoring'}</span>
                        <br />• Cash Device: <span className={deviceROIRef.current ? 'text-green-400' : 'text-yellow-400'}>{deviceROIRef.current ? 'Tracked' : 'Searching...'}</span>
                        <br />• Drawer: <span className={drawerStateRef.current === 'open' ? 'text-green-400' : 'text-gray-400'}>{drawerStateRef.current.toUpperCase()}</span>
                    </div>

                    <div className="flex items-center gap-2 text-sm text-green-400">
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                        <span>Auto-Theft Detection: ACTIVE</span>
                    </div>

                    <button
                        onClick={() => {
                            if (!isWebcamOn) {
                                setStatusText('❌ Please start monitoring first');
                                return;
                            }
                            if (!token) {
                                setStatusText('❌ Please login first');
                                return;
                            }
                            setTheftDetected(true);
                            setStatusText('🧪 TEST: Theft detection triggered manually');
                            if (!mediaRecorderRef.current || mediaRecorderRef.current.state !== 'recording') {
                                startShortRecording();
                            }
                            setTimeout(() => {
                                stopRecordingAndUpload();
                            }, 8000);
                            setTimeout(() => {
                                setTheftDetected(false);
                                setStatusText('Auto-monitoring enabled. Watching for suspicious behavior...');
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
                        <strong>Fraud Detection Scenarios:</strong>
                        <br />• <span className="text-yellow-400">Hand-to-Pocket Movement:</span> Cashier putting money in pocket
                        <br />• <span className="text-yellow-400">Hand-to-Hip Area:</span> Cashier concealing cash near hip
                        <br />• <span className="text-yellow-400">Suspicious Hand Position:</span> Hand moving toward body below shoulder level
                        <br />• <span className="text-yellow-400">Concealment Gesture:</span> Hand aligned with hip area (pocket region)
                        <br />
                        <br />
                        <strong>Detection Requirements:</strong>
                        <br />• Key landmarks (wrist, hip) should be reasonably visible
                        <br />• Temporal smoothing used to avoid flicker
                        <br />
                        <br />
                        <button
                            onClick={() => {
                                birdsEyePriorityRef.current = !birdsEyePriorityRef.current;
                                setStatusText(`Birds-eye priority: ${birdsEyePriorityRef.current ? 'ON' : 'OFF'}`);
                            }}
                            className={`px-3 py-2 text-sm rounded-md ${birdsEyePriorityRef.current ? 'bg-indigo-600 text-white' : 'bg-gray-700 hover:bg-gray-600'}`}
                            title="Prioritize top-down logic in theft detection"
                        >
                            Birds-eye Priority: {birdsEyePriorityRef.current ? 'ON' : 'OFF'}
                        </button>
                    </div>
                </div>
            </div>
        </div>
    );
};

export default MonitorCashiers;