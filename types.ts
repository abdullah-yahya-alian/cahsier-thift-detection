export interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface PoseLandmarkerResult {
  landmarks: Landmark[][];
  worldLandmarks: Landmark[][];
  segmentationMasks?: any[];
}

export interface Connection {
    start: number;
    end: number;
}

// These types correspond to the globals injected by the MediaPipe script
declare global {
  // We are defining this in the global scope because the MediaPipe library is loaded from a CDN.
  // This avoids TypeScript errors about 'mp' not being defined.
  var mp: {
    tasks: {
      vision: {
        PoseLandmarker: {
          createFromOptions: (
            filesetResolver: any,
            options: Record<string, any>
          ) => Promise<{ close: () => void; detectForVideo: (...args: any[]) => PoseLandmarkerResult; }>;
          POSE_CONNECTIONS: Connection[];
        };
        FilesetResolver: {
          forVisionTasks: (path: string) => Promise<any>;
        };
        DrawingUtils: {
          new (ctx: CanvasRenderingContext2D): {
            drawLandmarks: (landmarks: Landmark[], options?: any) => void;
            drawConnectors: (landmarks: Landmark[], connections: Connection[], options?: any) => void;
          };
          lerp: (x: number, y: number, z: number, a: number, b: number) => number;
        };
      }
    }
  }
}

// This is to make the linter happy, as we are not exporting a default value but we need this file to be a module.
export {};
