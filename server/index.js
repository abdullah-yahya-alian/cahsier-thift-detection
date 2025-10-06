import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import sqlite3 from 'sqlite3';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';
import { spawn } from 'child_process';
import { fileURLToPath } from 'url';

const app = express();
const PORT = process.env.PORT || 3003;
const JWT_SECRET = process.env.JWT_SECRET || 'dev_secret_change_me';
const DATA_DIR = path.resolve('data');
const CLIPS_DIR = path.join(DATA_DIR, 'clips');

// Dynamic FFmpeg path detection
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const findFFmpegPath = () => {
    const possiblePaths = [
        'C:\\FFmpeg\\bin\\ffmpeg.exe',
        'C:\\ffmpeg\\ffmpeg-8.0-essentials_build\\bin\\ffmpeg.exe',
        'C:\\ffmpeg\\bin\\ffmpeg.exe',
        'C:\\Program Files\\FFmpeg\\bin\\ffmpeg.exe',
        'C:\\Program Files (x86)\\FFmpeg\\bin\\ffmpeg.exe',
        'ffmpeg', // System PATH
        'C:\\ffmpeg\\ffmpeg-7.0-essentials_build\\bin\\ffmpeg.exe',
        'C:\\ffmpeg\\ffmpeg-6.0-essentials_build\\bin\\ffmpeg.exe',
        'C:\\ffmpeg\\ffmpeg-5.0-essentials_build\\bin\\ffmpeg.exe',
        'C:\\ffmpeg\\ffmpeg-4.0-essentials_build\\bin\\ffmpeg.exe',
        'C:\\ffmpeg\\ffmpeg-3.0-essentials_build\\bin\\ffmpeg.exe',
    ];

    // First, try to find FFmpeg in common locations
    for (const ffmpegPath of possiblePaths) {
        if (fs.existsSync(ffmpegPath)) {
            console.log(`✅ Found FFmpeg at: ${ffmpegPath}`);
            return ffmpegPath;
        }
    }

    // Try to find ffmpeg in PATH
    return new Promise((resolve) => {
        const testProcess = spawn('ffmpeg', ['-version'], { shell: true });
        testProcess.on('error', () => {
            console.log('❌ FFmpeg not found in PATH');
            resolve(null);
        });
        testProcess.on('close', (code) => {
            if (code === 0) {
                console.log('✅ Found FFmpeg in system PATH');
                resolve('ffmpeg');
            } else {
                console.log('❌ FFmpeg not found in PATH');
                resolve(null);
            }
        });
    });
};

// Function to add FFmpeg directory to PATH for this process
const addFFmpegToPath = (ffmpegPath) => {
    if (ffmpegPath && ffmpegPath !== 'ffmpeg' && ffmpegPath.endsWith('ffmpeg.exe')) {
        const ffmpegDir = path.dirname(ffmpegPath);
        const currentPath = process.env.PATH || '';
        if (!currentPath.includes(ffmpegDir)) {
            process.env.PATH = `${ffmpegDir};${currentPath}`;
            console.log(`📁 Added FFmpeg directory to PATH: ${ffmpegDir}`);
        }
    }
};

let FFMPEG_PATH = null;

fs.mkdirSync(CLIPS_DIR, { recursive: true });

// Initialize FFmpeg path detection
(async () => {
    FFMPEG_PATH = await findFFmpegPath();
    if (FFMPEG_PATH) {
        addFFmpegToPath(FFMPEG_PATH);
        console.log('🎬 FFmpeg initialized successfully for RTSP streaming');
    } else {
        console.log('⚠️  FFmpeg not found! RTSP streaming will not work.');
        console.log('Please install FFmpeg and add it to your PATH or place it in one of these locations:');
        console.log('- C:\\FFmpeg\\bin\\ffmpeg.exe');
        console.log('- C:\\ffmpeg\\ffmpeg-8.0-essentials_build\\bin\\ffmpeg.exe');
        console.log('- C:\\ffmpeg\\bin\\ffmpeg.exe');
        console.log('- C:\\ffmpeg\\ffmpeg-7.0-essentials_build\\bin\\ffmpeg.exe');
        console.log('- C:\\ffmpeg\\ffmpeg-6.0-essentials_build\\bin\\ffmpeg.exe');
        console.log('- C:\\ffmpeg\\ffmpeg-5.0-essentials_build\\bin\\ffmpeg.exe');
    }
})();

app.use(cors());
app.use(express.json());
app.use('/clips', express.static(CLIPS_DIR));

// SQLite setup
const db = new sqlite3.Database(path.join(DATA_DIR, 'clips.db'));
db.serialize(() => {
  db.run(`CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'cashier'
  )`);
  db.run(`CREATE TABLE IF NOT EXISTS clips (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    cashier_name TEXT NOT NULL,
    from_time TEXT NOT NULL,
    to_time TEXT NOT NULL,
    file_path TEXT NOT NULL,
    created_at TEXT NOT NULL
  )`);
});

// Seed demo users if none
db.get('SELECT COUNT(*) as count FROM users', (err, row) => {
  if (!row || row.count === 0) {
    const users = [
      { username: 'admin', password: 'admin123', role: 'admin' },
      { username: 'cashier1', password: 'cashier123', role: 'cashier' },
    ];
    users.forEach(u => {
      const hash = bcrypt.hashSync(u.password, 10);
      db.run('INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)', [u.username, hash, u.role]);
    });
    console.log('Seeded demo users.');
  }
});

function authMiddleware(req, res, next) {
  const header = req.headers.authorization || '';
  const token = header.startsWith('Bearer ') ? header.slice(7) : null;
  if (!token) return res.status(401).json({ error: 'Unauthorized' });
  try {
    req.user = jwt.verify(token, JWT_SECRET);
    next();
  } catch (e) {
    res.status(401).json({ error: 'Invalid token' });
  }
}

// RTSP Stream endpoint
app.get('/api/stream', (req, res) => {
    const rtspUrl = req.query.url;
    
    if (!rtspUrl) {
        return res.status(400).json({ error: 'RTSP URL is required' });
    }
    
    if (!FFMPEG_PATH) {
        return res.status(500).json({ error: 'FFmpeg not available' });
    }
    
    console.log(`🎥 Starting RTSP stream: ${rtspUrl}`);
    
    // Set headers for streaming
    res.setHeader('Content-Type', 'video/mp4');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');
    
    // FFmpeg command to convert RTSP to MP4 stream with optimized settings for low latency
    const ffmpegArgs = [
        '-i', rtspUrl,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-f', 'mp4',
        '-movflags', 'frag_keyframe+empty_moov+default_base_moof',
        '-fflags', '+genpts+flush_packets',
        '-r', '15', // Reduced framerate for better performance
        '-s', '640x480', // Reduced resolution for lower latency
        '-b:v', '500k', // Lower bitrate for faster streaming
        '-maxrate', '800k',
        '-bufsize', '1M',
        '-g', '15', // Keyframe interval
        '-keyint_min', '15',
        '-sc_threshold', '0',
        '-threads', '2', // Limit threads for better performance
        '-avoid_negative_ts', 'make_zero',
        '-fflags', '+genpts',
        'pipe:1'
    ];
    
    const ffmpegProcess = spawn(FFMPEG_PATH, ffmpegArgs, {
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true
    });
    
    let isStreaming = true;
    
    // Handle FFmpeg output with backpressure control
    ffmpegProcess.stdout.on('data', (chunk) => {
        if (isStreaming && !res.destroyed) {
            const canContinue = res.write(chunk);
            if (!canContinue) {
                // Pause FFmpeg output if response is backed up
                ffmpegProcess.stdout.pause();
                res.once('drain', () => {
                    ffmpegProcess.stdout.resume();
                });
            }
        }
    });
    
    // Handle errors with better logging
    ffmpegProcess.stderr.on('data', (data) => {
        const errorMsg = data.toString();
        // Only log important errors, not warnings
        if (errorMsg.includes('error') || errorMsg.includes('Error') || errorMsg.includes('failed')) {
            console.error(`FFmpeg error: ${errorMsg}`);
        }
    });
    
    ffmpegProcess.on('error', (error) => {
        console.error('FFmpeg process error:', error);
        isStreaming = false;
        if (!res.headersSent) {
            res.status(500).json({ error: 'Stream error: ' + error.message });
        } else {
            res.end();
        }
    });
    
    ffmpegProcess.on('close', (code) => {
        console.log(`FFmpeg process exited with code ${code}`);
        isStreaming = false;
        if (!res.headersSent) {
            if (code === 0) {
                res.status(200).end();
            } else {
                res.status(500).json({ error: `Stream ended with code ${code}` });
            }
        } else {
            res.end();
        }
    });
    
    // Handle client disconnect
    req.on('close', () => {
        console.log('Client disconnected, killing FFmpeg process');
        isStreaming = false;
        ffmpegProcess.kill('SIGTERM');
    });
    
    req.on('error', (error) => {
        console.error('Request error:', error);
        isStreaming = false;
        ffmpegProcess.kill('SIGTERM');
    });
    
    // Handle response errors
    res.on('error', (error) => {
        console.error('Response error:', error);
        isStreaming = false;
        ffmpegProcess.kill('SIGTERM');
    });
});

app.post('/api/login', (req, res) => {
  console.log("🔐 POST /api/login", req.body);
  const { username, password } = req.body;
  db.get('SELECT * FROM users WHERE username = ?', [username], (err, user) => {
    if (err || !user) {
      console.log("❌ Invalid credentials for user:", username);
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    if (!bcrypt.compareSync(password, user.password_hash)) {
      console.log("❌ Invalid password for user:", username);
      return res.status(401).json({ error: 'Invalid credentials' });
    }
    const token = jwt.sign({ id: user.id, username: user.username, role: user.role }, JWT_SECRET, { expiresIn: '12h' });
    console.log("✅ Login successful for user:", username, "role:", user.role);
    res.json({ token, user: { id: user.id, username: user.username, role: user.role } });
  });
});

const storage = multer.diskStorage({
  destination: function (req, file, cb) {
    cb(null, CLIPS_DIR);
  },
  filename: function (req, file, cb) {
    const unique = Date.now() + '-' + Math.round(Math.random() * 1e9);
    cb(null, unique + path.extname(file.originalname || '.webm'));
  }
});
const upload = multer({ storage });

app.post('/api/clips', authMiddleware, upload.single('clip'), (req, res) => {
  console.log('🎬 POST /api/clips - Upload request received');
  console.log('📋 Request body:', req.body);
  console.log('📁 Request file:', req.file ? {
    filename: req.file.filename,
    size: req.file.size,
    mimetype: req.file.mimetype,
    path: req.file.path
  } : 'No file');
  
  const { cashierName, fromTime, toTime } = req.body;
  if (!req.file) {
    console.log('❌ No file uploaded');
    return res.status(400).json({ error: 'No file uploaded' });
  }
  
  const record = {
    cashier_name: cashierName,
    from_time: fromTime,
    to_time: toTime,
    file_path: `/clips/${req.file.filename}`,
    created_at: new Date().toISOString(),
  };
  
  console.log('💾 Saving to database:', record);
  
  db.run('INSERT INTO clips (cashier_name, from_time, to_time, file_path, created_at) VALUES (?, ?, ?, ?, ?)',
    [record.cashier_name, record.from_time, record.to_time, record.file_path, record.created_at],
    function (err) {
      if (err) {
        console.log('❌ Database error:', err);
        return res.status(500).json({ error: 'DB error' });
      }
      console.log('✅ Clip saved successfully with ID:', this.lastID);
      res.json({ id: this.lastID, ...record });
    });
});

// Alternative low-latency RTSP stream endpoint
app.get('/api/stream-lowlatency', (req, res) => {
    const rtspUrl = req.query.url;
    
    if (!rtspUrl) {
        return res.status(400).json({ error: 'RTSP URL is required' });
    }
    
    if (!FFMPEG_PATH) {
        return res.status(500).json({ error: 'FFmpeg not available' });
    }
    
    console.log(`🎥 Starting low-latency RTSP stream: ${rtspUrl}`);
    
    // Set headers for streaming
    res.setHeader('Content-Type', 'video/mp4');
    res.setHeader('Cache-Control', 'no-cache');
    res.setHeader('Connection', 'keep-alive');
    res.setHeader('Access-Control-Allow-Origin', '*');
    
    // Ultra-low latency FFmpeg settings
    const ffmpegArgs = [
        '-i', rtspUrl,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-tune', 'zerolatency',
        '-profile:v', 'baseline',
        '-level', '3.0',
        '-f', 'mp4',
        '-movflags', 'frag_keyframe+empty_moov+default_base_moof+faststart',
        '-fflags', '+genpts+flush_packets+ignidx',
        '-r', '10', // Very low framerate for minimal latency
        '-s', '480x360', // Very low resolution
        '-b:v', '200k', // Very low bitrate
        '-maxrate', '300k',
        '-bufsize', '500k',
        '-g', '10', // Frequent keyframes
        '-keyint_min', '10',
        '-sc_threshold', '0',
        '-threads', '1', // Single thread
        '-avoid_negative_ts', 'make_zero',
        '-analyzeduration', '1000000', // 1 second analysis
        '-probesize', '1000000', // 1MB probe size
        'pipe:1'
    ];
    
    const ffmpegProcess = spawn(FFMPEG_PATH, ffmpegArgs, {
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true
    });
    
    let isStreaming = true;
    
    // Direct pipe for minimal latency
    ffmpegProcess.stdout.pipe(res, { end: false });
    
    ffmpegProcess.stderr.on('data', (data) => {
        const errorMsg = data.toString();
        if (errorMsg.includes('error') || errorMsg.includes('Error')) {
            console.error(`Low-latency FFmpeg error: ${errorMsg}`);
        }
    });
    
    ffmpegProcess.on('error', (error) => {
        console.error('Low-latency FFmpeg process error:', error);
        isStreaming = false;
        if (!res.headersSent) {
            res.status(500).json({ error: 'Low-latency stream error' });
        }
    });
    
    ffmpegProcess.on('close', (code) => {
        console.log(`Low-latency FFmpeg process exited with code ${code}`);
        isStreaming = false;
        res.end();
    });
    
    req.on('close', () => {
        console.log('Low-latency client disconnected');
        isStreaming = false;
        ffmpegProcess.kill('SIGTERM');
    });
});

// Test RTSP connection endpoint
app.get('/api/test-rtsp', (req, res) => {
    const rtspUrl = req.query.url;
    
    if (!rtspUrl) {
        return res.status(400).json({ error: 'RTSP URL is required' });
    }
    
    if (!FFMPEG_PATH) {
        return res.status(500).json({ error: 'FFmpeg not available' });
    }
    
    console.log(`🧪 Testing RTSP connection: ${rtspUrl}`);
    
    // Test RTSP connection with a short probe
    const testArgs = [
        '-i', rtspUrl,
        '-t', '5', // Test for 5 seconds
        '-f', 'null', // No output file
        '-'
    ];
    
    const testProcess = spawn(FFMPEG_PATH, testArgs, {
        stdio: ['ignore', 'pipe', 'pipe'],
        windowsHide: true
    });
    
    let output = '';
    let errorOutput = '';
    
    testProcess.stdout.on('data', (data) => {
        output += data.toString();
    });
    
    testProcess.stderr.on('data', (data) => {
        errorOutput += data.toString();
    });
    
    
    // Timeout after 10 seconds
    const timeout = setTimeout(() => {
        if (!testProcess.killed) {
            testProcess.kill('SIGTERM');
            if (!res.headersSent) {
                res.status(408).json({
                    success: false,
                    message: 'RTSP test timeout',
                    error: 'Connection test took too long'
                });
            }
        }
    }, 10000);
    
    // Clear timeout when process completes
    testProcess.on('close', (code) => {
        clearTimeout(timeout);
        console.log(`RTSP test completed with code: ${code}`);
        console.log('Test output:', output);
        console.log('Test errors:', errorOutput);
        
        if (!res.headersSent) {
            if (code === 0) {
                res.json({
                    success: true,
                    message: 'RTSP stream is accessible',
                    code: code,
                    output: output.substring(0, 500), // Limit output length
                    errorOutput: errorOutput.substring(0, 500)
                });
            } else {
                res.status(500).json({
                    success: false,
                    message: 'RTSP stream test failed',
                    code: code,
                    output: output.substring(0, 500),
                    errorOutput: errorOutput.substring(0, 500)
                });
            }
        }
    });
    
    testProcess.on('error', (error) => {
        clearTimeout(timeout);
        console.error('RTSP test process error:', error);
        if (!res.headersSent) {
            res.status(500).json({
                success: false,
                message: 'Failed to start RTSP test',
                error: error.message
            });
        }
    });
});

// FFmpeg status endpoint
app.get('/api/ffmpeg-status', (req, res) => {
    res.json({
        available: !!FFMPEG_PATH,
        path: FFMPEG_PATH,
        message: FFMPEG_PATH ? 'FFmpeg is available' : 'FFmpeg not found'
    });
});

app.get('/api/clips', authMiddleware, (req, res) => {
  const { cashier, from, to } = req.query;
  const filters = [];
  const values = [];
  if (cashier) { filters.push('cashier_name = ?'); values.push(cashier); }
  if (from) { filters.push('created_at >= ?'); values.push(from); }
  if (to) { filters.push('created_at <= ?'); values.push(to); }
  const where = filters.length ? ('WHERE ' + filters.join(' AND ')) : '';
  db.all(`SELECT * FROM clips ${where} ORDER BY created_at DESC`, values, (err, rows) => {
    if (err) return res.status(500).json({ error: 'DB error' });
    res.json(rows);
  });
});

// RTSP stream proxy endpoint
app.get('/api/stream', (req, res) => {
  const { url } = req.query;
  
  if (!url) {
    return res.status(400).json({ error: 'RTSP URL is required' });
  }
  
  console.log('📡 Starting RTSP stream proxy for:', url);
  
  // Set headers for streaming
  res.writeHead(200, {
    'Content-Type': 'video/mp4',
    'Cache-Control': 'no-cache',
    'Connection': 'keep-alive',
    'Access-Control-Allow-Origin': '*',
  });
  
  // Use FFmpeg to convert RTSP to MP4 stream
  const ffmpegPath = 'C:\\ffmpeg\\ffmpeg-8.0-essentials_build\\bin\\ffmpeg.exe';
  const ffmpeg = spawn(ffmpegPath, [
    '-i', url,
    '-c:v', 'libx264',
    '-preset', 'ultrafast',
    '-tune', 'zerolatency',
    '-f', 'mp4',
    '-movflags', 'frag_keyframe+empty_moov',
    '-'
  ]);
  
  ffmpeg.stdout.on('data', (data) => {
    res.write(data);
  });
  
  ffmpeg.stderr.on('data', (data) => {
    console.log('FFmpeg stderr:', data.toString());
  });
  
  ffmpeg.on('close', (code) => {
    console.log('FFmpeg process exited with code:', code);
    res.end();
  });
  
  ffmpeg.on('error', (err) => {
    console.error('FFmpeg error:', err);
    if (!res.headersSent) {
      res.status(500).json({ error: 'Streaming failed' });
    }
  });
  
  // Handle client disconnect
  req.on('close', () => {
    console.log('Client disconnected, killing FFmpeg process');
    ffmpeg.kill();
  });
});

app.listen(PORT, () => {
  console.log(`🚀 Server listening on http://localhost:${PORT}`);
  console.log(`📁 Clips directory: ${CLIPS_DIR}`);
  console.log(`🗄️ Database: ${path.join(DATA_DIR, 'clips.db')}`);
  console.log(`🔐 JWT Secret: ${JWT_SECRET.substring(0, 10)}...`);
});


