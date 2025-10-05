import express from 'express';
import cors from 'cors';
import multer from 'multer';
import path from 'path';
import fs from 'fs';
import sqlite3 from 'sqlite3';
import jwt from 'jsonwebtoken';
import bcrypt from 'bcryptjs';

const app = express();
const PORT = process.env.PORT || 3002;
const JWT_SECRET = process.env.JWT_SECRET || 'dev_secret_change_me';
const DATA_DIR = path.resolve('data');
const CLIPS_DIR = path.join(DATA_DIR, 'clips');

fs.mkdirSync(CLIPS_DIR, { recursive: true });

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

app.listen(PORT, () => {
  console.log(`🚀 Server listening on http://localhost:${PORT}`);
  console.log(`📁 Clips directory: ${CLIPS_DIR}`);
  console.log(`🗄️ Database: ${path.join(DATA_DIR, 'clips.db')}`);
  console.log(`🔐 JWT Secret: ${JWT_SECRET.substring(0, 10)}...`);
});


