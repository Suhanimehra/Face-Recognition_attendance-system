import sqlite3
from contextlib import closing

DB_PATH = 'face_attendance.db'

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with closing(get_db()) as db:
        db.executescript('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            enrollment TEXT UNIQUE NOT NULL,
            branch TEXT,
            year TEXT,
            class TEXT,
            email TEXT,
            registration_image BLOB
        );
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            name TEXT,
            time TEXT,
            class TEXT,
            image BLOB,
            FOREIGN KEY(user_id) REFERENCES users(id)
        );
        ''')
        db.commit()

def insert_user(name, enrollment, branch, year, class_, email, registration_image):
    db = get_db()
    db.execute('INSERT INTO users (name, enrollment, branch, year, class, email, registration_image) VALUES (?, ?, ?, ?, ?, ?, ?)',
               (name, enrollment, branch, year, class_, email, registration_image))
    db.commit()
    db.close()

def get_user_by_enrollment(enrollment):
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE enrollment = ?', (enrollment,)).fetchone()
    db.close()
    return user

def get_user_by_name(name):
    db = get_db()
    user = db.execute('SELECT * FROM users WHERE name = ?', (name,)).fetchone()
    db.close()
    return user

def insert_attendance(user_id, name, time, class_, image):
    db = get_db()
    db.execute('INSERT INTO attendance (user_id, name, time, class, image) VALUES (?, ?, ?, ?, ?)',
               (user_id, name, time, class_, image))
    db.commit()
    db.close()

def get_attendance_by_user_id(user_id):
    db = get_db()
    records = db.execute('SELECT * FROM attendance WHERE user_id = ?', (user_id,)).fetchall()
    db.close()
    return records 