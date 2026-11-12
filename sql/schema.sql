-- create database
CREATE DATABASE IF NOT EXISTS attendance_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
USE attendance_db;

-- Semesters table
CREATE TABLE IF NOT EXISTS semesters (
  id INT AUTO_INCREMENT PRIMARY KEY,
  semester VARCHAR(64) NOT NULL UNIQUE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Subjects
CREATE TABLE IF NOT EXISTS subjects (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  semester VARCHAR(64) NOT NULL,
  UNIQUE KEY unique_subject (name, semester)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Timetable
CREATE TABLE IF NOT EXISTS timetable (
  id INT AUTO_INCREMENT PRIMARY KEY,
  day_of_week VARCHAR(10) NOT NULL,
  subject_id INT NOT NULL,
  start_time TIME NOT NULL,
  end_time TIME NOT NULL,
  FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Teachers
CREATE TABLE IF NOT EXISTS teachers (
  id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  UNIQUE KEY (name)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Batches (for lab sections)
CREATE TABLE IF NOT EXISTS batches (
  id INT AUTO_INCREMENT PRIMARY KEY,
  subject_id INT NOT NULL,
  name VARCHAR(64) NOT NULL,          -- e.g. B1, B2
  usn_start VARCHAR(64) NOT NULL,
  usn_end VARCHAR(64) NOT NULL,
  FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Students
CREATE TABLE IF NOT EXISTS students (
  id INT AUTO_INCREMENT PRIMARY KEY,
  display_name VARCHAR(255) NOT NULL,
  usn VARCHAR(64) UNIQUE,
  semester VARCHAR(64),
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Embeddings: vector_text stores the serialized vector (e.g. comma-separated floats)
CREATE TABLE IF NOT EXISTS embeddings (
  id INT AUTO_INCREMENT PRIMARY KEY,
  student_id INT NOT NULL,
  vector_text LONGTEXT NOT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
  UNIQUE KEY u_student (student_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Sessions (teacher class instances) - we keep batch_id nullable
CREATE TABLE IF NOT EXISTS sessions (
  id INT AUTO_INCREMENT PRIMARY KEY,
  teacher_id INT,
  subject_id INT NOT NULL,
  date DATE NOT NULL,
  batch_id INT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY (teacher_id) REFERENCES teachers(id) ON DELETE SET NULL,
  FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
  FOREIGN KEY (batch_id) REFERENCES batches(id) ON DELETE SET NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Attendance
CREATE TABLE IF NOT EXISTS attendance (
  id INT AUTO_INCREMENT PRIMARY KEY,
  date DATE NOT NULL,
  subject_id INT NOT NULL,
  student_id INT NOT NULL,
  status ENUM('Present','Absent') NOT NULL,
  time_marked TIME NULL,
  attention_score FLOAT NULL,
  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  UNIQUE KEY u_attendance (date, subject_id, student_id),
  FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
  FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Index optimization
CREATE INDEX idx_students_semester ON students (semester);
CREATE INDEX idx_attendance_date ON attendance (date);
CREATE INDEX idx_sessions_date ON sessions (date);

