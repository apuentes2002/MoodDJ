-- Create database
DROP DATABASE IF EXISTS mooddj_soundcloud;
CREATE DATABASE mooddj_soundcloud;
USE mooddj_soundcloud;

-- Table: Users
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    soundcloud_id VARCHAR(100) UNIQUE NOT NULL,
    username VARCHAR(100),
    email VARCHAR(150),
    followers INT DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: Moods
CREATE TABLE moods (
    mood_id INT AUTO_INCREMENT PRIMARY KEY,
    mood_name ENUM('happy','sad','chill','energetic','focus','neutral') NOT NULL
);

-- Table: Tracks
CREATE TABLE tracks (
    track_id INT AUTO_INCREMENT PRIMARY KEY,
    soundcloud_track_id VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(200),
    artist VARCHAR(200),
    genre VARCHAR(100),
    bpm FLOAT,
    play_count INT,
    mood_score FLOAT,
    mood_id INT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (mood_id) REFERENCES moods(mood_id)
);

-- Junction Table: UserTracks (links users ↔ tracks ↔ moods)
CREATE TABLE user_tracks (
    user_track_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    track_id INT NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (track_id) REFERENCES tracks(track_id) ON DELETE CASCADE
);

