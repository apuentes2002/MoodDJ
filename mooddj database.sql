-- Create database
DROP DATABASE IF EXISTS mooddj;
CREATE DATABASE mooddj;
USE mooddj;

-- Table: Users
CREATE TABLE users (
    user_id INT AUTO_INCREMENT PRIMARY KEY,
    spotify_id VARCHAR(100) UNIQUE NOT NULL,
    display_name VARCHAR(100),
    email VARCHAR(150),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table: Moods
CREATE TABLE moods (
    mood_id INT AUTO_INCREMENT PRIMARY KEY,
    mood_name ENUM('happy','sad','excited','calm','neutral') NOT NULL
);

-- Table: Songs
CREATE TABLE songs (
    song_id INT AUTO_INCREMENT PRIMARY KEY,
    spotify_song_id VARCHAR(100) UNIQUE NOT NULL,
    title VARCHAR(200) NOT NULL,
    artist VARCHAR(200),
    album VARCHAR(200),
    duration_ms INT,
    valence FLOAT,
    energy FLOAT,
    tempo FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Junction Table: UserSongs (links users ↔ songs ↔ moods)
CREATE TABLE user_songs (
    user_song_id INT AUTO_INCREMENT PRIMARY KEY,
    user_id INT NOT NULL,
    song_id INT NOT NULL,
    mood_id INT NOT NULL,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (song_id) REFERENCES songs(song_id) ON DELETE CASCADE,
    FOREIGN KEY (mood_id) REFERENCES moods(mood_id) ON DELETE CASCADE
);

-- Insert default moods
INSERT INTO moods (mood_name) VALUES ('happy'), ('sad'), ('excited'), ('calm'), ('neutral');
