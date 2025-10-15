import requests
import mysql.connector

# SoundCloud API setup
CLIENT_ID = "your_client_id"
USER_ID = "soundcloud_user_id"

# Fetch liked tracks
url = f"https://api.soundcloud.com/users/{USER_ID}/likes?client_id={CLIENT_ID}"
response = requests.get(url)
data = response.json()

# Connect to MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yourpassword",
    database="mooddj_soundcloud"
)
cursor = db.cursor()

for item in data:
    track = item["track"]
    title = track["title"]
    artist = track["user"]["username"]
    genre = track.get("genre", "Unknown")
    bpm = track.get("bpm", 0)
    play_count = track.get("playback_count", 0)

    sql = """
    INSERT INTO tracks (soundcloud_track_id, title, artist, genre, bpm, play_count)
    VALUES (%s, %s, %s, %s, %s, %s)
    ON DUPLICATE KEY UPDATE play_count = VALUES(play_count)
    """
    cursor.execute(sql, (track["id"], title, artist, genre, bpm, play_count))

db.commit()
cursor.close()
db.close()
