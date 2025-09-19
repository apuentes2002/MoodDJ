import spotipy
from spotipy.oauth2 import SpotifyOAuth

# Spotify credentials
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
    client_id="",
    client_secret="",
    redirect_uri="http://localhost:8888/callback",
    scope="user-library-read playlist-read-private"
))
import mysql.connector

db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="yourpassword",
    database="mooddj"
)

cursor = db.cursor()

results = sp.current_user_saved_tracks(limit=10)

for item in results['items']:
    track = item['track']
    title = track['name']
    artist = track['artists'][0]['name']
    album = track['album']['name']
    duration_ms = track['duration_ms']
    spotify_id = track['id']

    sql = """
        INSERT INTO songs (spotify_song_id, title, artist, album, duration_ms)
        VALUES (%s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE title=VALUES(title)
    """
    cursor.execute(sql, (spotify_id, title, artist, album, duration_ms))
    db.commit()

    print(f"✅ Added {title} by {artist}")

features = sp.audio_features([spotify_id])[0]

valence = features['valence']
energy = features['energy']
tempo = features['tempo']

cursor.execute(
    "UPDATE songs SET valence=%s, energy=%s, tempo=%s WHERE spotify_song_id=%s",
    (valence, energy, tempo, spotify_id)
)
db.commit()
