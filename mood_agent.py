import os
import json
from pathlib import Path
from typing import List, Dict
import time
from youtube_search import YoutubeSearch  # Changed to youtube_search package

class MoodMusicAgent:
    def __init__(self):
        self.cache_directory = Path("music_cache")
        self.cache_directory.mkdir(exist_ok=True)
        
        # Modified mood mappings
        self.mood_mappings = {
            'happy': ['happy music playlist', 'upbeat music mix', 'feel good songs'],
            'sad': ['sad songs playlist', 'emotional music mix', 'melancholic songs'],
            'relaxed': ['relaxing music mix', 'chill music playlist', 'lofi mix'],
            'energetic': ['workout music mix', 'gym motivation music', 'hype music mix']
        }

    def search_songs(self, mood: str, limit: int = 10) -> List[Dict]:
        """Search for songs matching the given mood using YouTube"""
        songs = []
        search_terms = self.mood_mappings.get(mood, [mood])
        
        try:
            # Try each search term until we find results
            for term in search_terms:
                print(f"Searching for: {term}")
                
                # Using YoutubeSearch
                results = YoutubeSearch(term, max_results=limit).to_dict()
                
                if not results:
                    print(f"No results found for term: {term}")
                    continue
                    
                for video in results:
                    try:
                        song = {
                            'title': video['title'],
                            'url': f"https://youtube.com{video['url_suffix']}",
                            'duration': video.get('duration', 'Unknown'),
                            'channel': video.get('channel', 'Unknown'),
                            'views': video.get('views', 'N/A'),
                            'thumbnail': video.get('thumbnails', [None])[0],
                            'source': 'YouTube'
                        }
                        songs.append(song)
                        print(f"Found song: {song['title']}")
                        
                    except KeyError as e:
                        print(f"Error processing video data: {e}")
                        continue
                    
                if songs:
                    break
                    
        except Exception as e:
            print(f"Error during search: {e}")
            import traceback
            traceback.print_exc()
            
        print(f"Total songs found: {len(songs)}")
        return songs[:limit]

    def create_playlist(self, mood: str, num_songs: int = 5) -> List[Dict]:
        """Create a playlist based on mood"""
        if mood not in self.mood_mappings:
            raise ValueError(f"Unsupported mood. Choose from: {list(self.mood_mappings.keys())}")
            
        print(f"\nCreating playlist for mood: {mood}")
        songs = self.search_songs(mood, num_songs)
        
        if not songs:
            print("Warning: No songs found for this mood")
            return []
            
        return self._format_playlist(songs, mood)

    def _format_playlist(self, songs: List[Dict], mood: str) -> List[Dict]:
        """Format the playlist with additional metadata"""
        playlist = []
        for i, song in enumerate(songs, 1):
            playlist_entry = {
                **song,
                'playlist_position': i,
                'mood': mood,
                'added_date': time.strftime('%Y-%m-%d %H:%M:%S')
            }
            playlist.append(playlist_entry)
        return playlist

    def export_playlist(self, playlist: List[Dict], filename: str = "playlist.json"):
        """Export playlist to a JSON file"""
        if not playlist:
            print(f"Warning: Attempting to export empty playlist to {filename}")
            return
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(playlist, f, indent=2, ensure_ascii=False)
        print(f"Playlist exported to {filename}")