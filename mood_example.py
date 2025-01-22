import asyncio
from mood_agent import MoodMusicAgent

async def main():
    # Initialize the mood agent
    mood_agent = MoodMusicAgent()

    # Choose your mood
    mood = 'happy'  # can be 'happy', 'sad', 'relaxed', or 'energetic'
    
    try:
        # Create a playlist
        print(f"\nSearching for {mood} songs...")
        playlist = mood_agent.create_playlist(mood, num_songs=5)
        
        # Display the playlist
        print(f"\nYour {mood} playlist:")
        for song in playlist:
            print(f"{song['playlist_position']}. {song['title']} ({song['channel']})")
            print(f"   URL: {song['url']}")
            print(f"   Duration: {song['duration']}")
            print()
        
        # Export the playlist
        mood_agent.export_playlist(playlist, f"{mood}_playlist.json")
        print(f"\nPlaylist exported to {mood}_playlist.json")
            
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main()) 