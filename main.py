from agent import MusicAgent

def main():
    # Initialize the agent
    agent = MusicAgent()

    # Tifa's Theme inspired parameters - longer and louder version
    style = {
        'scale': 'major',        # Tifa's theme is primarily in major scale
        'length': 48,            # Doubled length for extended melody
        'tempo': 85,             # Keeping the waltz-like tempo
        'rhythm_complexity': 0.4, # Keeping the simple rhythm
        'octave': 4,             # Middle octave
        'key': 'F',              # F major
        'time_signature': (3, 4), # Waltz time signature
        'dynamics_range': (75, 110),  # Increased dynamics for louder sound
        'style_hints': {
            'waltz': True,        
            'legato': True,       
            'music_box': True,
            'forte': True         # Added hint for louder playing
        }
    }

    # Generate a longer melody
    melody = agent.generate_melody(style)

    # Generate harmonies for fuller sound
    harmonies = agent.harmonize(melody)
    
    # Generate additional harmonies
    extra_harmonies = agent.harmonize(melody)
    
    # Combine melody and all harmonies
    all_phrases = [melody] + harmonies + extra_harmonies

    # Export to MIDI file
    agent.export_midi(all_phrases, "tifa_theme_loud_extended.mid")

if __name__ == "__main__":
    main() 