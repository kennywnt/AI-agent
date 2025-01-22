from agent import MusicAgent

def main():
    # Initialize the agent
    agent = MusicAgent()

    # Mario Theme inspired parameters
    style = {
        'scale': 'major',        # Mario theme uses bright major scale
        'length': 32,            # Length for the iconic melody
        'tempo': 160,            # Fast, energetic tempo
        'rhythm_complexity': 0.7, # More bouncy, complex rhythm
        'octave': 5,             # Higher octave for that bright sound
        'key': 'C',              # C major like the original
        'dynamics_range': (85, 120),  # Loud, bouncy dynamics
        'style_hints': {
            'staccato': True,     # Short, bouncy notes
            'upbeat': True,       # Energetic feel
            'forte': True         # Strong, bold sound
        }
    }

    # Generate a Mario-style melody
    melody = agent.generate_melody(style)

    # Generate harmonies (Mario has strong accompaniment)
    harmonies = agent.harmonize(melody)
    
    # Generate additional harmonies for fuller sound
    extra_harmonies = agent.harmonize(melody)
    
    # Combine melody and all harmonies
    all_phrases = [melody] + harmonies + extra_harmonies

    # Export to MIDI file
    agent.export_midi(all_phrases, "mario_theme_inspired.mid")

if __name__ == "__main__":
    main() 