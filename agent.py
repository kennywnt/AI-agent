import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
from enum import Enum
import midiutil
import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import random

class MusicGenerationError(Exception):
    pass

class Note(Enum):
    C = 0
    CS = 1
    D = 2
    DS = 3
    E = 4
    F = 5
    FS = 6
    G = 7
    GS = 8
    A = 9
    AS = 10
    B = 11

@dataclass
class MusicalPhrase:
    notes: List[int]
    durations: List[float]
    velocities: List[int]
    scale: List[Note]
    tempo: int

class MusicAgent:
    def __init__(self, model_path: str = None):
        self.model = self._load_model(model_path)
        self.memory = []
        self.scales = {
            'major': [0, 2, 4, 5, 7, 9, 11],
            'minor': [0, 2, 3, 5, 7, 8, 10],
            'pentatonic': [0, 2, 4, 7, 9]
        }

    def analyze_melody(self, notes: List[int], durations: List[float]) -> Dict:
        # Analyze musical features
        intervals = np.diff(notes)
        rhythm_patterns = self._find_rhythm_patterns(durations)
        
        return {
            'key': self._detect_key(notes),
            'scale': self._detect_scale(notes),
            'intervals': self._analyze_intervals(intervals),
            'rhythm': rhythm_patterns,
            'complexity': self._calculate_complexity(notes, durations)
        }

    def generate_melody(self, style: Dict) -> MusicalPhrase:
        scale = self.scales[style.get('scale', 'major')]
        length = style.get('length', 16)
        
        # Generate base melody following style constraints
        notes = self._generate_notes(scale, length, style)
        durations = self._generate_rhythm(length, style)
        velocities = self.generate_dynamics(length, style)
        
        return MusicalPhrase(
            notes=notes,
            durations=durations,
            velocities=velocities,
            scale=scale,
            tempo=style.get('tempo', 120)
        )

    def harmonize(self, melody: MusicalPhrase) -> List[MusicalPhrase]:
        harmonies = []
        chord_prog = self._generate_chord_progression(melody)
        
        # Generate different harmony voices
        for voice_type in ['bass', 'tenor', 'alto']:
            harmony = self._generate_harmony_voice(
                melody, 
                chord_prog, 
                voice_type
            )
            harmonies.append(harmony)
            
        return harmonies

    def export_midi(self, phrases: List[MusicalPhrase], filename: str):
        midi = midiutil.MIDIFile(len(phrases))
        
        for track, phrase in enumerate(phrases):
            midi.addTempo(track, 0, phrase.tempo)
            
            time = 0
            for note, duration, velocity in zip(
                phrase.notes, 
                phrase.durations, 
                phrase.velocities
            ):
                midi.addNote(
                    track, 0, note, 
                    time, duration, 
                    velocity
                )
                time += duration
                
        with open(filename, 'wb') as f:
            midi.writeFile(f)

    def _detect_key(self, notes: List[int]) -> Note:
        # Implement key detection algorithm
        note_counts = np.bincount(np.array(notes) % 12, minlength=12)
        return Note(np.argmax(note_counts))

    def _detect_scale(self, notes: List[int]) -> str:
        unique_notes = set(note % 12 for note in notes)
        
        # Compare with known scales
        best_match = None
        best_score = 0
        
        for scale_name, scale in self.scales.items():
            score = len(unique_notes.intersection(scale))
            if score > best_score:
                best_score = score
                best_match = scale_name
                
        return best_match

    def _generate_notes(self, scale: List[int], length: int, style: Dict) -> List[int]:
        notes = []
        prev_note = np.random.choice(scale)
        
        for _ in range(length):
            # Generate next note based on style and previous note
            weights = self._calculate_transition_weights(prev_note, scale, style)
            next_note = np.random.choice(scale, p=weights)
            notes.append(next_note)
            prev_note = next_note
            
        return notes

    def _generate_rhythm(self, length: int, style: Dict) -> List[float]:
        base_durations = [0.25, 0.5, 1.0]  # sixteenth, eighth, quarter notes
        complexity = style.get('rhythm_complexity', 0.5)
        
        durations = []
        remaining_time = length
        
        while remaining_time > 0:
            if remaining_time < min(base_durations):
                break
                
            available_durations = [d for d in base_durations if d <= remaining_time]
            weights = self._calculate_rhythm_weights(available_durations, complexity)
            duration = np.random.choice(available_durations, p=weights)
            
            durations.append(duration)
            remaining_time -= duration
            
        return durations

    def _calculate_complexity(self, notes: List[int], durations: List[float]) -> float:
        # Calculate melodic complexity
        interval_complexity = np.std(np.diff(notes))
        rhythm_complexity = np.std(durations)
        
        # Normalize and combine
        return (interval_complexity + rhythm_complexity) / 2

    def _generate_harmony_voice(
        self, 
        melody: MusicalPhrase, 
        chord_prog: List[List[int]], 
        voice_type: str
    ) -> MusicalPhrase:
        harmony_notes = []
        
        for melody_note, chord in zip(melody.notes, chord_prog):
            if voice_type == 'bass':
                note = chord[0]  # Root note
            elif voice_type == 'tenor':
                note = chord[1]  # Third
            else:  # alto
                note = chord[2]  # Fifth
                
            harmony_notes.append(note)
            
        return MusicalPhrase(
            notes=harmony_notes,
            durations=melody.durations.copy(),
            velocities=[80] * len(harmony_notes),  # Slightly quieter than melody
            scale=melody.scale,
            tempo=melody.tempo
        )

    def _load_model(self, model_path: str):
        if model_path:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        else:
            return None

    def _find_rhythm_patterns(self, durations: List[float]) -> Dict:
        # Analyze and identify common rhythm patterns
        patterns = {}
        window_size = 4  # Look for patterns up to 4 beats
        
        for i in range(len(durations) - window_size + 1):
            pattern = tuple(durations[i:i + window_size])
            patterns[pattern] = patterns.get(pattern, 0) + 1
            
        return patterns

    def _analyze_intervals(self, intervals: List[int]) -> Dict:
        # Analyze the frequency and types of intervals
        interval_counts = {}
        for interval in intervals:
            interval_counts[abs(interval)] = interval_counts.get(abs(interval), 0) + 1
            
        return {
            'counts': interval_counts,
            'avg_size': np.mean(np.abs(intervals)),
            'direction_changes': np.sum(np.diff(intervals) != 0)
        }

    def _calculate_transition_weights(self, prev_note: int, scale: List[int], style: Dict) -> np.ndarray:
        # Calculate probability weights for next note transitions
        weights = np.ones(len(scale))
        
        # Adjust weights based on interval preferences
        for i, note in enumerate(scale):
            interval = abs(note - prev_note)
            # Prefer smaller intervals
            weights[i] *= np.exp(-0.2 * interval)
            
        # Normalize weights
        return weights / np.sum(weights)

    def _calculate_rhythm_weights(self, durations: List[float], complexity: float) -> np.ndarray:
        # Calculate probability weights for rhythm choices
        weights = np.ones(len(durations))
        
        # Adjust weights based on complexity
        for i, duration in enumerate(durations):
            if complexity < 0.3:  # Prefer longer durations for simple rhythms
                weights[i] *= duration
            elif complexity > 0.7:  # Prefer shorter durations for complex rhythms
                weights[i] *= (1 / duration)
                
        # Normalize weights
        return weights / np.sum(weights)

    def _generate_chord_progression(self, melody: MusicalPhrase) -> List[List[int]]:
        # Generate basic chord progression based on melody
        chord_progression = []
        scale = melody.scale
        
        for note in melody.notes:
            # Simple triad construction
            root = note
            third = (note + 4) % 12  # Major third
            fifth = (note + 7) % 12  # Perfect fifth
            chord_progression.append([root, third, fifth])
            
        return chord_progression

    def evaluate_phrase(self, phrase: MusicalPhrase) -> Dict:
        """Evaluate musical qualities of a phrase"""
        return {
            'complexity': self._calculate_complexity(phrase.notes, phrase.durations),
            'rhythm_variety': self._calculate_rhythm_variety(phrase.durations),
            'melodic_range': self._calculate_melodic_range(phrase.notes),
            'consonance': self._calculate_consonance(phrase.notes)
        }

    def _calculate_rhythm_variety(self, durations: List[float]) -> float:
        """Calculate rhythm variation score"""
        unique_durations = len(set(durations))
        return unique_durations / len(durations)

    def _calculate_melodic_range(self, notes: List[int]) -> Dict:
        """Calculate melodic range statistics"""
        return {
            'range': max(notes) - min(notes),
            'avg_pitch': np.mean(notes),
            'std_pitch': np.std(notes)
        }

    def _calculate_consonance(self, notes: List[int]) -> float:
        """Calculate consonance score based on musical intervals"""
        consonant_intervals = {0, 3, 4, 7, 8, 9}  # Perfect and major intervals
        intervals = [abs(notes[i] - notes[i-1]) % 12 for i in range(1, len(notes))]
        consonant_count = sum(1 for i in intervals if i in consonant_intervals)
        return consonant_count / len(intervals)

    def save_model(self, path: str):
        """Save the trained model and its state"""
        if self.model is not None:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scales': self.scales,
                'memory': self.memory
            }, path)
        
    def load_model(self, path: str):
        """Load a trained model and its state"""
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.scales = checkpoint.get('scales', self.scales)
            self.memory = checkpoint.get('memory', [])

    def set_generation_constraints(self, constraints: Dict):
        """Set constraints for melody generation"""
        self.constraints = {
            'min_pitch': constraints.get('min_pitch', 48),  # C3
            'max_pitch': constraints.get('max_pitch', 84),  # C6
            'min_duration': constraints.get('min_duration', 0.25),
            'max_duration': constraints.get('max_duration', 4.0),
            'allowed_intervals': constraints.get('allowed_intervals', range(-12, 13)),
            'phrase_length': constraints.get('phrase_length', 8)
        }

    def _apply_constraints(self, phrase: MusicalPhrase) -> MusicalPhrase:
        """Apply musical constraints to a generated phrase"""
        notes = np.clip(phrase.notes, self.constraints['min_pitch'], self.constraints['max_pitch'])
        durations = np.clip(phrase.durations, self.constraints['min_duration'], self.constraints['max_duration'])
        
        # Ensure valid intervals
        for i in range(1, len(notes)):
            interval = notes[i] - notes[i-1]
            if interval not in self.constraints['allowed_intervals']:
                notes[i] = notes[i-1] + min(self.constraints['allowed_intervals'], 
                                          key=lambda x: abs(x-interval))
        
        return MusicalPhrase(
            notes=list(notes),
            durations=list(durations),
            velocities=phrase.velocities,
            scale=phrase.scale,
            tempo=phrase.tempo
        )

    def analyze_composition(self, phrase: MusicalPhrase) -> Dict:
        """Comprehensive analysis of a musical phrase"""
        return {
            'structure': self._analyze_structure(phrase),
            'harmony': self._analyze_harmony(phrase),
            'statistics': self._calculate_statistics(phrase),
            'style_features': self._extract_style_features(phrase)
        }

    def _analyze_structure(self, phrase: MusicalPhrase) -> Dict:
        """Analyze the structural elements of the phrase"""
        return {
            'motifs': self._identify_motifs(phrase.notes),
            'phrases': self._segment_phrases(phrase),
            'repetitions': self._find_repetitions(phrase.notes)
        }

    def _analyze_harmony(self, phrase: MusicalPhrase) -> Dict:
        """Analyze harmonic content"""
        return {
            'chord_progression': self._identify_chords(phrase),
            'tonality': self._analyze_tonality(phrase.notes),
            'voice_leading': self._analyze_voice_leading(phrase)
        }

    def _calculate_statistics(self, phrase: MusicalPhrase) -> Dict:
        """Calculate various musical statistics"""
        return {
            'pitch_histogram': np.bincount(phrase.notes),
            'rhythm_density': len(phrase.notes) / sum(phrase.durations),
            'average_velocity': np.mean(phrase.velocities),
            'pitch_range': max(phrase.notes) - min(phrase.notes)
        }

    def _extract_style_features(self, phrase: MusicalPhrase) -> Dict:
        """Extract style-related features"""
        return {
            'articulation': self._analyze_articulation(phrase),
            'dynamics': self._analyze_dynamics(phrase),
            'tempo_variations': self._analyze_tempo_variations(phrase)
        }

    def _validate_phrase(self, phrase: MusicalPhrase) -> bool:
        """Validate a musical phrase"""
        if not phrase.notes or not phrase.durations:
            raise MusicGenerationError("Empty phrase")
        
        if len(phrase.notes) != len(phrase.durations):
            raise MusicGenerationError("Mismatched notes and durations")
        
        if not all(0 <= note <= 127 for note in phrase.notes):
            raise MusicGenerationError("Invalid MIDI note numbers")
        
        if not all(duration > 0 for duration in phrase.durations):
            raise MusicGenerationError("Invalid note durations")
        
        return True

    def _validate_style(self, style: Dict) -> bool:
        """Validate style parameters"""
        required_keys = ['scale', 'tempo', 'rhythm_complexity']
        if not all(key in style for key in required_keys):
            raise MusicGenerationError(f"Missing required style parameters: {required_keys}")
        
        if style['tempo'] <= 0:
            raise MusicGenerationError("Tempo must be positive")
        
        if not 0 <= style['rhythm_complexity'] <= 1:
            raise MusicGenerationError("Rhythm complexity must be between 0 and 1")
        
        return True

    def generate_dynamics(self, length, style):
        """Generate velocity values (dynamics) for the notes"""
        # Default base velocity (medium loudness)
        base_velocity = 80
        
        # Generate slight variations in velocity for more natural sound
        velocities = []
        for _ in range(length):
            # Add random variation of ±10 to the base velocity
            velocity = base_velocity + random.randint(-10, 10)
            # Ensure velocity stays within MIDI bounds (0-127)
            velocity = max(0, min(127, velocity))
            velocities.append(velocity)
        
        return velocities