import numpy as np
from scipy.io.wavfile import write
import json
import scipy.signal as signal

# Expanded note frequencies
NOTE_FREQ = {
    'C0': 16.35, 'C#0': 17.32, 'D0': 18.35, 'D#0': 19.45, 'E0': 20.60, 'F0': 21.83, 
    'F#0': 23.12, 'G0': 24.50, 'G#0': 25.96, 'A0': 27.50, 'A#0': 29.14, 'B0': 30.87,
    'C1': 32.70, 'C#1': 34.65, 'D1': 36.71, 'D#1': 38.89, 'E1': 41.20, 'F1': 43.65, 
    'F#1': 46.25, 'G1': 49.00, 'G#1': 51.91, 'A1': 55.00, 'A#1': 58.27, 'B1': 61.74,
    'C2': 65.41, 'C#2': 69.30, 'D2': 73.42, 'D#2': 77.78, 'E2': 82.41, 'F2': 87.31, 
    'F#2': 92.50, 'G2': 98.00, 'G#2': 103.83, 'A2': 110.00, 'A#2': 116.54, 'B2': 123.47,
    'C3': 130.81, 'C#3': 138.59, 'D3': 146.83, 'D#3': 155.56, 'E3': 164.81, 'F3': 174.61, 
    'F#3': 185.00, 'G3': 196.00, 'G#3': 207.65, 'A3': 220.00, 'A#3': 233.08, 'B3': 246.94,
    'C4': 261.63, 'C#4': 277.18, 'D4': 293.66, 'D#4': 311.13, 'E4': 329.63, 'F4': 349.23, 
    'F#4': 369.99, 'G4': 392.00, 'G#4': 415.30, 'A4': 440.00, 'A#4': 466.16, 'B4': 493.88,
    'C5': 523.25, 'C#5': 554.37, 'D5': 587.33, 'D#5': 622.25, 'E5': 659.25, 'F5': 698.46, 
    'F#5': 739.99, 'G5': 783.99, 'G#5': 830.61, 'A5': 880.00, 'A#5': 932.33, 'B5': 987.77,
    'C6': 1046.50, 'C#6': 1108.73, 'D6': 1174.66, 'D#6': 1244.51, 'E6': 1318.51, 'F6': 1396.91, 
    'F#6': 1479.98, 'G6': 1567.98, 'G#6': 1661.22, 'A6': 1760.00, 'A#6': 1864.66, 'B6': 1975.53,
    'C7': 2093.00, 'C#7': 2217.46, 'D7': 2349.32, 'D#7': 2489.02, 'E7': 2637.02, 'F7': 2793.83, 
    'F#7': 2959.96, 'G7': 3135.96, 'G#7': 3322.44, 'A7': 3520.00, 'A#7': 3729.31, 'B7': 3951.07,
    'C8': 4186.01, 'REST': 0.0
}

class RealisticInstrumentSynthesizer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def generate_violin(self, freq, duration, volume):
        """
        More realistic violin sound synthesis using multiple techniques
        - Combines multiple harmonic components
        - Adds bow noise and overtone complexity
        - Uses ADSR envelope with more nuanced shaping
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Harmonic series with decreasing amplitude
        harmonics = [
            (1.0, freq),     # Fundamental frequency
            (0.5, freq * 2),  # First overtone
            (0.25, freq * 3), # Second overtone
            (0.125, freq * 4) # Third overtone
        ]
        
        # Generate wave with harmonic components
        wave = np.zeros_like(t)
        for amp, harmonic_freq in harmonics:
            wave += amp * np.sin(2 * np.pi * harmonic_freq * t)
        
        # Add subtle bow noise and vibrato
        bow_noise = 0.05 * np.random.normal(0, 1, len(t))
        vibrato = 0.02 * np.sin(2 * np.pi * 6 * t) * np.sin(2 * np.pi * freq * t)
        wave += bow_noise + vibrato
        
        # Advanced ADSR envelope
        envelope = self._create_violin_envelope(duration)
        
        return volume * wave * envelope

    def _create_violin_envelope(self, duration):
        """
        Create a more nuanced ADSR envelope for violin-like articulation
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # More complex envelope parameters
        attack = 0.05
        decay = 0.1
        sustain = 0.7
        release = 0.2
        
        envelope = np.ones_like(t)
        
        # Attack phase (quick rise)
        attack_samples = int(attack * self.sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 2
        
        # Decay phase (smooth reduction to sustain level)
        decay_samples = int(decay * self.sample_rate)
        if decay_samples > 0:
            decay_curve = np.linspace(1, sustain, decay_samples) ** 1.5
            envelope[attack_samples:attack_samples+decay_samples] = decay_curve
        
        # Sustain phase
        sustain_start = int((attack + decay) * self.sample_rate)
        sustain_end = int((duration - release) * self.sample_rate)
        envelope[sustain_start:sustain_end] = sustain
        
        # Release phase (gradual fade-out)
        release_samples = int(release * self.sample_rate)
        if release_samples > 0:
            release_curve = np.linspace(sustain, 0, release_samples) ** 0.5
            envelope[-release_samples:] = release_curve
        
        return envelope

    def generate_cello(self, freq, duration, volume):
        """
        Realistic cello sound with rich, warm tone
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Rich harmonic content typical of cello
        harmonics = [
            (1.0, freq),     # Fundamental
            (0.4, freq * 2),  # First overtone
            (0.2, freq * 3),  # Second overtone
            (0.1, freq * 4)   # Third overtone
        ]
        
        wave = np.zeros_like(t)
        for amp, harmonic_freq in harmonics:
            wave += amp * np.sin(2 * np.pi * harmonic_freq * t)
        
        # Add subtle string resonance and body noise
        body_resonance = 0.1 * np.sin(2 * np.pi * (freq/2) * t)
        string_noise = 0.03 * np.random.normal(0, 1, len(t))
        wave += body_resonance + string_noise
        
        # Warm, smooth envelope
        envelope = self._create_cello_envelope(duration)
        
        return volume * wave * envelope

    def _create_cello_envelope(self, duration):
        """
        Cello-specific envelope with warm attack and smooth release
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        attack = 0.1
        decay = 0.15
        sustain = 0.8
        release = 0.3
        
        envelope = np.ones_like(t)
        
        # Soft, warm attack
        attack_samples = int(attack * self.sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 1.5
        
        # Gradual decay
        decay_samples = int(decay * self.sample_rate)
        if decay_samples > 0:
            decay_curve = np.linspace(1, sustain, decay_samples) ** 1.3
            envelope[attack_samples:attack_samples+decay_samples] = decay_curve
        
        # Sustain phase
        sustain_start = int((attack + decay) * self.sample_rate)
        sustain_end = int((duration - release) * self.sample_rate)
        envelope[sustain_start:sustain_end] = sustain
        
        # Smooth release
        release_samples = int(release * self.sample_rate)
        if release_samples > 0:
            release_curve = np.linspace(sustain, 0, release_samples) ** 0.7
            envelope[-release_samples:] = release_curve
        
        return envelope

    def generate_acoustic_guitar(self, freq, duration, volume):
        """
        More realistic acoustic guitar sound
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        # Guitar-like harmonic content
        harmonics = [
            (1.0, freq),     # Fundamental
            (0.3, freq * 2),  # Second harmonic
            (0.15, freq * 3), # Third harmonic
            (0.07, freq * 4)  # Fourth harmonic
        ]
        
        wave = np.zeros_like(t)
        for amp, harmonic_freq in harmonics:
            wave += amp * np.sin(2 * np.pi * harmonic_freq * t)
        
        # Add string pluck characteristics
        pluck_noise = 0.1 * np.random.normal(0, 1, len(t)) * np.exp(-10 * t)
        wood_resonance = 0.05 * np.sin(2 * np.pi * (freq/4) * t)
        wave += pluck_noise + wood_resonance
        
        # Guitar-specific envelope
        envelope = self._create_guitar_envelope(duration)
        
        return volume * wave * envelope

    def _create_guitar_envelope(self, duration):
        """
        Guitar-specific quick decay envelope
        """
        t = np.linspace(0, duration, int(self.sample_rate * duration), False)
        
        attack = 0.01
        decay = 0.2
        sustain = 0.5
        release = 0.1
        
        envelope = np.ones_like(t)
        
        # Very quick attack
        attack_samples = int(attack * self.sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        
        # Quick decay
        decay_samples = int(decay * self.sample_rate)
        if decay_samples > 0:
            decay_curve = np.linspace(1, sustain, decay_samples) ** 2
            envelope[attack_samples:attack_samples+decay_samples] = decay_curve
        
        # Short sustain, quick release
        envelope[int((attack + decay) * self.sample_rate):] *= np.exp(-5 * t[int((attack + decay) * self.sample_rate):])
        
        return envelope

def create_advanced_music(json_file, output_file="advanced_output.wav"):
    with open(json_file, 'r') as f:
        config = json.load(f)

    sample_rate = config.get("sample_rate", 44100)
    total_duration = config["total_duration"]
    tempo = config["tempo"]
    beat_duration = 60 / tempo

    full_wave = np.zeros(int(sample_rate * total_duration))
    synth = RealisticInstrumentSynthesizer(sample_rate)

    for instrument in config["instruments"]:
        pattern = instrument["pattern"]
        volume = instrument["volume"]
        pos = 0

        while pos < total_duration:
            for event in pattern:
                duration = event["duration"]
                samples = int(sample_rate * duration)

                vol_variation = volume * (0.95 + np.random.random() * 0.1)
                time_offset = np.random.uniform(-0.005, 0.005)
                pos = max(0, min(pos + time_offset, total_duration))

                # Expanded instrument generation
                if instrument["type"] == "violin":
                    base_freq = NOTE_FREQ.get(event["note"], 0.0)
                    freq = base_freq * (2 ** (instrument.get("pitch_shift", 0) / 12))
                    wave = synth.generate_violin(freq, duration, vol_variation)
                elif instrument["type"] == "cello":
                    base_freq = NOTE_FREQ.get(event["note"], 0.0)
                    freq = base_freq * (2 ** (instrument.get("pitch_shift", 0) / 12))
                    wave = synth.generate_cello(freq, duration, vol_variation)
                elif instrument["type"] == "acoustic_guitar":
                    base_freq = NOTE_FREQ.get(event["note"], 0.0)
                    freq = base_freq * (2 ** (instrument.get("pitch_shift", 0) / 12))
                    wave = synth.generate_acoustic_guitar(freq, duration, vol_variation)
                # Keep previous instrument types
                elif instrument["type"] == "piano":
                    base_freq = NOTE_FREQ.get(event["note"], 0.0)
                    freq = base_freq * (2 ** (instrument.get("pitch_shift", 0) / 12))
                    wave = generate_wave(freq, duration, sample_rate, vol_variation)
                # ... other instrument types from previous implementation

                start = int(pos * sample_rate)
                end = min(start + samples, len(full_wave))
                if start < len(full_wave) and end > start:
                    wave = wave[:end - start]
                    full_wave[start:end] += wave

                pos += duration
                if pos >= total_duration and not instrument["repeat"]:
                    break

    # Normalization and mastering
    full_wave = full_wave / (np.max(np.abs(full_wave)) + 0.1)
    
    # Optional: Soft clipping for warmth
    full_wave = np.tanh(full_wave)
    
    write(output_file, sample_rate, full_wave.astype(np.float32))
    print(f"Advanced music saved as '{output_file}'")

# Keep other existing functions from the previous implementation

if __name__ == "__main__":
    create_advanced_music("advanced_music_config.json")