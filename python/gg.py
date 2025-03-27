import numpy as np
from scipy.io.wavfile import write
import json

NOTE_FREQ = {
    # Octave 0
    "C0": 16.35, "C#0": 17.32, "D0": 18.35, "D#0": 19.45, "E0": 20.60, "F0": 21.83, "F#0": 23.12, "G0": 24.50, "G#0": 25.96, "A0": 27.50, "A#0": 29.14, "B0": 30.87,
    # Octave 1
    "C1": 32.70, "C#1": 34.65, "D1": 36.71, "D#1": 38.89, "E1": 41.20, "F1": 43.65, "F#1": 46.25, "G1": 49.00, "G#1": 51.91, "A1": 55.00, "A#1": 58.27, "B1": 61.74,
    # Octave 2
    "C2": 65.41, "C#2": 69.30, "D2": 73.42, "D#2": 77.78, "E2": 82.41, "F2": 87.31, "F#2": 92.50, "G2": 98.00, "G#2": 103.83, "A2": 110.00, "A#2": 116.54, "B2": 123.47,
    # Octave 3
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56, "E3": 164.81, "F3": 174.61, "F#3": 185.00, "G3": 196.00, "G#3": 207.65, "A3": 220.00, "A#3": 233.08, "B3": 246.94,
    # Octave 4 (Middle C)
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13, "E4": 329.63, "F4": 349.23, "F#4": 369.99, "G4": 392.00, "G#4": 415.30, "A4": 440.00, "A#4": 466.16, "B4": 493.88,
    # Octave 5
    "C5": 523.25, "C#5": 554.37, "D5": 587.33, "D#5": 622.25, "E5": 659.25, "F5": 698.46, "F#5": 739.99, "G5": 783.99, "G#5": 830.61, "A5": 880.00, "A#5": 932.33, "B5": 987.77,
    # Octave 6
    "C6": 1046.50, "C#6": 1108.73, "D6": 1174.66, "D#6": 1244.51, "E6": 1318.51, "F6": 1396.91, "F#6": 1479.98, "G6": 1567.98, "G#6": 1661.22, "A6": 1760.00, "A#6": 1864.66, "B6": 1975.53,
    # Octave 7
    "C7": 2093.00, "C#7": 2217.46, "D7": 2349.32, "D#7": 2489.02, "E7": 2637.02, "F7": 2793.83, "F#7": 2959.96, "G7": 3135.96, "G#7": 3322.44, "A7": 3520.00, "A#7": 3729.31, "B7": 3951.07,
    # Octave 8
    "C8": 4186.01, "C#8": 4434.92, "D8": 4698.63, "D#8": 4978.03, "E8": 5274.04, "F8": 5587.65, "F#8": 5919.91, "G8": 6271.93, "G#8": 6644.88, "A8": 7040.00, "A#8": 7458.62, "B8": 7902.13,
    # Octave 9 (just C9 for extreme high)
    "C9": 8372.02,
    # Special lo-fi/detuned notes for Joji vibe
    "A4_flat": 435.00,  # Slightly flat A4
    "E4_detune": 327.00,  # Slightly off E4
    "B3_lofi": 245.00,   # Subtle lo-fi B3
    "G#2_sad": 102.50,   # Slightly detuned G#2
    "REST": 0.0
}

class RealisticInstrumentSynthesizer:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    def generate_violin(self, freq, duration, volume):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        wave = np.zeros_like(t)
        if freq > 0:
            harmonics = [(1.0, freq), (0.5, freq * 2), (0.25, freq * 3), (0.125, freq * 4)]
            for amp, harmonic_freq in harmonics:
                wave += amp * np.sin(2 * np.pi * harmonic_freq * t)
            bow_noise = 0.05 * np.random.normal(0, 1, len(t))
            vibrato = 0.02 * np.sin(2 * np.pi * 6 * t) * np.sin(2 * np.pi * freq * t)
            wave += bow_noise + vibrato
        envelope = self._create_violin_envelope(duration)
        return volume * wave * envelope

    def _create_violin_envelope(self, duration):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        envelope = np.ones_like(t)
        attack, decay, sustain, release = 0.05, 0.1, 0.7, 0.2
        attack_samples = int(attack * self.sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 2
        decay_samples = int(decay * self.sample_rate)
        if decay_samples > 0:
            envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain, decay_samples) ** 1.5
        sustain_start = int((attack + decay) * self.sample_rate)
        sustain_end = int((duration - release) * self.sample_rate)
        envelope[sustain_start:sustain_end] = sustain
        if release_samples := int(release * self.sample_rate):
            envelope[-release_samples:] = np.linspace(sustain, 0, release_samples) ** 0.5
        return envelope

    def generate_cello(self, freq, duration, volume):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        wave = np.zeros_like(t)
        if freq > 0:
            harmonics = [(1.0, freq), (0.4, freq * 2), (0.2, freq * 3), (0.1, freq * 4)]
            for amp, harmonic_freq in harmonics:
                wave += amp * np.sin(2 * np.pi * harmonic_freq * t)
            body_resonance = 0.1 * np.sin(2 * np.pi * (freq/2) * t)
            string_noise = 0.03 * np.random.normal(0, 1, len(t))
            wave += body_resonance + string_noise
        envelope = self._create_cello_envelope(duration)
        return volume * wave * envelope

    def _create_cello_envelope(self, duration):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        envelope = np.ones_like(t)
        attack, decay, sustain, release = 0.1, 0.15, 0.8, 0.3
        attack_samples = int(attack * self.sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 1.5
        decay_samples = int(decay * self.sample_rate)
        if decay_samples > 0:
            envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain, decay_samples) ** 1.3
        sustain_start = int((attack + decay) * self.sample_rate)
        sustain_end = int((duration - release) * self.sample_rate)
        envelope[sustain_start:sustain_end] = sustain
        if release_samples := int(release * self.sample_rate):
            envelope[-release_samples:] = np.linspace(sustain, 0, release_samples) ** 0.7
        return envelope

    def generate_acoustic_guitar(self, freq, duration, volume):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        wave = np.zeros_like(t)
        if freq > 0:
            harmonics = [(1.0, freq), (0.3, freq * 2), (0.15, freq * 3), (0.07, freq * 4)]
            for amp, harmonic_freq in harmonics:
                wave += amp * np.sin(2 * np.pi * harmonic_freq * t)
            pluck_noise = 0.1 * np.random.normal(0, 1, len(t)) * np.exp(-10 * t)
            wood_resonance = 0.05 * np.sin(2 * np.pi * (freq/4) * t)
            wave += pluck_noise + wood_resonance
        envelope = self._create_guitar_envelope(duration)
        return volume * wave * envelope

    def _create_guitar_envelope(self, duration):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        envelope = np.ones_like(t)
        attack, decay, sustain, release = 0.01, 0.2, 0.5, 0.1
        attack_samples = int(attack * self.sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        decay_samples = int(decay * self.sample_rate)
        if decay_samples > 0:
            envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain, decay_samples) ** 2
        envelope[int((attack + decay) * self.sample_rate):] *= np.exp(-5 * t[int((attack + decay) * self.sample_rate):])
        return envelope

    def generate_bass(self, freq, duration, volume):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        wave = np.zeros_like(t)
        if freq > 0:
            harmonics = [(1.0, freq), (0.6, freq * 2), (0.3, freq * 3), (0.1, freq * 4)]
            for amp, harmonic_freq in harmonics:
                wave += amp * np.sin(2 * np.pi * harmonic_freq * t)
            body_resonance = 0.15 * np.sin(2 * np.pi * (freq/2) * t)
            string_noise = 0.02 * np.random.normal(0, 1, len(t))
            wave += body_resonance + string_noise
        envelope = self._create_bass_envelope(duration)
        return volume * wave * envelope

    def _create_bass_envelope(self, duration):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        envelope = np.ones_like(t)
        attack, decay, sustain, release = 0.03, 0.1, 0.85, 0.15
        attack_samples = int(attack * self.sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 1.5
        decay_samples = int(decay * self.sample_rate)
        if decay_samples > 0:
            envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain, decay_samples) ** 1.2
        sustain_start = int((attack + decay) * self.sample_rate)
        sustain_end = int((duration - release) * self.sample_rate)
        envelope[sustain_start:sustain_end] = sustain
        if release_samples := int(release * self.sample_rate):
            envelope[-release_samples:] = np.linspace(sustain, 0, release_samples) ** 0.8
        return envelope

    def generate_drum(self, freq, duration, volume, drum_type="kick"):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        wave = np.zeros_like(t)
        if drum_type == "kick":
            # Low-frequency thump with quick decay
            wave = np.sin(2 * np.pi * 60 * t) * np.exp(-15 * t)  # 60 Hz base
            wave += 0.5 * np.random.normal(0, 1, samples) * np.exp(-20 * t)  # Noise burst
        elif drum_type == "snare":
            # Sharp noise with tonal component
            wave = np.sin(2 * np.pi * 200 * t) * np.exp(-10 * t)  # 200 Hz tone
            wave += np.random.normal(0, 1, samples) * np.exp(-12 * t)  # White noise
        envelope = self._create_drum_envelope(duration)
        return volume * wave * envelope

    def _create_drum_envelope(self, duration):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        envelope = np.exp(-15 * t)  # Fast decay for punchy drums
        return envelope

    def generate_electric_guitar(self, freq, duration, volume):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        wave = np.zeros_like(t)
        if freq > 0:
            # Sawtooth-like wave with distortion
            harmonics = [(1.0, freq), (0.8, freq * 2), (0.6, freq * 3), (0.4, freq * 4), (0.2, freq * 5)]
            for amp, harmonic_freq in harmonics:
                wave += amp * np.sin(2 * np.pi * harmonic_freq * t)
            # Soft clipping for distortion
            wave = np.tanh(wave * 2)
            noise = 0.05 * np.random.normal(0, 1, len(t))
            wave += noise
        envelope = self._create_electric_guitar_envelope(duration)
        return volume * wave * envelope

    def _create_electric_guitar_envelope(self, duration):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        envelope = np.ones_like(t)
        attack, decay, sustain, release = 0.01, 0.15, 0.7, 0.2
        attack_samples = int(attack * self.sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
        decay_samples = int(decay * self.sample_rate)
        if decay_samples > 0:
            envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain, decay_samples) ** 1.5
        sustain_start = int((attack + decay) * self.sample_rate)
        sustain_end = int((duration - release) * self.sample_rate)
        envelope[sustain_start:sustain_end] = sustain
        if release_samples := int(release * self.sample_rate):
            envelope[-release_samples:] = np.linspace(sustain, 0, release_samples) ** 0.8
        return envelope

    def generate_synth_pad(self, freq, duration, volume):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        wave = np.zeros_like(t)
        if freq > 0:
            # Smooth, lush sound with multiple detuned waves
            wave += np.sin(2 * np.pi * freq * t)
            wave += 0.8 * np.sin(2 * np.pi * (freq * 1.01) * t)  # Slight detune
            wave += 0.6 * np.sin(2 * np.pi * (freq * 0.99) * t)  # Slight detune
            wave += 0.3 * np.sin(2 * np.pi * (freq * 2) * t)  # Octave up
        envelope = self._create_synth_pad_envelope(duration)
        return volume * wave * envelope

    def _create_synth_pad_envelope(self, duration):
        samples = int(self.sample_rate * duration)
        t = np.linspace(0, duration, samples, False)
        envelope = np.ones_like(t)
        attack, decay, sustain, release = 0.2, 0.3, 0.6, 0.4
        attack_samples = int(attack * self.sample_rate)
        if attack_samples > 0:
            envelope[:attack_samples] = np.linspace(0, 1, attack_samples) ** 1.5
        decay_samples = int(decay * self.sample_rate)
        if decay_samples > 0:
            envelope[attack_samples:attack_samples+decay_samples] = np.linspace(1, sustain, decay_samples)
        sustain_start = int((attack + decay) * self.sample_rate)
        sustain_end = int((duration - release) * self.sample_rate)
        envelope[sustain_start:sustain_end] = sustain
        if release_samples := int(release * self.sample_rate):
            envelope[-release_samples:] = np.linspace(sustain, 0, release_samples) ** 0.5
        return envelope

def generate_wave(freq, duration, sample_rate, volume):
    samples = int(sample_rate * duration)
    t = np.linspace(0, duration, samples, False)
    wave = np.sin(2 * np.pi * freq * t) if freq > 0 else np.zeros_like(t)
    envelope = np.exp(-t * 2)
    return volume * wave * envelope

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
                nominal_samples = int(sample_rate * duration)

                vol_variation = volume * (0.9 + np.random.random() * 0.2)
                time_offset = np.random.uniform(-0.01, 0.01)
                pos = max(0, min(pos + time_offset, total_duration))

                start = int(pos * sample_rate)
                end = min(start + nominal_samples, len(full_wave))
                target_samples = end - start

                if target_samples <= 0 or start >= len(full_wave):
                    break

                wave = None
                try:
                    base_freq = NOTE_FREQ.get(event["note"], 0.0)
                    freq = base_freq * (2 ** (instrument.get("pitch_shift", 0) / 12))
                    if instrument["type"] == "violin":
                        wave = synth.generate_violin(freq, duration, vol_variation)
                    elif instrument["type"] == "cello":
                        wave = synth.generate_cello(freq, duration, vol_variation)
                    elif instrument["type"] == "acoustic_guitar":
                        wave = synth.generate_acoustic_guitar(freq, duration, vol_variation)
                    elif instrument["type"] == "bass":
                        wave = synth.generate_bass(freq, duration, vol_variation)
                    elif instrument["type"] == "piano":
                        wave = generate_wave(freq, duration, sample_rate, vol_variation)
                    elif instrument["type"] == "drum_kick":
                        wave = synth.generate_drum(freq, duration, vol_variation, drum_type="kick")
                    elif instrument["type"] == "drum_snare":
                        wave = synth.generate_drum(freq, duration, vol_variation, drum_type="snare")
                    elif instrument["type"] == "electric_guitar":
                        wave = synth.generate_electric_guitar(freq, duration, vol_variation)
                    elif instrument["type"] == "synth_pad":
                        wave = synth.generate_synth_pad(freq, duration, vol_variation)
                    if wave is None:
                        wave = np.zeros(nominal_samples)
                except Exception as e:
                    print(f"Error generating wave for {instrument['type']}: {e}")
                    wave = np.zeros(nominal_samples)

                if len(wave) > target_samples:
                    wave = wave[:target_samples]
                elif len(wave) < target_samples:
                    wave = np.pad(wave, (0, target_samples - len(wave)), 'constant')

                full_wave[start:end] += wave

                pos += duration
                if pos >= total_duration and not instrument["repeat"]:
                    break

    max_amplitude = np.max(np.abs(full_wave))
    if max_amplitude > 0:
        full_wave = full_wave / (max_amplitude * 1.1)
    full_wave = np.tanh(full_wave * 1.5)
    
    write(output_file, sample_rate, full_wave.astype(np.float32))
    print(f"Advanced music saved as '{output_file}'")

if __name__ == "__main__":
    json_file = "track.json"
    create_advanced_music(json_file)