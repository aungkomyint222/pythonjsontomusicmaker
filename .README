# AI-Powered Music Production Tool

This project is an advanced music production application designed to generate realistic and expressive music using AI techniques. The application leverages JSON-based configuration files to define musical compositions, instruments, and patterns, making it highly customizable and user-friendly.

## Features

- **Realistic Instrument Synthesis**: Includes support for instruments like violin, cello, acoustic guitar, bass, electric guitar, piano, and synth pads. Each instrument is modeled with realistic harmonic content, noise, and ADSR envelopes.
- **JSON-Based Music Configuration**: Users can define their music compositions in JSON files, specifying instruments, patterns, tempo, and other parameters.
- **Dynamic Sound Generation**: Supports features like pitch shifting, volume variation, and time offsets for a more human-like performance.
- **Multi-Instrument Support**: Allows layering of multiple instruments to create rich and complex compositions.
- **Drum and Percussion**: Includes drum synthesis for kick and snare sounds with realistic decay and noise characteristics.
- **Soft Clipping and Normalization**: Ensures the final audio output is polished and ready for use.

## How It Works

1. **Define Music in JSON**: Create a JSON file (e.g., `track.json` or `music_config.json`) to specify the instruments, patterns, and other parameters for your composition.
2. **Run the Application**: Use the Python script to process the JSON file and generate a `.wav` file as the output.
3. **Listen to the Output**: The generated audio file can be played back or used in other music production workflows.

## Example JSON Configuration

Here’s an example of a JSON configuration file:

```json
{
    "sample_rate": 44100,
    "total_duration": 30,
    "tempo": 100,
    "instruments": [
        {
            "type": "electric_guitar",
            "volume": 0.7,
            "pattern": [
                {"note": "A4", "duration": 0.6},
                {"note": "F#4", "duration": 0.6},
                {"note": "E4", "duration": 0.6}
            ],
            "repeat": true
        }
    ]
}