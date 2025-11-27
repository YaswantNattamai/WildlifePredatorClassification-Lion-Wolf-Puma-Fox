import os
import librosa
import numpy as np

# Parameters
SAMPLE_RATE = 22050
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512
DURATION = 5
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def preprocess_audio(file_path, sample_rate=SAMPLE_RATE, duration=DURATION):
    """Load an MP3 file and convert it to Mel-Spectrogram (dB)."""
    y, sr = librosa.load(file_path, sr=sample_rate)

    # Pad/trim
    if len(y) < SAMPLES_PER_TRACK:
        y = np.pad(y, (0, SAMPLES_PER_TRACK - len(y)), mode="constant")
    else:
        y = y[:SAMPLES_PER_TRACK]

    # Mel-Spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db.T   # shape = (time_frames, n_mels)

def process_folder(input_folder, output_file="mel_spectrograms.npy"):
    """Process all MP3s in a folder and save as a NumPy array."""
    spectrograms = []
    file_names = []

    for file in os.listdir(input_folder):
        if file.endswith(".mp3"):
            file_path = os.path.join(input_folder, file)
            mel_spec = preprocess_audio(file_path)
            spectrograms.append(mel_spec)
            file_names.append(file)

    spectrograms = np.array(spectrograms, dtype=object)  # keep variable lengths if needed
    np.save(output_file, spectrograms)
    print(f"Saved {len(spectrograms)} spectrograms to {output_file}")
    return file_names

# Example usage
input_folder = "Lion_Sound"   # folder containing mp3 files
process_folder(input_folder, "Lion_mel_spectrograms.npy")
