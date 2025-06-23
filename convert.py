import os
from pydub import AudioSegment
from tqdm import tqdm

# Set your folder paths
input_folder = "archive/911_recordings"
output_folder = "wav_folder"

# Make sure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Loop through all .mp3 files
for filename in tqdm(os.listdir(input_folder), desc="Converting MP3s"):
    if filename.lower().endswith(".mp3"):
        input_path = os.path.join(input_folder, filename)
        output_filename = os.path.splitext(filename)[0] + ".wav"
        output_path = os.path.join(output_folder, output_filename)

        # Skip if already converted
        if os.path.exists(output_path):
            continue

        try:
            # Load and convert
            sound = AudioSegment.from_mp3(input_path)
            sound = sound.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            sound.export(output_path, format="wav")

            print(f"Converted: {filename} → {output_filename}")

        except Exception as e:
            print(f"Failed to convert {filename}: {e}")
