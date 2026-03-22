from faster_whisper import WhisperModel
import os, json

audio_dir = "audios"
json_dir = "transcripts"

os.makedirs(json_dir, exist_ok=True)

mp3_files = [f for f in os.listdir(audio_dir) if f.endswith(".mp3")]

model = WhisperModel(
    "large-v3",
    device="cuda",
    compute_type="float16"
)

for file in mp3_files:
    print(f"Processing: {file}")

    input_path = os.path.join(audio_dir, file)
    base_name = os.path.splitext(file)[0]
    output_path = os.path.join(json_dir, f"{base_name}.json")

    segments, info = model.transcribe(
        input_path,
        language="en",
        beam_size=5,
        vad_filter=True
    )

    segments = list(segments)

    data = {
        "lecture": base_name,
        "language": info.language,
        "duration": info.duration,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            }
            for seg in segments
        ]
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"Saved → {output_path}")
