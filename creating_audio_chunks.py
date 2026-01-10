"""
Audio File Splitter - Split large audio files into chunks for OpenAI Whisper API
Splits files larger than 25MB into smaller pieces
"""

import os
from openai import OpenAI

# ============================================================================
# CONFIGURATION
# ============================================================================

# Add your OpenAI API key here
OPENAI_API_KEY = "your-api-key-here"  # Replace with your actual API key

# Whisper model to use
WHISPER_MODEL = "gpt-4o-transcribe"

# Optional prompt
TRANSCRIPTION_PROMPT = "This is a german language class in which the professor is speaking in English, Hindi and in German from this audio file extract all the words which the professor is speaking in German. Also if the professor is giving any conjugation of the word in German or any sentence structure in German then also extract that"

# Maximum chunk size in MB (keep under 25MB for API limit)
MAX_CHUNK_SIZE_MB = 20  # Use 20MB to have some buffer

# ============================================================================


def get_audio_duration_estimate(file_size_mb):
    """
    Rough estimate of audio duration based on file size.
    WAV files are typically ~10MB per minute.
    """
    minutes = file_size_mb / 10
    return minutes


def split_audio_file(input_file, chunk_size_mb=20):
    """
    Split audio file into chunks.
    This is a simple byte-based split for WAV files.

    Args:
        input_file: Path to input audio file
        chunk_size_mb: Target size for each chunk in MB

    Returns:
        List of chunk file paths
    """
    print(f"[INFO] Splitting audio file: {input_file}")

    # Get file size
    file_size = os.path.getsize(input_file)
    file_size_mb = file_size / (1024 * 1024)

    print(f"[INFO] Total file size: {file_size_mb:.2f} MB")

    # Calculate number of chunks needed
    chunk_size_bytes = chunk_size_mb * 1024 * 1024
    num_chunks = int(file_size / chunk_size_bytes) + 1

    print(f"[INFO] Will create {num_chunks} chunks")

    # Create output directory for chunks
    base_name = os.path.splitext(input_file)[0]
    chunks_dir = f"{base_name}_chunks"
    os.makedirs(chunks_dir, exist_ok=True)

    chunk_files = []

    try:
        with open(input_file, "rb") as f:
            # Read WAV header (first 44 bytes)
            wav_header = f.read(44)

            for i in range(num_chunks):
                chunk_file = os.path.join(chunks_dir, f"chunk_{i+1}.wav")

                print(f"[INFO] Creating chunk {i+1}/{num_chunks}: {chunk_file}")

                with open(chunk_file, "wb") as chunk_f:
                    # Write WAV header
                    chunk_f.write(wav_header)

                    # Write audio data
                    remaining = chunk_size_bytes
                    while remaining > 0:
                        data = f.read(min(remaining, 8192))
                        if not data:
                            break
                        chunk_f.write(data)
                        remaining -= len(data)

                chunk_files.append(chunk_file)

                # Check chunk size
                chunk_size_mb_actual = os.path.getsize(chunk_file) / (1024 * 1024)
                print(f"[INFO] Chunk size: {chunk_size_mb_actual:.2f} MB")

        print(f"[SUCCESS] Split into {len(chunk_files)} chunks")
        return chunk_files

    except Exception as e:
        print(f"[ERROR] Failed to split file: {e}")
        return []


def transcribe_audio_chunk(audio_path, api_key, model, prompt=""):
    """Transcribe a single audio chunk."""
    try:
        client = OpenAI(api_key=api_key)

        with open(audio_path, "rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=model,
                language="de",
                response_format="text",
                prompt=prompt,
            )

            return transcription

    except Exception as e:
        print(f"[ERROR] Transcription failed for {audio_path}: {e}")
        return None


def main():
    """Main function."""

    # ========================================================================
    # CONFIGURATION - Update this path
    # ========================================================================

    # Path to your large audio file
    audio_path = r"C:\Users\manik\Desktop\language_pantheon_App\WhatsApp Audio 2026-01-10 at 14.38.41.wav"

    # Output file
    output_text_file = "german_words_complete.txt"

    # ========================================================================

    print("=" * 60)
    print("[DE] German Word Extractor - Large File Splitter")
    print("=" * 60)

    # Check API key
    if OPENAI_API_KEY == "your-api-key-here":
        print("\n[ERROR] Please add your OpenAI API key in the script!")
        return

    # Check if input file exists
    if not os.path.exists(audio_path):
        print(f"[ERROR] File not found: {audio_path}")
        return

    # Get file size
    file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
    print(f"[INFO] Audio file size: {file_size_mb:.2f} MB")

    if file_size_mb <= 25:
        print("[INFO] File is small enough, no splitting needed!")
        print("[TIP] Use the regular script instead")
        return

    try:
        # Step 1: Split the audio file
        print("\n[STEP 1] Splitting audio file...")
        chunk_files = split_audio_file(audio_path, MAX_CHUNK_SIZE_MB)

        if not chunk_files:
            print("[ERROR] Failed to split audio file")
            return

        # Step 2: Transcribe each chunk
        print("\n[STEP 2] Transcribing chunks...")
        all_transcriptions = []

        for i, chunk_file in enumerate(chunk_files):
            print(f"\n[INFO] Processing chunk {i+1}/{len(chunk_files)}...")

            transcription = transcribe_audio_chunk(
                chunk_file, OPENAI_API_KEY, WHISPER_MODEL, TRANSCRIPTION_PROMPT
            )

            if transcription:
                all_transcriptions.append(transcription)
                print(f"[SUCCESS] Chunk {i+1} transcribed successfully")
            else:
                print(f"[WARNING] Chunk {i+1} transcription failed")

        # Step 3: Combine all transcriptions
        print("\n[STEP 3] Combining transcriptions...")
        complete_text = "\n".join(all_transcriptions)

        # Save to file
        with open(output_text_file, "w", encoding="utf-8") as f:
            f.write(complete_text)

        print(f"[SAVE] Complete transcription saved to: {output_text_file}")
        print(f"\n[TEXT] Transcribed text:\n{'-'*50}\n{complete_text}\n{'-'*50}")

        # Cost estimate
        estimated_minutes = get_audio_duration_estimate(file_size_mb)
        estimated_cost = estimated_minutes * 0.006

        print(f"\n[COST] API Usage Info:")
        print(f"   Processed {len(chunk_files)} chunks")
        print(f"   Estimated audio duration: ~{estimated_minutes:.1f} minutes")
        print(f"   Estimated cost: ${estimated_cost:.3f}")

        print("\n[SUCCESS] Process completed successfully!")

    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")


if __name__ == "__main__":
    main()
