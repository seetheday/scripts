#!/bin/bash

# --- Configuration ---
MODEL="$HOME/whisper/models/ggml-base.en.bin"
INPUT_DIR="$HOME/Reaper/MacBook/final audio"
OUTPUT_DIR="$HOME/Reaper/MacBook/transcripts"
LOCKFILE=/tmp/transcribe.lock

if [ -e "$LOCKFILE" ] && kill -0 $(cat "$LOCKFILE") 2>/dev/null; then
   echo "Transcription already running, exiting."
   exit 1
fi

echo $$ > "$LOCKFILE"

# Make sure output directory exists
mkdir -p "$OUTPUT_DIR"

# --- Process each MP3 file ---
for FILE in "$INPUT_DIR"/*.mp3; do
   # Skip if no MP3 files found (when glob doesn't match anything)
   [[ ! -f "$FILE" ]] && continue
   
   BASENAME=$(basename "$FILE" .mp3)
   OUTPUT_FILE_BASE="$OUTPUT_DIR/${BASENAME}"
   TRANSCRIPT_FILE="${OUTPUT_FILE_BASE}.txt"
   
   # Skip if transcription already exists
   if [[ -f "$TRANSCRIPT_FILE" ]]; then
       echo "Skipping $FILE — transcript already exists at $TRANSCRIPT_FILE"
       continue
   fi
   
   echo "Transcribing $FILE -> $TRANSCRIPT_FILE"
   ~/whisper/build/bin/whisper-cli -m "$MODEL" -f "$FILE" -otxt -of "$OUTPUT_FILE_BASE"
   
   # Check if transcription was successful
   if [[ -f "$TRANSCRIPT_FILE" ]]; then
       echo "✓ Successfully transcribed: $TRANSCRIPT_FILE"
   else
       echo "✗ Failed to transcribe: $FILE"
   fi
done

rm -f "$LOCKFILE"
echo "All done!"
