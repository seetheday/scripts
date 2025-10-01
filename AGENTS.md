# Repository Guidelines

## Project Structure & Module Organization
All automation lives in the repository root as standalone shell scripts. `transcribe.sh` drives Whisper-based audio transcription from `~/Reaper/MacBook/final audio` into transcripts under `~/Reaper/MacBook/transcripts`. `kdenlive-whisper.sh` prepares a Python virtual environment in `myenv/` and then launches Kdenlive. `open-discover.sh` restarts KDE's Discover application, and `hetzner.server.ip` stores reference connection data. Treat `myenv/` as disposable build tooling; no source files should be committed there.

## Build, Test, and Development Commands
- `bash transcribe.sh` &mdash; run the full transcription pass for any new `.mp3` assets.
- `bash kdenlive-whisper.sh` &mdash; recreate the `myenv` virtualenv and open Kdenlive for editing.
- `bash open-discover.sh` &mdash; relaunch Discover if package operations stall.
Use `chmod +x <script>` when adding new helpers so they remain executable.

## Coding Style & Naming Conventions
Author scripts in POSIX-compatible Bash with 4-space indentation and descriptive snake_case variable names (e.g., `OUTPUT_FILE_BASE`). Guard concurrent execution with lock files when jobs touch shared resources. Keep executable filenames lowercase with hyphens, mirroring the existing patterns.

## Testing Guidelines
Lint new or edited scripts with `shellcheck script.sh` before committing, and run `bash -n script.sh` to catch syntax errors. For `transcribe.sh`, perform a dry run using a short sample `.mp3` and confirm it writes to `~/Reaper/MacBook/transcripts` without clobbering existing outputs. Document manual verification steps in commit messages when automation is not practical.

## Commit & Pull Request Guidelines
Use concise, imperative commit subjects (e.g., "Add whisper batch lockfile"). Describe the affected scripts, rationale, and any manual checks in the body. Pull requests should summarise behavioural changes, reference related tickets or context, and include logs or screenshots when they clarify workflow impacts (such as new transcript paths or UI behaviour).

## Security & Configuration Tips
Protect personal paths and server addresses; avoid hard-coding credentials alongside `hetzner.server.ip`. When sharing logs, redact full audio filenames and private directories. Consider moving environment-specific settings into `.env`-style files ignored by Git if additional configuration emerges.
