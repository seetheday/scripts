# Episode Draft Publisher

Automates creation of Sanity draft documents for new Maple History Podcast episodes based on the Acast RSS feed. The script fetches fresh feed entries, locates local artwork/transcripts by GUID, uploads assets to Sanity, and stores the draft as `drafts.episode-<guid>` so you can review and publish in Sanity Studio.

## Requirements
- Python 3.11+
- Access to the podcast RSS feed
- Sanity project ID, dataset, API version, and a read/write API token
- Local directories containing episode artwork and transcripts

## Initial Setup
1. Copy the sample configuration: `cp publish_episode_draft.sample.toml publish_episode_draft.toml` (or place it in `~/.config/maple_history/config.toml`).
2. Edit the new file with your feed URL, Sanity project values, and filesystem paths.
3. Export your Sanity token (example: `export SANITY_API_TOKEN="<token>"`). Keep the token outside version control.
4. Make the script executable: `chmod +x publish_episode_draft.py`.

### TOML Configuration Notes
- **acast.feed_url**: The RSS feed to poll.
- **paths.***: Absolute or `~`-expanded paths for artwork, transcripts, state, logs, and lockfile. The state file remembers processed GUIDs.
- **sanity.token_env_var**: Name of the environment variable holding your API token.
- **defaults**: Optional fallback tags, historical period, or season if the feed lacks those fields.
- **runtime.timeout_seconds**: HTTP timeout for API calls.

## File Naming Conventions
- Artwork: `acast-<guid>-artwork-<friendly-name>.jpg`
- Transcript: `acast-<guid>-transcript-<friendly-name>.txt`

Only the GUID and kind (`artwork` / `transcript`) are used programmatically. Anything after the second hyphen is treated as a human-friendly label and ignored by the script; include as much descriptive text as you want.

## Usage
- Dry run (no uploads): `./publish_episode_draft.py --dry-run`
- Process a specific GUID: `./publish_episode_draft.py --guid <acast-guid>`
- Default run (using config file in current directory): `./publish_episode_draft.py`

The script writes logs to the configured log file and maintains a JSON state under `paths.state_file`. It will skip GUIDs already recorded there.

## Scheduling
Add a cron entry (e.g., every 15 minutes) on the machine that stores the assets:
```
*/15 * * * * /usr/bin/env bash -lc 'cd /home/simon/scripts && source ~/.profile && SANITY_API_TOKEN=... ./publish_episode_draft.py'
```
Ensure the environment exports the token and points to the correct configuration.

## Troubleshooting
- Missing assets: the script logs a warning and still creates a draft without artwork/transcript.
- Lockfile present: remove the file indicated in the log if a crash leaves it behind.
- Invalid config: rerun with `--verbose` to see detailed parsing and payload logs.
