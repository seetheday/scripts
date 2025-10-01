#!/usr/bin/env python3
"""Automate creation of Sanity draft documents for new Acast podcast episodes."""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import email.utils
import html
import json
import logging
import mimetypes
import os
import re
import sys
import textwrap
import unicodedata
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit("Python 3.11+ is required (missing tomllib module)") from exc

import requests

CONFIG_SEARCH_LOCATIONS = (
    Path.cwd() / "publish_episode_draft.toml",
    Path.home() / ".config" / "maple_history" / "config.toml",
)

DEFAULT_STATE_FILE = Path.home() / ".maple_history" / "state.json"
DEFAULT_LOG_FILE = Path.home() / ".maple_history" / "publisher.log"
DEFAULT_LOCK_FILE = Path("/tmp/maple_history_publish.lock")

ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
NAMESPACES = {"itunes": ITUNES_NS}


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we may not have permission to signal it.
        return True
    return True


@dataclasses.dataclass
class EpisodeMetadata:
    guid: str
    title: str
    description_html: str
    link: Optional[str]
    publish_date: dt.datetime
    episode_number: int
    season: int
    duration: Optional[str]
    duration_seconds: Optional[int]
    enclosure_url: Optional[str]
    acast_episode_id: str


@dataclasses.dataclass
class Config:
    feed_url: str
    artwork_dir: Path
    transcript_dir: Path
    sanity_project_id: str
    sanity_dataset: str
    sanity_api_version: str
    sanity_token_env: str
    polling_interval_minutes: int
    state_file: Path
    log_file: Path
    lock_file: Path
    default_tags: List[str]
    default_historical_period: Optional[str]
    default_season: Optional[int]
    timeout_seconds: int
    dry_run: bool = False


class LockFile:
    """Simple PID-based lock file for cron safety."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._fd: Optional[int] = None

    def acquire(self) -> None:
        if self.path.exists():
            try:
                with self.path.open("r", encoding="ascii") as fh:
                    pid_text = fh.read().strip()
                if pid_text.isdigit() and _pid_running(int(pid_text)):
                    raise RuntimeError(f"Lock file already present: {self.path}")
            except OSError:
                raise RuntimeError(f"Lock file already present: {self.path}")
            # Stale lock; remove and continue
            try:
                self.path.unlink()
            except OSError:
                raise RuntimeError(f"Unable to remove stale lock file: {self.path}")
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("ascii"))
        os.close(fd)
        # Re-open read-only so removal on exit works even if process crashes after PID write
        self._fd = os.open(self.path, os.O_RDONLY)

    def release(self) -> None:
        if self._fd is not None:
            os.close(self._fd)
        if self.path.exists():
            try:
                self.path.unlink()
            except OSError:
                logging.warning("Unable to remove lock file %s", self.path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create Sanity drafts for new Maple History Podcast episodes.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="Path to TOML configuration file.")
    parser.add_argument("--guid", help="Process a specific Acast GUID only.")
    parser.add_argument("--dry-run", action="store_true", help="Do not upload assets or mutate Sanity data.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase log verbosity to DEBUG for this run.",
    )
    return parser.parse_args()


def load_config(path: Optional[Path], dry_run: bool) -> Config:
    config_path = resolve_config_path(path)
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)

    acast_cfg = data.get("acast", {})
    paths_cfg = data.get("paths", {})
    sanity_cfg = data.get("sanity", {})
    defaults_cfg = data.get("defaults", {})
    runtime_cfg = data.get("runtime", {})

    feed_url = acast_cfg.get("feed_url")
    if not feed_url:
        raise ValueError("Configuration missing acast.feed_url")

    artwork_dir = Path(paths_cfg.get("artwork_dir", "")).expanduser()
    transcript_dir = Path(paths_cfg.get("transcript_dir", "")).expanduser()

    sanity_project = sanity_cfg.get("project_id")
    sanity_dataset = sanity_cfg.get("dataset")
    sanity_api_version = sanity_cfg.get("api_version", "2023-10-01")
    sanity_token_env = sanity_cfg.get("token_env_var", "SANITY_API_TOKEN")

    polling_interval = int(acast_cfg.get("polling_interval_minutes", 15))

    state_file = Path(paths_cfg.get("state_file", DEFAULT_STATE_FILE)).expanduser()
    log_file = Path(paths_cfg.get("log_file", DEFAULT_LOG_FILE)).expanduser()
    lock_file = Path(paths_cfg.get("lock_file", DEFAULT_LOCK_FILE)).expanduser()

    default_tags = list(defaults_cfg.get("tags", []))
    default_historical_period = defaults_cfg.get("historical_period")
    default_season = defaults_cfg.get("season")
    timeout_seconds = int(runtime_cfg.get("timeout_seconds", 20))

    if not sanity_project or not sanity_dataset:
        raise ValueError("Configuration missing sanity.project_id or sanity.dataset")

    return Config(
        feed_url=feed_url,
        artwork_dir=artwork_dir,
        transcript_dir=transcript_dir,
        sanity_project_id=sanity_project,
        sanity_dataset=sanity_dataset,
        sanity_api_version=sanity_api_version,
        sanity_token_env=sanity_token_env,
        polling_interval_minutes=polling_interval,
        state_file=state_file,
        log_file=log_file,
        lock_file=lock_file,
        default_tags=default_tags,
        default_historical_period=default_historical_period,
        default_season=default_season,
        timeout_seconds=timeout_seconds,
        dry_run=dry_run,
    )


def resolve_config_path(explicit: Optional[Path]) -> Path:
    if explicit:
        path = explicit.expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        return path
    for candidate in CONFIG_SEARCH_LOCATIONS:
        path = candidate.expanduser()
        if path.exists():
            return path
    raise FileNotFoundError(
        "No configuration file found. Provide --config or create publish_episode_draft.toml."
    )


def setup_logging(log_file: Path, verbose: bool) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    handlers = [logging.FileHandler(log_file, encoding="utf-8")]
    console_handler = logging.StreamHandler(sys.stdout)
    handlers.append(console_handler)

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=handlers,
    )


def load_state(state_path: Path) -> Dict[str, Dict[str, str]]:
    if not state_path.exists():
        state_path.parent.mkdir(parents=True, exist_ok=True)
        return {"processed_guids": {}, "last_checked": None}
    try:
        with state_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError:
        logging.warning("State file %s is invalid JSON; starting fresh", state_path)
        return {"processed_guids": {}, "last_checked": None}


def save_state(state_path: Path, state: Dict[str, Dict[str, str]]) -> None:
    tmp_path = state_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as fh:
        json.dump(state, fh, indent=2)
    tmp_path.replace(state_path)


def fetch_feed(url: str, timeout: int) -> ET.Element:
    logging.debug("Fetching RSS feed %s", url)
    response = requests.get(url, timeout=timeout, headers={"User-Agent": "maple-history-publisher/1.0"})
    response.raise_for_status()
    root = ET.fromstring(response.content)
    return root


def parse_feed(root: ET.Element) -> List[EpisodeMetadata]:
    channel = root.find("channel")
    if channel is None:
        raise ValueError("RSS feed missing channel element")

    episodes: List[EpisodeMetadata] = []
    for item in channel.findall("item"):
        guid = (item.findtext("guid") or "").strip()
        if not guid:
            logging.debug("Skipping item without guid")
            continue

        title = (item.findtext("title") or "").strip()
        description_html = item.findtext("description") or ""
        link = (item.findtext("link") or None)
        pub_date_raw = item.findtext("pubDate")
        publish_date = parse_pub_date(pub_date_raw)

        season = extract_int_field(item, "itunes:season")
        episode_number = extract_int_field(item, "itunes:episode")

        if season is None:
            logging.debug("Item %s missing itunes:season", guid)
        if episode_number is None:
            logging.debug("Item %s missing itunes:episode", guid)

        duration_text = find_namespaced_text(item, "itunes:duration")
        duration_seconds = parse_duration(duration_text)

        enclosure = item.find("enclosure")
        enclosure_url = None
        if enclosure is not None:
            enclosure_url = enclosure.attrib.get("url")
            if not duration_seconds:
                length = enclosure.attrib.get("length")
                duration_seconds = parse_duration(length)

        acast_episode_id = guid

        episodes.append(
            EpisodeMetadata(
                guid=guid,
                title=title,
                description_html=description_html,
                link=link,
                publish_date=publish_date,
                episode_number=episode_number or -1,
                season=season or -1,
                duration=duration_text,
                duration_seconds=duration_seconds,
                enclosure_url=enclosure_url,
                acast_episode_id=acast_episode_id,
            )
        )
    return episodes


def parse_pub_date(raw: Optional[str]) -> dt.datetime:
    if not raw:
        return dt.datetime.now(dt.timezone.utc)
    try:
        parsed = email.utils.parsedate_to_datetime(raw)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed.astimezone(dt.timezone.utc)
    except (TypeError, ValueError):
        logging.warning("Unable to parse pubDate %s; defaulting to now", raw)
        return dt.datetime.now(dt.timezone.utc)


def extract_int_field(item: ET.Element, tag: str) -> Optional[int]:
    text = find_namespaced_text(item, tag)
    if not text:
        return None
    try:
        return int(text.strip())
    except ValueError:
        return None


def find_namespaced_text(item: ET.Element, tag: str) -> Optional[str]:
    element = item.find(tag, NAMESPACES)
    if element is not None and element.text:
        return element.text
    return None


def parse_duration(text_value: Optional[str]) -> Optional[int]:
    if not text_value:
        return None
    text_value = text_value.strip()
    if not text_value:
        return None
    if text_value.isdigit():
        return int(text_value)
    parts = text_value.split(":")
    try:
        parts_int = [int(part) for part in parts]
    except ValueError:
        return None
    total = 0
    for part in parts_int:
        total = total * 60 + part
    return total


def html_to_plain_text(html_value: str) -> str:
    # Remove tags and unescape HTML entities.
    text = re.sub(r"<[^>]+>", " ", html_value)
    text = html.unescape(text)
    cleaned = re.sub(r"\s+", " ", text).strip()
    return cleaned


def html_to_portable_text_blocks(html_value: str) -> List[Dict[str, object]]:
    plain = html_to_plain_text(html_value)
    if not plain:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n{2,}|\r{2,}", plain) if p.strip()]
    if not paragraphs:
        paragraphs = [plain]
    blocks = []
    for paragraph in paragraphs:
        blocks.append(
            {
                "_type": "block",
                "style": "normal",
                "markDefs": [],
                "children": [
                    {
                        "_type": "span",
                        "text": paragraph,
                        "marks": [],
                    }
                ],
            }
        )
    return blocks


def slugify(value: str, max_length: int = 96) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value).strip("-")
    value = value.lower()
    if not value:
        value = "episode"
    if len(value) > max_length:
        value = value[:max_length]
        value = value.rstrip("-") or "episode"
    return value


def ensure_episode_numbers(meta: EpisodeMetadata, config: Config) -> Tuple[int, int]:
    if meta.season > 0 and meta.episode_number > 0:
        return meta.episode_number, meta.season

    episode = meta.episode_number if meta.episode_number > 0 else None
    season = meta.season if meta.season > 0 else None

    pattern = re.compile(r"S(?P<season>\d+)E(?P<episode>\d+)", re.IGNORECASE)
    match = pattern.search(meta.title)
    if match:
        season = season or int(match.group("season"))
        episode = episode or int(match.group("episode"))

    if season is None:
        season = config.default_season
    if episode is None:
        raise ValueError(f"Unable to determine episode number for {meta.guid}")
    if season is None:
        raise ValueError(f"Unable to determine season for {meta.guid}")

    return episode, season


def find_local_asset(directory: Path, guid: str, kind: str) -> Optional[Path]:
    if not directory.exists():
        logging.warning("Directory %s does not exist", directory)
        return None
    pattern = f"acast-{guid}-{kind}-"
    for path in directory.glob(f"acast-{guid}-{kind}-*"):
        if path.is_file():
            logging.debug("Matched %s for %s", path, guid)
            return path
    logging.info("No %s file found for GUID %s in %s", kind, guid, directory)
    return None


def extract_alt_text(path: Path, guid: str, default_title: str) -> str:
    stem = path.stem
    pattern = re.compile(rf"^acast-{re.escape(guid)}-artwork-(.+)$", re.IGNORECASE)
    match = pattern.match(stem)
    if match:
        friendly = match.group(1).replace("-", " ").replace("_", " ")
        friendly = friendly.strip().title()
        if friendly:
            return friendly
    return f"Episode artwork for {default_title}"


def read_transcript(path: Path) -> str:
    with path.open("r", encoding="utf-8") as fh:
        return fh.read().strip()


def ensure_duration(meta: EpisodeMetadata) -> Tuple[str, Optional[int]]:
    duration_text = meta.duration
    duration_seconds = meta.duration_seconds

    if duration_text and duration_text.strip():
        cleaned = duration_text.strip()
    elif duration_seconds is not None:
        cleaned = seconds_to_timestamp(duration_seconds)
    else:
        cleaned = ""
    if cleaned and duration_seconds is None:
        duration_seconds = parse_duration(cleaned)
    return cleaned, duration_seconds


def seconds_to_timestamp(total_seconds: int) -> str:
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:d}:{seconds:02d}"


def derive_short_description(text_value: str, max_length: int = 160) -> str:
    if len(text_value) <= max_length:
        return text_value
    return textwrap.shorten(text_value, width=max_length, placeholder="…")


def collect_document(meta: EpisodeMetadata, config: Config, transcript_path: Optional[Path], transcript_text: Optional[str], artwork_ref: Optional[str], alt_text: Optional[str], slug: str, short_description: str, description_blocks: List[Dict[str, object]], duration_text: str, duration_seconds: Optional[int], tags: List[str]) -> Dict[str, object]:
    document_id = f"episode-{meta.guid}"
    document = {
        "_id": f"drafts.{document_id}",
        "_type": "episode",
        "title": meta.title,
        "slug": {"_type": "slug", "current": slug},
        "episodeNumber": meta.episode_number,
        "season": meta.season,
        "publishDate": meta.publish_date.isoformat(),
        "description": description_blocks,
        "shortDescription": short_description,
        "acastEpisodeId": meta.acast_episode_id,
        "tags": tags,
    }
    if duration_text:
        document["duration"] = duration_text
    if duration_seconds is not None:
        document["durationSeconds"] = duration_seconds
    if config.default_historical_period:
        document["historicalPeriod"] = config.default_historical_period
    if transcript_text:
        document["transcript"] = transcript_text
    if artwork_ref:
        document["artwork"] = {
            "_type": "image",
            "asset": {"_type": "reference", "_ref": artwork_ref},
        }
        if alt_text:
            document["artwork"]["alt"] = alt_text
    if transcript_path and not transcript_text:
        document.setdefault("transcript", "")
    return document


def upload_artwork(session: requests.Session, config: Config, token: str, path: Path) -> str:
    url = f"https://{config.sanity_project_id}.api.sanity.io/{config.sanity_api_version}/assets/images/{config.sanity_dataset}"
    headers = {"Authorization": f"Bearer {token}"}
    mime_type, _ = mimetypes.guess_type(path.name)
    mime_type = mime_type or "application/octet-stream"
    with path.open("rb") as fh:
        files = {"file": (path.name, fh, mime_type)}
        response = session.post(url, headers=headers, files=files, timeout=config.timeout_seconds)
    response.raise_for_status()
    data = response.json()
    document = data.get("document")
    if not document or "_id" not in document:
        raise ValueError("Unexpected Sanity asset response")
    return document["_id"]


def mutate_document(session: requests.Session, config: Config, token: str, document: Dict[str, object]) -> Dict[str, object]:
    url = f"https://{config.sanity_project_id}.api.sanity.io/{config.sanity_api_version}/data/mutate/{config.sanity_dataset}"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {"mutations": [{"createOrReplace": document}]}
    response = session.post(url, headers=headers, timeout=config.timeout_seconds, json=payload)
    response.raise_for_status()
    return response.json()


def process_episode(meta: EpisodeMetadata, config: Config, session: requests.Session, token: Optional[str]) -> bool:
    logging.info("Processing GUID %s | %s", meta.guid, meta.title)

    try:
        episode_number, season = ensure_episode_numbers(meta, config)
    except ValueError as err:
        logging.error("%s", err)
        return False
    meta.episode_number = episode_number
    meta.season = season

    duration_text, duration_seconds = ensure_duration(meta)

    description_blocks = html_to_portable_text_blocks(meta.description_html)
    plain_text = html_to_plain_text(meta.description_html)
    short_description = derive_short_description(plain_text or meta.title)
    slug_base = slugify(meta.title)

    transcript_path = find_local_asset(config.transcript_dir, meta.guid, "transcript")
    transcript_text = read_transcript(transcript_path) if transcript_path else None

    artwork_path = find_local_asset(config.artwork_dir, meta.guid, "artwork")
    alt_text = None
    artwork_ref = None
    if artwork_path:
        alt_text = extract_alt_text(artwork_path, meta.guid, meta.title)
        if not config.dry_run:
            if not token:
                raise RuntimeError("Sanity token required to upload artwork")
            artwork_ref = upload_artwork(session, config, token, artwork_path)
        else:
            logging.info("Dry run: skipping artwork upload for %s", artwork_path)
    else:
        logging.warning("Missing artwork for GUID %s", meta.guid)

    if config.dry_run:
        logging.info("Dry run: skipping Sanity mutation for GUID %s", meta.guid)
        return True

    if not token:
        raise RuntimeError("Sanity token environment variable is not set")

    document = collect_document(
        meta=meta,
        config=config,
        transcript_path=transcript_path,
        transcript_text=transcript_text,
        artwork_ref=artwork_ref,
        alt_text=alt_text,
        slug=slug_base,
        short_description=short_description,
        description_blocks=description_blocks,
        duration_text=duration_text,
        duration_seconds=duration_seconds,
        tags=config.default_tags,
    )

    logging.debug("Sanity document payload for %s: %s", meta.guid, document)

    result = mutate_document(session, config, token, document)
    logging.info("Sanity mutation result for %s: %s", meta.guid, result)
    return True


def main() -> int:
    args = parse_args()

    try:
        config = load_config(args.config, dry_run=args.dry_run)
    except Exception as exc:
        print(f"Error loading configuration: {exc}", file=sys.stderr)
        return 2

    setup_logging(config.log_file, args.verbose)
    logging.info("publisher starting | dry_run=%s", config.dry_run)

    lock = LockFile(config.lock_file)
    try:
        lock.acquire()
    except RuntimeError as err:
        logging.warning("%s", err)
        return 0

    try:
        state = load_state(config.state_file)
        processed = state.get("processed_guids", {})

        try:
            root = fetch_feed(config.feed_url, config.timeout_seconds)
        except Exception as exc:
            logging.error("Failed to fetch RSS feed: %s", exc)
            return 1

        episodes = parse_feed(root)
        if not episodes:
            logging.info("No episodes found in feed")
            return 0

        token = None
        if not config.dry_run:
            token = os.environ.get(config.sanity_token_env)
            if not token:
                logging.error("Environment variable %s is not set", config.sanity_token_env)
                return 2

        session = requests.Session()

        episodes_to_process = []
        if args.guid:
            episodes_to_process = [ep for ep in episodes if ep.guid == args.guid]
            if not episodes_to_process:
                logging.warning("GUID %s not found in feed", args.guid)
        else:
            for ep in episodes:
                if ep.guid not in processed:
                    episodes_to_process.append(ep)

        if not episodes_to_process:
            logging.info("No new episodes to process")
            state["last_checked"] = dt.datetime.now(dt.timezone.utc).isoformat()
            save_state(config.state_file, state)
            return 0

        success_any = False
        for ep in sorted(episodes_to_process, key=lambda x: x.publish_date):
            try:
                success = process_episode(ep, config, session, token)
            except Exception as exc:  # Catch-all to keep loop alive
                logging.error("Failed to process %s: %s", ep.guid, exc)
                success = False
            if success:
                processed[ep.guid] = dt.datetime.now(dt.timezone.utc).isoformat()
                success_any = True

        state["processed_guids"] = processed
        state["last_checked"] = dt.datetime.now(dt.timezone.utc).isoformat()
        save_state(config.state_file, state)

        return 0 if success_any else 1
    finally:
        lock.release()


if __name__ == "__main__":
    sys.exit(main())
