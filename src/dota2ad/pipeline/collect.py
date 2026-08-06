#!/usr/bin/env python3
"""
Collect Dota 2 Ability Draft replays: discover → download → parse.

Three concurrent stages connected by bounded queues:
  Discover (main thread) → Download (N threads) → Parse (M threads)

Error policy:
  - Per-match failures raise SkipMatch and never halt the run. They are split
    into *permanent* (the match/replay is genuinely gone — OpenDota 404, replay
    CDN 403/404/410, an unparseable demo) and *transient* (5xx/522/429/network,
    a corrupt download — the match is valid, the service just hiccupped).
  - Systemic failures abort the whole run: a rejected API key, missing JAR /
    java, or disk full raise FatalPipelineError; a long run of *consecutive*
    failures trips a circuit breaker.

Resume correctness:
  Matches complete out of seq order (concurrency), so max(success seq) can sit
  *above* an unfinished lower seq. To never lose a valid match to a transient
  blip, we persist the contiguous *resolved* frontier (.resume_seq) — the
  highest seq below which every discovered match is resolved (succeeded, or
  permanently/terminally skipped). `--resume` restarts from that frontier, so a
  transient skip leaves the frontier held and gets retried; a permanent skip
  advances it. A transient skip that keeps failing across TRANSIENT_RETRY_CAP
  attempts is given up on (treated as resolved) so it can't stall the frontier.
"""

import argparse
import bz2
import contextlib
import errno
import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from datetime import datetime, UTC
from pathlib import Path
from queue import Empty, Full, Queue

import requests
from dotenv import load_dotenv

from dota2ad.core import default_paths

load_dotenv()

OPENDOTA_API = "https://api.opendota.com/api"
STEAM_MATCH_HISTORY_BY_SEQ = (
    "https://api.steampowered.com/IDOTA2Match_570/GetMatchHistoryBySequenceNum/v1"
)
ABILITY_DRAFT_MODE = 18
STEAM_RATE_LIMIT_DELAY = 5.0
OPENDOTA_EXPLORER_BATCH = 500  # /explorer LIMIT per page (bulk pulls read-time out)
OPENDOTA_API_KEY = os.environ.get("OPENDOTA_API_KEY")
OPENDOTA_RATE_LIMIT_DELAY = 0.1 if OPENDOTA_API_KEY else 1.0

# A transient (retryable) per-match failure that recurs this many times across
# runs is given up on, so one broken match can't stall the resume frontier.
TRANSIENT_RETRY_CAP = 3

DEFAULT_CLARITY_JAR = Path(
    "dota2-ad-parser/target/clarity-ad-parser-1.0-SNAPSHOT-shaded.jar"
)

SENTINEL = None


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class SkipMatch(Exception):
    """A per-match failure — skip this match and keep going (never fatal).

    `permanent=True`  : the match/replay is genuinely gone (OpenDota 404, replay
                        CDN 403/404/410, an unparseable demo). Safe to advance
                        the resume frontier past it.
    `permanent=False` : a transient infra hiccup (5xx/522/429/network, corrupt
                        download). The match is valid; leave the frontier held
                        so `--resume` retries it.
    `details`         : the already-fetched OpenDota match record, when the skip
                        happened after the details fetch (replay CDN 403/404).
                        Persisted beside the .gone marker; build-dataset
                        aggregates these into dataset/gone_matches.jsonl and
                        experiments/random-mechanism/retrieval.py characterizes
                        the retrieval censoring from them.
    """

    def __init__(self, message: str, *, permanent: bool, details: dict | None = None):
        super().__init__(message)
        self.permanent = permanent
        self.details = details


class FatalPipelineError(Exception):
    """An unrecoverable, pipeline-wide failure — abort immediately.

    Raised for: a rejected API key (auth), a missing Clarity JAR / java, disk
    full. A single bad match is NEVER fatal — that is SkipMatch. (Sustained
    consecutive failures also abort, via the circuit breaker in PipelineStats.)
    """


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


class RateLimiter:
    """Enforces a minimum delay between calls."""

    def __init__(self, min_interval: float):
        self.min_interval = min_interval
        self.lock = threading.Lock()
        self.last_call = 0.0

    def wait(self):
        with self.lock:
            now = time.monotonic()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                time.sleep(self.min_interval - elapsed)
            self.last_call = time.monotonic()


class PipelineStats:
    """Thread-safe counters + two consecutive-failure circuit breakers.

    Skips come in two flavours that need very different thresholds:

    * *Transient* (5xx / network / disk / corrupt): infra is broken. These don't
      cluster in healthy operation, so a short run (`abort_after_failures`) trips.
    * *Permanent* (replay genuinely gone — CDN 403/404): a normal ~7% of matches,
      and they *do* cluster into contiguous bands (40+ in a row observed). Counting
      these toward the transient breaker false-aborts mid-window, so they get a
      separate, much higher threshold (`abort_after_permanent`) that a natural band
      never reaches but a Valve-wide 403/404 outage does.

    `record_skip` returns the abort reason when a breaker trips, else None. A
    success — or a skip of the *other* flavour — resets the relevant run (the CDN
    isn't uniformly down if some matches resolve as genuinely-gone, and infra
    isn't stuck if some replays come back permanent).
    """

    def __init__(
        self, abort_after_failures: int = 25, abort_after_permanent: int = 300
    ):
        self._lock = threading.Lock()
        self.downloaded = 0
        self.succeeded = 0
        self.skipped = 0
        self.abort_after_failures = abort_after_failures
        self.abort_after_permanent = abort_after_permanent
        self._consecutive_failures = 0
        self._consecutive_permanent = 0

    def inc_downloaded(self):
        with self._lock:
            self.downloaded += 1

    def record_success(self):
        """A match finished successfully — reset both breakers."""
        with self._lock:
            self.succeeded += 1
            self._consecutive_failures = 0
            self._consecutive_permanent = 0

    def record_skip(self, *, permanent: bool) -> str | None:
        """A match was skipped. Returns the abort reason if a breaker tripped."""
        with self._lock:
            self.skipped += 1
            if permanent:
                self._consecutive_permanent += 1
                self._consecutive_failures = 0
                if self._consecutive_permanent >= self.abort_after_permanent:
                    return (
                        f"{self._consecutive_permanent} consecutive replays "
                        "permanently unavailable — a CDN-wide 403/404 outage, not "
                        "isolated gone replays"
                    )
                return None
            self._consecutive_failures += 1
            self._consecutive_permanent = 0
            if self._consecutive_failures >= self.abort_after_failures:
                return (
                    f"{self._consecutive_failures} consecutive transient failures "
                    "— a systemic problem (rejected key / API or CDN outage / disk "
                    "full), not isolated bad replays"
                )
            return None


class ResumeCursor:
    """Persists the contiguous resolved-seq frontier for gap-free `--resume`.

    Matches complete out of order, so max(success seq) can sit above an
    unfinished lower seq. We instead track the highest seq below which *every*
    discovered match is resolved (succeeded or terminally skipped), and persist
    it. Resume restarts from frontier+1, re-covering any hole left by a
    transient skip while never re-scanning fully-resolved prefixes.
    """

    def __init__(self, path: Path):
        self._path = path
        self._lock = threading.Lock()
        self._discovered: list[int] = []  # seqs in discovery (increasing) order
        self._resolved: set[int] = set()
        self._idx = 0  # frontier pointer into _discovered

    def discovered(self, seq: int):
        with self._lock:
            self._discovered.append(seq)

    def resolve(self, seq: int):
        """Mark a seq done (success or terminal skip) and persist the frontier
        if it advanced."""
        with self._lock:
            self._resolved.add(seq)
            advanced = False
            while (
                self._idx < len(self._discovered)
                and self._discovered[self._idx] in self._resolved
            ):
                self._idx += 1
                advanced = True
            if advanced:
                self._path.write_text(str(self._discovered[self._idx - 1]))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def log_error(
    error_dir: Path, match_id: int, reason: str, exc: Exception | None = None
):
    """Append an error report for a match. The `[attempt]` marker lets
    count_attempts tally how many times this match has failed across runs."""
    error_dir.mkdir(parents=True, exist_ok=True)
    err_file = error_dir / f"{match_id}.txt"
    with open(err_file, "a", encoding="utf-8") as f:
        f.write(f"[attempt] {reason}\n")
        if exc is not None:
            f.write("".join(traceback.format_exception(exc)))
            f.write("\n")


def count_attempts(error_dir: Path, match_id: int) -> int:
    """How many times this match has been logged as failed (across all runs)."""
    f = error_dir / f"{match_id}.txt"
    if not f.exists():
        return 0
    return f.read_text(encoding="utf-8").count("[attempt] ")


def gone_marker(error_dir: Path, match_id: int) -> Path:
    """Marker that a match is permanently unavailable (genuinely-gone replay,
    unknown match, or unparseable demo). Discovery skips these so a resume never
    re-attempts them — re-attempts waste an OpenDota call + a 403 each, and since
    already-collected matches are skipped *silently* in discovery, a band of
    re-attempted gone replays reads to the breaker as one long consecutive-failure
    run (a false CDN-outage abort)."""
    return error_dir / f"{match_id}.gone"


def write_json_atomic(path: Path, obj) -> None:
    """Write JSON via a temp file + atomic os.replace, so an interrupted write
    (SIGKILL on shutdown, power loss) can never leave a truncated file that the
    already-collected check would mistake for complete. Same-dir temp keeps the
    replace atomic (one filesystem)."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def retry_after_seconds(headers, default: int = 60, cap: int = 120) -> int:
    """Parse a `Retry-After` header into a bounded sleep.

    `Retry-After` may be an integer (seconds) or an HTTP-date; we only handle
    the integer form (the APIs we hit use it) and fall back to `default`
    otherwise. Capped so a hostile/huge value can't stall the run for hours.
    """
    raw = headers.get("Retry-After")
    try:
        seconds = int(raw) if raw is not None else default
    except ValueError:
        seconds = default
    return max(0, min(seconds, cap))


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------


def fetch_match_details(
    match_id: int,
    opendota_limiter: RateLimiter,
    max_retries: int = 5,
) -> dict:
    """Fetch match details from OpenDota with rate limiting and retries.

    Raises:
        FatalPipelineError: rejected with 401/403 — bad/blocked API key (abort).
        SkipMatch(permanent=True): unknown match (404).
        SkipMatch(permanent=False): transient (429/5xx/network) didn't recover.
    """
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        opendota_limiter.wait()
        try:
            params = {"api_key": OPENDOTA_API_KEY} if OPENDOTA_API_KEY else {}
            resp = requests.get(
                f"{OPENDOTA_API}/matches/{match_id}", params=params, timeout=30
            )
            resp.raise_for_status()
            return resp.json()

        except requests.exceptions.HTTPError as e:
            last_exc = e
            code = e.response.status_code
            if code == 429:
                wait = retry_after_seconds(e.response.headers)
                print(
                    f"OpenDota rate limited, waiting {wait}s "
                    f"(attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            if code in (401, 403):
                raise FatalPipelineError(
                    f"OpenDota rejected the request (HTTP {code}) — bad/blocked API key"
                ) from e
            if code >= 500:
                wait = min(2 ** (attempt + 1), 30)
                print(
                    f"OpenDota server error {code} for {match_id}, "
                    f"retrying in {wait}s (attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            # Other 4xx (e.g. 404 unknown match) — permanent, skip + advance.
            raise SkipMatch(
                f"OpenDota HTTP {code} for match {match_id}", permanent=True
            ) from e

        except requests.exceptions.RequestException as e:
            last_exc = e
            wait = min(2 ** (attempt + 1), 30)
            print(
                f"OpenDota network error for {match_id}: {e}, "
                f"retrying in {wait}s (attempt {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue

    raise SkipMatch(
        f"OpenDota fetch for {match_id} did not recover after {max_retries} retries",
        permanent=False,
    ) from last_exc


def get_seq_num_from_match_id(
    match_id: int, opendota_limiter: RateLimiter
) -> int | None:
    """Get the sequence number for a given match ID via OpenDota."""
    print(f"Fetching sequence number for match {match_id}...")
    try:
        md = fetch_match_details(match_id, opendota_limiter)
    except (FatalPipelineError, SkipMatch) as e:
        print(f"Error: {e}", file=sys.stderr)
        return None
    if "match_seq_num" in md:
        seq_num = md["match_seq_num"]
        print(f"Match {match_id} has sequence number {seq_num}")
        return seq_num
    print(f"Could not find sequence number for match {match_id}", file=sys.stderr)
    return None


def get_seq_num_from_date(
    date_str: str, opendota_limiter: RateLimiter
) -> int | None:
    """Find a match near the given date and return its seq num."""
    target_ts = int(
        datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=UTC).timestamp()
    )
    print(f"Finding match near {date_str} (unix {target_ts})...")

    sql = (
        f"SELECT match_id FROM public_matches "
        f"WHERE start_time > {target_ts} ORDER BY start_time ASC LIMIT 1"
    )
    opendota_limiter.wait()
    try:
        params = {"sql": sql}
        if OPENDOTA_API_KEY:
            params["api_key"] = OPENDOTA_API_KEY
        resp = requests.get(f"{OPENDOTA_API}/explorer", params=params, timeout=30)
        resp.raise_for_status()
        rows = resp.json().get("rows", [])
    except requests.exceptions.RequestException as e:
        print(f"Error querying OpenDota explorer: {e}", file=sys.stderr)
        return None

    if not rows:
        print(f"No matches found after {date_str}", file=sys.stderr)
        return None

    match_id = rows[0]["match_id"]
    print(f"Found match {match_id} near {date_str}")
    return get_seq_num_from_match_id(match_id, opendota_limiter)


def explorer_query(sql: str, opendota_limiter: RateLimiter, max_retries: int = 8) -> list[dict]:
    """Run an OpenDota /explorer SQL query, retrying on read-timeout / network /
    5xx. The explorer has brief transient-500 outage windows (verified: a query
    that 500s for ~30s succeeds 6/6 minutes later), so the retry budget (~150s of
    capped backoff) is sized to ride those out. Raises RuntimeError if it can't."""
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        opendota_limiter.wait()
        try:
            params = {"sql": sql}
            if OPENDOTA_API_KEY:
                params["api_key"] = OPENDOTA_API_KEY
            resp = requests.get(f"{OPENDOTA_API}/explorer", params=params, timeout=90)
            resp.raise_for_status()
            data = resp.json()
            if data.get("err"):
                last_exc = RuntimeError(data["err"])
                wait = min(2 ** (attempt + 1), 30)
                print(
                    f"OpenDota explorer query error ({data['err']}), retrying in "
                    f"{wait}s (attempt {attempt + 1}/{max_retries})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            return data.get("rows", [])
        except requests.exceptions.RequestException as e:
            last_exc = e
            wait = min(2 ** (attempt + 1), 30)
            print(
                f"OpenDota explorer network error: {e}, retrying in {wait}s "
                f"(attempt {attempt + 1}/{max_retries})",
                file=sys.stderr,
            )
            time.sleep(wait)
            continue
    raise RuntimeError(f"explorer query failed after {max_retries} retries: {last_exc}")


def find_floor_match_id(target_ts: int, opendota_limiter: RateLimiter) -> int | None:
    """Smallest AD match_id with start_time >= target_ts — the old-first paging
    floor — in one indexed query. `ORDER BY start_time ASC LIMIT 1` uses the
    start_time index (the same shape as get_seq_num_from_date); ordering by
    match_id over a start_time filter instead read-times-out. None if no match."""
    rows = explorer_query(
        f"SELECT match_id FROM public_matches "
        f"WHERE game_mode = {ABILITY_DRAFT_MODE} AND start_time > {target_ts} "
        f"ORDER BY start_time ASC LIMIT 1",
        opendota_limiter,
    )
    return rows[0]["match_id"] if rows else None


def discover_matches_explorer(
    limit: int,
    output_dir: Path,
    opendota_limiter: RateLimiter,
    fatal_error: threading.Event,
    stats: PipelineStats,
    retention_days: int,
):
    """Generator yielding (match_id, match_id) for AD matches **oldest-first**,
    via OpenDota /explorer paging — bypasses the slow Steam sequential scan.

    Starts at the ~retention-old floor (one indexed start_time query) and pages
    forward by ascending match_id, so the matches nearest expiry are collected
    first. Skips already-collected matches (both detail files present) and any
    past the retention cutoff. Stops when /explorer returns no newer AD matches.
    Genuinely-gone replays (~7%, CDN 403/404) are per-match permanent-skips along
    the way; a match too new to have its replay published yet 403s as well —
    OpenDota issues the replay_salt before Valve uploads the file — and is
    re-yielded on a later run once it lands (no detail files yet → not skipped).
    match_id doubles as the monotonic cursor key for the resume frontier."""
    cutoff = int(time.time()) - retention_days * 86400
    try:
        floor = find_floor_match_id(cutoff, opendota_limiter)
    except RuntimeError as e:
        print(f"Explorer discovery stopped (floor lookup): {e}", file=sys.stderr)
        return
    if floor is None:
        print("OpenDota explorer returned no AD matches.")
        return
    print(f"Explorer old-first floor (~{retention_days}d): match_id {floor}")

    cursor = floor - 1  # page match_id > cursor; -1 so the floor match is included
    found = 0
    while found < limit and not fatal_error.is_set():
        sql = (
            f"SELECT match_id, start_time FROM public_matches "
            f"WHERE game_mode = {ABILITY_DRAFT_MODE} AND match_id > {cursor} "
            f"ORDER BY match_id ASC LIMIT {OPENDOTA_EXPLORER_BATCH}"
        )
        try:
            rows = explorer_query(sql, opendota_limiter)
        except RuntimeError as e:
            print(f"Explorer discovery stopped: {e}", file=sys.stderr)
            return
        if not rows:
            print("Explorer caught up to the newest AD matches.")
            return
        for row in rows:
            if found >= limit or fatal_error.is_set():
                return
            if (row.get("start_time") or 0) < cutoff:
                continue  # older than retention (replay expired) — skip forward
            mid = row["match_id"]
            md = output_dir / str(mid)
            if (md / "match_details.json").exists() and (md / "draft_details.json").exists():
                continue  # already collected
            if gone_marker(output_dir / "errors", mid).exists():
                continue  # permanently unavailable — don't re-attempt
            yield mid, mid
            found += 1
        cursor = rows[-1]["match_id"]
        print(
            f"Discovered {found}/{limit} AD matches (explorer, oldest-first) "
            f"(downloaded={stats.downloaded} parsed={stats.succeeded} skipped={stats.skipped})"
        )


# ---------------------------------------------------------------------------
# Stage 1 — Discover (main thread)
# ---------------------------------------------------------------------------


def discover_matches(
    api_key: str,
    limit: int,
    start_seq_num: int | None,
    fatal_error: threading.Event,
    stats: PipelineStats,
):
    """Generator yielding (match_id, match_seq_num) for AD matches, in
    increasing seq order."""
    current_seq_num = start_seq_num
    found = 0
    consecutive_429s = 0

    while found < limit:
        if fatal_error.is_set():
            return

        params: dict = {"matches_requested": 100}
        if api_key:
            params["key"] = api_key
        if current_seq_num:
            params["start_at_match_seq_num"] = current_seq_num

        try:
            response = requests.get(
                STEAM_MATCH_HISTORY_BY_SEQ, params=params, timeout=30
            )
            response.raise_for_status()
            consecutive_429s = 0
            data = response.json()

            batch = data.get("result", {}).get("matches", [])
            if not batch:
                print("No more matches from Steam API")
                return

            ad_matches = [m for m in batch if m.get("game_mode") == ABILITY_DRAFT_MODE]
            for m in ad_matches:
                if found >= limit:
                    return
                yield m["match_id"], m["match_seq_num"]
                found += 1

            print(
                f"Discovered {found}/{limit} AD matches "
                f"(downloaded={stats.downloaded} parsed={stats.succeeded} skipped={stats.skipped})"
            )

            current_seq_num = batch[-1]["match_seq_num"] + 1
            time.sleep(STEAM_RATE_LIMIT_DELAY)

        except requests.exceptions.HTTPError as e:
            code = e.response.status_code
            if code == 429:
                consecutive_429s += 1
                wait = retry_after_seconds(e.response.headers)
                print(
                    f"Steam API rate limited, waiting {wait}s "
                    f"(attempt {consecutive_429s})...",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            if code in (401, 403):
                # Rejected key — systemic, abort the whole run.
                print(f"FATAL: Steam API rejected the key (HTTP {code})", file=sys.stderr)
                fatal_error.set()
                return
            print(f"Steam API HTTP error: {e}", file=sys.stderr)
            return

        except requests.exceptions.RequestException as e:
            print(f"Steam API error: {e}", file=sys.stderr)
            return


# ---------------------------------------------------------------------------
# Stage 2 — Download (pure logic, no queue awareness)
# ---------------------------------------------------------------------------


def download_replay(
    match_id: int, replay_url: str, bz2_path: Path, dem_path: Path
) -> int:
    """Download the replay's .dem.bz2 and decompress it to .dem, retrying the
    whole operation on any recoverable failure. Returns compressed bytes.

    Decompression doubles as an integrity check: a truncated/corrupt download
    fails there and is re-downloaded.

    Raises:
        SkipMatch(permanent=True): CDN 403/404/410 — replay genuinely gone.
        SkipMatch(permanent=False): 5xx/network/corrupt didn't recover (retry
            on --resume; the match is valid, the CDN just hiccupped).
        FatalPipelineError: disk full.
    """
    max_retries = 3
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        if attempt > 0:
            wait = min(2**attempt, 30)
            print(f"  Match {match_id}: retry {attempt} after {wait}s ({last_exc})")
            time.sleep(wait)
        try:
            resp = requests.get(replay_url, stream=True, timeout=120)
            resp.raise_for_status()
            downloaded = 0
            with open(bz2_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
            # Decompress now — doubles as an integrity check; a truncated or
            # corrupt download fails here and is retried (re-downloaded).
            with bz2.BZ2File(bz2_path, "rb") as fin, open(dem_path, "wb") as fout:
                for block in iter(lambda: fin.read(1024 * 1024), b""):
                    fout.write(block)
            return downloaded

        except requests.exceptions.HTTPError as e:
            if e.response.status_code in (403, 404, 410):
                raise SkipMatch(
                    f"replay unavailable (HTTP {e.response.status_code})", permanent=True
                ) from e
            last_exc = e  # 5xx / unexpected status — transient, retry
        except requests.exceptions.RequestException as e:
            last_exc = e  # network drop / timeout — transient, retry
        except OSError as e:
            if e.errno == errno.ENOSPC:
                raise FatalPipelineError("disk full while writing replay") from e
            last_exc = e  # corrupt bz2 (invalid data stream) — transient, retry
        except EOFError as e:
            last_exc = e  # truncated bz2 stream — transient, retry

    raise SkipMatch(
        f"replay download/decompress failed after {max_retries} retries: {last_exc}",
        permanent=False,
    ) from last_exc


def process_download(
    match_id: int,
    opendota_limiter: RateLimiter,
    tmp_dir: Path,
    keep_replays_dir: Path | None = None,
) -> tuple[dict, Path]:
    """Fetch OpenDota details, download, decompress. Returns (match_details, dem_path).

    If `keep_replays_dir` is set, the compressed .dem.bz2 is persisted to
    `<keep_replays_dir>/<match_id>/<match_id>.dem.bz2` alongside a copy of
    `match_details.json`, so the archive stays directly reusable for re-parsing.

    Raises SkipMatch for per-match failures; FatalPipelineError only propagates
    up from fetch_match_details (rejected key) / download_replay (disk full).
    """
    md = fetch_match_details(match_id, opendota_limiter)

    replay_url = md.get("replay_url")
    if not replay_url:
        # OpenDota may not have a replay_url yet for a fresh match — transient.
        raise SkipMatch("no replay_url available", permanent=False)

    bz2_path = tmp_dir / f"{match_id}.dem.bz2"
    dem_path = tmp_dir / f"{match_id}.dem"

    try:
        downloaded = download_replay(match_id, replay_url, bz2_path, dem_path)
        print(f"  Match {match_id}: downloaded {downloaded / 1024 / 1024:.1f} MB")
        if keep_replays_dir is not None:
            archive_dir = keep_replays_dir / str(match_id)
            archive_dir.mkdir(parents=True, exist_ok=True)
            archive_tmp = archive_dir / f"{match_id}.dem.bz2.tmp"
            shutil.copy2(bz2_path, archive_tmp)
            os.replace(archive_tmp, archive_dir / f"{match_id}.dem.bz2")
            write_json_atomic(archive_dir / "match_details.json", md)
    except BaseException as e:
        if isinstance(e, SkipMatch) and e.permanent and e.details is None:
            e.details = md   # replay gone, details in hand — keep them for the censoring record
        _cleanup(bz2_path, dem_path)
        raise
    finally:
        _cleanup(bz2_path)

    return md, dem_path


# ---------------------------------------------------------------------------
# Stage 3 — Parse (pure logic, no queue awareness)
# ---------------------------------------------------------------------------


def run_clarity(dem_path: Path, clarity_jar: Path) -> dict:
    """Run the Clarity JAR and return parsed JSON.

    Raises:
        FatalPipelineError: java/JRE not found (setup problem — abort).
        subprocess.CalledProcessError: clarity exited non-zero.
        subprocess.TimeoutExpired: clarity took too long.
        json.JSONDecodeError: clarity output wasn't valid JSON.
    """
    try:
        result = subprocess.run(
            [
                "java",
                "--add-opens",
                "java.base/java.util=ALL-UNNAMED",
                "--add-opens",
                "java.base/java.lang=ALL-UNNAMED",
                "-jar",
                str(clarity_jar),
                "--in",
                str(dem_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=120,
        )
    except FileNotFoundError as e:
        raise FatalPipelineError("java not found — install a JRE") from e
    return json.loads(result.stdout)


def process_parse(
    match_id: int,
    md: dict,
    dem_path: Path,
    output_dir: Path,
    clarity_jar: Path,
):
    """Run Clarity, validate, write output JSONs.

    Raises SkipMatch when this replay can't be parsed — `permanent=True` for an
    unparseable/incomplete demo, `permanent=False` for a clarity timeout (may be
    transient load). FatalPipelineError only propagates from run_clarity (java
    missing).
    """
    try:
        clarity_json = run_clarity(dem_path, clarity_jar)
    except subprocess.CalledProcessError as e:
        msg = "clarity exited non-zero"
        if e.stderr:
            msg += f"\n    stderr: {e.stderr[:500]}"
        raise SkipMatch(msg, permanent=True) from e
    except subprocess.TimeoutExpired as e:
        raise SkipMatch(f"clarity timed out: {e}", permanent=False) from e
    except json.JSONDecodeError as e:
        raise SkipMatch(f"clarity JSON decode error: {e}", permanent=True) from e

    pool_items = clarity_json.get("pool_items") or []
    picks = clarity_json.get("picks") or []

    if not pool_items or not picks:
        raise SkipMatch(
            f"incomplete clarity output (pool={len(pool_items)} picks={len(picks)})",
            permanent=True,
        )

    match_dir = output_dir / str(match_id)
    match_dir.mkdir(parents=True, exist_ok=True)

    write_json_atomic(match_dir / "match_details.json", md)
    write_json_atomic(match_dir / "draft_details.json", clarity_json)

    print(f"  Match {match_id}: OK (pool={len(pool_items)} picks={len(picks)})")


# ---------------------------------------------------------------------------
# Cleanup helper
# ---------------------------------------------------------------------------


def _cleanup(*paths: Path):
    for p in paths:
        with contextlib.suppress(OSError):
            p.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Worker threads (thin loops: queue plumbing + error categorization)
# ---------------------------------------------------------------------------


def _trip_breaker(reason: str, fatal_error: threading.Event):
    """Print the circuit-breaker abort message and set the fatal flag."""
    print(
        f"ABORT: {reason}. Halting; fix and re-run with --resume.",
        file=sys.stderr,
    )
    fatal_error.set()


def _put_sentinels(q: Queue, count: int, fatal_error: threading.Event):
    """Send `count` shutdown sentinels, stopping early if the run is aborting.

    Workers self-terminate on `fatal_error` (via their get() timeout), so once
    we're aborting there is no need to deliver sentinels — and blocking here on
    a full queue with no consumers left would hang the main thread.
    """
    for _ in range(count):
        while not fatal_error.is_set():
            try:
                q.put(SENTINEL, timeout=1)
                break
            except Full:
                continue


def _handle_skip(
    exc: SkipMatch,
    match_id: int,
    seq: int,
    cursor: ResumeCursor,
    stats: PipelineStats,
    error_dir: Path,
    fatal_error: threading.Event,
):
    """Log a per-match skip, advance/hold the resume frontier, feed the breaker.

    Permanent skips (or transient ones that have exhausted TRANSIENT_RETRY_CAP
    attempts across runs) resolve the seq so the frontier moves past them.
    Transient skips under the cap leave the seq unresolved so `--resume` retries
    it. Trips the circuit breaker after too many consecutive failures.
    """
    log_error(error_dir, match_id, str(exc), exc=exc)
    give_up = exc.permanent or count_attempts(error_dir, match_id) >= TRANSIENT_RETRY_CAP
    if exc.permanent:
        gone_marker(error_dir, match_id).touch()
        if exc.details is not None:
            write_json_atomic(error_dir / f"{match_id}.details.json", exc.details)
        kind = "permanent"
    elif give_up:
        kind = f"transient, gave up after {TRANSIENT_RETRY_CAP} attempts"
    else:
        kind = "transient, will retry on --resume"
    print(f"  Match {match_id}: SKIP ({kind}) — {exc}")
    if give_up:
        cursor.resolve(seq)
    reason = stats.record_skip(permanent=exc.permanent)
    if reason:
        _trip_breaker(reason, fatal_error)


def downloader_worker(
    match_queue: Queue,
    parse_queue: Queue,
    tmp_dir: Path,
    opendota_limiter: RateLimiter,
    fatal_error: threading.Event,
    stats: PipelineStats,
    cursor: ResumeCursor,
    error_dir: Path,
    keep_replays_dir: Path | None = None,
):
    """Pull (match_id, seq) from match_queue, download, push to parse_queue."""
    while not fatal_error.is_set():
        try:
            item = match_queue.get(timeout=1)
        except Empty:
            continue  # re-check fatal_error, then keep waiting
        if item is SENTINEL:
            return

        match_id, seq = item
        try:
            md, dem_path = process_download(
                match_id, opendota_limiter, tmp_dir, keep_replays_dir,
            )
            stats.inc_downloaded()
            item_out = (match_id, seq, md, dem_path)
            while not fatal_error.is_set():
                try:
                    parse_queue.put(item_out, timeout=1)
                    break
                except Full:
                    continue
            else:
                _cleanup(dem_path)
        except SkipMatch as e:
            _handle_skip(e, match_id, seq, cursor, stats, error_dir, fatal_error)
            if fatal_error.is_set():
                return
        except FatalPipelineError as e:
            log_error(error_dir, match_id, str(e), exc=e)
            print(f"FATAL: Match {match_id}: {e}", file=sys.stderr)
            fatal_error.set()
            return
        except Exception as e:
            log_error(error_dir, match_id, f"unexpected: {e}", exc=e)
            print(f"UNEXPECTED FATAL: Match {match_id}: {e!r}", file=sys.stderr)
            traceback.print_exc()
            fatal_error.set()
            return


def parser_worker(
    parse_queue: Queue,
    output_dir: Path,
    clarity_jar: Path,
    fatal_error: threading.Event,
    stats: PipelineStats,
    cursor: ResumeCursor,
    error_dir: Path,
):
    """Pull (match_id, seq, md, dem_path) from parse_queue, parse, write output."""
    while not fatal_error.is_set():
        try:
            item = parse_queue.get(timeout=1)
        except Empty:
            continue  # re-check fatal_error, then keep waiting
        if item is SENTINEL:
            return

        match_id, seq, md, dem_path = item
        try:
            process_parse(match_id, md, dem_path, output_dir, clarity_jar)
            stats.record_success()
            cursor.resolve(seq)
        except SkipMatch as e:
            _handle_skip(e, match_id, seq, cursor, stats, error_dir, fatal_error)
            if fatal_error.is_set():
                return
        except FatalPipelineError as e:
            log_error(error_dir, match_id, str(e), exc=e)
            print(f"FATAL: Match {match_id}: {e}", file=sys.stderr)
            fatal_error.set()
            return
        except Exception as e:
            log_error(error_dir, match_id, f"unexpected: {e}", exc=e)
            print(f"UNEXPECTED FATAL: Match {match_id}: {e!r}", file=sys.stderr)
            traceback.print_exc()
            fatal_error.set()
            return
        finally:
            _cleanup(dem_path)


# ---------------------------------------------------------------------------
# Pipeline orchestration
# ---------------------------------------------------------------------------


def run_pipeline(
    n: int,
    output_dir: Path,
    clarity_jar: Path,
    api_key: str | None,
    start_seq_num: int | None,
    num_downloaders: int,
    num_parsers: int,
    keep_replays_dir: Path | None = None,
    abort_after_failures: int = 25,
    source: str = "steam",
    retention_days: int = 14,
):
    if not clarity_jar.exists():
        print(f"Clarity JAR not found: {clarity_jar}", file=sys.stderr)
        return 1

    tmp_dir = Path(tempfile.mkdtemp(prefix="dota2-pipeline-"))
    print(f"Temp dir: {tmp_dir}")
    print(
        f"Pipeline: {n} matches, {num_downloaders} downloader(s), "
        f"{num_parsers} parser(s), discovery={source}"
    )
    print(f"Output: {output_dir.absolute()}")

    match_queue: Queue = Queue(maxsize=10)
    parse_queue: Queue = Queue(maxsize=4)
    fatal_error = threading.Event()
    error_dir = output_dir / "errors"
    stats = PipelineStats(abort_after_failures=abort_after_failures)
    cursor = ResumeCursor(output_dir / ".resume_seq")
    opendota_limiter = RateLimiter(OPENDOTA_RATE_LIMIT_DELAY)

    output_dir.mkdir(parents=True, exist_ok=True)

    if keep_replays_dir is not None:
        keep_replays_dir.mkdir(parents=True, exist_ok=True)
        print(f"Keeping .dem.bz2 archives in: {keep_replays_dir.absolute()}")
    print(
        f"Per-match failures are skipped; the run aborts on a rejected key, "
        f"missing JAR/java, {stats.abort_after_failures} consecutive transient "
        f"failures, or {stats.abort_after_permanent} consecutive unavailable replays."
    )

    downloaders = [
        threading.Thread(
            target=downloader_worker,
            args=(
                match_queue,
                parse_queue,
                tmp_dir,
                opendota_limiter,
                fatal_error,
                stats,
                cursor,
                error_dir,
                keep_replays_dir,
            ),
            name=f"downloader-{i}",
            daemon=True,
        )
        for i in range(num_downloaders)
    ]
    parsers = [
        threading.Thread(
            target=parser_worker,
            args=(
                parse_queue,
                output_dir,
                clarity_jar,
                fatal_error,
                stats,
                cursor,
                error_dir,
            ),
            name=f"parser-{i}",
            daemon=True,
        )
        for i in range(num_parsers)
    ]

    for t in downloaders + parsers:
        t.start()

    try:
        # Stage 1: Discover matches (main thread)
        discovered = 0
        if source == "explorer":
            disc = discover_matches_explorer(
                n, output_dir, opendota_limiter, fatal_error, stats, retention_days)
        else:
            assert api_key is not None  # main() requires the key for --source steam
            disc = discover_matches(api_key, n, start_seq_num, fatal_error, stats)
        for match_id, seq in disc:
            if fatal_error.is_set():
                break

            cursor.discovered(seq)
            match_out = output_dir / str(match_id)
            if (match_out / "match_details.json").exists() and (
                match_out / "draft_details.json"
            ).exists():
                print(f"  Match {match_id}: already parsed, skipping")
                cursor.resolve(seq)
                continue

            while not fatal_error.is_set():
                try:
                    match_queue.put((match_id, seq), timeout=1)
                    break
                except Full:
                    continue
            else:
                break
            discovered += 1

        print(f"\nDiscovery done: {discovered} matches queued")

        if fatal_error.is_set():
            # Workers self-terminate on fatal_error (via their get() timeout);
            # wait briefly for in-flight work, then daemon threads die at exit.
            for t in downloaders + parsers:
                t.join(timeout=5)
        else:
            # Normal shutdown: sentinel each stage, then join. Sentinel puts are
            # fatal-aware so a late abort can't hang us on a full queue, and the
            # workers' get() timeout guarantees join() returns either way.
            _put_sentinels(match_queue, len(downloaders), fatal_error)
            for t in downloaders:
                t.join()
            _put_sentinels(parse_queue, len(parsers), fatal_error)
            for t in parsers:
                t.join()

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"Succeeded: {stats.succeeded}")
    print(f"Skipped:   {stats.skipped}")
    print(f"Output:    {output_dir.absolute()}")

    if error_dir.exists() and any(error_dir.iterdir()):
        print(f"Errors:    {error_dir.absolute()}")

    if fatal_error.is_set():
        print("\nPipeline aborted (systemic failure). Re-run with --resume.", file=sys.stderr)
        return 1

    # Success if we collected something, or there was simply nothing new to do
    # (everything in range was already parsed). Only a run that queued new
    # matches yet parsed none of them is a real failure.
    if stats.succeeded > 0 or discovered == 0:
        return 0
    return 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def resume_start_seq(output_dir: Path) -> int | None:
    """Where `--resume` should restart: the persisted frontier+1. None if
    there is no frontier to resume from."""
    cursor_file = output_dir / ".resume_seq"
    if cursor_file.exists():
        try:
            return int(cursor_file.read_text().strip()) + 1
        except (ValueError, OSError):
            pass
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Collect AD replays: discover → download → parse"
    )
    parser.add_argument(
        "-n",
        "--num-matches",
        type=int,
        default=10,
        help="Number of AD matches to process (default: 10)",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: <DOTA2AD_ROOT>/parsed)",
    )
    parser.add_argument(
        "--downloaders",
        type=int,
        default=3,
        help="Number of downloader threads (default: 3)",
    )
    parser.add_argument(
        "--parsers",
        type=int,
        default=2,
        help="Number of parser threads (default: 2)",
    )
    parser.add_argument(
        "--clarity-jar",
        type=Path,
        default=DEFAULT_CLARITY_JAR,
        help="Path to Clarity shaded JAR",
    )
    parser.add_argument(
        "--steam-api-key",
        type=str,
        help="Steam API key (or set STEAM_API_KEY env var)",
    )
    parser.add_argument(
        "--start-seq-num",
        type=int,
        help="Starting match sequence number",
    )
    parser.add_argument(
        "--start-match-id",
        type=int,
        help="Start from the sequence number of this match ID",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start from a match near this date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the .resume_seq frontier in the output directory",
    )
    parser.add_argument(
        "--keep-replays-dir",
        type=Path,
        default=None,
        help="Directory to persist downloaded .dem.bz2 archives at "
        "<dir>/<match_id>/<match_id>.dem.bz2, for re-parsing later "
        "without re-downloading. Defaults to <DOTA2AD_ROOT>/cache/replays.",
    )
    parser.add_argument(
        "--no-keep-replays",
        action="store_true",
        help="Don't persist downloaded replays (skip populating the replay cache).",
    )
    parser.add_argument(
        "--max-consecutive-failures",
        type=int,
        default=25,
        help="Abort the whole run if this many matches fail in a row — signals a "
        "systemic problem (rejected key, API/CDN outage, disk full). Isolated "
        "per-match failures (unavailable replays, unparseable demos) are always "
        "skipped, never fatal (default: 25).",
    )
    parser.add_argument(
        "--source",
        choices=["steam", "explorer"],
        default="explorer",
        help="Discovery source. 'explorer' (default) pages OpenDota /explorer "
        "for AD match_ids oldest-first (fast, no API key required, "
        "replay-retention-bounded; auto-skips already-collected so --resume "
        "is implicit); 'steam' scans GetMatchHistoryBySequenceNum (slow but "
        "~complete, needs STEAM_API_KEY). The --start-*/--resume options "
        "apply to 'steam' only.",
    )
    parser.add_argument(
        "--retention-days",
        type=int,
        default=14,
        help="Explorer source: the old-first floor age. Replays are reliably "
        "downloadable through ~14d, with a fuzzy 15-16d boundary where "
        "individual matches expire (those get per-match-skipped on a 403/404). "
        "Default 14 = the reliable edge; push to 15 to chase the fuzzy tail.",
    )

    args = parser.parse_args()
    output_dir = args.output_dir or default_paths().parsed

    start_opts = sum(
        1
        for x in [args.start_seq_num, args.start_match_id, args.start_date, args.resume]
        if x
    )
    if start_opts > 1:
        print(
            "Error: only one of --start-seq-num, --start-match-id, "
            "--start-date, or --resume can be specified",
            file=sys.stderr,
        )
        return 1

    api_key = args.steam_api_key or os.environ.get("STEAM_API_KEY")
    if args.source == "steam" and not api_key:
        print(
            "Error: --source steam needs a Steam API key "
            "(--steam-api-key or STEAM_API_KEY env)",
            file=sys.stderr,
        )
        return 1

    opendota_limiter = RateLimiter(OPENDOTA_RATE_LIMIT_DELAY)
    start_seq_num = args.start_seq_num

    # --start-*/--resume resolve a Steam sequence cursor; explorer pages
    # oldest-first and auto-skips already-collected, so it ignores them.
    if args.source == "steam":
        if args.start_match_id:
            start_seq_num = get_seq_num_from_match_id(args.start_match_id, opendota_limiter)
            if start_seq_num is None:
                print("Failed to get seq num from match ID", file=sys.stderr)
                return 1

        if args.start_date:
            start_seq_num = get_seq_num_from_date(args.start_date, opendota_limiter)
            if start_seq_num is None:
                print("Failed to get seq num from date", file=sys.stderr)
                return 1

        if args.resume:
            start_seq_num = resume_start_seq(output_dir)
            if start_seq_num is None:
                print(
                    "No existing matches found in output dir, nothing to resume from",
                    file=sys.stderr,
                )
                return 1
            print(f"Resuming from seq_num {start_seq_num}")

    if args.no_keep_replays:
        keep_replays_dir = None
    else:
        keep_replays_dir = args.keep_replays_dir or default_paths().replays

    return run_pipeline(
        n=args.num_matches,
        output_dir=output_dir,
        clarity_jar=args.clarity_jar,
        api_key=api_key,
        start_seq_num=start_seq_num,
        num_downloaders=args.downloaders,
        num_parsers=args.parsers,
        keep_replays_dir=keep_replays_dir,
        abort_after_failures=args.max_consecutive_failures,
        source=args.source,
        retention_days=args.retention_days,
    )


if __name__ == "__main__":
    sys.exit(main())
