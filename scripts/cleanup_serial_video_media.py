"""Preview or remove media belonging to terminal serial video series."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
from typing import Any

import pymysql
from dotenv import load_dotenv
from pymysql.cursors import DictCursor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "agentic" / "src"))

from agentic.runtime.serial_paths import serial_series_artifact_root  # noqa: E402


MEDIA_SUFFIXES = {
    ".avi",
    ".bmp",
    ".flac",
    ".gif",
    ".jpeg",
    ".jpg",
    ".m4a",
    ".m4v",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".ogg",
    ".png",
    ".wav",
    ".webm",
    ".webp",
}


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "List local media for completed/discarded serial video series. "
            "Pass --delete to remove the listed files."
        )
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "agentic" / "output",
        help="Media root to inspect (default: agentic/output).",
    )
    parser.add_argument(
        "--delete",
        action="store_true",
        help="Delete the listed media files. Without this flag, only preview them.",
    )
    return parser.parse_args()


def _media_root(value: Path) -> Path:
    root = value.expanduser().resolve()
    try:
        root.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"Media root must stay inside the project: {root}") from exc
    if root == PROJECT_ROOT:
        raise ValueError("Media root cannot be the project root itself.")
    return root


def _database_connection() -> pymysql.connections.Connection:
    load_dotenv(PROJECT_ROOT / "media_overload.env", override=False)
    host = os.getenv("mysql_host", "").strip()
    user = os.getenv("serial_video_mysql_user", "").strip()
    password = os.getenv("serial_video_mysql_password", "")
    database = os.getenv("serial_video_mysql_database", "").strip()
    if not all((host, user, password, database)):
        raise ValueError(
            "Set mysql_host, serial_video_mysql_user, serial_video_mysql_password, "
            "and serial_video_mysql_database in media_overload.env."
        )
    if not re.fullmatch(r"[A-Za-z0-9_]+", database):
        raise ValueError("serial_video_mysql_database may contain only letters, digits, and underscores.")
    return pymysql.connect(
        host=host,
        port=int(os.getenv("mysql_port", "3306")),
        user=user,
        password=password,
        database=database,
        charset="utf8mb4",
        cursorclass=DictCursor,
        connect_timeout=5,
        read_timeout=30,
        write_timeout=30,
    )


def _inside_root(path: Path, root: Path) -> Path | None:
    if path.is_symlink():
        return None
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError):
        return None
    if not resolved.is_file() or resolved.suffix.lower() not in MEDIA_SUFFIXES:
        return None
    return resolved


def _mount_points() -> set[Path]:
    if os.name == "nt":
        return set()
    if not sys.platform.startswith("linux"):
        raise OSError(
            "Cleanup cannot verify filesystem mount boundaries on this platform."
        )
    try:
        mount_table = Path("/proc/self/mountinfo").read_text(encoding="utf-8")
    except OSError as exc:
        raise OSError("Cleanup cannot read /proc/self/mountinfo safely.") from exc

    points: set[Path] = set()
    for line in mount_table.splitlines():
        fields = line.split()
        if len(fields) < 6:
            raise OSError("Cleanup found an invalid entry in /proc/self/mountinfo.")
        mount_point = re.sub(
            r"\\([0-7]{3})",
            lambda match: chr(int(match.group(1), 8)),
            fields[4],
        )
        points.add(Path(mount_point))
    return points


def _is_mount_point(
    path: Path,
    root_device: int,
    mount_points: set[Path],
) -> bool:
    try:
        resolved = path.resolve(strict=True)
        if resolved in mount_points or os.path.ismount(resolved):
            return True
        return resolved.stat().st_dev != root_device
    except (OSError, RuntimeError):
        return True


def _has_mounted_path_or_ancestor(path: Path, root: Path) -> bool:
    try:
        mount_points = _mount_points()
        root_device = root.stat().st_dev
    except OSError:
        return True

    current = path
    while current != root:
        try:
            resolved = current.resolve(strict=True)
            resolved.relative_to(root)
            if resolved != current:
                return True
            if _is_mount_point(resolved, root_device, mount_points):
                return True
        except (OSError, RuntimeError, ValueError):
            return True
        current = current.parent
    return False


def _series_artifact_media_files(
    media_root: Path,
    series_id: str,
    *,
    require_canonical_root: bool,
) -> set[Path]:
    artifact_root = serial_series_artifact_root(media_root, series_id)
    try:
        resolved_root = artifact_root.resolve(strict=True)
        resolved_root.relative_to(media_root)
        if require_canonical_root and resolved_root != artifact_root.absolute():
            return set()
    except (OSError, RuntimeError, ValueError):
        return set()
    if not resolved_root.is_dir():
        return set()
    if _has_mounted_path_or_ancestor(resolved_root, media_root):
        return set()

    files: set[Path] = set()
    pending_directories = [resolved_root]
    visited_directories: set[Path] = set()
    while pending_directories:
        directory = pending_directories.pop()
        try:
            resolved_directory = directory.resolve(strict=True)
            resolved_directory.relative_to(resolved_root)
        except (OSError, RuntimeError, ValueError):
            continue
        if resolved_directory in visited_directories:
            continue
        visited_directories.add(resolved_directory)
        if _has_mounted_path_or_ancestor(resolved_directory, media_root):
            continue

        try:
            with os.scandir(resolved_directory) as entries:
                for entry in entries:
                    path = Path(entry.path)
                    try:
                        is_directory = entry.is_dir(follow_symlinks=False)
                    except OSError:
                        continue
                    if is_directory:
                        pending_directories.append(path)
                        continue
                    resolved = _inside_root(path, media_root)
                    if resolved is None or _has_mounted_path_or_ancestor(
                        resolved, media_root
                    ):
                        continue
                    try:
                        resolved.relative_to(resolved_root)
                    except ValueError:
                        continue
                    files.add(resolved)
        except OSError:
            continue
    return files


def _database_media_path(raw_path: Any, media_root: Path) -> Path | None:
    raw = str(raw_path or "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    resolved = _inside_root(path, media_root)
    if resolved is None or _has_mounted_path_or_ancestor(resolved, media_root):
        return None
    return resolved


def _is_completed_series_retryable(
    series_id: str,
    episodes: list[dict[str, Any]],
    publications: dict[tuple[str, str], dict[str, set[str]]],
) -> str | None:
    for episode in episodes:
        episode_id = str(episode["episode_id"])
        status = str(episode["status"] or "").lower()
        if status == "blocked":
            return f"episode {episode['episode_number']} is blocked and can be resumed"
        if status != "ready":
            continue
        platforms = publications.get((series_id, episode_id), {})
        if not platforms:
            return f"episode {episode['episode_number']} has no publication receipts"
        if any("reported_success" not in statuses for statuses in platforms.values()):
            return f"episode {episode['episode_number']} has a platform without a success receipt"
    return None


def _collect_media(
    connection: pymysql.connections.Connection,
    media_root: Path,
) -> tuple[dict[str, set[Path]], dict[str, str]]:
    _mount_points()
    with connection.cursor() as cursor:
        cursor.execute("SELECT series_id, lifecycle_status FROM series ORDER BY series_id")
        all_series = cursor.fetchall()
        terminal_series = [
            row
            for row in all_series
            if str(row["lifecycle_status"]) in {"completed", "discarded"}
        ]
        cursor.execute(
            """SELECT e.series_id, e.episode_id, e.episode_number, e.version,
                      e.is_current, e.status, e.final_video_path
               FROM episodes e ORDER BY e.series_id, e.episode_number, e.version"""
        )
        episodes = cursor.fetchall()
        cursor.execute(
            """SELECT e.series_id, p.episode_id, p.platform, p.status,
                      e.is_current, p.media_path
               FROM episode_publications p JOIN episodes e ON e.episode_id = p.episode_id"""
        )
        publication_rows = cursor.fetchall()
        cursor.execute(
            """SELECT e.series_id, r.generated_output_path, r.output_path
               FROM render_attempts r JOIN segments g ON g.segment_id = r.segment_id
               JOIN episodes e ON e.episode_id = g.episode_id"""
        )
        render_rows = cursor.fetchall()
        cursor.execute(
            """SELECT series_id, path FROM reference_assets"""
        )
        reference_rows = cursor.fetchall()

    episode_by_series: dict[str, list[dict[str, Any]]] = {}
    for episode in episodes:
        if episode["is_current"]:
            episode_by_series.setdefault(str(episode["series_id"]), []).append(episode)

    publication_statuses: dict[tuple[str, str], dict[str, set[str]]] = {}
    for row in publication_rows:
        if not row["is_current"]:
            continue
        key = (str(row["series_id"]), str(row["episode_id"]))
        platform = str(row["platform"])
        publication_statuses.setdefault(key, {}).setdefault(platform, set()).add(
            str(row["status"] or "").lower()
        )

    eligible: dict[str, str] = {}
    preserved: dict[str, str] = {}
    for row in terminal_series:
        series_id = str(row["series_id"])
        lifecycle = str(row["lifecycle_status"])
        if lifecycle == "completed":
            reason = _is_completed_series_retryable(
                series_id,
                episode_by_series.get(series_id, []),
                publication_statuses,
            )
            if reason:
                preserved[series_id] = reason
                continue
        eligible[series_id] = lifecycle
    protected_series_ids = {
        str(row["series_id"]) for row in all_series
    }.difference(eligible)

    paths_by_series: dict[str, set[Path]] = {series_id: set() for series_id in eligible}
    referenced_paths_by_series: dict[str, set[Path]] = {}

    def record_path(series_id: str, raw_path: Any) -> None:
        resolved = _database_media_path(raw_path, media_root)
        if resolved is None:
            return
        referenced_paths_by_series.setdefault(series_id, set()).add(resolved)
        if series_id in eligible:
            paths_by_series[series_id].add(resolved)

    for episode in episodes:
        series_id = str(episode["series_id"])
        record_path(series_id, episode["final_video_path"])
    for row in render_rows:
        series_id = str(row["series_id"])
        record_path(series_id, row["generated_output_path"])
        record_path(series_id, row["output_path"])
    for row in reference_rows:
        series_id = str(row["series_id"])
        record_path(series_id, row["path"])
    for row in publication_rows:
        series_id = str(row["series_id"])
        record_path(series_id, row["media_path"])

    for series_id in eligible:
        paths_by_series[series_id].update(
            _series_artifact_media_files(
                media_root,
                series_id,
                require_canonical_root=True,
            )
        )

    protected_paths: set[Path] = set()
    for series_id, paths in referenced_paths_by_series.items():
        if series_id not in eligible:
            protected_paths.update(paths)
    for series_id in protected_series_ids:
        protected_paths.update(
            _series_artifact_media_files(
                media_root,
                series_id,
                require_canonical_root=False,
            )
        )
    for series_id in eligible:
        paths_by_series[series_id].difference_update(protected_paths)

    return paths_by_series, preserved


def _format_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> int:
    args = _arguments()
    try:
        media_root = _media_root(args.output_dir)
        connection = _database_connection()
        try:
            paths_by_series, preserved = _collect_media(connection, media_root)
        finally:
            connection.close()
    except (OSError, ValueError, pymysql.MySQLError) as exc:
        print(f"Serial video cleanup could not continue: {exc}", file=sys.stderr)
        return 1

    unique_paths = set().union(*paths_by_series.values()) if paths_by_series else set()
    total_bytes = 0
    for path in unique_paths:
        try:
            if path.is_file():
                total_bytes += path.stat().st_size
        except OSError:
            continue
    action = "Deleting" if args.delete else "Previewing"
    print(f"{action} {len(unique_paths)} media file(s) ({total_bytes / 1024 / 1024:.1f} MiB).")
    for series_id, paths in sorted(paths_by_series.items()):
        print(f"[{series_id}] {len(paths)} file(s)")
        for path in sorted(paths):
            print(f"  {_format_path(path)}")
    for series_id, reason in sorted(preserved.items()):
        print(f"[preserved: {series_id}] {reason}")

    if not args.delete:
        print("Preview only. Re-run with --delete to remove these files.")
        return 0

    failures = 0
    deleted_bytes = 0
    deleted_count = 0
    for path in sorted(unique_paths):
        try:
            resolved = _inside_root(path, media_root)
            if (
                resolved is None
                or resolved != path
                or _has_mounted_path_or_ancestor(resolved, media_root)
            ):
                continue
            size = resolved.stat().st_size
            resolved.unlink()
            deleted_bytes += size
            deleted_count += 1
            print(f"Deleted {_format_path(resolved)}")
        except OSError as exc:
            failures += 1
            print(f"Could not delete {_format_path(path)}: {exc}", file=sys.stderr)
    print(
        f"Deleted {deleted_count} media file(s) "
        f"({deleted_bytes / 1024 / 1024:.1f} MiB)."
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
