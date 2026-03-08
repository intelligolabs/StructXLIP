#!/usr/bin/env python3
"""Collect image paths from Sketchy JSON files, zip them, and optionally upload to HF."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple
import zipfile


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Package local image files referenced by Sketchy train/test JSON into a ZIP."
    )
    parser.add_argument(
        "--train-json",
        type=Path,
        default=Path(
            "/mnt/data/zruan/workspace_novel/zruan/StructXLIP/datasets/train/Sketchy.json"
        ),
        help="Path to train Sketchy JSON.",
    )
    parser.add_argument(
        "--test-json",
        type=Path,
        default=Path(
            "/mnt/data/zruan/workspace_novel/zruan/StructXLIP/datasets/test/Sketchy.json"
        ),
        help="Path to test Sketchy JSON.",
    )
    parser.add_argument(
        "--base-main",
        type=Path,
        default=Path("/mnt/data/zruan/workspace_novel/zruan"),
        help="Main root for resolving relative paths (e.g. LOST/...).",
    )
    parser.add_argument(
        "--base-goal",
        type=Path,
        default=Path("/mnt/data/zruan/workspace_novel/zruan/GOAL"),
        help="GOAL root for resolving OtherDatasets/... paths.",
    )
    parser.add_argument(
        "--output-zip",
        type=Path,
        default=Path(
            "/mnt/data/zruan/workspace_novel/zruan/StructXLIP/datasets/sketchy_images_bundle.zip"
        ),
        help="Output ZIP path.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "/mnt/data/zruan/workspace_novel/zruan/StructXLIP/datasets/sketchy_images_bundle_manifest.json"
        ),
        help="Where to write manifest JSON.",
    )
    parser.add_argument(
        "--compression",
        choices=["stored", "deflated"],
        default="stored",
        help="ZIP compression type. 'stored' is fastest for image-heavy datasets.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only scan/resolve and write manifest. Do not create ZIP.",
    )
    parser.add_argument(
        "--hf-repo-id",
        type=str,
        default="",
        help="Optional HF repo id (e.g. username/sketchy-images). If set, upload ZIP.",
    )
    parser.add_argument(
        "--hf-repo-type",
        choices=["dataset", "model", "space"],
        default="dataset",
        help="HF repo type for upload.",
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default="",
        help="Optional path in HF repo. Defaults to output zip filename.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default="",
        help="Optional HF token. If omitted, uses local cached login/token env.",
    )
    parser.add_argument(
        "--hf-private",
        action="store_true",
        help="Create HF repo as private (if creation is needed).",
    )
    return parser.parse_args()


def iter_strings(value: object) -> Iterator[str]:
    if isinstance(value, str):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            yield from iter_strings(item)
        return
    if isinstance(value, dict):
        for item in value.values():
            yield from iter_strings(item)


def is_image_path(text: str) -> bool:
    return Path(text).suffix.lower() in IMAGE_SUFFIXES


def resolve_local_file(path_text: str, base_main: Path, base_goal: Path) -> Optional[Path]:
    p = Path(path_text)
    candidates: List[Path] = []
    if p.is_absolute():
        candidates.append(p)
    else:
        # Most train paths are anonymized to OtherDatasets/... and live under base_goal.
        if path_text.startswith("OtherDatasets/"):
            candidates.append(base_goal / path_text)
        # Most test paths live under base_main (e.g. LOST/...).
        candidates.append(base_main / path_text)
        # Fallback for mixed cases.
        candidates.append(base_goal / path_text)

    for c in candidates:
        if c.exists() and c.is_file():
            return c.resolve()
    return None


def choose_arcname(
    source_text: str,
    resolved_file: Path,
    base_main: Path,
    base_goal: Path,
) -> str:
    raw = source_text.lstrip("./").replace("\\", "/")
    if not Path(raw).is_absolute() and raw:
        return raw

    try:
        return resolved_file.relative_to(base_goal).as_posix()
    except ValueError:
        pass
    try:
        return resolved_file.relative_to(base_main).as_posix()
    except ValueError:
        pass
    return f"external/{resolved_file.name}"


def gather_entries(
    json_paths: Sequence[Path], base_main: Path, base_goal: Path
) -> Tuple[List[Tuple[Path, str]], List[str], Dict[str, int]]:
    source_paths: List[str] = []
    for jp in json_paths:
        with jp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        for s in iter_strings(data):
            if is_image_path(s):
                source_paths.append(s)

    unique_source_paths = list(dict.fromkeys(source_paths))
    entries: List[Tuple[Path, str]] = []
    missing: List[str] = []
    seen_files: Set[Path] = set()
    used_arcnames: Set[str] = set()

    for src in unique_source_paths:
        resolved = resolve_local_file(src, base_main=base_main, base_goal=base_goal)
        if resolved is None:
            missing.append(src)
            continue
        if resolved in seen_files:
            continue

        arcname = choose_arcname(
            source_text=src, resolved_file=resolved, base_main=base_main, base_goal=base_goal
        )
        if arcname in used_arcnames:
            digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:10]
            p = Path(arcname)
            arcname = f"{p.stem}_{digest}{p.suffix}"

        seen_files.add(resolved)
        used_arcnames.add(arcname)
        entries.append((resolved, arcname))

    stats = {
        "image_path_strings_total": len(source_paths),
        "image_path_strings_unique": len(unique_source_paths),
        "resolved_files_unique": len(entries),
        "missing_unique_paths": len(missing),
    }
    return entries, missing, stats


def write_zip(entries: Sequence[Tuple[Path, str]], output_zip: Path, compression: str) -> int:
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    if compression == "stored":
        method = zipfile.ZIP_STORED
        kwargs = {}
    else:
        method = zipfile.ZIP_DEFLATED
        kwargs = {"compresslevel": 1}

    with zipfile.ZipFile(output_zip, mode="w", compression=method, allowZip64=True, **kwargs) as zf:
        for src, arcname in entries:
            zf.write(src, arcname=arcname)
    return output_zip.stat().st_size


def write_manifest(
    manifest_path: Path,
    json_paths: Sequence[Path],
    output_zip: Path,
    stats: Dict[str, int],
    total_size_bytes: int,
    missing: Sequence[str],
    uploaded: bool,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "json_files": [str(p) for p in json_paths],
        "output_zip": str(output_zip),
        "stats": stats,
        "resolved_total_size_bytes": total_size_bytes,
        "resolved_total_size_gb": round(total_size_bytes / 1024 / 1024 / 1024, 3),
        "missing_samples": list(missing[:200]),
        "uploaded_to_hf": uploaded,
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def upload_to_hf(
    output_zip: Path,
    repo_id: str,
    repo_type: str,
    path_in_repo: str,
    token: str,
    private: bool,
) -> None:
    from huggingface_hub import HfApi, create_repo

    create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
        token=token or None,
    )
    api = HfApi(token=token or None)
    api.upload_file(
        path_or_fileobj=str(output_zip),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        repo_type=repo_type,
        token=token or None,
    )


def main() -> None:
    args = parse_args()

    json_paths = [args.train_json, args.test_json]
    for p in json_paths:
        if not p.exists():
            raise FileNotFoundError(f"JSON file not found: {p}")

    entries, missing, stats = gather_entries(
        json_paths=json_paths, base_main=args.base_main, base_goal=args.base_goal
    )
    total_size_bytes = sum(src.stat().st_size for src, _ in entries)

    uploaded = False
    zip_size_bytes = 0
    if args.dry_run:
        print("Dry run mode: skip zip creation/upload.")
    else:
        zip_size_bytes = write_zip(entries, output_zip=args.output_zip, compression=args.compression)
        print(f"Wrote ZIP: {args.output_zip}")
        print(f"ZIP size bytes: {zip_size_bytes}")

        if args.hf_repo_id:
            path_in_repo = args.hf_path or args.output_zip.name
            upload_to_hf(
                output_zip=args.output_zip,
                repo_id=args.hf_repo_id,
                repo_type=args.hf_repo_type,
                path_in_repo=path_in_repo,
                token=args.hf_token,
                private=args.hf_private,
            )
            uploaded = True
            print(f"Uploaded to HF: {args.hf_repo_type} {args.hf_repo_id}/{path_in_repo}")

    write_manifest(
        manifest_path=args.manifest,
        json_paths=json_paths,
        output_zip=args.output_zip,
        stats=stats,
        total_size_bytes=total_size_bytes,
        missing=missing,
        uploaded=uploaded,
    )

    print("Summary:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print(f"  resolved_total_size_bytes: {total_size_bytes}")
    print(f"  resolved_total_size_gb: {round(total_size_bytes / 1024 / 1024 / 1024, 3)}")
    print(f"  missing_examples_written_to_manifest: {min(len(missing), 200)}")
    if not args.dry_run:
        print(f"  zip_size_bytes: {zip_size_bytes}")


if __name__ == "__main__":
    main()
