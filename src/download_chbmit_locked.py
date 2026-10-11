from __future__ import annotations

import argparse
import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chbmit_locked_protocol.json")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--summaries-only", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config)
    cfg = json.loads(config_path.read_text())
    config_hash = hashlib.sha256(
        json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    root = Path(cfg["data_dir"])
    root.mkdir(parents=True, exist_ok=True)
    snapshot = root / "locked_protocol_snapshot.json"
    if snapshot.exists() and json.loads(snapshot.read_text()) != cfg:
        raise RuntimeError(
            "the locked protocol differs from its data snapshot; create a new "
            "protocol version and data directory rather than overwriting it"
        )
    snapshot.write_text(json.dumps(cfg, indent=2) + "\n")
    base = cfg["base_url"].rstrip("/")
    download(base + "/RECORDS", root / "RECORDS")
    download(base + "/RECORDS-WITH-SEIZURES", root / "RECORDS-WITH-SEIZURES")
    all_records = (root / "RECORDS").read_text().strip().splitlines()
    seizure_records = set((root / "RECORDS-WITH-SEIZURES").read_text().strip().splitlines())
    rng = np.random.default_rng(int(cfg["seed"]))
    manifest = []
    for subject in cfg["subjects"]:
        subject_dir = root / subject
        subject_dir.mkdir(exist_ok=True)
        summary_rel = f"{subject}/{subject}-summary.txt"
        summary_path = subject_dir / f"{subject}-summary.txt"
        if not args.manifest_only:
            download(f"{base}/{summary_rel}", summary_path)
        subject_all = [r for r in all_records if r.startswith(subject + "/")]
        subject_seizure = [r for r in subject_all if r in seizure_records]
        subject_non = [r for r in subject_all if r not in seizure_records]
        n_non = min(int(cfg["nonseizure_records_per_subject"]), len(subject_non))
        if n_non:
            positions = np.linspace(0, len(subject_non) - 1, n_non, dtype=int)
            selected_non = [subject_non[i] for i in positions]
        else:
            selected_non = []
        context_map: dict[str, list[str]] = {}
        if cfg.get("include_immediate_predecessor_for_seizure_context", False):
            position = {record: index for index, record in enumerate(subject_all)}
            for target in subject_seizure:
                index = position[target]
                if index > 0:
                    predecessor = subject_all[index - 1]
                    context_map.setdefault(predecessor, []).append(target)
        kind_by_record = {record: "context" for record in context_map}
        kind_by_record.update({record: "nonseizure" for record in selected_non})
        kind_by_record.update({record: "seizure" for record in subject_seizure})
        selected = [(record, kind_by_record[record]) for record in subject_all if record in kind_by_record]
        for rel, kind in selected:
            filename = rel.split("/", 1)[1]
            destination = subject_dir / filename
            manifest.append(
                {
                    "subject": subject,
                    "record": Path(filename).stem,
                    "kind": kind,
                    "relative_path": rel,
                    "local_path": str(destination),
                    "url": f"{base}/{rel}",
                    "seed": int(cfg["seed"]),
                    "protocol_version": cfg["protocol_version"],
                    "config_sha256": config_hash,
                    "context_for": context_map.get(rel, []),
                }
            )
            if not args.manifest_only and not args.summaries_only:
                download(f"{base}/{rel}", destination)
    (root / "locked_manifest.json").write_text(json.dumps(manifest, indent=2))
    counts = {}
    for row in manifest:
        counts[row["kind"]] = counts.get(row["kind"], 0) + 1
    print(f"manifest records: {counts}; path={root / 'locked_manifest.json'}")


def download(url: str, destination: Path, maximum_attempts: int = 12) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"exists {destination}")
        return
    print(f"download {url}")
    partial = destination.with_suffix(destination.suffix + ".part")
    retryable = (TimeoutError, socket.timeout, urllib.error.URLError, OSError)
    for attempt in range(1, maximum_attempts + 1):
        mode = "ab" if partial.exists() else "wb"
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url)
        if offset:
            request.add_header("Range", f"bytes={offset}-")
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                if offset and response.status != 206:
                    mode = "wb"
                with partial.open(mode) as stream:
                    while True:
                        chunk = response.read(4 * 1024 * 1024)
                        if not chunk:
                            break
                        stream.write(chunk)
            partial.replace(destination)
            return
        except retryable as exc:
            if attempt == maximum_attempts:
                raise
            delay = min(2 * attempt, 15)
            saved = partial.stat().st_size if partial.exists() else 0
            print(
                f"retry {attempt}/{maximum_attempts} after {type(exc).__name__}; "
                f"saved_bytes={saved}; waiting_seconds={delay}",
                flush=True,
            )
            time.sleep(delay)


if __name__ == "__main__":
    main()
