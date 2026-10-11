from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chbmit_subset.json")
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    data_dir = Path(cfg["data_dir"])
    data_dir.mkdir(parents=True, exist_ok=True)
    base_url = cfg["base_url"].rstrip("/")

    download(base_url + "/RECORDS-WITH-SEIZURES", data_dir / "RECORDS-WITH-SEIZURES")
    download(base_url + "/RECORDS", data_dir / "RECORDS")

    seizure_records = (data_dir / "RECORDS-WITH-SEIZURES").read_text().strip().splitlines()
    all_records = (data_dir / "RECORDS").read_text().strip().splitlines()

    for subject in cfg["subjects"]:
        subject_dir = data_dir / subject
        subject_dir.mkdir(parents=True, exist_ok=True)
        download(f"{base_url}/{subject}/{subject}-summary.txt", subject_dir / f"{subject}-summary.txt")

        subj_seizure = [r for r in seizure_records if r.startswith(subject + "/")]
        subj_all = [r for r in all_records if r.startswith(subject + "/")]
        subj_non = [r for r in subj_all if r not in set(subj_seizure)]
        selected = (
            subj_seizure[: int(cfg["max_seizure_records_per_subject"])]
            + subj_non[: int(cfg["max_nonseizure_records_per_subject"])]
        )
        for record in selected:
            edf_name = record.split("/", 1)[1]
            download(f"{base_url}/{subject}/{edf_name}", subject_dir / edf_name)
            seizure_url = f"{base_url}/{subject}/{edf_name}.seizures"
            try:
                download(seizure_url, subject_dir / f"{edf_name}.seizures")
            except Exception:
                pass


def download(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"exists {dest}")
        return
    print(f"download {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=60) as response:
        with tmp.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
    tmp.replace(dest)


if __name__ == "__main__":
    main()
