#!/usr/bin/env python3
"""Download and checksum the pinned public Porto dataset. Standard library only."""
import argparse
import hashlib
import json
import os
import tarfile
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, default=ROOT / "data")
    parser.add_argument("--archive", type=Path, help="use an already downloaded release archive")
    args = parser.parse_args()
    spec = json.loads((ROOT / "configs/porto_medium_v02_release.json").read_text())
    args.destination.mkdir(parents=True, exist_ok=True)
    target = args.destination / spec["dataset_directory"]
    if target.exists():
        raise FileExistsError(f"{target} exists; run student_benchmark.py verify on it")
    with tempfile.TemporaryDirectory(prefix="porto-download-", dir=args.destination) as tmp:
        tmp = Path(tmp)
        archive = args.archive
        if archive is None:
            archive = tmp / spec["asset_name"]
            request = urllib.request.Request(spec["url"], headers={"User-Agent": "PWC-student-kit"})
            with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as out:
                while block := response.read(1024 * 1024):
                    out.write(block)
        if digest(archive) != spec["sha256"]:
            raise ValueError("release archive SHA-256 mismatch; archive was not extracted")
        with tarfile.open(archive, "r:gz") as tar:
            members = tar.getmembers()
            for m in members:
                p = Path(m.name)
                if p.is_absolute() or ".." in p.parts or not p.parts or p.parts[0] != spec["dataset_directory"]:
                    raise ValueError(f"unsafe archive path: {m.name}")
                if not (m.isfile() or m.isdir()):
                    raise ValueError(f"unexpected archive member type: {m.name}")
            tar.extractall(tmp, members=members, filter="data")
        extracted = tmp / spec["dataset_directory"]
        if digest(extracted / "MANIFEST.json") != spec["manifest_sha256"]:
            raise ValueError("manifest checksum mismatch")
        os.replace(extracted, target)
    print(f"Downloaded and archive-verified: {target}")
    print("Next: python scripts/public_benchmark/student_benchmark.py verify --data " + str(target))


if __name__ == "__main__":
    main()
