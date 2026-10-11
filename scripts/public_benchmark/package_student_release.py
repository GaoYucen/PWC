#!/usr/bin/env python3
"""Build deterministic release archives from an explicitly supplied public dataset."""
import argparse
import gzip
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def archive(output, entries):
    with output.open("wb") as raw, gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
            for path, name in sorted(entries, key=lambda x: x[1]):
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                info.mode = 0o644
                info.mtime = 0
                with path.open("rb") as f:
                    tar.addfile(info, f)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/porto-medium-hourly-v0.2")
    parser.add_argument("--output", type=Path, default=ROOT / "releases")
    parser.add_argument("--kit", action="store_true", help="include Git-indexed source and the public data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((args.data / "MANIFEST.json").read_text())
    entries = []
    for name, item in manifest["files"].items():
        path = args.data / name
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"dataset checksum mismatch: {name}")
        entries.append((path, f"{args.data.name}/{name}"))
    entries.append((args.data / "MANIFEST.json", f"{args.data.name}/MANIFEST.json"))
    data_asset = args.output / "porto-medium-hourly-v0.2.tar.gz"
    if not args.kit:
        archive(data_asset, entries)
        spec = {"dataset_directory": args.data.name, "asset_name": data_asset.name,
                "release_tag": "porto-medium-v0.2",
                "url": f"https://github.com/GaoYucen/PWC/releases/download/porto-medium-v0.2/{data_asset.name}",
                "bytes": data_asset.stat().st_size, "sha256": sha256(data_asset),
                "manifest_sha256": sha256(args.data / "MANIFEST.json")}
        (ROOT / "configs/porto_medium_v02_release.json").write_text(json.dumps(spec, indent=2) + "\n")
        print(json.dumps(spec, indent=2))
    else:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        tracked = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0")
        source = []
        for name in filter(None, tracked):
            path = ROOT / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"unexpected source entry: {name}")
            source.append((path, f"PWC-student-kit/{name}"))
        source.extend((p, f"PWC-student-kit/data/{n}") for p, n in entries)
        provenance = args.output / "KIT_PROVENANCE.json"
        provenance.write_text(json.dumps({"repository": "https://github.com/GaoYucen/PWC",
                                          "code_commit": head,
                                          "dataset_manifest_sha256": sha256(args.data / "MANIFEST.json")},
                                         indent=2) + "\n")
        source.append((provenance, "PWC-student-kit/KIT_PROVENANCE.json"))
        output = args.output / "porto-medium-v0.2-student-kit.tar.gz"
        archive(output, source)
        print(json.dumps({"asset": output.name, "bytes": output.stat().st_size,
                          "sha256": sha256(output)}, indent=2))


if __name__ == "__main__":
    main()
