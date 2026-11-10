"""Resume the official Food-101 archive, verify checksum, extract chosen classes."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from foodcal.data import ARCHIVE_BYTES, ARCHIVE_MD5, ARCHIVE_URL, extract_subset, file_hash, load_config, make_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--output", default="data/processed/food101_10")
    args = parser.parse_args()
    os.chdir(ROOT)
    raw = Path("data/raw")
    raw.mkdir(exist_ok=True, parents=True)
    archive = raw / "food-101.tar.gz"
    partial = raw / "food-101.tar.gz.part"
    status = Path("runs/setup/food101_download.json")
    status.parent.mkdir(parents=True, exist_ok=True)
    with (raw / ".food101-download.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("Food-101 setup is already running")
        if not archive.exists():
            last = 0
            for attempt in range(4):
                offset = partial.stat().st_size if partial.exists() else 0
                if offset == ARCHIVE_BYTES:
                    break
                if offset > ARCHIVE_BYTES:
                    raise ValueError("Partial file exceeds official archive size")
                headers = {"User-Agent": "food-calorie-course-project"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                try:
                    with urllib.request.urlopen(urllib.request.Request(ARCHIVE_URL, headers=headers), timeout=30) as response:
                        if offset and (response.status != 206 or not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-")):
                            raise ValueError("Server did not honor resume range; partial data preserved")
                        with partial.open("ab" if offset else "wb") as handle:
                            while True:
                                chunk = response.read(1024*1024)
                                if not chunk:
                                    break
                                handle.write(chunk)
                                offset += len(chunk)
                                if time.monotonic()-last >= 10:
                                    value = {"state": "downloading", "bytes": offset, "total_bytes": ARCHIVE_BYTES,
                                             "percent": round(offset/ARCHIVE_BYTES*100, 2), "url": ARCHIVE_URL}
                                    status.write_text(json.dumps(value, indent=2) + "\n")
                                    print(json.dumps(value), flush=True)
                                    last = time.monotonic()
                    if partial.stat().st_size != ARCHIVE_BYTES:
                        raise OSError("Download ended before the full archive arrived")
                    break
                except OSError as exc:
                    print(f"Attempt {attempt+1}: {type(exc).__name__}: {exc}", flush=True)
                    if attempt == 3:
                        status.write_text(json.dumps({"state": "interrupted", "bytes": partial.stat().st_size if partial.exists() else 0,
                                                      "total_bytes": ARCHIVE_BYTES, "error": str(exc)}, indent=2)+"\n")
                        raise
                    time.sleep(1)
            print("Verifying official archive checksum...", flush=True)
            if file_hash(partial, "md5") != ARCHIVE_MD5:
                raise ValueError("Official archive MD5 mismatch; partial file kept for inspection")
            partial.rename(archive)
        elif archive.stat().st_size != ARCHIVE_BYTES or file_hash(archive, "md5") != ARCHIVE_MD5:
            raise ValueError("Existing archive is not the verified official file")
        status.write_text(json.dumps({"state": "verified", "bytes": ARCHIVE_BYTES, "url": ARCHIVE_URL,
                                     "md5": ARCHIVE_MD5}, indent=2)+"\n")
        if args.prepare:
            config = load_config()
            print("Extracting the selected ten classes...", flush=True)
            count = extract_subset(archive, raw / "food-101", config["classes"])
            report = make_manifest(raw / "food-101", args.output, config)
            status.write_text(json.dumps({"state": "prepared", "bytes": ARCHIVE_BYTES,
                                         "selected_images": count, "manifest": str(Path(args.output)/"manifest.csv"),
                                         "counts": report["counts"], "md5": ARCHIVE_MD5}, indent=2)+"\n")
            print(json.dumps(report["counts"]), flush=True)


if __name__ == "__main__":
    main()
