from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "config" / "index_manifest_v1.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_directory(path: Path) -> str:
    entries: list[str] = []
    for item in sorted(path.rglob("*")):
        if not item.is_file() or item.name == "LOCK":
            continue
        entries.append(f"{sha256_file(item)}{item.relative_to(path).as_posix()}")
    payload = "".join(f"{entry}\n" for entry in sorted(entries)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def require_equal(label: str, actual: str, expected: str) -> bool:
    if actual == expected:
        print(f"[PASS] {label}: {actual}")
        return True
    print(f"[FAIL] {label}: expected={expected}, actual={actual}")
    return False


def main() -> int:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    paragraph_cache = ROOT / manifest["corpus"]["paragraph_cache"]
    build_script = ROOT / manifest["build"]["script"]
    index_dir = ROOT / manifest["index"]["uri"]
    failures = 0

    for label, path in (
        ("paragraph_cache", paragraph_cache),
        ("build_script", build_script),
        ("index_dir", index_dir),
    ):
        if not path.exists():
            print(f"[FAIL] {label}: missing at {path}")
            failures += 1

    if failures:
        return 1

    if not require_equal(
        "paragraph_cache_sha256",
        sha256_file(paragraph_cache),
        manifest["corpus"]["paragraph_cache_sha256"],
    ):
        failures += 1
    if not require_equal(
        "build_script_sha256",
        sha256_file(build_script),
        manifest["build"]["script_sha256"],
    ):
        failures += 1
    if not require_equal(
        "index_directory_sha256",
        sha256_directory(index_dir),
        manifest["index"]["directory_sha256"],
    ):
        failures += 1

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
