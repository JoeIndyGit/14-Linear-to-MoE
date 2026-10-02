"""Refresh checksums after a reviewed change to submission files."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = {".git", "__pycache__", ".venv", "venv", ".ipynb_checkpoints", "runs", "dist"}


def update_manifest():
    files = sorted(p for p in ROOT.rglob("*") if p.is_file()
        and not EXCLUDED.intersection(p.relative_to(ROOT).parts)
        and p.name != "MANIFEST.sha256" and p.suffix not in {".zip", ".pyc"})
    text = "\n".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " +
                     p.relative_to(ROOT).as_posix() for p in files) + "\n"
    (ROOT / "MANIFEST.sha256").write_text(text)
    print(f"Updated manifest for {len(files)} files.")


if __name__ == "__main__":
    update_manifest()
