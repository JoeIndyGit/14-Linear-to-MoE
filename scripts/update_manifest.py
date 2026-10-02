"""Refresh checksums after a reviewed change to submission files."""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXCLUDED = {".git", "__pycache__", ".venv", "venv", ".ipynb_checkpoints", "runs", "dist"}
TEXT_SUFFIXES = {".py", ".md", ".csv", ".json", ".ipynb", ".txt", ".yml", ".yaml", ".cff", ".svg"}


def update_manifest():
    files = sorted(p for p in ROOT.rglob("*") if p.is_file()
        and not EXCLUDED.intersection(p.relative_to(ROOT).parts)
        and p.name != "MANIFEST.sha256" and p.suffix not in {".zip", ".pyc"})
    # Match .gitattributes text eol=lf before hashing. csv.writer otherwise
    # emits CRLF even on Linux; Git checkout normalizes those bytes to LF.
    normalized = 0
    for path in files:
        if path.suffix in TEXT_SUFFIXES or path.name in {".gitignore", ".gitattributes", "LICENSE"}:
            data = path.read_bytes()
            canonical = data.replace(b"\r\n", b"\n")
            if canonical != data:
                path.write_bytes(canonical)
                normalized += 1
    text = "\n".join(hashlib.sha256(p.read_bytes()).hexdigest() + "  " +
                     p.relative_to(ROOT).as_posix() for p in files) + "\n"
    (ROOT / "MANIFEST.sha256").write_text(text)
    print(f"Updated manifest for {len(files)} files; normalized {normalized} text files to LF.")


if __name__ == "__main__":
    update_manifest()
