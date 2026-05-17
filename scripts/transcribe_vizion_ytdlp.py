"""Transcrit via yt-dlp (telechargement sous-titres VTT) puis convertit en .md.

Plus fiable que youtube-transcript-api (evite les rate-limits YouTube).
"""
from __future__ import annotations

import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
URLS_FILE = ROOT / "scripts" / "vizion_urls.txt"
OUT_DIR = ROOT / "scripts" / "transcripts"
TMP_DIR = ROOT / "scripts" / "transcripts" / "_tmp"
OUT_DIR.mkdir(exist_ok=True)
TMP_DIR.mkdir(exist_ok=True)

YT_DLP = ROOT / "venv" / "Scripts" / "yt-dlp.exe"


def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|/v/|/embed/|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def parse_vtt(vtt_path: Path) -> str:
    """Parse un fichier VTT et retourne le texte propre (sans timecodes ni balises)."""
    content = vtt_path.read_text(encoding="utf-8", errors="ignore")
    lines = content.splitlines()
    out_lines = []
    seen = set()
    for line in lines:
        # Skip header, timestamps, lignes vides
        if not line.strip():
            continue
        if line.startswith("WEBVTT") or line.startswith("Kind:") or line.startswith("Language:"):
            continue
        if "-->" in line:
            continue
        if line.strip().isdigit():
            continue
        # Retire les balises <c>, <00:00:00.000>, etc.
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if not clean:
            continue
        # Skip doublons consecutifs (sous-titres auto se repetent)
        if clean in seen:
            continue
        seen.add(clean)
        out_lines.append(clean)
    return "\n".join(out_lines)


def transcribe_with_ytdlp(url: str) -> Path | None:
    vid = extract_video_id(url)
    if not vid:
        print(f"  [ERR] URL invalide : {url}")
        return None

    out_path = OUT_DIR / f"{vid}.md"
    if out_path.exists():
        print(f"  [SKIP] {vid} deja transcrit")
        return out_path

    print(f"  [...] {vid}")

    # Telecharge les sous-titres FR (manuels d'abord, puis auto si pas dispo)
    cmd = [
        str(YT_DLP),
        "--write-sub", "--write-auto-sub",
        "--sub-langs", "fr,fr-FR,fr.*",
        "--skip-download",
        "--sub-format", "vtt",
        "-o", str(TMP_DIR / "%(id)s.%(ext)s"),
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        print(f"  [ERR] timeout pour {vid}")
        return None

    # Cherche le fichier .vtt cree
    vtt_files = list(TMP_DIR.glob(f"{vid}*.vtt"))
    if not vtt_files:
        # Pas de sous-titres FR ? On retry avec n'importe quelle langue
        cmd2 = [str(YT_DLP), "--write-auto-sub", "--sub-langs", "all", "--skip-download",
                "--sub-format", "vtt", "-o", str(TMP_DIR / "%(id)s.%(ext)s"), url]
        try:
            subprocess.run(cmd2, capture_output=True, text=True, timeout=120)
        except subprocess.TimeoutExpired:
            pass
        vtt_files = list(TMP_DIR.glob(f"{vid}*.vtt"))

    if not vtt_files:
        print(f"  [ERR] Aucun sous-titre pour {vid}")
        return None

    # Prend le FR en priorite si plusieurs
    vtt_files.sort(key=lambda p: 0 if "fr" in p.name.lower() else 1)
    vtt_path = vtt_files[0]

    text = parse_vtt(vtt_path)
    if not text:
        print(f"  [ERR] VTT vide pour {vid}")
        return None

    md = [
        f"# Transcript : {vid}",
        f"",
        f"URL : https://youtube.com/watch?v={vid}",
        f"Source: {vtt_path.name}",
        "",
        "---",
        "",
        text,
    ]
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"  [OK] {vid} -> {out_path.name} ({len(text)} chars)")

    # Cleanup
    for f in vtt_files:
        try:
            f.unlink()
        except Exception:
            pass

    return out_path


def main() -> None:
    if not URLS_FILE.exists():
        print(f"Pas de {URLS_FILE}")
        return

    urls = [line.strip() for line in URLS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]

    print(f"=== Transcription yt-dlp de {len(urls)} videos ===\n")
    success = 0
    for i, url in enumerate(urls):
        if transcribe_with_ytdlp(url):
            success += 1
        if i < len(urls) - 1:
            time.sleep(1)

    print(f"\n=== {success}/{len(urls)} OK ===")


if __name__ == "__main__":
    main()
