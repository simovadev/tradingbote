"""Transcrit les videos Vizion FR via youtube-transcript-api.

Usage :
    python scripts/transcribe_vizion.py <url1> <url2> ...
OU mettre les URLs dans scripts/vizion_urls.txt (une par ligne) puis :
    python scripts/transcribe_vizion.py

Output : scripts/transcripts/<video_id>.md
"""
from __future__ import annotations

import argparse
import re
import sys
from datetime import timedelta
from pathlib import Path

# Ajoute la racine du projet au path pour les imports si besoin
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))


URLS_FILE = ROOT / "scripts" / "vizion_urls.txt"
OUT_DIR = ROOT / "scripts" / "transcripts"
OUT_DIR.mkdir(exist_ok=True)


def extract_video_id(url: str) -> str | None:
    """Extrait l'ID YouTube d'une URL (gere youtube.com/watch?v=, youtu.be/, etc.)"""
    patterns = [
        r"(?:v=|/v/|/embed/|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",   # ID brut
    ]
    for pat in patterns:
        m = re.search(pat, url)
        if m:
            return m.group(1)
    return None


def format_time(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds)))


def transcribe_video(url: str) -> Path | None:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound

    vid = extract_video_id(url)
    if not vid:
        print(f"  [ERR] URL invalide : {url}")
        return None

    out_path = OUT_DIR / f"{vid}.md"
    if out_path.exists():
        print(f"  [SKIP] {vid} deja transcrit ({out_path.name})")
        return out_path

    print(f"  [...] Transcription de {vid}")

    try:
        api = YouTubeTranscriptApi()
        # Cherche en francais d'abord, puis n'importe quelle langue
        try:
            transcript = api.fetch(vid, languages=["fr", "fr-FR"])
        except NoTranscriptFound:
            # Fallback : prend la 1ere disponible
            transcript_list = api.list(vid)
            available = list(transcript_list)
            if not available:
                print(f"  [ERR] Aucune transcription disponible pour {vid}")
                return None
            print(f"  [!] Pas de FR, fallback sur {available[0].language_code}")
            transcript = available[0].fetch()
    except TranscriptsDisabled:
        print(f"  [ERR] Transcripts desactives sur {vid}")
        return None
    except Exception as e:
        print(f"  [ERR] {vid} : {e}")
        return None

    # Construit le markdown
    lines = [
        f"# Transcript YouTube : {vid}",
        f"",
        f"URL : https://youtube.com/watch?v={vid}",
        f"",
        "---",
        "",
    ]

    # FetchedTranscript se comporte comme un iterable de FetchedTranscriptSnippet
    for snippet in transcript:
        # snippet.start, snippet.duration, snippet.text
        ts = format_time(snippet.start)
        text = snippet.text.replace("\n", " ").strip()
        lines.append(f"**[{ts}]** {text}")
        lines.append("")

    # Texte continu sans timecodes (utile pour lecture rapide)
    lines.append("\n---\n\n## Texte complet (sans timecodes)\n")
    full_text = " ".join(s.text.replace("\n", " ").strip() for s in transcript)
    lines.append(full_text)

    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  [OK] {vid} -> {out_path.name} ({len(list(transcript))} segments)")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Transcrit des videos YouTube en .md")
    ap.add_argument("urls", nargs="*", help="URLs YouTube a transcrire")
    args = ap.parse_args()

    urls = args.urls
    # Si pas d'URLs en argument, on lit le fichier
    if not urls and URLS_FILE.exists():
        urls = [
            line.strip()
            for line in URLS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")
        ]
        print(f"Lecture de {len(urls)} URLs depuis {URLS_FILE.name}\n")

    if not urls:
        print(f"Aucune URL fournie.")
        print(f"Usage 1 : python scripts/transcribe_vizion.py <url1> <url2> ...")
        print(f"Usage 2 : creer {URLS_FILE} avec une URL par ligne puis relancer")
        return

    import time
    print(f"=== Transcription de {len(urls)} videos ===\n")
    success = 0
    for i, url in enumerate(urls):
        if transcribe_video(url):
            success += 1
        # Delai entre videos pour eviter le rate-limit YouTube
        # (sauf si SKIP, on garde le delai au cas ou)
        if i < len(urls) - 1:
            time.sleep(2)

    print(f"\n=== {success}/{len(urls)} transcriptions OK ===")
    print(f"Fichiers dans : {OUT_DIR}")


if __name__ == "__main__":
    main()
