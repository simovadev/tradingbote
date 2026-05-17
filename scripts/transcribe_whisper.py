"""Transcrit les videos sans sous-titres via Whisper local.

Pipeline:
1. Telecharge l'audio uniquement via yt-dlp (mp3, faible bitrate pour vitesse)
2. Whisper transcrit en francais
3. Sauvegarde en .md
4. Supprime l'audio temporaire
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
AUDIO_DIR = ROOT / "scripts" / "transcripts" / "_audio"
OUT_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

YT_DLP = ROOT / "venv" / "Scripts" / "yt-dlp.exe"

# Modele Whisper : small = bon equilibre vitesse/qualite
# (tiny = ultra-rapide mais qualite limitee, medium/large = lent)
WHISPER_MODEL = "small"


def extract_video_id(url: str) -> str | None:
    m = re.search(r"(?:v=|/v/|/embed/|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def download_audio(url: str, vid: str) -> Path | None:
    """Telecharge l'audio en mp3 via yt-dlp."""
    out_path = AUDIO_DIR / f"{vid}.mp3"
    if out_path.exists():
        return out_path
    cmd = [
        str(YT_DLP),
        "-x", "--audio-format", "mp3",
        "--audio-quality", "5",   # qualite moyenne (5/10) = plus rapide
        "-o", str(AUDIO_DIR / "%(id)s.%(ext)s"),
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if out_path.exists():
            return out_path
        print(f"  [DL ERR] yt-dlp output: {result.stderr[-200:]}")
        return None
    except subprocess.TimeoutExpired:
        print(f"  [DL TIMEOUT] {vid}")
        return None


def transcribe_audio(audio_path: Path, model) -> str | None:
    """Transcrit un mp3 via Whisper."""
    try:
        result = model.transcribe(
            str(audio_path),
            language="fr",
            verbose=False,
            fp16=False,    # CPU
        )
        return result["text"].strip()
    except Exception as e:
        print(f"  [WHISPER ERR] {e}")
        return None


def main() -> None:
    # Charge la liste des URLs manquantes
    if not URLS_FILE.exists():
        print("Pas de fichier URLs")
        return

    urls = [line.strip() for line in URLS_FILE.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")]

    # Filtre celles deja transcrites
    todo = []
    for url in urls:
        vid = extract_video_id(url)
        if vid and not (OUT_DIR / f"{vid}.md").exists():
            todo.append((url, vid))

    if not todo:
        print("Toutes les videos sont deja transcrites !")
        return

    print(f"=== {len(todo)} videos a transcrire via Whisper ===\n")
    print(f"Chargement du modele Whisper '{WHISPER_MODEL}'... (peut prendre 1 min la 1ere fois)\n")

    import whisper
    model = whisper.load_model(WHISPER_MODEL)
    print("Modele charge\n")

    success = 0
    for i, (url, vid) in enumerate(todo):
        t0 = time.time()
        print(f"[{i+1}/{len(todo)}] {vid}")

        # 1. Download audio
        audio_path = download_audio(url, vid)
        if audio_path is None:
            print(f"  [SKIP] download echec")
            continue
        print(f"  [DL OK] {audio_path.stat().st_size//1024} KB")

        # 2. Whisper transcribe
        text = transcribe_audio(audio_path, model)
        if not text:
            continue

        # 3. Sauvegarde .md
        md_content = f"""# Transcript Whisper : {vid}

URL : https://youtube.com/watch?v={vid}
Source: Whisper {WHISPER_MODEL} (transcription IA)

---

{text}
"""
        out_path = OUT_DIR / f"{vid}.md"
        out_path.write_text(md_content, encoding="utf-8")

        elapsed = time.time() - t0
        chars = len(text)
        print(f"  [OK] {chars} chars en {elapsed:.0f}s -> {out_path.name}")
        success += 1

        # 4. Cleanup audio
        try:
            audio_path.unlink()
        except Exception:
            pass

    print(f"\n=== {success}/{len(todo)} transcriptions Whisper OK ===")


if __name__ == "__main__":
    main()
