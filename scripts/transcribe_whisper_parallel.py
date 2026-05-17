"""Transcription Whisper en PARALLELE - ProcessPool (vrai parallelisme CPU).

Chaque process charge son propre modele Whisper et bosse independamment.
- Modele 'base' = 145 MB par process. 6 process = ~1 GB RAM (gerable).
- Sur CPU multi-core, 6 process ~ 4-5x plus rapide que sequentiel.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).parent.parent
URLS_FILE = ROOT / "scripts" / "vizion_urls.txt"
OUT_DIR = ROOT / "scripts" / "transcripts"
AUDIO_DIR = ROOT / "scripts" / "transcripts" / "_audio"
OUT_DIR.mkdir(exist_ok=True)
AUDIO_DIR.mkdir(exist_ok=True)

YT_DLP = ROOT / "venv" / "Scripts" / "yt-dlp.exe"

# Modele Whisper. 'base' = bon equilibre. 'tiny' = ultra rapide.
WHISPER_MODEL = "base"

# Nombre de processus paralleles
N_WORKERS = 6


def log(msg: str):
    print(msg, flush=True)


def extract_video_id(url: str) -> str | None:
    m = re.search(r"(?:v=|/v/|/embed/|/shorts/|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


# Modele global PAR PROCESS (chaque worker charge le sien)
_WORKER_MODEL = None


def get_worker_model():
    """Charge le modele Whisper UNE FOIS par worker process."""
    global _WORKER_MODEL
    if _WORKER_MODEL is None:
        import whisper
        _WORKER_MODEL = whisper.load_model(WHISPER_MODEL)
    return _WORKER_MODEL


def download_audio(url: str, vid: str) -> Path | None:
    """Telecharge l'audio mp3 via yt-dlp."""
    out_path = AUDIO_DIR / f"{vid}.mp3"
    if out_path.exists():
        return out_path
    cmd = [
        str(YT_DLP),
        "-x", "--audio-format", "mp3",
        "--audio-quality", "9",   # qualite basse = rapide (suffit pour Whisper)
        "--no-warnings",
        "-o", str(AUDIO_DIR / "%(id)s.%(ext)s"),
        url,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if out_path.exists():
            return out_path
        log(f"  [DL ERR] {vid}: {result.stderr[-150:] if result.stderr else 'no output'}")
        return None
    except subprocess.TimeoutExpired:
        log(f"  [DL TIMEOUT] {vid}")
        return None
    except Exception as e:
        log(f"  [DL EX] {vid}: {e}")
        return None


def transcribe_one(url: str, idx: int, total: int) -> bool:
    """Pipeline complet pour une video : download + whisper + save."""
    vid = extract_video_id(url)
    if not vid:
        return False

    out_path = OUT_DIR / f"{vid}.md"
    if out_path.exists():
        log(f"  [{idx}/{total}] [SKIP] {vid}")
        return True

    t0 = time.time()
    log(f"  [{idx}/{total}] [...] {vid}")

    # 1. Download audio
    audio_path = download_audio(url, vid)
    if audio_path is None:
        return False
    dl_time = time.time() - t0
    log(f"  [{idx}/{total}] [DL] {vid} {audio_path.stat().st_size // 1024}KB en {dl_time:.0f}s")

    # 2. Whisper transcribe (chaque process a son propre modele)
    model = get_worker_model()
    t_whisper = time.time()
    try:
        result = model.transcribe(
            str(audio_path),
            language="fr",
            verbose=False,
            fp16=False,
        )
        text = result["text"].strip()
    except Exception as e:
        log(f"  [{idx}/{total}] [WHISPER ERR] {vid}: {e}")
        try:
            audio_path.unlink()
        except Exception:
            pass
        return False

    whisper_time = time.time() - t_whisper

    if not text:
        return False

    # 3. Sauvegarde markdown
    md = f"""# Transcript Whisper : {vid}

URL : https://youtube.com/watch?v={vid}
Source: Whisper {WHISPER_MODEL}

---

{text}
"""
    out_path.write_text(md, encoding="utf-8")
    total_time = time.time() - t0
    log(f"  [{idx}/{total}] [OK] {vid} ({len(text)} chars, whisper={whisper_time:.0f}s, total={total_time:.0f}s)")

    # 4. Cleanup audio
    try:
        audio_path.unlink()
    except Exception:
        pass

    return True


def main():
    if not URLS_FILE.exists():
        log("Pas de fichier URLs")
        return

    urls = [l.strip() for l in URLS_FILE.read_text(encoding="utf-8").splitlines()
            if l.strip() and not l.startswith("#")]

    # Filtre les manquantes
    todo = []
    for url in urls:
        vid = extract_video_id(url)
        if vid and not (OUT_DIR / f"{vid}.md").exists():
            todo.append(url)

    if not todo:
        log("Toutes les videos sont deja transcrites !")
        return

    log(f"=== {len(todo)} videos a transcrire avec {N_WORKERS} process paralleles ===")
    log(f"Modele: {WHISPER_MODEL} (RAM par worker ~ 150 MB)\n")

    t_start = time.time()
    success = 0
    failures = []

    # ProcessPoolExecutor : chaque process est independant
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(transcribe_one, url, i + 1, len(todo)): url
                   for i, url in enumerate(todo)}
        for fut in as_completed(futures):
            try:
                if fut.result():
                    success += 1
                else:
                    failures.append(futures[fut])
            except Exception as e:
                log(f"  [ERR] {futures[fut]}: {e}")
                failures.append(futures[fut])

    elapsed = time.time() - t_start
    log(f"\n=== {success}/{len(todo)} transcriptions OK en {elapsed:.0f}s ===")
    if failures:
        log(f"Echecs:")
        for u in failures:
            log(f"  {u}")


if __name__ == "__main__":
    main()
