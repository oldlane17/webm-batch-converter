#!/usr/bin/env python3
"""
convert_to_webm.py


Convert all video files in an input folder to WebM (VP9 + Opus) using ffmpeg.


Usage:
python convert_to_webm.py /path/to/input_folder /path/to/output_folder


Options:
-r, --recursive : Recurse into subfolders and preserve folder structure.
-w, --workers N : Number of parallel workers (default: number of CPUs).
--crf N : Quality for libvpx-vp9 (lower = better quality, default 32).
--overwrite : Overwrite existing output files.
--pretend : Print actions but don't run ffmpeg.


Requirements:
- ffmpeg must be installed and on PATH.


This script tries to be safe with filenames, logs progress, and parallelizes conversions.
"""


from __future__ import annotations
import argparse
import concurrent.futures
import multiprocessing
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List
import tempfile
import math


# Recognized video file extensions (lowercase)
_VIDEO_EXTS = {
'.mp4', '.mov', '.mkv', '.avi', '.flv', '.wmv', '.m4v', '.mpeg', '.mpg', '.webm', '.3gp', '.ts', '.mts'
}




def find_video_files(folder: Path, recursive: bool) -> List[Path]:
    files = []
    if recursive:
        for p in folder.rglob('*'):
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTS:
                files.append(p)
    else:
        for p in folder.iterdir():
            if p.is_file() and p.suffix.lower() in _VIDEO_EXTS:
                files.append(p)
    return sorted(files)




def ffmpeg_installed() -> bool:
    return shutil.which('ffmpeg') is not None


def make_output_path(input_path: Path, input_root: Path, output_root: Path) -> Path:
    """Preserve relative path from input_root into output_root and change extension to .webm"""
    rel = input_path.relative_to(input_root)
    out_rel = rel.with_suffix('.webm')
    return output_root.joinpath(out_rel)


def _get_source_bitrate_kbps(input_path: Path) -> int | None:
    """Return the video stream bitrate in kb/s if available, else try to estimate from filesize/duration."""
    try:
        proc = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=bit_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
             str(input_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        out = proc.stdout.decode().strip()
        if out and out != 'N/A':
            try:
                bit_rate = int(float(out))
                if bit_rate > 0:
                    return int(round(bit_rate / 1000.0))
            except Exception:
                pass

        # fallback: try container duration then compute kbps from filesize
        proc2 = subprocess.run(
            ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
             '-of', 'default=nokey=1:noprint_wrappers=1', str(input_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        dur_out = proc2.stdout.decode().strip()
        duration = None
        try:
            duration = float(dur_out)
        except Exception:
            duration = None

        if duration and duration > 0:
            size_bytes = input_path.stat().st_size
            kbps = int(round((size_bytes * 8) / 1000.0 / duration))
            return kbps
    except Exception:
        pass
    return None

def convert_one(input_path: Path, output_path: Path, crf: int, overwrite: bool, pretend: bool) -> tuple[Path, bool, str]:
    """
    Two-pass VP9 conversion targeting a bitrate slightly lower than the source average.
    Returns (input_path, success, message)
    NOTE: 'crf' argument is ignored for two-pass mode (we target bitrate instead).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return input_path, False, f"skipped (exists): {output_path}"

    # Null device depends on platform
    null_dev = 'NUL' if os.name == 'nt' else '/dev/null'

    # Measure source bitrate (kbps)
    source_kbps = _get_source_bitrate_kbps(input_path)
    if source_kbps is None or source_kbps <= 0:
        # fallback conservative default (kbps)
        source_kbps = 1200

    # Choose a target bitrate: 75% of source by default
    multiplier = 0.75
    target_kbps = max(200, int(math.floor(source_kbps * multiplier)))

    # Create a unique temporary directory for passlog to avoid collisions in parallel runs
    tmpdir = Path(tempfile.mkdtemp(prefix='vp9-pass-'))
    passlog_base = tmpdir / 'ffpass'  # ffmpeg will append -0.log etc.

    # Common flags for VP9 two-pass encoding
    vp9_common = [
        '-c:v', 'libvpx-vp9',
        '-b:v', f'{target_kbps}k',
        '-tile-columns', '4',
        '-frame-parallel', '1',
        '-speed', '1',
        '-passlogfile', str(passlog_base)
    ]

    # First pass: no audio, output to null device
    pass1_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(input_path),
    ] + vp9_common + [
        '-pass', '1',
        '-an',
        '-f', 'null',
        null_dev,
    ]

    # Second pass: encode audio and write final webm
    pass2_cmd = [
        'ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(input_path),
    ] + vp9_common + [
        '-pass', '2',
        '-c:a', 'libopus',
        '-b:a', '64k',
        '-vf', 'scale=iw:ih',
        '-f', 'webm',
        str(output_path),
    ]

    if pretend:
        # cleanup tmpdir and return the printed commands instead of running
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return input_path, True, 'pretend pass1: ' + ' '.join(pass1_cmd) + '\npretend pass2: ' + ' '.join(pass2_cmd)

    try:
        # Run first pass
        subprocess.run(pass1_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # Run second pass
        subprocess.run(pass2_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        # cleanup
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return input_path, True, f'converted (two-pass {target_kbps}k) -> {output_path}'
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors='ignore') if e.stderr else str(e)
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return input_path, False, f'ffmpeg error: {err[:500]}'
    except Exception as e:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return input_path, False, f'exception: {e}'


def chunked(iterable: Iterable, size: int) -> Iterable[List]:
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
        break
    yield chunk

def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Convert videos in a folder to WebM (VP9 + Opus)')
    parser.add_argument('input_folder', type=Path, help='Path to input folder with videos')
    parser.add_argument('output_folder', type=Path, help='Path where converted videos will be placed')
    parser.add_argument('-r', '--recursive', action='store_true', help='Recurse into subfolders and preserve structure')
    parser.add_argument('-w', '--workers', type=int, default=max(1, multiprocessing.cpu_count()),
    help='Number of parallel workers (default: number of CPUs)')
    parser.add_argument('--crf', type=int, default=32, help='CRF for libvpx-vp9 (lower => better quality). Default 32')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing files in output')
    parser.add_argument('--pretend', action='store_true', help="Don't run ffmpeg, just print commands")
    args = parser.parse_args(argv)


    input_folder: Path = args.input_folder.expanduser().resolve()
    output_folder: Path = args.output_folder.expanduser().resolve()

    if not input_folder.exists() or not input_folder.is_dir():
        print(f"Input folder does not exist or is not a directory: {input_folder}")
        return 2


    output_folder.mkdir(parents=True, exist_ok=True)

    if not ffmpeg_installed() and not args.pretend:
        print('ffmpeg not found on PATH. Please install ffmpeg before running this script.')
        return 3
    
    files = find_video_files(input_folder, args.recursive)
    if not files:
        print('No video files found in', input_folder)
        return 0

    print(f'Found {len(files)} video file(s). Starting conversion with {args.workers} worker(s).')


    tasks = []
    for f in files:
        outp = make_output_path(f, input_folder, output_folder)
        tasks.append((f, outp))


    successes = 0
    failures = 0


    # Use ThreadPoolExecutor because ffmpeg is an external process and CPU usage is in the process; threads are fine.
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        future_to_task = {ex.submit(convert_one, inp, outp, args.crf, args.overwrite, args.pretend): (inp, outp) for inp, outp in tasks}
        for future in concurrent.futures.as_completed(future_to_task):
            inp, outp = future_to_task[future]
            try:
                _inp, ok, message = future.result()
            except Exception as e:
                ok = False
                message = f'exception while converting: {e}'
            if ok:
                successes += 1
                print(f'[OK] {inp.name}: {message}')
            else:
                failures += 1
                print(f'[ERR] {inp.name}: {message}')



    print('\nSummary:')
    print(f' Converted: {successes}')
    print(f' Failed/Skipped: {failures}')
    return 0




if __name__ == '__main__':
    raise SystemExit(main())
