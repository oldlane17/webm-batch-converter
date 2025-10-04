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


def _get_source_stats(input_path: Path) -> tuple[int | None, float | None, int]:
    """
    Return (source_kbps, duration_seconds, filesize_bytes)
    - source_kbps: average overall bitrate in kilobits-per-second (kb/s) if possible (video stream bit_rate or computed),
      or None on error.
    - duration_seconds: duration float or None.
    - filesize_bytes: int
    """
    filesize = input_path.stat().st_size
    duration = None
    source_kbps = None

    try:
        # Try to get stream bit_rate and duration via ffprobe (video stream)
        proc = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=bit_rate,duration', '-of', 'default=noprint_wrappers=1:nokey=1',
             str(input_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        lines = [l.strip() for l in proc.stdout.decode().splitlines() if l.strip() != '']
        # lines may contain bit_rate on first non-empty line, duration on second
        bit_rate = None
        if len(lines) >= 1:
            try:
                br = float(lines[0])
                if br > 0:
                    bit_rate = int(br)
            except Exception:
                bit_rate = None
        if len(lines) >= 2:
            try:
                d = float(lines[1])
                if d > 0:
                    duration = d
            except Exception:
                duration = None

        if bit_rate and bit_rate > 0:
            source_kbps = int(round(bit_rate / 1000.0))
    except Exception:
        # fall through to other heuristics
        pass

    # If we didn't get duration yet, try format duration
    if duration is None:
        try:
            proc2 = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration',
                 '-of', 'default=nokey=1:noprint_wrappers=1', str(input_path)],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            d_out = proc2.stdout.decode().strip()
            if d_out:
                duration = float(d_out)
        except Exception:
            duration = None

    # If still no source_kbps, compute from filesize/duration (total container bitrate)
    if (source_kbps is None or source_kbps == 0) and duration and duration > 0:
        try:
            kbps = int(round((filesize * 8) / 1000.0 / duration))
            source_kbps = kbps
        except Exception:
            source_kbps = None

    return source_kbps, duration, filesize

def convert_one(input_path: Path, output_path: Path, crf: int, overwrite: bool, pretend: bool) -> tuple[Path, bool, str]:
    """
    Smart conversion:
      - For low-bitrate/small videos, use CRF single-pass (smaller outputs).
      - For larger/high-bitrate videos, use two-pass VP9 targeting 70% of source_kbps.
    The 'crf' parameter acts as a fallback CRF for single-pass mode (unless two-pass chosen).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return input_path, False, f"skipped (exists): {output_path}"

    # Platform null device
    null_dev = 'NUL' if os.name == 'nt' else '/dev/null'

    # Get source stats
    source_kbps, duration, filesize = _get_source_stats(input_path)

    debug_lines = []
    debug_lines.append(f"source_kbps={source_kbps} kb/s, duration={duration}, filesize={filesize} bytes")

    # If we fail to measure, assume a conservative default
    if source_kbps is None or source_kbps <= 0:
        source_kbps = 1200
        debug_lines.append("measured source_kbps missing -> fallback 1200 kb/s")

    # Heuristic decision: if source_kbps is small, prefer CRF single-pass to avoid overhead inflation.
    SMALL_THRESHOLD_KBPS = 400  # configurable: below this we prefer CRF single-pass
    if source_kbps < SMALL_THRESHOLD_KBPS:
        # Use CRF single-pass with a relatively high CRF to keep file small.
        # Choose a CRF that is no lower than the provided 'crf' but biased towards smaller output.
        chosen_crf = max(crf, 30)  # default to at least 30 for tiny/low-bitrate inputs
        debug_lines.append(f"Choosing CRF single-pass mode with crf={chosen_crf} because source_kbps < {SMALL_THRESHOLD_KBPS}")

        cmd = [
            'ffmpeg', '-hide_banner', '-loglevel', 'error',
            '-i', str(input_path),
            '-c:v', 'libvpx-vp9',
            '-crf', str(chosen_crf),
            '-b:v', '0',
            '-tile-columns', '4',
            '-frame-parallel', '1',
            '-speed', '1',
            '-vf', 'scale=iw:ih',
            '-c:a', 'libopus',
            '-b:a', '64k',
            '-f', 'webm',
        ]
        if overwrite:
            cmd.insert(1, '-y')
        else:
            cmd.insert(1, '-n')

        cmd.append(str(output_path))

        if pretend:
            return input_path, True, 'pretend: ' + ' '.join(cmd) + '\n' + '\n'.join(debug_lines)

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return input_path, True, f'converted (crf={chosen_crf}) -> {output_path}\\n' + '\\n'.join(debug_lines)
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors='ignore') if e.stderr else str(e)
            return input_path, False, f'ffmpeg error: {err[:500]}\\n' + '\\n'.join(debug_lines)
        except Exception as e:
            return input_path, False, f'exception: {e}\\n' + '\\n'.join(debug_lines)

    # Otherwise use two-pass
    multiplier = 0.7
    target_kbps = max(64, int(math.floor(source_kbps * multiplier)))
    # never exceed source_kbps
    if target_kbps >= source_kbps:
        target_kbps = max(64, int(source_kbps * 0.9))

    debug_lines.append(f"Choosing 2-pass mode: source_kbps={source_kbps}, multiplier={multiplier}, target_kbps={target_kbps}")

    tmpdir = Path(tempfile.mkdtemp(prefix='vp9-pass-'))
    passlog_base = tmpdir / 'ffpass'

    vp9_common = [
        '-c:v', 'libvpx-vp9',
        '-b:v', f'{target_kbps}k',
        '-tile-columns', '4',
        '-frame-parallel', '1',
        '-speed', '1',
        '-passlogfile', str(passlog_base)
    ]

    pass1_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(input_path)] + vp9_common + [
        '-pass', '1', '-an', '-f', 'null', null_dev
    ]
    pass2_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(input_path)] + vp9_common + [
        '-pass', '2', '-c:a', 'libopus', '-b:a', '64k', '-vf', 'scale=iw:ih', '-f', 'webm', str(output_path)
    ]

    if pretend:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return input_path, True, 'pretend pass1: ' + ' '.join(pass1_cmd) + '\\npretend pass2: ' + ' '.join(pass2_cmd) + '\\n' + '\\n'.join(debug_lines)

    try:
        subprocess.run(pass1_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(pass2_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return input_path, True, f'converted (two-pass {target_kbps}k) -> {output_path}\\n' + '\\n'.join(debug_lines)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors='ignore') if e.stderr else str(e)
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return input_path, False, f'ffmpeg error: {err[:500]}\\n' + '\\n'.join(debug_lines)
    except Exception as e:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return input_path, False, f'exception: {e}\\n' + '\\n'.join(debug_lines)


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
