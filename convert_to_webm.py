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

def convert_one(input_path: Path, output_path: Path, crf: int, overwrite: bool, pretend: bool) -> tuple[Path, bool, str]:
    """
    Convert a single file to WebM (VP9 + Opus).
    Returns (input_path, success, message)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return input_path, False, f"skipped (exists): {output_path}"


    # Build ffmpeg command. We use libvpx-vp9 and libopus.
    # -y to overwrite handled only when overwrite is True
    cmd = [
    'ffmpeg',
    '-hide_banner',
    '-loglevel', 'error',
    '-i', str(input_path),
    '-c:v', 'libvpx-vp9',
    '-crf', str(crf),
    '-b:v', '0', # CRF-based for libvpx-vp9
    '-tile-columns', '4',
    '-frame-parallel', '1',
    '-speed', '1', # tradeoff speed/quality, 0=best, up to 5 fastest; 1 is slow but high quality
    '-c:a', 'libopus',
    '-b:a', '64k',
    '-f', 'webm',
    ]


    if overwrite:
        cmd.insert(1, '-y')
    else:
        cmd.insert(1, '-n') # don't overwrite


    cmd.append(str(output_path))


    if pretend:
        return input_path, True, 'pretend: ' + ' '.join(cmd)


    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return input_path, True, f'converted -> {output_path}'
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors='ignore') if e.stderr else str(e)
        return input_path, False, f'ffmpeg error: {err[:500]}'
    except Exception as e:
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
