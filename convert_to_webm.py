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
        proc = subprocess.run(
            ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
             '-show_entries', 'stream=bit_rate,duration', '-of', 'default=noprint_wrappers=1:nokey=1',
             str(input_path)],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        lines = [l.strip() for l in proc.stdout.decode().splitlines() if l.strip() != '']
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
        pass

    # fallback: try format duration
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

    # compute from filesize/duration if needed (container avg bitrate)
    if (source_kbps is None or source_kbps == 0) and duration and duration > 0:
        try:
            kbps = int(round((filesize * 8) / 1000.0 / duration))
            source_kbps = kbps
        except Exception:
            source_kbps = None

    return source_kbps, duration, filesize


def convert_one(input_path: Path, output_path: Path, crf: int, overwrite: bool, pretend: bool) -> tuple[Path, bool, str]:
    """
    Adaptive conversion:
      - For very small / low-bitrate inputs: use single-pass CRF with high CRF and low audio bitrate (keeps outputs small).
      - Otherwise: two-pass VP9 but compute target bitrate from original filesize so target_total_kbps < original_total_kbps.
    Returns (input_path, success, message).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not overwrite:
        return input_path, False, f"skipped (exists): {output_path}"

    null_dev = 'NUL' if os.name == 'nt' else '/dev/null'

    # Gather stats
    source_kbps, duration, filesize = _get_source_stats(input_path)
    debug = []
    debug.append(f"measured: source_kbps={source_kbps}, duration={duration}, filesize={filesize}")

    if source_kbps is None or source_kbps <= 0 or duration is None or duration <= 0:
        # conservative fallback if measurement fails
        source_kbps = int(max(120, (filesize * 8) / 1000.0 / (duration if duration and duration > 0 else 1)))
        debug.append(f"fallback source_kbps -> {source_kbps}")

    # Heuristics and tunables:
    # If the input total avg bitrate is small, prefer CRF single-pass to avoid encoder overhead.
    SMALL_TOTAL_KBPS = 500   # if original total kbps < this, use CRF single-pass
    CRF_FOR_SMALL = max(crf, 34)  # higher CRF -> smaller file for small inputs
    AUDIO_KBPS_SMALL = 32     # use 32 kb/s audio for tiny files
    AUDIO_KBPS_NORMAL = 64    # usual audio bitrate for larger files

    if source_kbps < SMALL_TOTAL_KBPS:
        # Single-pass CRF optimized for small files
        chosen_crf = CRF_FOR_SMALL
        audio_kbps = AUDIO_KBPS_SMALL
        debug.append(f"Choosing single-pass CRF mode (crf={chosen_crf}, audio={audio_kbps} kb/s) because source_kbps < {SMALL_TOTAL_KBPS}")

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
            '-b:a', f'{audio_kbps}k',
            '-f', 'webm',
        ]
        if overwrite:
            cmd.insert(1, '-y')
        else:
            cmd.insert(1, '-n')
        cmd.append(str(output_path))

        if pretend:
            return input_path, True, 'pretend: ' + ' '.join(cmd) + '\n' + '\n'.join(debug)

        try:
            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            return input_path, True, f'converted (crf={chosen_crf}, audio={audio_kbps}k) -> {output_path}\n' + '\n'.join(debug)
        except subprocess.CalledProcessError as e:
            err = e.stderr.decode(errors='ignore') if e.stderr else str(e)
            return input_path, False, f'ffmpeg error: {err[:500]}\n' + '\n'.join(debug)
        except Exception as e:
            return input_path, False, f'exception: {e}\n' + '\n'.join(debug)

    # Otherwise compute a target total kbps that is guaranteed lower than original
    desired_ratio = 0.60   # target total bitrate = 60% of original total bitrate (tune this)
    orig_total_kbps = source_kbps
    target_total_kbps = max(64, int(math.floor(orig_total_kbps * desired_ratio)))
    # choose audio kbps for normal files
    audio_kbps = AUDIO_KBPS_NORMAL
    # make sure we leave some room for audio; video_kbps = target_total - audio_kbps
    video_kbps = max(32, int(target_total_kbps - audio_kbps))
    # Do not exceed original video bitrate estimate (if known)
    # We only have orig_total_kbps; ensure video_kbps < orig_total_kbps
    if video_kbps >= orig_total_kbps:
        video_kbps = max(32, int(orig_total_kbps * 0.9))

    debug.append(f"Choosing two-pass mode: orig_total_kbps={orig_total_kbps}, target_total_kbps={target_total_kbps}, video_kbps={video_kbps}, audio_kbps={audio_kbps}")

    # Build two-pass commands with passlog in tmp dir
    tmpdir = Path(tempfile.mkdtemp(prefix='vp9-pass-'))
    passlog_base = tmpdir / 'ffpass'

    vp9_common = [
        '-c:v', 'libvpx-vp9',
        '-b:v', f'{video_kbps}k',
        '-tile-columns', '4',
        '-frame-parallel', '1',
        '-speed', '1',
        '-passlogfile', str(passlog_base)
    ]

    pass1_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-i', str(input_path)] + vp9_common + [
        '-pass', '1', '-an', '-f', 'null', null_dev
    ]
    pass2_cmd = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-i', str(input_path)] + vp9_common + [
        '-pass', '2', '-c:a', 'libopus', '-b:a', f'{audio_kbps}k', '-vf', 'scale=iw:ih', '-f', 'webm', str(output_path)
    ]

    if pretend:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return input_path, True, 'pretend pass1: ' + ' '.join(pass1_cmd) + '\npretend pass2: ' + ' '.join(pass2_cmd) + '\n' + '\n'.join(debug)

    try:
        subprocess.run(pass1_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(pass2_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return input_path, True, f'converted (two-pass video={video_kbps}k audio={audio_kbps}k) -> {output_path}\n' + '\n'.join(debug)
    except subprocess.CalledProcessError as e:
        err = e.stderr.decode(errors='ignore') if e.stderr else str(e)
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return input_path, False, f'ffmpeg error: {err[:500]}\n' + '\n'.join(debug)
    except Exception as e:
        try:
            shutil.rmtree(tmpdir)
        except Exception:
            pass
        return input_path, False, f'exception: {e}\n' + '\n'.join(debug)


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
