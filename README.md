# webm-batch-converter


Convert a directory of videos to WebM (VP9 + Opus) using `ffmpeg`.


## Features


- Converts common video formats (mp4, mov, mkv, avi, ...) to `.webm`.
- Preserves relative folder structure when using `-r`.
- Parallel conversion using multiple workers.
- Safe-by-default: won't overwrite outputs unless `--overwrite` is specified.
- `--pretend` mode to print ffmpeg commands without executing them.


## Requirements


- Python 3.8+
- `ffmpeg` installed and available on the system PATH (not a pip package). Install ffmpeg from your OS's package manager or from https://ffmpeg.org/.


This script uses only Python standard library modules — no pip packages are required.


## Installation


```bash
# Clone the repo
git clone https://github.com/yourusername/webm-batch-converter.git
cd webm-batch-converter


# (Optional) create a venv
python -m venv .venv
source .venv/bin/activate # macOS / Linux
.venv/Scripts/activate # Windows (use forward slashes in docs to avoid escaping issues)


# No pip packages to install. Ensure ffmpeg is installed:
ffmpeg -version
```

## Usage
```python
# Basic usage (non-recursive):
python convert_to_webm.py /path/to/input_folder /path/to/output_folder


# Recurse and preserve folder structure with 4 parallel workers and higher quality:
python convert_to_webm.py /path/to/input_folder /path/to/output_folder -r -w 4 --crf 28


# Preview commands without running (safe dry-run):
python convert_to_webm.py /path/to/input_folder /path/to/output_folder --pretend


# Overwrite existing outputs:
python convert_to_webm.py /path/to/input_folder /path/to/output_folder --overwrite
```

## Options

-r, --recursive : Recurse into subfolders and preserve folder structure.

-w, --workers N : Number of parallel workers (default: number of CPUs).

--crf N : Quality for libvpx-vp9 (lower = better quality, default 32).

--overwrite : Overwrite existing output files.

--pretend : Print actions but don't run ffmpeg.
