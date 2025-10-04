#!/usr/bin/env python3
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
