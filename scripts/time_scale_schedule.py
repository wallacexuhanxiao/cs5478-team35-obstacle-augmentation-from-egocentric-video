"""Copy an Oracle schedule with its source frames played on a slower clock.

EgoEngine's release plays 30 fps TACO/Aria frames at 20 Hz (1.5x longer). This
opt-in variant keeps every frame, window and Anchor boundary; only the playback
fps changes, and the result records the scale. Results on it are labelled
"source time scaled", never reported as the original source clock.
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--schedule', type=Path, required=True)
    p.add_argument('--scale', type=float, default=1.5)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.scale <= 0:
        p.error('scale must be positive')
    data = json.loads(args.schedule.read_text())
    if 'source_time_scale' in data:
        p.error('schedule is already time-scaled')
    data['source_fps'] = float(data['fps'])
    data['fps'] = float(data['fps']) / args.scale
    data['source_time_scale'] = float(args.scale)
    data['source_time_scale_provenance'] = (
        f'Opt-in: {args.schedule} with its {data["num_frames"]} source frames played at {data["fps"]:g} Hz instead of '
        f'{data["source_fps"]:g} Hz ({args.scale:g}x longer), as the EgoEngine release does for TACO/Aria.')
    args.output.write_text(json.dumps(data, indent=2))
    print('TIME_SCALED', args.output, data['source_fps'], '->', data['fps'], flush=True)


if __name__ == '__main__':
    main()
