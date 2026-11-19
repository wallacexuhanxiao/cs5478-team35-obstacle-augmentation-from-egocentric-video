"""Visual-only prepared package for an externally simulated rollout of a released case.

Copies the case's processed scene and visual inputs under a new name so that
`physics_to_retarget.py --multipath-run` can write a new `retarget.json` and
background without touching the release.
"""
import argparse
import json
import os
import shutil
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--case', required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.source_root / 'manifest.json').read_text())
    info = next(c for c in manifest['cases'] if c['task'] == args.case)
    new = dict(info, task=args.name, processed_dir=str(Path(info['processed_dir']).parent / args.name),
               visual_dir=str(Path(info['visual_dir']).parent / args.name),
               augmentation=dict(source_case=args.case, kind='visual package for an external PhysX rollout'))
    if args.out.exists():
        shutil.rmtree(args.out)
    shutil.copytree(args.source_root / info['processed_dir'], args.out / new['processed_dir'])
    shutil.copytree(args.source_root / info['visual_dir'], args.out / new['visual_dir'])
    os.symlink((args.source_root / 'assets').resolve(), args.out / 'assets')
    (args.out / 'manifest.json').write_text(json.dumps(dict(created_from=str(args.source_root.resolve()), cases=[new]), indent=2))
    print('VISUAL_PACKAGE', args.out, flush=True)


if __name__ == '__main__':
    main()
