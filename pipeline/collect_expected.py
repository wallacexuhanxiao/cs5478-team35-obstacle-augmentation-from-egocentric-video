"""Collect a compact reference of a finished sample run into samples/<case>/expected/ (results.json, the box /
certificate / object-pose JSONs and the edited videos), with absolute paths replaced by paths relative to the run.

usage: python pipeline/collect_expected.py work/mustard_1 samples/mustard_1/expected
"""
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np

run, out = Path(sys.argv[1]).resolve(), Path(sys.argv[2])
repo = run.parents[1]
out.mkdir(parents=True, exist_ok=True)


def rel(text):
    text = text.replace(str(run), 'work/' + run.name).replace(str(repo), '.')
    return re.sub(r'/[^\s"\']*/(envs|scratch|home)/[^\s"\']*', '<path>', text)


def trace(path):
    frames = json.loads(Path(path).read_text())
    return (np.array([f['wrist_xyz_wxyz'][:3] for f in frames]), np.array([f['objects']['tool']['pose'][:3] for f in frames]))


results = {}
w0, o0 = trace(run / 'reproduce' / 'final' / 'actual_trace.json')
for name in ['reproduce'] + sorted(p.name for p in run.glob('detour_*')):
    r = json.loads((run / name / 'final' / 'pipeline_result.json').read_text())
    w, o = trace(run / name / 'final' / 'actual_trace.json')
    n = min(len(w), len(w0))
    item = dict(passed=bool(r.get('full_pipeline_success') and r.get('task_success')),
                continuous_rollout_state_resets=r.get('state_resets_between_segments'),
                final_object_error=r.get('task', {}).get('final_errors'),
                executed_duration_s=r.get('actual_duration_s'), source_duration_s=r.get('source_duration_s'),
                original_source_duration_s=r.get('original_source_duration_s'),
                max_bottle_lift_m=float(o[:, 2].max() - o[0, 2]))
    if name != 'reproduce':
        d = np.linalg.norm(w[:n] - w0[:n], axis=1)
        item.update(max_wrist_deviation_from_reproduction_m=float(d.max()), at_frame=int(d.argmax()),
                    max_wrist_raise_over_reproduction_m=float((w[:n, 2] - w0[:n, 2]).max()))
        k = name.split('_', 1)[1]
        cert = run / f'certificate_{k}.json'
        if cert.exists():
            item['box_certificate'] = json.loads(cert.read_text())
            item['box_certificate']['box'] = f'boxes/chosen/{k}.json'
        item['objects_placed'] = sorted(p.stem for p in (run / f'composite_{k}').glob('*.json'))
    results[name] = item
(out / 'results.json').write_text(json.dumps(results, indent=1))

for sub in ['boxes/chosen'] + [p.relative_to(run).as_posix() for p in run.glob('composite_*')]:
    for f in (run / sub).glob('*.json'):
        dst = out / sub / f.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(rel(f.read_text()))
for f in run.glob('certificate_*.json'):
    (out / f.name).write_text(rel(f.read_text()))
(out / 'SUMMARY.txt').write_text(rel((run / 'SUMMARY.txt').read_text()))
videos = out / 'videos'
videos.mkdir(exist_ok=True)
for v in sorted(run.glob('render/*/*/visual.mp4')):
    shutil.copy(v, videos / f'{v.parent.name}.mp4')
print(json.dumps(results, indent=1))
print('VIDEOS', sorted(p.name for p in videos.glob('*.mp4')))
