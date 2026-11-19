"""Run EgoEngine's official `run_mjwp_mixed.py` at a sim_dt it was not tuned for.

SPIDER's Config fixes trace_dt = 1/50 s (recording cadence only) and asserts it
is a multiple of sim_dt; the mixed runner exposes no flag for it. Our scenes
step at 1/120 s, so round trace_dt to the nearest multiple of sim_dt before
the official config check. Nothing else in the official runner is changed.

usage: python egoengine_mixed_launcher.py /path/to/run_mjwp_mixed.py --sim-dt ... (official flags)
"""
import importlib.util
import sys
from pathlib import Path


def main():
    runner = Path(sys.argv.pop(1)).resolve()
    sys.path.insert(0, str(runner.parent))
    spec = importlib.util.spec_from_file_location('egoengine_run_mjwp_mixed', runner)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    official = module.process_config

    def process_config(cfg):
        steps = max(1, round(cfg.trace_dt / cfg.sim_dt))
        cfg.trace_dt = steps * cfg.sim_dt
        return official(cfg)
    module.process_config = process_config
    import tyro
    module.main(tyro.cli(module.MixedArgs))


if __name__ == '__main__':
    main()
