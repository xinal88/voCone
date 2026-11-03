"""Write one training config per arm from Arena-Training's sb_training_config.yaml.

The arms differ only in agent name and architecture; everything else (reward, budget, seeds,
curriculum, PPO) is shared, and the script checks that before writing.

    python make_configs.py [path/to/Arena-Training/configs/sb_training_config.yaml]   # -> configs/vo_cone_<ARM>.yaml
"""

from __future__ import annotations

import difflib
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_TEMPLATE = HERE.parent / "Arena-Training" / "configs" / "sb_training_config.yaml"
OUT = HERE / "configs"

ARMS = ("AGENT_5_VO", "AGENT_5", "AGENT_5_TTC")  # launch order: VO first
SEED = 42
N_TIMESTEPS = 1_000_000
# Fixed curriculum schedule in total control ticks (PREREGISTRATION.md, decision 7): the
# pedestrian stage starts at 150k for every arm instead of on a success threshold.
STAGE_TICKS = [0, 50_000, 100_000, 150_000]
CHECKPOINT_TICKS = [250_000, 500_000, 750_000]

# DRL-VO heading term at the unit's defaults, shared by every arm (PREREGISTRATION.md, decision 2).
HEADING_REWARD = """      active_heading_direction:
        r_angle: 0.6
        theta_m: 0.5235987755982988  # pi / 6
        theta_min: 1000
        ped_min_dist: 8.0
        iters: 60
"""


def sub(pattern: str, replacement: str, text: str) -> str:
    out, count = re.subn(pattern, replacement, text, count=1, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"template changed, pattern not found: {pattern}")
    return out


def shared(text: str) -> str:
    text = sub(r"^(\s+seed:) null", rf"\1 {SEED}", text)
    text = sub(r"^(\s+env_seed:) null", rf"\1 {SEED}", text)
    # The budget is algorithm.parameters.total_timesteps (default 10M); general.n_timesteps is not read.
    text = sub(r"^(\s+total_batch_size: .*\n)", rf"\1        total_timesteps: {N_TIMESTEPS}  # training budget in control ticks\n", text)
    text = sub(r"^(\s+n_timesteps:) \d+.*$", rf"\1 {N_TIMESTEPS}  # not read by the trainer, kept equal to total_timesteps", text)
    text = sub(r"^(\s+tensorboard_log:) null", r"\1 tensorboard", text)
    text = sub(r"^(\s+lower_threshold: .*\n)", rf"\1      stage_ticks: {STAGE_TICKS}  # fixed schedule, replaces the thresholds\n", text)
    text = sub(r"^(  callbacks:\n)", rf"\1    checkpoint_ticks: {CHECKPOINT_TICKS}  # tick_<T>.zip snapshots\n", text)
    text = sub(r"^    stop_training_on_threshold:\n(?:      .*\n){3}", "    stop_training_on_threshold: null  # success early stop off\n", text)
    text = sub(r"^(      collision:\n        reward: -15\n)", r"\1" + HEADING_REWARD, text)
    return text


def arm(text: str, name: str) -> str:
    text = sub(r"^(  name:) null", rf"\1 {name}_vo_cone", text)
    return sub(r"^(\s+architecture_name:) AGENT_1", rf"\1 {name}", text)


def main() -> None:
    template = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_TEMPLATE
    base = shared(template.read_text())
    configs = {name: arm(base, name) for name in ARMS}
    reference = configs[ARMS[0]].splitlines()
    for name, text in configs.items():
        changed = [
            line
            for line in difflib.unified_diff(reference, text.splitlines(), lineterm="", n=0)
            if line[:1] in "+-" and not line.startswith(("+++", "---"))
        ]
        if len(changed) not in (0, 4):
            raise AssertionError(f"{name} differs from {ARMS[0]} beyond name and architecture: {changed}")
    OUT.mkdir(exist_ok=True)
    for name, text in configs.items():
        path = OUT / f"vo_cone_{name}.yaml"
        path.write_text(text)
        print(path)


if __name__ == "__main__":
    main()
