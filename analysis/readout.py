"""Readout for the velocity-obstacle cone experiment.

Reads the per-episode `combined_metrics.parquet` of one or more benchmark runs, pairs the
episodes of every arm by (stage, episode), and reports the delta of the new arm against each
control with a paired bootstrap interval, per split and metric. The verdict follows the task:
chase when one held-out interval of the verdict metric lies above zero (better) and none lies
below, drop when none lies above zero, otherwise inconclusive.

    python readout.py runs/*/combined_metrics.parquet --splits splits.yaml --out readout
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


@dataclass(frozen=True)
class Metric:
    name: str
    higher_is_better: bool
    only_both_succeeded: bool = False


METRICS = (
    Metric("zero_contact_success", True),
    Metric("success", True),
    Metric("contacts", False),
    Metric("time_to_goal", False, only_both_succeeded=True),
    Metric("ped_path_deflection_m", False),
)


def load(paths: list[Path], arm_column: str, contact_column: str) -> pd.DataFrame:
    frame = pd.concat([pd.read_parquet(p).assign(_source=str(p)) for p in paths], ignore_index=True)
    frame = frame.rename(columns={arm_column: "arm"})
    frame["success"] = frame["success"].astype(float)
    frame["contacts"] = frame[contact_column].fillna(0).astype(float)
    frame["zero_contact_success"] = ((frame["success"] > 0) & (frame["contacts"] == 0)).astype(float)
    return frame


def assign_splits(frame: pd.DataFrame, splits: list[dict]) -> pd.DataFrame:
    """Each split is {name, stage (regex), held_out}; the first matching split wins, unmatched stages are dropped."""

    def match(stage: str) -> tuple[str | None, bool]:
        for split in splits:
            if re.fullmatch(split["stage"], str(stage)):
                return split["name"], bool(split.get("held_out", True))
        return None, False

    matched = frame["stage"].map(match)
    frame = frame.assign(split=matched.str[0], held_out=matched.str[1])
    return frame[frame["split"].notna()]


def paired_bootstrap(diff: np.ndarray, resamples: int, rng: np.random.Generator) -> tuple[float, float]:
    index = rng.integers(0, len(diff), size=(resamples, len(diff)))
    means = diff[index].mean(axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def readout(frame: pd.DataFrame, arm: str, controls: list[str], resamples: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for (split, held_out), group in frame.groupby(["split", "held_out"], sort=True):
        wide = group.pivot_table(index=["stage", "episode"], columns="arm", values=[m.name for m in METRICS], aggfunc="first")
        for control in controls:
            for metric in METRICS:
                if (metric.name, arm) not in wide or (metric.name, control) not in wide:
                    continue
                pair = wide[[(metric.name, arm), (metric.name, control)]].dropna()
                if metric.only_both_succeeded:
                    ok = wide[("success", arm)].gt(0) & wide[("success", control)].gt(0)
                    pair = pair[ok.reindex(pair.index, fill_value=False)]
                if pair.empty:
                    continue
                new, old = pair.iloc[:, 0].to_numpy(), pair.iloc[:, 1].to_numpy()
                diff = new - old
                low, high = paired_bootstrap(diff, resamples, rng)
                better_low, better_high = (low, high) if metric.higher_is_better else (-high, -low)
                rows.append(
                    {
                        "split": split,
                        "held_out": held_out,
                        "control": control,
                        "metric": metric.name,
                        "pairs": len(diff),
                        "arm_mean": new.mean(),
                        "control_mean": old.mean(),
                        "delta": diff.mean(),
                        "ci_low": low,
                        "ci_high": high,
                        "better": "above" if better_low > 0 else "below" if better_high < 0 else "spans",
                    }
                )
    return pd.DataFrame(rows)


def verdict(table: pd.DataFrame, metric: str) -> str:
    held = table[table["held_out"] & (table["metric"] == metric)]
    if held.empty:
        return "no held-out data"
    if (held["better"] == "above").any() and not (held["better"] == "below").any():
        return "chase"
    if not (held["better"] == "above").any():
        return "drop"
    return "inconclusive"


def to_markdown(table: pd.DataFrame) -> str:
    lines = ["| split | held-out | control | metric | pairs | arm | control | Δ [95% CI] | |", "|---|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        lines.append(
            f"| {r.split} | {'yes' if r.held_out else 'no'} | {r.control} | {r.metric} | {r.pairs} | {r.arm_mean:.3f} | "
            f"{r.control_mean:.3f} | {r.delta:+.3f} [{r.ci_low:+.3f}, {r.ci_high:+.3f}] | {r.better} |"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("metrics", nargs="+", type=Path, help="combined_metrics.parquet files")
    parser.add_argument("--splits", type=Path, required=True, help="YAML list of {name, stage regex, held_out}")
    parser.add_argument("--arm", default="AGENT_5_VO")
    parser.add_argument("--controls", nargs="+", default=["AGENT_5", "AGENT_5_TTC"])
    parser.add_argument("--arm-column", default="planner", help="column naming the contestant")
    parser.add_argument("--contact-column", default="collision_amount_pedestrian")
    parser.add_argument("--verdict-metric", default="zero_contact_success")
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("readout"))
    args = parser.parse_args()

    frame = assign_splits(load(args.metrics, args.arm_column, args.contact_column), yaml.safe_load(args.splits.read_text()))
    table = readout(frame, args.arm, args.controls, args.resamples, args.seed)
    result = verdict(table, args.verdict_metric)
    args.out.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out / "readout.csv", index=False)
    (args.out / "readout.md").write_text(f"Verdict on {args.verdict_metric}: **{result}**\n\n{to_markdown(table)}\n")
    print(f"verdict: {result}")
    print(to_markdown(table))


if __name__ == "__main__":
    main()
