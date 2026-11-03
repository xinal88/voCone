#!/usr/bin/env bash
# VO-cone experiment: set up the Arena workspace and start training, AGENT_5_VO first.
#
#   bash start_training.sh setup              # once, after Arena's install.sh: training feature, simulator, GPU
#   bash start_training.sh prepare            # apply our patches, copy the arm configs into the workspace
#   bash start_training.sh train AGENT_5_VO   # then AGENT_5 (minimal run), AGENT_5_TTC later
#   bash start_training.sh evaluate           # minimal held-out benchmark, VO vs AGENT_5
#
# ARENA_WS_DIR points at the Arena workspace (default ../arena_ws). Run `train` inside tmux: one arm takes days. SIM and ROBOT must stay the same for every arm.
set -euo pipefail

VO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WS=${ARENA_WS_DIR:-$VO/arena_ws}
TRAIN=$WS/src/Arena/arena_training
RL=$TRAIN/deps/rosnav_rl
SIM=${SIM:-gazebo}
ROBOT=${ROBOT:-jackal}

in_arena() {  # run arena CLI lines inside the workspace shell, the way install.sh does
    (cd "$WS" && printf '%s\n' "$@" 'exit' | source arena)
}

apply_once() {  # apply a patch unless it is already applied
    local repo=$1 patch=$2
    if git -C "$repo" apply --reverse --check "$patch" 2>/dev/null; then
        echo "already applied: $(basename "$patch")"
    else
        git -C "$repo" apply "$patch" && echo "applied: $(basename "$patch")"
    fi
}

case "${1:-}" in
    setup)
        [ -e "$WS/arena" ] || { echo "no Arena workspace at $WS, run Arena's install.sh first" >&2; exit 1; }
        git -C "$WS/src/Arena" submodule update --init --depth 1 arena_evaluation
        in_arena 'arena feature training install' "arena feature $SIM install" 'arena feature docker gpu on'
        ;;
    prepare)
        [ -d "$RL/rosnav_rl" ] || { echo "rosnav_rl not found at $RL, run setup first" >&2; exit 1; }
        apply_once "$RL" "$VO/patches/rosnav-rl.patch"
        apply_once "$TRAIN" "$VO/patches/Arena-Training.patch"
        cp -v "$VO"/experiment/configs/vo_cone_*.yaml "$TRAIN/configs/"
        in_arena 'arena build'
        ;;
    train)
        arm=${2:?usage: start_training.sh train AGENT_5_VO|AGENT_5|AGENT_5_TTC}
        config=$TRAIN/configs/vo_cone_$arm.yaml
        [ -f "$config" ] || { echo "missing $config, run prepare first" >&2; exit 1; }
        mkdir -p "$VO/experiment/logs"
        log=$VO/experiment/logs/train_${arm}_$(date +%Y%m%d_%H%M).log
        echo "training $arm on sim=$SIM robot=$ROBOT, log: $log"
        in_arena "arena train sim:=$SIM robot:=$ROBOT robot.mobile:=rosnav_rl train_config:=vo_cone_$arm.yaml" 2>&1 | tee "$log"
        ;;
    evaluate)
        BENCH=$WS/src/Arena/arena_evaluation/arena_evaluation/configs/benchmark
        cp -v "$VO/experiment/suite_vo_min.yaml" "$BENCH/suites/vo_min.yaml"
        cp -v "$VO/experiment/contest_vo_min.yaml" "$BENCH/contests/vo_min.yaml"
        mkdir -p "$VO/experiment/logs"
        in_arena 'arena evaluation benchmark --suite vo_min --contest vo_min' 2>&1 | tee "$VO/experiment/logs/eval_vo_min_$(date +%Y%m%d_%H%M).log"
        ;;
    *)
        sed -n '2,9p' "$0"
        exit 1
        ;;
esac
