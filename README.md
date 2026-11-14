# Voronoi-A* and TSP-OR DQN

This repository contains a finite-segment Voronoi-A* planner and a local
navigation policy that combines obstacle-motion prediction, a BiLSTM Dueling
DQN, dual-criterion replay and event-triggered replanning.

## Installation and verification

Use Python 3.10 or later. Run commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Train and evaluate the current implementation

`configs/method.json` specifies the learner and the Maze A training schedule.
Algorithm 3 uses one-step Double DQN with Huber loss. Eqs. (18)–(19) use one
buffer of 200,000 transitions and sample 64 entries with replacement. The
proposed replay uses no importance-sampling correction. For the Table VIII
replay comparisons, use `--replay-mode per` for proportional PER with its
importance-sampling schedule, and `--replay-mode uniform` for uniform replay.
Pass the same replay mode when evaluating the resulting checkpoint.

```bash
python train.py --map-file grid_map2.txt --seed 0 --device cpu \
  --save-path outputs/maze_a/seed0/final.pth --log-dir outputs/maze_a/seed0/train

python -m experiments.proposed_evaluation \
  --map-file grid_map2.txt --training-map grid_map2.txt \
  --checkpoint outputs/maze_a/seed0/final.pth --device cpu --robot-radius 0.50 \
  --scenarios 300 --eval-seed 10000 --levels 1 2 3 4 \
  --output-dir outputs/maze_a/seed0/evaluation
```

Use training seeds 0, 1 and 2 in separate runs. On Maze A, the default schedule
samples Levels 1–4 for 1,000 episodes with at most 2,000 control steps each.
A training run records the final checkpoint and its checksum, learner settings,
effective curriculum, source snapshots, and per-episode logs. The grid
evaluator records each scenario and timing result. For Table VII, throughput
counts successful tasks per second of total episode wall time. The cycle-time
results include both the mean with triggered replanning and the mean after
subtracting triggered planning time. Initial planning is included in episode
wall time.

The four encoder and head combinations can be selected with `--architecture`.
Use `--prediction-mode none` to omit prediction cues and their reward penalty,
or `--navigation-mode direct_goal` to train without intermediate sub-goals.
The grid evaluator can also evaluate fixed-checkpoint controller ablations with
`--controller`.

## Table XI(b) continuous closed-loop protocol

The holonomic tracker is in
`revision_experiments/continuous_transfer/tracking_loop.py`. Its [protocol
file](configs/table_xv_b.json) defines the seven groups, 300 scenarios per
seed, 0.50-cell radius, 0.02-s integration interval, 4.84-s moving-action window,
and episode caps of 2,000 decisions and 9,680 simulated seconds. Move refusal
rate is the number of rejected moves divided by rejected plus issued moves.
The output reports fractions and the per-episode records used to aggregate them.
`motion_control.py` is a separate unicycle example.

Run group C2 with archive directory label 0, for example:

```bash
python -m revision_experiments.continuous_transfer.tracking_loop \
  --group C2 --training-seed 0 --allow-historical-checkpoint \
  --output-dir results/table_xi_b/C2_seed0
```

Repeat with the other groups and seeds. For weights from a new training run,
pass its path with `--checkpoint` and omit `--allow-historical-checkpoint`.
To aggregate three complete runs for one group, pass their output directories
to `python tools/seed_statistics.py --continuous-dirs DIR0 DIR1 DIR2 --output summary.json`.

## Archived checkpoint provenance

[Archive metadata](checkpoints/archive_metadata.json) records SHA-256 and
the provenance fields embedded in each of the nine released checkpoints.
The embedded code protocol is `jas0644-aligned-v2`, the replay protocol is
`submitted-eq18-eq19-one-step`, and the recorded training environment has a
radius of 0.0 cells. The three Maze A files record 330,488, 328,231 and 305,422
training interactions. The directory names provide seed labels. The payloads
contain source hashes, without a training commit, embedded seed, full learner
configuration, or original run logs.

The archive examples explicitly opt into evaluation of these historical
weights under the current evaluation protocol. Archive metadata and new
evaluation output are separate from the original paper's raw records. The
train/evaluate commands above cover TSP-OR DQN and its released variants.
The original comparison runs and artifacts for M-DQN, IDDQN, RecurrentPPO,
APF-L and IGFL are not included in this repository.
