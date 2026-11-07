# Local View Navigation Based on Global-Local Collaborative Framework in High-Density Scenes

Autonomous Mobile Robots (AMRs) deployed in intelligent warehouses and similar scenarios must navigate through large-scale, obstacle-dense environments where only the permanent structural layout is known a priori. In such settings, the environment is cluttered with unknown temporary static obstacles and unpredictable dynamic entities, creating a high-density, dynamically evolving scenario. However, existing hybrid planners that rely on precomputed geometric paths are prone to frequent global re-planning when unexpected obstacles invalidate the assumed path structure. Meanwhile, standard end-to-end Deep Reinforcement Learning (DRL) methods are limited by slow convergence and weak long-range guidance in large maps. To overcome these limitations, we propose a novel global--local collaborative navigation framework to provide long-range topological guidance and to avoid fast-changing obstacles. First, the global planner constructs a Voronoi diagram from the structural map known a priori. Second, a Voronoi-A* search is performed to extract sparse key path points that serve as flexible sub-goals. Third, for local control, we introduce the Temporal-State Processing and Optimized Replay DQN (TSP-OR DQN), which integrates temporal feature modeling, motion prediction, and a prioritized replay mechanism to handle partial observability. Finally, an event-triggered global re-planning rule is introduced to regenerate key path points when the active sub-goal cannot be reached within a distance-adaptive arrival budget. Extensive evaluation across four environments and four obstacle density levels demonstrates that our framework achieves competitive or superior performance against representative baselines across most tested scenarios. \textcolor{blue}{Across three independent training seeds, the proposed method retains mean success rates above 60\% in the hardest Indoor ($61.2\% \pm 1.8\%$), Maze~A ($63.2\% \pm 2.1\%$), and unseen Maze~B ($71.9\% \pm 5.1\%$) Level~4 settings, while achieving $92.6\% \pm 1.6\%$ in Outdoor Level~4.} Furthermore, the learned policy exhibits strong generalization to unseen maps and novel obstacle configurations.

## Install and verify

Python 3.8+; validation used Python 3.8 and PyTorch 2.3 on CPU. Run commands from the repository root.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
```

## Train and evaluate

The default is 1,000 episodes, uniformly sampled L1–L4, up to 2,000 control steps per episode. Maze A is `grid_map2.txt`. Seeds 0, 1 and 2 identify independent training runs.

```bash
python train.py --map-file grid_map2.txt --seed 0 --device cpu \
  --save-path outputs/seed0/final.pth --log-dir outputs/seed0/train
python -m experiments.proposed_evaluation \
  --map-file grid_map2.txt --training-map grid_map2.txt \
  --checkpoint outputs/seed0/final.pth --device cpu \
  --scenarios 300 --eval-seed 10000 --levels 1 2 3 4 \
  --output-dir outputs/seed0/evaluation
```

Training records the learner settings, seed, effective curriculum, source snapshots alongside JSONL episode logs.
