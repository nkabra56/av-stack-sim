# Trained policy provenance

Both `.zip` files here are PPO policies (Stable-Baselines3 2.9.0, `MlpPolicy`, default
architecture) trained on `core/rl/parking_env.py`'s `ParkingEnv` -- see DESIGN.md
section 10's "learned parking policy" future extension and
`core/validation/rl_comparison.py`.

Committed here (unlike `core/data/kitti/`/`core/data/ngsim/`, these are locally
generated artifacts, not third-party data, so there's no attribution/license to carry)
so `tests/test_rl_comparison.py` can evaluate a real trained policy quickly, without
needing to retrain (several minutes of wall-clock time each) as part of the fast test
suite -- the same "commit the real artifact, keep collection/training separate from
the fast suite" pattern the KITTI/NGSIM excerpts already use.

**Reproducing them**:

```
python -m core.rl.train perpendicular_open --timesteps 300000 --seed 0 --save core/data/rl/parking_policy_perpendicular_open.zip
python -m core.rl.train perpendicular_flanked --timesteps 500000 --seed 0 --save core/data/rl/parking_policy_perpendicular_flanked.zip
```

**Measured results** (`core/validation/rl_comparison.py`, 5 seeds each, ground-truth
`ParkingEnv` evaluation vs. the real noisy `ParkingHarness` baseline under MPC):

| scenario | RL success | RL collision | RL mean steps | baseline success | baseline collision | baseline mean steps |
|---|---|---|---|---|---|---|
| `perpendicular_open` (no obstacles) | 100% | 0% | 65 | 100% | 0% | 261 |
| `perpendicular_flanked` (two flanking parked cars) | 100% | 0% | 65 | 100% | 0% | 365 |

Both policies reach the goal reliably and never collide, on both the obstacle-free
scenario and the one with real obstacles to route around -- and reach it in noticeably
fewer steps than the baseline in both cases, likely because they aren't bound by Hybrid
A*'s own path-following behavior or Pure Pursuit/MPC's tracking-error margins, free
instead to cut directly toward the goal. Both scenarios converged to a near-identical
`ep_rew_mean`/`ep_len_mean` during training (~109 / 65) -- plausible, not a training bug:
`perpendicular_flanked`'s obstacles sit off the direct start-to-goal line (y=2.5, while
the direct path stays near y=0-1), so a policy free to choose its own path never
actually needs a meaningfully different, longer route around them.

**Not evidence the learned policy is "better"** in any general sense: it never has to
handle sensor noise (trains and evaluates on ground truth -- see `rl_comparison.py`'s
own docstring for why this comparison isn't strictly apples-to-apples), and it's
untested on the project's genuinely tight scenarios (`parallel_between_cars`,
`perpendicular_obstructed_lane`) where a reverse-gear cusp or a materially longer
detour would be required -- a policy free to cut a direct diagonal line to the goal
hasn't been tested anywhere that's actually blocked from doing so.
