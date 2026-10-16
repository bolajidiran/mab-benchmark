# Plug-ins

Environments (datasets and simulators), algorithms and metrics are looked up by name
in `mab_benchmark.registry`. Add your own at run time:

```python
from mab_benchmark import registry
registry.register_algorithm("my-ucb", MyUCB, defaults={"c": 2.0})
registry.register_environment("my-logs", MyLoggedBandit, kind="stationary", defaults={"K": 40, "T": 5000})
registry.register_metric("p90_regret", lambda regrets, info: float(np.percentile(regrets, 90)))
```

or ship them in your own package through entry points, without editing mab-benchmark:

```toml
[project.entry-points."mab_benchmark.algorithms"]
my-ucb = "my_package.algs:MyUCB"

[project.entry-points."mab_benchmark.environments"]
my-logs = "my_package.envs:MyLoggedBandit"     # set BENCHMARK_KIND = "contextual" etc. on the class

[project.entry-points."mab_benchmark.metrics"]
p90_regret = "my_package.metrics:p90_regret"
```

Built-in names are never replaced by an entry point.

**Contracts.** An environment is constructed as `Env(**params, seed=seed)` and has
attributes `K`, `T`, methods `pull` and `regret_increment` (taking the round `t` when
its kind is `nonstationary` or `contextual`), `context(t)` when contextual, and
`expected_random_regret()` for NCR. An algorithm subclasses `BanditAlgorithm` and is
constructed as `Alg(n_arms=K, **params)` (plus `context_dim` when contextual). A metric
is `f(regrets, info) -> float`, where `info` has `K`, `T`, `n_runs` and
`random_regret` (per run, or `None`).

Algorithms included beyond the baselines: `FG-TS`, `KL-UCB` (Garivier & Cappe, 2011),
`SW-TS` (sliding-window Thompson sampling; Trovo et al., 2020), `LinTS` (Agrawal &
Goyal, 2013) and `Lin`, the greedy linear model with epsilon = 0.
