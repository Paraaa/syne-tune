import itertools
import logging
from argparse import ArgumentParser
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
from benchmarking.baselines import (
    MethodArguments,
    methods,
)
from benchmarking.benchmarks import (
    benchmark_definitions,
)
from syne_tune.config_space import Float

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.blackbox_repository.simulated_tabular_backend import (
    BlackboxRepositoryBackend,
)
from syne_tune.backend.local_backend import LocalBackend
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from syne_tune.experiments import load_experiment


def run(
    method_names,
    benchmark_names,
    seeds,
    max_num_evaluations=None,
    n_workers: int = 4,
):
    logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend.simulator_backend.simulator_backend").setLevel(
        logging.WARNING
    )


    combinations = list(itertools.product(method_names, seeds, benchmark_names))

    print(f"Going to evaluate: {combinations}")
    exp_names = []
    for method, seed, benchmark_name in tqdm(combinations):

        if method in ["LLMKD"]:
            # continue 
            n_workers = 1

        if "ackley" in benchmark_name:
            config_space = {f"x{i}": Float(-32.768, 32.768) for i in range(20)}
        elif "hartmann3" in benchmark_name:
            config_space = {f"x{i}": Float(0.0, 1.0) for i in range(3)}
        elif "hartmann6" in benchmark_name:
            config_space = {f"x{i}": Float(0.0, 1.0) for i in range(6)}
        elif "levy" in benchmark_name:
            config_space = {f"x{i}": Float(-10.0, 10.0) for i in range(10)}
        elif "rastrigin" in benchmark_name:
            config_space = {f"x{i}": Float(-5.12, 5.12) for i in range(10)}
        elif "rosenbrock" in benchmark_name:
            config_space = {f"x{i}": Float(-2.048, 2.048) for i in range(8)}
        else:
            raise ValueError(f"Unknown benchmark name: {benchmark_name}")


        mode = "min"
        metric = "F1"

        np.random.seed(seed)

        print(f"Starting experiment ({method}/{benchmark_name}/{seed})")

        backend = LocalBackend(
            entry_point=f"./benchmarking/local_benchmarks/{benchmark_name}.py"
        )

        # 5 candidates initially to be evaluated
        num_random_candidates = 5
        random_state = np.random.RandomState(seed)
        points_to_evaluate = [
            {
                k: v.sample(random_state=random_state)
                for k, v in config_space.items()
            }
            for _ in range(num_random_candidates)
        ]
        scheduler = methods[method](
            MethodArguments(
                config_space=config_space,
                metric=metric,
                mode=mode,
                random_seed=seed,
                resource_attr="training_iteration",
                num_brackets=1,
                use_surrogates="lcbench" in benchmark_name,
                points_to_evaluate=points_to_evaluate,
            )
        )

        stop_criterion = StoppingCriterion(
            max_num_evaluations=max_num_evaluations,
        )
        tuner = Tuner(
            trial_backend=backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=n_workers,
            # metadata={
            #     "seed": seed,
            #     "algorithm": method,
            #     "benchmark": benchmark_name,
            # },
        )
        tuner.run()
        exp_names.append(tuner.name)
        save_results(
            tuner,
            method=method,
            metric=metric,
            seed=seed,
            config_space=config_space,
            benchmark_name=benchmark_name,
        )
    return exp_names


def save_results(tuner, method, metric, seed, config_space, benchmark_name):

    df = load_experiment(tuner.name).results
    configs = []
    runtime_traj = []
    F1 = []

    for _, trial_df in df.groupby("trial_id"):
        runtime_traj.append(float(trial_df.st_tuner_time.iloc[-1]))
        F1.append(trial_df[metric].values[-1])
        config = {}
        for hyper in config_space.keys():
            c = trial_df.iloc[0]["config_" + hyper]
            config[hyper] = c
        configs.append(config)
    result = {
        "configs": configs,
        "runtime_traj": runtime_traj,
        "F1": F1,
        metric: F1, # Saving the same thing with its metric name
    }
    if method == "LLMKD":
        method = "LLMKD-k-10"
    results = pd.DataFrame(result)
    dir = f"./results/{benchmark_name}/{method}/observed_fvals/"

    os.makedirs(dir, exist_ok=True)
    results.to_csv(f"{dir}/{method}_{metric}_{seed}.csv", index=False)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        required=False,
        default=0,
        help="seed to run",
    )
    parser.add_argument(
        "--run_all_seeds",
        type=int,
        required=False,
        default=0,
        help="If 1 runs all seeds between [0, args.seed] if 0 run only args.seed.",
    )

    parser.add_argument(
        "--method",
        type=str,
        required=False,
        help="a method to run from baselines.py, run all by default.",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        required=False,
        help="a benchmark to run from benchmarks.py, run all by default.",
    )
    parser.add_argument(
        "--n_workers",
        help="number of workers to use when tuning.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--max_num_evaluations",
        help="number of evaluations to use when tuning.",
        type=int,
        default=None,
    )

    args, _ = parser.parse_known_args()
    if args.run_all_seeds:
        seeds = list(range(args.seed))
    else:
        seeds = [args.seed]
    method_names = [args.method] if args.method is not None else list(methods.keys())
    
    benchmark_definitions = ["rastrigin", "hartmann3", "hartmann6", "levy", "ackley", "rosenbrock"]

    benchmark_names = (
        [args.benchmark]
        if args.benchmark is not None
        else benchmark_definitions
    )

    run(
        method_names=method_names,
        benchmark_names=benchmark_names,
        seeds=seeds,
        n_workers=args.n_workers,
        max_num_evaluations=args.max_num_evaluations,
    )

