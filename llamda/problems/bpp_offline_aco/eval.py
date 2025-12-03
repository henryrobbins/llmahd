# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/problems/bpp_offline_aco/eval.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import os
import logging
import argparse

import numpy as np

from aco import ACO  # type: ignore
from gen_inst import BPPInstance, load_dataset, dataset_conf  # type: ignore
from llamda.utils import load_heuristic_from_code


POSSIBLE_FUNC_NAMES = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]


N_ITERATIONS = 15
N_ANTS = 20
SAMPLE_COUNT = 200


def solve(inst: BPPInstance, heuristics, mode="sample"):
    heu = heuristics(inst.demands.copy(), inst.capacity)  # normalized in ACO
    assert tuple(heu.shape) == (inst.n, inst.n)
    assert 0 < heu.max() < np.inf
    aco = ACO(
        inst.demands,
        heu.astype(float),
        capacity=inst.capacity,
        n_ants=N_ANTS,
        greedy=False,
    )
    if mode == "sample":
        obj, _ = aco.sample_only(SAMPLE_COUNT)
    else:
        obj, _ = aco.run(N_ITERATIONS)
    return obj


if __name__ == "__main__":

    print("[*] Running ...")

    parser = argparse.ArgumentParser()
    parser.add_argument("problem_size", type=int)
    parser.add_argument("mood", type=str, choices=["train", "val"])
    parser.add_argument(
        "method", type=str, nargs="?", default="aco", choices=["sample", "aco"]
    )
    parser.add_argument(
        "--code-path", type=str, required=True, help="Path to individual's code file"
    )
    args = parser.parse_args()

    problem_size = args.problem_size
    mood = args.mood
    method = args.method
    heuristics = load_heuristic_from_code(args.code_path, POSSIBLE_FUNC_NAMES)

    basepath = os.path.dirname(__file__)
    # automacially generate dataset if nonexists
    if not os.path.isfile(
        os.path.join(basepath, f"dataset/train{dataset_conf['train'][0]}_dataset.npz")
    ):
        from gen_inst import generate_datasets

        generate_datasets()

    if mood == "train":
        dataset_path = os.path.join(
            basepath, f"dataset/{mood}{problem_size}_dataset.npz"
        )
        dataset = load_dataset(dataset_path)
        n_instances = len(dataset)

        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")

        objs = []
        for i, instance in enumerate(dataset):
            obj = solve(instance, heuristics, mode=method)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj)

        print("[*] Average:")
        print(np.mean(objs))

    else:  # mood == 'val'
        for problem_size in dataset_conf["val"]:
            dataset_path = os.path.join(
                basepath, f"dataset/{mood}{problem_size}_dataset.npz"
            )
            dataset = load_dataset(dataset_path)
            n_instances = dataset[0].n
            logging.info(f"[*] Evaluating {dataset_path}")

            objs = []
            for i, instance in enumerate(dataset):
                obj = solve(instance, heuristics, mode=method)
                objs.append(obj)

            print(f"[*] Average for {problem_size}: {np.mean(objs)}")
