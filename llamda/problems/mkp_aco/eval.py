# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/problems/mkp_aco/eval.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import os
import argparse
import logging

import numpy as np
import torch
from aco import ACO  # type: ignore

from llamda.utils import load_heuristic_from_code


POSSIBLE_FUNC_NAMES = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

N_ITERATIONS = 50
N_ANTS = 10


def solve(prize: np.ndarray, weight: np.ndarray, heuristics):
    n, m = weight.shape
    heu = heuristics(prize.copy(), weight.copy()) + 1e-9
    assert heu.shape == (n,)
    heu[heu < 1e-9] = 1e-9
    aco = ACO(
        torch.from_numpy(prize), torch.from_numpy(weight), torch.from_numpy(heu), N_ANTS
    )
    obj, _ = aco.run(N_ITERATIONS)
    return obj


if __name__ == "__main__":

    print("[*] Running ...")

    parser = argparse.ArgumentParser()
    parser.add_argument("problem_size", type=int)
    parser.add_argument("mood", type=str, choices=["train", "val"])
    parser.add_argument(
        "--code-path", type=str, required=True, help="Path to individual's code file"
    )
    args = parser.parse_args()

    problem_size = args.problem_size
    mood = args.mood
    heuristics = load_heuristic_from_code(args.code_path, POSSIBLE_FUNC_NAMES)

    basepath = os.path.dirname(__file__)
    # automacially generate dataset if nonexists
    if not os.path.isfile(os.path.join(basepath, f"dataset/train50_dataset.npz")):
        from gen_inst import generate_datasets

        generate_datasets()

    if mood == "train":
        dataset_path = os.path.join(
            basepath, f"dataset/{mood}{problem_size}_dataset.npz"
        )
        dataset = np.load(dataset_path)
        prizes, weights = dataset["prizes"], dataset["weights"]
        n_instances = prizes.shape[0]

        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")

        objs = []
        for i, (prize, weight) in enumerate(zip(prizes, weights)):
            obj = solve(prize, weight, heuristics)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj.item())

        print("[*] Average:")
        print(np.mean(objs))

    else:  # mood == 'val'
        for problem_size in [100, 300, 500]:
            dataset_path = os.path.join(
                basepath, f"dataset/{mood}{problem_size}_dataset.npz"
            )
            dataset = np.load(dataset_path)
            prizes, weights = dataset["prizes"], dataset["weights"]
            n_instances = prizes.shape[0]
            logging.info(f"[*] Evaluating {dataset_path}")

            objs = []
            for i, (prize, weight) in enumerate(zip(prizes, weights)):
                obj = solve(prize, weight, heuristics)
                objs.append(obj.item())

            print(f"[*] Average for {problem_size}: {np.mean(objs)}")
