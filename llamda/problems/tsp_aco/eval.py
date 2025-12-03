# Adapted from ReEvo: https://github.com/ai4co/reevo/blob/main/problems/tsp_aco/eval.py
# Licensed under the MIT License (see THIRD-PARTY-LICENSES.txt)

import argparse
import logging
from os import path

import numpy as np
from aco import ACO  # type: ignore
from scipy.spatial import distance_matrix

from llamda.utils import load_heuristic_from_code


POSSIBLE_FUNC_NAMES = ["heuristics", "heuristics_v1", "heuristics_v2", "heuristics_v3"]

N_ITERATIONS = 100
N_ANTS = 30


def solve(node_pos, heuristics):
    dist_mat = distance_matrix(node_pos, node_pos)
    dist_mat[np.diag_indices_from(dist_mat)] = 1  # set diagonal to a large number
    heu = heuristics(dist_mat.copy()) + 1e-9
    heu[heu < 1e-9] = 1e-9
    aco = ACO(dist_mat, heu, n_ants=N_ANTS)
    obj = aco.run(N_ITERATIONS)
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

    basepath = path.join(path.dirname(__file__), "dataset")
    if not path.isfile(path.join(basepath, "train50_dataset.npy")):
        from gen_inst import generate_datasets

        generate_datasets()

    if mood == "train":
        dataset_path = path.join(basepath, f"{mood}{problem_size}_dataset.npy")
        node_positions = np.load(dataset_path)
        n_instances = node_positions.shape[0]
        print(f"[*] Dataset loaded: {dataset_path} with {n_instances} instances.")

        objs = []
        for i, node_pos in enumerate(node_positions):
            obj = solve(node_pos, heuristics)
            print(f"[*] Instance {i}: {obj}")
            objs.append(obj)

        print("[*] Average:")
        print(np.mean(objs))

    else:
        for problem_size in [20, 50, 100]:
            dataset_path = path.join(basepath, f"{mood}{problem_size}_dataset.npy")
            node_positions = np.load(dataset_path)
            logging.info(f"[*] Evaluating {dataset_path}")
            n_instances = node_positions.shape[0]
            objs = []
            for i, node_pos in enumerate(node_positions):
                obj = solve(node_pos, heuristics)
                objs.append(obj.item())
            print(f"[*] Average for {problem_size}: {np.mean(objs)}")
