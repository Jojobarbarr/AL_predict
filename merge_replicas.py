from pathlib import Path
from utils import str_to_int
import json
import numpy as np
import pickle as pkl
from argparse import ArgumentParser
from graphics import plot_merge_replicas


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to the config file")
    parser.add_argument(
        "-i", "--skip_incomplete", action="store_true", help="Skip incomplete replicas"
    )
    parser.add_argument(
        "-p",
        "--precise",
        action="store_true",
        help="Print the mean of the last (1-MEAN_MASK) generations",
    )
    parser.add_argument(
        "-m",
        "--mean_mask",
        type=float,
        default=0.75,
        help="Proportion of generations to compute the mean on",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf8") as json_file:
        config = json.load(json_file)

    generations = str_to_int(config["Simulation"]["Generations"])
    plot_points = int(generations // str_to_int(config["Simulation"]["Plot points"]))
    result_dir = Path(config["Paths"]["Save"])
    population = str_to_int(config["Simulation"]["Population size"])
    replicas = result_dir.glob("*")
    folders = [
        folder.stem for folder in replicas if folder.is_dir() and folder.stem[0] != "_"
    ]

    x_values = np.array(
        [generation for generation in range(0, generations + 1, plot_points)],
        dtype=np.int64,
    )
    # folders = ["1", "2", "3", "4"]
    non_coding_proportion = np.zeros((len(folders), len(x_values)))
    livings = np.zeros((len(folders), len(x_values)))
    min_generation = np.inf
    for replica_index, replica in enumerate(folders):
        replica_dir = result_dir / replica
        try:
            for generation_index, generation in enumerate(x_values):
                if generation == 0:
                    generation = 1
                with open(
                    replica_dir / "logs" / f"generation_{generation}.pkl", "rb"
                ) as pkl_file:
                    try:
                        stats = pkl.load(pkl_file)
                    except EOFError:
                        print(
                            f"EOFError while loading generation {generation}, last generation could be corrupted."
                        )
                        break
                non_coding_proportion[replica_index, generation_index] = np.array(
                    [genome["Non coding proportion"] for genome in stats["genome"]]
                ).mean()
                livings[replica_index, generation_index] = stats["population"][
                    "Living child proportion"
                ]
        except FileNotFoundError:
            non_coding_proportion[replica_index, generation_index:] = np.nan
            livings[replica_index, generation_index:] = np.nan
            min_generation = min(min_generation, generation)
    min_generation = min(min_generation, generations)
    with open(
        result_dir / "_iterative_model" / "config.json", "r", encoding="utf8"
    ) as json_file:
        config = json.load(json_file)
        iterations = config["Iterations"]
        time_acceleration = config["Time acceleration"]

    x_iterative_model = np.array(
        [x * time_acceleration for x in range(0, iterations + 1)]
    )
    x_iterative_model = x_iterative_model[
        x_iterative_model - time_acceleration <= min_generation
    ]
    nc_proportions_iterative_model = np.load(
        result_dir / "_iterative_model" / "nc_proportions.npy", allow_pickle=True
    )[: len(x_iterative_model)]
    Nes_iterative_model = np.load(
        result_dir / "_iterative_model" / "Nes.npy", allow_pickle=True
    )[: len(x_iterative_model)]
    livings_iterative_model = Nes_iterative_model / population

    save_dir = result_dir / "_plots"
    plot_merge_replicas(
        x_values,
        non_coding_proportion.mean(axis=0),
        None,
        x_iterative_model,
        nc_proportions_iterative_model,
        save_dir,
        "Non coding proportion",
    )

    plot_merge_replicas(
        x_values,
        livings.mean(axis=0),
        None,
        x_iterative_model,
        livings_iterative_model,
        save_dir,
        "Living children proportion",
    )

    print(
        f"Mean non coding proportion value once equilibrium is reached: {non_coding_proportion[:, int(args.mean_mask * len(x_values)):].mean()}"
    )
    print(
        f"Mean living children proportion value once equilibrium is reached: {livings[:, int(args.mean_mask * len(x_values)):].mean()}"
    )
    print(
        f"Be careful, these values are computed on the last {100 * (1 - args.mean_mask)}% generations. Check that equilibrium is reached before."
    )
