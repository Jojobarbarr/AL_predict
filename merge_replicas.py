from pathlib import Path
from utils import str_to_int
import json
import numpy as np
import pickle as pkl
from argparse import ArgumentParser
from graphics import plot_simulation


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to the config file")
    parser.add_argument(
        "-i", "--skip_incomplete", action="store_true", help="Skip incomplete replicas"
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf8") as json_file:
        config = json.load(json_file)

    generations = str_to_int(config["Simulation"]["Generations"])
    plot_points = generations // str_to_int(config["Simulation"]["Plot points"])
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
    non_coding_proportion = np.zeros((len(folders), len(x_values)))
    livings = np.zeros((len(folders), len(x_values)))

    inclompete_mask = np.zeros(len(folders), dtype=bool)
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
            if args.skip_incomplete:
                inclompete_mask[replica_index] = True

    save_dir = result_dir / "_plots"
    plot_simulation(
        x_values,
        non_coding_proportion[~inclompete_mask].mean(axis=0),
        None,
        save_dir,
        "Non coding proportion",
        ylim=-1,
    )
    plot_simulation(
        x_values,
        livings[~inclompete_mask].mean(axis=0),
        None,
        save_dir,
        "Living children proportion",
        ylim=-1,
    )
