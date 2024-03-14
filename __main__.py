import argparse

import json
from pathlib import Path
import matplotlib.pyplot as plt


from mutagenese import Mutagenese
from simulation import Simulation
from wright_fisher import WrightFisher

if __name__ == "__main__":
    plt.rcParams.update(
        {"figure.max_open_warning": 0}
    )  # To disable warning when plotting using multithreading.

    arg_parser = argparse.ArgumentParser(
        prog="AL_predict",
        description="Evolution model from a math model over genome structure.",
    )

    arg_parser.add_argument(
        "config_file",
        type=Path,
        help="Configuration file for the experiments, relative paths from the folder where the command is launch or absolute path.",
    )
    arg_parser.add_argument(
        "-p",
        "--only_plot",
        action="store_true",
        help="If used, the execution will only execute the plotting part.",
    )
    arg_parser.add_argument(
        "-s",
        "--save",
        action="store_true",
        help="If used, the initial population will be saved in a file.",
    )
    arg_parser.add_argument(
        "-l",
        "--load",
        type=Path,
        default="",
        help="If used, the initial population will be loaded from a specified file.",
    )
    arg_parser.add_argument(
        "-t",
        "--plot_in_time",
        action="store_true",
        help="If used, the plotting will be done during execution.",
    )
    arg_parser.add_argument(
        "-o",
        "--overwrite",
        action="store_true",
        help="If used, the save directory will be overwritten.",
    )

    args = arg_parser.parse_args()

    with open(args.config_file, "r", encoding="utf8") as json_file:
        config = json.load(json_file)

    if config["Experiment"]["Type"] == "Mutagenese":
        experiment = Mutagenese(config)

    elif config["Experiment"]["Type"] == "Simulation":
        if config["Simulation"]["Replication model"] == "Wright-Fisher":
            experiment = WrightFisher(
                config, args.load, args.plot_in_time, args.overwrite
            )
        elif config["Simulation"]["Replication model"] == "Moran":
            pass

    if args.save:
        experiment.save_population("initial_population.pkl")

    experiment.run(
        only_plot=args.only_plot,
    )
