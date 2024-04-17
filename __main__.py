import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

from mutagenese import Mutagenese
from wright_fisher import WrightFisher

if __name__ == "__main__":
    plt.rcParams.update(
        {"figure.max_open_warning": 0}
    )  # To disable warning when plotting using multithreading.

    arg_parser = argparse.ArgumentParser(
        prog="AL_predict",
        description="Evolution model from a math model about genome structure.",
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
        default=".",
        help="If used, the initial population will be loaded from a specified file (must contain only one individual).",
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
    arg_parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        default=".",
        help="If used, the simulation will be loaded from a checkpoint and saved in the first replica folder, unless a replica is specified with the -r option.",
    )
    arg_parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="If used, the program will print more regularly the advancement of the simulation. Can slow down the simulation.",
    )

    args = arg_parser.parse_args()

    with open(args.config_file, "r", encoding="utf8") as json_file:
        config = json.load(json_file)

    if config["Experiment"]["Type"] == "Mutagenese":
        experiment = Mutagenese(config, args)

    elif config["Experiment"]["Type"] == "Simulation":

        if config["Simulation"]["Replication model"] == "Wright-Fisher":
            experiment = WrightFisher(config, args)
        elif config["Simulation"]["Replication model"] == "Moran":
            raise NotImplementedError(
                "Moran model is not implemented. Maybe you can do it ;-) ?"
            )

        if args.save:
            experiment.save_population("initial_population.pkl")

    experiment.run(only_plot=args.only_plot)
