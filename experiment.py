import shutil
from argparse import Namespace
from pathlib import Path
from typing import Any


class Experiment:
    """Base class for the experiments, it contains the configuration of the experiment.

    Attributes:
        mutations_config: Configuration of the mutations.
        genome_config: Configuration of the genome.
        mutation_rates_config: Configuration of the mutation rates.
        mutagenese_config: Configuration of the mutagenese.
        simulation_config: Configuration of the simulation.
        home_dir: Path to the home directory.
        save_path: Path to the save directory.
        checkpoints_path: Path to the checkpoints directory.
    """

    def __init__(
        self,
        config: dict[str, Any],
        args: Namespace,
    ) -> None:
        """Initialize the experiment with the configuration and the arguments.

        Args:
            config (dict[str, Any]): The configuration file of the experiment.
            args (Namespace): The arguments of the replication.
        """
        self.verbose = args.verbose
        self.mutations_config = config["Mutations"]
        self.genome_config = config["Genome"]
        self.mutation_rates_config = config["Mutation rates"]
        self.mutagenese_config = config["Mutagenese"]
        self.simulation_config = config["Simulation"]

        self.checkpoints_path = args.checkpoint
        self.save_path: Path = Path(config["Paths"]["Save"])
        self.create_save_directory(args.overwrite, args.only_plot)
        self.write_readme(config, args)

    def create_save_directory(
        self,
        overwrite: bool,
        only_plot: bool,
    ) -> None:
        """Create the save directory according to the previous experiment and the arguments.

        Args:
            overwrite (bool): If True, the save directory will overwrite the previous one.
            only_plot (bool): If True, the save directory will be the previous one but no overwrite should occur.
        """
        folders = self.save_path.glob("*")
        folders = [
            int(folder.stem)
            for folder in folders
            if folder.is_dir() and folder.stem[0] != "_"
        ] + [0]
        last_folder = max(folders)  # folders is never empty beacause of ' + [0]'

        if overwrite and last_folder > 0:
            self.save_path /= str(last_folder)
            shutil.rmtree(self.save_path)
            self.save_path.mkdir()
            print(f"Save folder is: {self.save_path} (Overwriting a previous folder)")

        elif only_plot and last_folder > 0:
            self.save_path /= str(last_folder)
            print(f"Save folder is: {self.save_path} (Existent folder)")

        else:
            self.save_path /= str(last_folder + 1)
            self.save_path.mkdir(parents=True)
            print(f"Save folder is: {self.save_path} (Creating a new folder)")

    def write_readme(
        self,
        config: dict[str, Any],
        args: Namespace,
    ) -> None:
        """Write the README.md file with the configuration and the arguments.

        Args:
            config (dict[str, Any]): The configuration file of the experiment.
            args (Namespace): The arguments of the replication.
        """
        with open(self.save_path / "README.md", "w", encoding="utf8") as readme:
            readme.write("# Experiment\n")
            readme.write("## Configuration\n")
            for key, value in config.items():
                readme.write(f"### {key}\n")
                if isinstance(value, dict):
                    for key2, value2 in value.items():
                        readme.write(f"#### {key2}\n")
                        readme.write(f"{value2}\n")
                else:
                    readme.write(f"{value}\n")
                readme.write("\n")
            readme.write("## Arguments\n")
            for key, value in vars(args).items():
                readme.write(f"### {key}\n")
                readme.write(f"{value}\n")
                readme.write("\n")
