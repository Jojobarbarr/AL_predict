from typing import Any
from pathlib import Path
import re
import shutil


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
        self, config: dict[str, Any], overwrite: bool = False, only_plot: bool = False
    ):
        self.mutations_config = config["Mutations"]
        self.genome_config = config["Genome"]
        self.mutation_rates_config = config["Mutation rates"]
        self.mutagenese_config = config["Mutagenese"]
        self.simulation_config = config["Simulation"]

        self.checkpointing = config["Paths"]["Checkpointing"]
        self.checkpoint_number = config["Paths"]["Checkpoint number"]
        self.checkpoints_path = Path(config["Paths"]["Checkpoint"])
        self.save_path: Path = Path(config["Paths"]["Save"])
        self.create_save_directory(overwrite, only_plot)

    def create_save_directory(self, overwrite: bool = False, only_plot: bool = False):
        folders = self.save_path.glob("*")
        folders = [int(folder.stem) for folder in folders if folder.is_dir()] + [0]
        last_folder = max(folders)
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


#     def create_save_directory(
#         self,
#         overwrite: bool = False,
#         only_plot: bool = False,
#     ):
#         """Create the save directory."""
#         folders = self.save_path.glob("./[0-9]/")
#         folders = list(folders)
#         if len(folders) > 0:
#             last_replica = max([folder for folder in folders], key=extract_number)
#             replica_number, folder = extract_number(last_replica)
#             if overwrite:
#                 shutil.rmtree(folder)
#                 self.save_path = folder
#                 self.save_path.mkdir()
#                 print(
#                     f"Save folder is: {self.save_path} (Overwriting a previous folder)"
#                 )
#             elif only_plot:
#                 self.save_path = folder.parent / str(replica_number)
#                 print(f"Save folder is: {self.save_path} (Existent folder)")
#             else:
#                 self.save_path = folder.parent / str(replica_number + 1)
#                 self.save_path.mkdir()
#                 print(f"Save folder is: {self.save_path} (Creating a new folder)")
#         else:
#             save_name = "1"
#             self.save_path = self.save_path / save_name
#             self.save_path.mkdir(parents=True)
#             print(f"Save folder is: {self.save_path} (Creating a new folder)")


# def extract_number(folder):
#     print(folder.name)
#     replica_number = re.findall(r"\d+$", folder.name)
#     print(replica_number)
#     return (int(replica_number[0]) if replica_number else -1, folder)
