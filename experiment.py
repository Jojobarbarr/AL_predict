from typing import Any
from pathlib import Path


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

    def __init__(self, config: dict[str, Any]):
        self.mutations_config = config["Mutations"]
        self.genome_config = config["Genome"]
        self.mutation_rates_config = config["Mutation rates"]
        self.mutagenese_config = config["Mutagenese"]
        self.simulation_config = config["Simulation"]

        self.home_dir = Path(config["Paths"]["Home"])
        self.save_path = Path(config["Paths"]["Save"])
        self.checkpoints_path = Path(config["Paths"]["Checkpoint"])
        del config
