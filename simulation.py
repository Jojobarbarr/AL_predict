import json
import pickle as pkl
from argparse import Namespace
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

import graphics
from experiment import Experiment
from genome import Genome
from mutations import Mutation
from utils import L_M, MUTATIONS, str_to_int


class Simulation(Experiment):
    """Handles the simulation experience.

    Attributes:
        generations (int): number of generations.
        plot_point (int): number of generations between two plots.
        population (int): number of individuals in the population.
        checkpoint (int): number of generations between two checkpoints.
        genomes (npt.NDArray[np.object_]): array of genomes.
        vec_genome_length (np.vectorize): vectorized version of the genome_length method.
        vec_blend_genomes (np.vectorize): vectorized version of the blend_genomes method.
        vec_clone (np.vectorize): vectorized version of the clone method.
        num_workers (int): number of workers for multiprocessing.
        mutation_rates (npt.NDArray[np.float32]): mutation rates.
        total_mutation_rate (float): sum of the mutation rates.
        biases_mutation (npt.NDArray[np.float32]): biases for the mutation rates.
        mutations (npt.NDArray[Mutation]): array of mutations.
    """

    def __init__(
        self,
        config: dict[str, Any],
        args: Namespace,
    ) -> None:
        """Simulation initialization.

        Args:
            config (dict[str, Any]): configuration.
            load_file (Path, optional): If provided, the population is loaded (quickier than creating it). Defaults to Path("").

        Raises:
            FileNotFoundError: If the file to load the population is not found, an exception is raised and execution stops.
        """
        super().__init__(config, args)

        ## Simulation initialization
        print("Initializing simulation")
        init_start = perf_counter()

        self.rng: np.random.Generator = np.random.default_rng()
        # self.rng: np.random.Generator = np.random.default_rng(42)

        self.plot_in_time = args.plot_in_time

        self.generations = str_to_int(self.simulation_config["Generations"])
        self.plot_number = str_to_int(self.simulation_config["Plot points"])
        self.plot_point = self.generations // self.plot_number

        self.population = str_to_int(self.simulation_config["Population size"])

        self.init_mutations()
        self.genomes = np.empty(self.population, dtype=Genome)

        self.homogeneous = self.genome_config["Homogeneous"]
        self.orientation = self.genome_config["Orientation"]
        self.g = str_to_int(self.genome_config["g"])
        self.z_c = self.set_z("z_c")

        # # https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
        # self.vec_get_genome_length = np.vectorize(self.get_genome_length)
        # self.vec_get_genome_z_nc = np.vectorize(self.get_genome_z_nc)
        # self.vec_compute_genomes_stats = np.vectorize(self.compute_genomes_stats)
        # self.vec_blend_genomes = np.vectorize(self.blend_genomes)
        # self.vec_clone = np.vectorize(self.clone)
        self.initial_generation: int = 0
        self.livings: np.ndarray[Any, np.dtype[np.bool_]] = np.ones(
            self.population, dtype=bool
        )
        if args.checkpoint == Path(""):
            self.init_genomes(args.load)
        else:
            self.livings, self.initial_generation = self.load_from_checkpoint(
                args.checkpoint
            )

        init_end = perf_counter()
        print(f"Simulation initialized in {init_end - init_start} s")

    def init_mutations(self):
        """Initialize the mutations."""
        mutation_types = self.mutations_config["Mutation types"]

        # Mutation rates
        self.mutation_rates = np.array(
            [
                float(self.mutation_rates_config[mutation_type])
                for mutation_type in mutation_types
            ],
            dtype=np.float32,
        )
        self.total_mutation_rate = sum(self.mutation_rates)
        self.biases_mutation = self.mutation_rates / self.total_mutation_rate

        # Mutations
        l_m = int(self.mutations_config["l_m"])
        self.mutations = np.array(
            [
                (
                    MUTATIONS[mutation_type](l_m=l_m, self.mutation_length_distribution)
                    if mutation_type in L_M
                    else MUTATIONS[mutation_type](self.mutation_length_distribution)
                )
                for mutation_type in mutation_types
            ],
            dtype=Mutation,
        )

    def init_genomes(self, load_file: Path) -> None:
        """Initialize the population generating it or loading an individual then cloning it.

        Args:
            load_file (Path): The file to load the individual from. If empty, population is generated.
        """
        # Load or create the population
        if load_file != Path(""):
            with open(load_file, "rb") as pkl_file:
                genome = pkl.load(pkl_file)

            self.genomes = np.array(
                [genome.clone() for _ in range(self.population)],
                dtype=Genome,
            )

        else:
            self.generate_genomes()

    def generate_genomes(self) -> None:
        """Generate the population of genomes according to the configuration file."""
        z_nc = self.set_z("z_nc")

        self.genomes = np.array(
            [
                Genome(self.g, self.z_c, z_nc, self.homogeneous, self.orientation)
                for _ in range(self.population)
            ],
            dtype=Genome,
        )

    def set_z(self, z_type: str) -> int:
        if z_type == "z_c":
            z_factor = str_to_int(self.genome_config["beta"])
        else:
            z_factor = str_to_int(self.genome_config["alpha"])
        if self.genome_config[f"{z_type}_auto"]:
            return z_factor * str_to_int(self.genome_config["g"])
        return str_to_int(self.genome_config[z_type])

    def clone(self, parent: Genome) -> Genome:
        """Clones the parent genome as a deep copy. This function is used through self.vec_clone, its vectorized equivalent.

        Args:
            parent (Genome): The parent genome.

        Returns:
            Genome: The child genome.
        """
        return parent.clone()

    def blend_genomes(self, genome: Genome) -> Genome:
        """Blend the genomes to homogenize the non coding lengths. This function is used through self.vec_blend_genomes, its vectorized equivalent.

        Args:
            genome (Genome): the genome to blend.
        """
        return genome.blend()

    def get_genome_z_nc(self, genome: Genome) -> int:
        """Get genome z_nc. This function is used through self.vec_get_genome_z_nc, its vectorized equivalent.

        Args:
            genome (Genome): The genome to get z_nc from.

        Returns:
            int: z_nc, number of non conding bases.
        """
        return genome.z_nc

    def compute_genomes_stats(self, genome: Genome) -> dict[str, Any]:
        """Compute the genome stats. This function is used through self.vec_compute_genomes_stats, its vectorized equivalent.

        Args:
            genome (Genome): The genome to compute stats from.

        Returns:
            Genome: The genome with computed stats.
        """
        genome.compute_stats()
        return genome.stats.d_stats

    def get_genome_length(self, genome: Genome) -> int:
        """Get genome length. This function is used through self.vec_get_genome_length, its vectorized equivalent.

        Args:
            genome (Genome): The genome to get length from.

        Returns:
            int: The genome length.
        """
        return genome.length

    def save_population(self, file: str):
        """Save the population in a pickle file.
        Args:
            file (str): pickle file name saved in self.save_path / "populations"
        """
        save_dir = self.save_path / "populations"
        save_dir.mkdir(parents=True, exist_ok=True)
        savefile = save_dir / file
        with open(savefile, "wb") as pkl_file:
            pkl.dump(self.genomes, pkl_file, protocol=pkl.HIGHEST_PROTOCOL)
        print(f"\nPopulation saved in {savefile}\n")

    def mutation_is_deleterious(
        self,
        mutation_event: Mutation,
        genome: Genome,
    ) -> tuple[bool, bool]:
        mutation_event.genome = genome
        if mutation_event.is_neutral():
            return (False, mutation_event.apply())
        return True, False

    def format_time(self, time: float) -> str:
        if time < 60:
            return f"{time:.0f} s"
        if time < 3600:
            return f"{time // 60:.0f} min {time % 60:.0f} s"
        if time < 86400:
            return f"{time // 3600:.0f} h {(time % 3600) // 60:.0f} min"
        return f"{time // 86400:.0f} j {(time % 86400) // 3600:.0f} h"

    def save_checkpoint(
        self,
        generation: int,
        livings: np.ndarray[Any, np.dtype[np.bool_]],
    ):
        save_dir = self.save_path / "checkpoints"
        save_dir.mkdir(parents=True, exist_ok=True)
        to_dump = {
            "self.genomes": self.genomes,
            "livings": livings,
            "generation": generation,
        }
        with open(save_dir / f"generation_{generation}.pkl", "wb") as pkl_file:
            pkl.dump(to_dump, pkl_file, protocol=pkl.HIGHEST_PROTOCOL)
        print()

    def load_from_checkpoint(
        self,
        path_to_checkpoint: Path,
    ) -> tuple[np.ndarray[Any, np.dtype[np.bool_]], int]:
        """Load the simulation from a checkpoint.

        Args:
            path_to_checkpoint (Path): The path to the pickle file containing the checkpoints information.

        Returns:
            tuple[np.ndarray[Any, np.dtype[np.bool_]], int]: the current living mask and the generation.
        """
        with open(path_to_checkpoint, "rb") as pkl_file:
            loaded = pkl.load(pkl_file)
        self.genomes = loaded["self.genomes"]
        livings = loaded["livings"]
        generation = loaded["generation"]
        return livings, generation

    def plot_mutation_info(self):
        dict_mutation_info = {}
        for mutation in self.mutations:
            with open(
                self.save_path / "mutation_info" / f"{str(mutation)}.json",
                "r",
                encoding="utf8",
            ) as json_file:
                dict_mutation_info[str(mutation)] = json.load(json_file)
        graphics.plot_mutation_info(
            dict_mutation_info, self.save_path / "mutation_info", "Total mutations"
        )
        graphics.plot_mutation_info(
            {
                mutation: neutral_proportion
                for mutation, neutral_proportion in dict_mutation_info.items()
                if mutation != "Deletion" and mutation != "Duplication"
            },
            self.save_path / "mutation_info",
            "Neutral mutations proportion",
        )
        graphics.plot_mutation_info(
            {
                mutation: neutral_proportion
                for mutation, neutral_proportion in dict_mutation_info.items()
                if mutation == "Deletion" or mutation == "Duplication"
            },
            self.save_path / "mutation_info",
            "Neutral mutations proportion",
            suffix="_deletion_and_duplication",
        )
        graphics.plot_mutation_info(
            {
                mutation: length_mean
                for mutation, length_mean in dict_mutation_info.items()
                if mutation != "Inversion"
            },
            self.save_path / "mutation_info",
            "Length mean",
        )
        graphics.plot_mutation_info(
            {
                mutation: length_mean
                for mutation, length_mean in dict_mutation_info.items()
                if mutation != "Inversion"
            },
            self.save_path / "mutation_info",
            "Length standard deviation",
        )
