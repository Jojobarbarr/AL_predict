from tqdm import tqdm
import multiprocessing as mp
import json
import concurrent.futures
import pickle as pkl
import random as rd
from collections import defaultdict
from configparser import ConfigParser
from pathlib import Path
from time import perf_counter
import numpy as np
import numpy.typing as npt

import graphics
from experiment import Experiment
from genome import Genome
from mutations import Mutation
from utils import MUTATIONS, str_to_int, L_M

PROFILE = False


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

    def __init__(self, config: ConfigParser, load_file: Path = Path("")) -> None:
        """Simulation initialization.

        Args:
            config (ConfigParser): configuration.
            load_file (Path, optional): If provided, the population is loaded (quickier than creating it). Defaults to Path("").

        Raises:
            FileNotFoundError: If the file to load the population is not found, an exception is raised and execution stops.
        """
        super().__init__(config)

        # ## Multiprocessing initialization
        # self.init_multiprocessing()

        self.replication_model = self.simulation_config["Replication model"]

        ## Simulation initialization
        print("Initializing simulation")
        init_start = perf_counter()

        self.generations = str_to_int(self.simulation_config["Generations"])
        self.plot_point = self.generations // int(self.simulation_config["Plot points"])

        self.population = str_to_int(self.simulation_config["Population size"])

        self.init_mutations()
        self.genomes = np.empty(self.population, dtype=Genome)
        if self.checkpoints_path == Path(""):
            self.init_genomes(load_file)

        self.saving_checkpoint = self.simulation_config["Save checkpoints"]
        self.checkpoint = str_to_int(self.simulation_config["Checkpoints"])

        # Vectorize the clone method to apply it efficiently to the whole population.
        # https://numpy.org/doc/stable/reference/generated/numpy.vectorize.html
        self.vec_genome_length = np.vectorize(self.genome_length)
        self.vec_compute_genomes_stats = np.vectorize(self.compute_genomes_stats)
        self.vec_blend_genomes = np.vectorize(self.blend_genomes)
        self.vec_clone = np.vectorize(self.clone)

        init_end = perf_counter()
        print(f"Simulation initialized in {init_end - init_start} s")

    def init_multiprocessing(self):
        self.num_workers = min(mp.cpu_count(), 32)
        self.num_workers = 8

        mp.set_start_method("fork")

    def init_mutations(self):
        mutation_types = self.mutations_config["Mutation types"]

        # Mutation rates
        self.mutation_rates = np.array(
            [
                float(self.mutation_rates_config[f"{mutation_type} rate"])
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
                    MUTATIONS[mutation_type](l_m=l_m)
                    if mutation_type in L_M
                    else MUTATIONS[mutation_type]()
                )
                for mutation_type in mutation_types
            ],
            dtype=Mutation,
        )

    def init_genomes(self, load_file: Path):
        self.blend = self.simulation_config["Blend"]

        # Load or create the population
        if load_file != Path(""):
            try:
                print(f"Loading population {load_file}")
                with open(load_file, "rb") as pkl_file:
                    self.genomes = pkl.load(pkl_file)
                print(f"Population {load_file} loaded")
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"File {load_file} not found. Please provide a valid file."
                ) from exc
        else:
            print("Creating population")
            self.generate_genomes()
            print("Population created")

    def generate_genomes(self):
        homogeneous = self.genome_config["Homogeneous"]
        orientation = self.genome_config["Orientation"]
        g = str_to_int(self.genome_config["g"])
        z_c = self.set_z("z_c")
        z_nc = self.set_z("z_nc")

        self.genomes = np.array([Genome(g, z_c, z_nc, homogeneous, orientation) for _ in range(self.population)], dtype=Genome)  # type: ignore TODO: optim?

    def set_z(self, z_type: str) -> int:
        z_factor = str_to_int(self.genome_config[f"{z_type}_factor"])
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

    def run(
        self,
        only_plot: bool = False,
    ):
        if not only_plot:
            main_start_time = perf_counter()

            # Flag to forget first generation in the time performance computation.
            first_gen_not_passed = True

            time_perfs = []
            sum_time_perfs = 0
            if self.checkpoints_path != Path(""):
                self.load_from_checkpoint(
                    self.simulation_config["Generation Checkpoint"]
                )
            if self.replication_model == "Wright-Fisher":
                went_to_last_gen = True
                last_gen_with_data = self.generations
                # Initialize the set of genomes to ensure every genome stats will be computed at the first generation.
                genomes_changed = np.ones(self.population, dtype=bool)
                genomes_lengths = np.array(
                    [genome.length for genome in self.genomes], dtype=np.int32
                )
                total_bases_number = genomes_lengths.sum()
                biases_genomes = np.array(
                    [
                        genome_length / total_bases_number
                        for genome_length in genomes_lengths
                    ],
                    np.float32,
                )
                parents_indices = [0 for _ in range(self.population)]
                for genome in self.genomes:
                    genome.compute_stats()

                for generation in range(1, self.generations + 1):
                    start_time = perf_counter()
                    try:
                        (
                            genomes_changed,
                            genomes_lengths,
                            biases_genomes,
                            total_bases_number,
                            parents_indices,
                        ) = self.generation_step_wright_fisher(
                            generation,
                            genomes_changed,
                            genomes_lengths,
                            biases_genomes,
                            total_bases_number,
                            parents_indices,
                        )
                    except RuntimeError as exc:
                        print(exc)
                        went_to_last_gen = False
                        last_gen_with_data = generation - 1
                        break
                    end_time = perf_counter()
                    time_perfs.append(end_time - start_time)
                    if generation % self.plot_point == 0:
                        if first_gen_not_passed:
                            # the first gen is longer than the others because of the compute_stats() method applied to all genomes.
                            time_perfs = time_perfs[1:]
                            first_gen_not_passed = False

                        sum_over_last_generations = sum(time_perfs)
                        generation_elapsed = len(time_perfs)
                        try:
                            average_time_perf_over_last_gens = (
                                sum_over_last_generations / generation_elapsed
                            )
                        except ZeroDivisionError:
                            average_time_perf_over_last_gens = sum_over_last_generations
                        sum_time_perfs += sum_over_last_generations
                        try:
                            average_time_perf = sum_time_perfs / (generation - 1)
                        except ZeroDivisionError:
                            average_time_perf = sum_time_perfs

                        print(
                            f"\nGeneration {generation}"
                            f" - Mean elapsed time by generation: {average_time_perf:.3f} s/generation"
                            f" - Last {generation_elapsed} generations: {average_time_perf_over_last_gens:.3f} s/generation"
                        )
                        print(
                            f"{self.format_time(end_time - main_start_time)} elapsed since the beginning of the simulation "
                            f"- \033[1mEstimated remaining time: {self.format_time(average_time_perf * (self.generations - generation))}\033[0m"
                        )
                        time_perfs = []

                    if (
                        self.simulation_config["Save checkpoints"]
                        and generation % self.checkpoint == 0
                    ):
                        self.save_checkpoint(
                            generation,
                            genomes_changed,
                            genomes_lengths,
                            biases_genomes,
                            total_bases_number,
                        )

                print(f"Generation {generation} - End of simulation")
                self.plot_simulation_wright_fisher(
                    went_to_last_gen, last_gen_with_data, only_plot
                )

            elif self.replication_model == "Moran":
                for iteration in range(1, self.generations + 1):
                    start_time = perf_counter()

                    self.iteration_step_moran(iteration)

                    end_time = perf_counter()
                    time_perfs.append(end_time - start_time)
                    if iteration % self.plot_point == 0:
                        if first_gen_not_passed:
                            # the first gen is longer than the others because of the compute_stats() method applied to all genomes.
                            time_perfs = time_perfs[1:]
                            first_gen_not_passed = False

                        sum_over_last_generations = sum(time_perfs)
                        generation_elapsed = len(time_perfs)
                        average_time_perf_over_last_gens = (
                            sum_over_last_generations / generation_elapsed
                        )
                        sum_time_perfs += sum_over_last_generations
                        average_time_perf = sum_time_perfs / (iteration - 1)
                        print(
                            f"\nIteration {iteration}"
                            f" - Mean elapsed time by generation: {average_time_perf:.3f} s/iteration"
                            f" - Last {generation_elapsed} iterations: {average_time_perf_over_last_gens:.3f} s/iteration"
                        )
                        print(
                            f"{self.format_time(end_time - main_start_time)} elapsed since the beginning of the simulation "
                            f"- \033[1mEstimated remaining time: {self.format_time(average_time_perf * (self.generations - iteration))}\033[0m"
                        )
                        time_perfs = []

                print(
                    f"Iteration {iteration} (~Generation {iteration // self.population})- End of simulation"
                )
                self.plot_simulation_moran(only_plot)

    def iteration_step_moran(self, iteration: int):
        dead_genome_index = rd.choice(range(len(self.genomes)))
        living = False
        structure_has_changed = False
        replication_counter = 0
        while not living:
            replication_counter += 1
            son_index = rd.choice(range(len(self.genomes)))
            if son_index != dead_genome_index:
                son = self.genomes[son_index]
                mutation_number = rd.binomialvariate(
                    son.length, p=self.total_mutation_rate
                )
                mutation_events = rd.choices(self.mutations, k=mutation_number)
                for mutation_event in mutation_events:
                    if self.mutation_is_deleterious(mutation_event, son_index):
                        break
                    structure_has_changed = True
                living = True
        if self.blend and structure_has_changed:
            son = son.blend()
        self.genomes[dead_genome_index] = son.clone()

        if iteration % self.plot_point == 0 or iteration == 1:
            z_nc_array = np.array(
                [genome.z_nc for genome in self.genomes], dtype=np.int_
            )
            z_nc_array.sort()

            self.genomes = self.vec_compute_genomes_stats(self.genomes)
            genomes_stats = np.array(
                [genome.stats.d_stats for genome in self.genomes], dtype=dict
            )

            population_stats = {
                "z_nc_array": z_nc_array,
            }
            graphics.save_checkpoint(
                self.save_path / "logs", genomes_stats, population_stats, iteration
            )

    def generation_step_wright_fisher(
        self,
        generation: int,
        previous_changed_genomes_mask: npt.NDArray[np.bool_],
        genomes_lengths: npt.NDArray[np.int_],
        genomes_biases: npt.NDArray[np.float_],
        total_bases_number: int,
        parents_indices: list[int],
    ) -> tuple[
        npt.NDArray[np.bool_],
        npt.NDArray[np.int_],
        npt.NDArray[np.float_],
        int,
        list[int],
    ]:
        if generation % self.plot_point == 0 or generation == 1:
            z_nc_array = np.array(
                [genome.z_nc for genome in self.genomes], dtype=np.int_
            )
            z_nc_array.sort()

        # All individuals are alive at the beginning of a generation step.
        living_genomes_mask = np.ones(self.population, dtype=bool)
        # List that will store all the genomes affected by a neutral deletion that changed their structure.
        changed_genomes_mask = np.zeros(self.population, dtype=bool)

        (
            mutation_events,
            mutant_genomes,
            genomes_lengths,
            genomes_biases,
            total_bases_number,
        ) = self.compute_random_trial(
            previous_changed_genomes_mask,
            genomes_lengths,
            genomes_biases,
            total_bases_number,
        )

        # The first mutation that was drawn affects the first genome that was drawn...
        # As both are random, it doesn't introduce bias.
        for mutation_event, genome_index in zip(mutation_events, mutant_genomes):
            if living_genomes_mask[genome_index]:
                if self.mutation_is_deleterious(mutation_event, genome_index):
                    living_genomes_mask[genome_index] = False
                else:
                    # the mutation changed the structure of the genome.
                    changed_genomes_mask[genome_index] = True

        changed_genomes_mask = np.logical_and(changed_genomes_mask, living_genomes_mask)

        if not living_genomes_mask.any():
            raise RuntimeError(
                f"Generation {generation} - All individuals are dead.\n"
                f"Last checkpoint at generation: {generation - ((generation - 1) % self.plot_point)}"
            )

        # Wright-Fisher model: random draw with replacement of individuals. Population size is constant.
        parents_indices = rd.choices(
            range(len(self.genomes[living_genomes_mask])), k=self.population
        )

        # Create the set to map the parents with a structure change to their son (if any)
        next_generation_structure_change_mask = np.zeros(self.population, dtype=bool)
        for son_index, parent_index in enumerate(parents_indices):
            if changed_genomes_mask[parent_index]:
                next_generation_structure_change_mask[son_index] = True

        if self.blend:
            if changed_genomes_mask.any():
                self.genomes[changed_genomes_mask] = self.vec_blend_genomes(
                    self.genomes[changed_genomes_mask]
                )

        if generation % self.plot_point == 0 or generation == 1:
            # Statistics only needs to be compute if structure has changed.
            ##
            self.genomes = self.vec_compute_genomes_stats(self.genomes)
            genomes_stats = np.array(
                [genome.stats.d_stats for genome in self.genomes], dtype=dict
            )

            count_offspring = defaultdict(lambda: 0)
            count_offspring_alives = defaultdict(lambda: 0)
            for son_index, parent_index in enumerate(parents_indices):
                count_offspring[parent_index] += 1
                if living_genomes_mask[son_index]:
                    count_offspring_alives[parent_index] += 1
            effective_fitness_array = np.array(
                [
                    count_offspring_alives[parent_index] / count_offspring[parent_index]
                    for parent_index in parents_indices
                ],
                dtype=np.int_,
            )

            living_percentage = living_genomes_mask.sum() / self.population * 100
            structure_changed_percentage = (
                changed_genomes_mask.sum() / self.population * 100
            )

            population_stats = {
                "Living percentage": living_percentage,
                "Structure changed percentage": structure_changed_percentage,
                "z_nc_array": z_nc_array,
                "Effective fitness array": effective_fitness_array,
            }
            graphics.save_checkpoint(
                self.save_path / "logs", genomes_stats, population_stats, generation
            )
            if self.saving_checkpoint:
                if generation % (self.checkpoint + 1) == 0:
                    self.save_checkpoint(
                        generation,
                        changed_genomes_mask,
                        genomes_lengths,
                        genomes_biases,
                        total_bases_number,
                    )

        # As several individuals maybe clones, must ensure every instance is independant, a copy.
        self.genomes = self.vec_clone(
            self.genomes[living_genomes_mask][parents_indices]
        )

        return (
            next_generation_structure_change_mask,
            genomes_lengths,
            genomes_biases,
            total_bases_number,
            parents_indices,
        )

    def compute_random_trial(
        self,
        previous_changed_genomes_mask: npt.NDArray[np.bool_],
        genomes_lengths: npt.NDArray[np.int_],
        genomes_biases: npt.NDArray[np.float_],
        total_bases_number: int,
    ):
        # The lengths are only computed for genomes that were changed by a mutation.
        if previous_changed_genomes_mask.any():
            changed_genomes_lengths = self.vec_genome_length(
                self.genomes[previous_changed_genomes_mask]
            )
            deltas = (
                changed_genomes_lengths - genomes_lengths[previous_changed_genomes_mask]
            )
            delta = deltas.sum()
            total_bases_number += delta
            genomes_lengths[previous_changed_genomes_mask] = changed_genomes_lengths

            if delta != 0:
                genomes_biases = genomes_lengths / total_bases_number
            else:
                # The normalized length are only computed for genomes that were changed by a mutation.
                genomes_biases[previous_changed_genomes_mask] = (
                    genomes_lengths[previous_changed_genomes_mask] / total_bases_number
                )

        # Determine the total number of event with a binomial law over all the bases.
        mutation_number = rd.binomialvariate(
            total_bases_number, p=self.total_mutation_rate
        )
        # Determine which mutations happens with a biased (over mutation rate) choice with replacement.
        mutation_events = np.random.choice(
            self.mutations, size=mutation_number, p=self.biases_mutation
        )
        # Determine whiwh genomes are affected with a biased (over the genome length) choice with replacement.
        mutant_genomes = np.random.choice(
            range(self.population), size=mutation_number, p=genomes_biases
        )
        return (
            mutation_events,
            mutant_genomes,
            genomes_lengths,
            genomes_biases,
            total_bases_number,
        )

    def compute_genomes_stats(self, genome: Genome) -> Genome:
        genome.compute_stats()
        return genome

    def genome_length(self, genome: Genome) -> int:
        return genome.length

    def mutation_is_deleterious(
        self,
        mutation_event: Mutation,
        genome_index: int,
    ) -> bool:
        mutation_event.genome = self.genomes[genome_index]
        if mutation_event.is_neutral():
            mutation_event.apply()
            return False
        return True

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
        genomes_changed: npt.NDArray[np.bool_],
        genomes_lengths: npt.NDArray[np.int_],
        biases_genomes: npt.NDArray[np.float_],
        total_bases_number: int,
    ):
        self.checkpoints_path.mkdir(parents=True, exist_ok=True)
        to_dump = {
            "self.genomes": self.genomes,
            "genomes_changed": genomes_changed,
            "genomes_lengths": genomes_lengths,
            "biases_genomes": biases_genomes,
            "total_bases_number": total_bases_number,
        }
        with open(
            self.checkpoints_path / f"generation_{generation}.pkl", "wb"
        ) as pkl_file:
            pkl.dump(to_dump, pkl_file, protocol=pkl.HIGHEST_PROTOCOL)
        print()

    def load_from_checkpoint(
        self, generation
    ) -> tuple[
        npt.NDArray[np.bool_], npt.NDArray[np.int_], npt.NDArray[np.float_], int
    ]:
        if generation < 0:
            files = self.checkpoints_path.glob("*.pkl")
            generation = max([int(file.stem.split("_")[1]) for file in files])
        with open(
            self.checkpoints_path / f"generation_{generation}.pkl", "rb"
        ) as pkl_file:
            loaded = pkl.load(pkl_file)
        self.genomes = loaded["self.genomes"]
        genomes_changed = loaded["genomes_changed"]
        genomes_lengths = loaded["genomes_lengths"]
        biases_genomes = loaded["biases_genomes"]
        total_bases_number = loaded["total_bases_number"]
        return (genomes_changed, genomes_lengths, biases_genomes, total_bases_number)

    def plot_simulation_wright_fisher(
        self,
        went_to_last_gen: bool = True,
        last_gen_with_data: int = 0,
        only_plot: bool = False,
    ):
        g = str_to_int(self.genome_config["g"])
        if went_to_last_gen:
            x_values = np.array(
                [
                    generation
                    for generation in range(0, self.generations + 1, self.plot_point)
                ],
                dtype=np.int64,
            )
        else:
            x_values = np.array(
                [
                    generation
                    for generation in range(0, last_gen_with_data + 1, self.plot_point)
                ],
                dtype=np.int64,
            )

        genomes_non_coding_proportion_means = np.empty(len(x_values), dtype=np.float32)
        genomes_non_coding_proportion_vars = np.empty(len(x_values), dtype=np.float32)

        genomes_nc_lengths_means = np.empty((len(x_values), g), dtype=np.float32)
        genomes_nc_lengths_vars = np.empty((len(x_values), g), dtype=np.float32)

        population_living_percentages = np.empty(len(x_values), dtype=np.float32)

        population_struc_change_percentages = np.empty(len(x_values), dtype=np.float32)

        population_z_ncs = np.empty((len(x_values), self.population), dtype=np.float32)

        population_effective_fitnesss = np.empty(len(x_values), dtype=np.float32)

        for index, generation in tqdm(
            enumerate(x_values),
            "Plotting generations",
            len(x_values),
            unit=" plot points",
        ):
            if generation == 0:
                generation = 1
            with open(
                self.save_path / "logs" / f"generation_{generation}.pkl", "rb"
            ) as pkl_file:
                stats = pkl.load(pkl_file)

            genome_raw_stats = stats["genome"]
            population_raw_stats = stats["population"]

            genomes_non_coding_proportion = np.array(
                [genome["Non coding proportion"] for genome in genome_raw_stats],
                dtype=np.float32,
            )
            genomes_non_coding_proportion_mean = np.mean(genomes_non_coding_proportion)
            genomes_non_coding_proportion_var = (
                np.var(genomes_non_coding_proportion)
                * len(genomes_non_coding_proportion)
                / (len(genomes_non_coding_proportion) - 1)
            )
            genomes_non_coding_proportion_means[index] = (
                genomes_non_coding_proportion_mean
            )
            genomes_non_coding_proportion_vars[index] = (
                genomes_non_coding_proportion_var
            )

            genomes_nc_lengths = np.array(
                [genome["Non coding length list"] for genome in genome_raw_stats],
                dtype=np.ndarray,
            )
            genomes_nc_lengths_mean = np.mean(genomes_nc_lengths, axis=0)
            genomes_nc_lengths_var = (
                np.var(genomes_nc_lengths, axis=0)
                * len(genomes_nc_lengths)
                / (len(genomes_nc_lengths) - 1)
            )
            genomes_nc_lengths_means[index] = genomes_nc_lengths_mean
            genomes_nc_lengths_vars[index] = genomes_nc_lengths_var

            population_living_percentages[index] = population_raw_stats[
                "Living percentage"
            ]

            population_z_ncs[index] = population_raw_stats["z_nc_array"]

            population_struc_change_percentages[index] = population_raw_stats[
                "Structure changed percentage"
            ]

            population_effective_fitnesss[index] = np.mean(
                population_raw_stats["Effective fitness array"]
            )

        save_dir = self.save_path / "plots"
        graphics.plot_simulation(
            x_values,
            genomes_non_coding_proportion_means,
            genomes_non_coding_proportion_vars,
            save_dir,
            "Non coding proportion",
        )
        graphics.plot_simulation(
            x_values,
            population_living_percentages,
            None,
            save_dir,
            "Population living percentage",
        )
        graphics.plot_simulation(
            x_values,
            population_struc_change_percentages,
            None,
            save_dir,
            "Population structure changed percentage",
        )
        graphics.plot_simulation(
            x_values,
            population_effective_fitnesss,
            None,
            save_dir,
            "Population effective fitness",
        )

        for stat_list_mean, stat_list_var, name in zip(
            (genomes_nc_lengths_means, population_z_ncs),
            (genomes_nc_lengths_vars, None),
            (
                "Genomes non coding lengths",
                "Population z_nc",
            ),
        ):
            for quantile in (0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 1):
                index = int((g - 1) * quantile)
                mean_list = stat_list_mean[:, index]
                var_list = (
                    stat_list_var[:, index] if stat_list_var is not None else None
                )
                graphics.plot_simulation(
                    x_values,
                    mean_list,
                    var_list,
                    save_dir,
                    f"{name} - {quantile * 100}th percentile",
                )
        if not only_plot:
            save_mutation_info = self.save_path / "mutation_info"
            save_mutation_info.mkdir(parents=True, exist_ok=True)
            for mutation in self.mutations:
                mutation.stats.compute()
                with open(
                    save_mutation_info / f"{str(mutation)}.json", "w", encoding="utf8"
                ) as json_file:
                    json.dump(mutation.stats.d_stats, json_file, indent=2)
            self.plot_mutation_info()

        print("Simulation plots saved ! ;-)")

        print("\n\n")
        print("Plotting generation plots...")
        # threads = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
            futures = [
                executor.submit(self.plot_one_gen, generation)
                for generation in x_values
            ]
            for _ in tqdm(
                concurrent.futures.as_completed(futures),
                desc="Plotting generations",
                total=len(x_values),
                unit=" plot points",
            ):
                pass
        # for index, generation in tqdm(
        #     enumerate(x_value),
        #     "Plotting generations",
        #     len(x_value),
        #     unit=" plot points",
        # ):
        #     thread = threading.Thread(target=self.plot_one_gen, args=(generation,))
        #     threads.append(thread)
        #     thread.start()

        # for thread in threads:
        #     thread.join()

    def plot_one_gen(self, generation: int):
        if generation == 0:
            generation = 1
        with open(
            self.save_path / "logs" / f"generation_{generation}.pkl", "rb"
        ) as pkl_file:
            stats = pkl.load(pkl_file)

        genome_raw_stats = stats["genome"]
        population_raw_stats = stats["population"]

        genomes_non_coding_proportion = np.array(
            [genome["Non coding proportion"] for genome in genome_raw_stats],
            dtype=np.float32,
        )

        graphics.plot_generation(
            genomes_non_coding_proportion,
            generation,
            0.49995,
            0.50005,
            1000,
            "Non coding proportion",
            self.save_path / "generation_plots",
        )

        genomes_nc_lengths = np.array(
            [genome["Non coding length list"] for genome in genome_raw_stats],
            dtype=np.ndarray,
        )
        genomes_nc_lengths_mean = np.mean(genomes_nc_lengths, axis=0)
        graphics.plot_generation(
            genomes_nc_lengths_mean,
            generation,
            950,
            1050,
            1000,
            "Genomes non coding lengths",
            self.save_path / "generation_plots",
        )

        population_z_nc = population_raw_stats["z_nc_array"]
        graphics.plot_generation(
            population_z_nc,
            generation,
            1e6 - 100,
            1e6 + 100,
            500,
            "Population z_nc",
            self.save_path / "generation_plots",
        )

        population_effective_fitness = population_raw_stats["Effective fitness array"]
        graphics.plot_generation(
            population_effective_fitness,
            generation,
            0.99995,
            1.00005,
            1000,
            "Population effective fitness",
            self.save_path / "generation_plots",
        )
        return generation

    def plot_simulation_moran(
        self,
        only_plot: bool = False,
    ):
        g = str_to_int(self.genome_config["g"])
        x_values = np.array(
            [
                iteration
                for iteration in range(0, self.generations + 1, self.plot_point)
            ],
            dtype=np.int64,
        )

        genomes_non_coding_proportion_means = np.empty(len(x_values), dtype=np.float32)
        genomes_non_coding_proportion_vars = np.empty(len(x_values), dtype=np.float32)

        genomes_nc_lengths_means = np.empty((len(x_values), g), dtype=np.float32)
        genomes_nc_lengths_vars = np.empty((len(x_values), g), dtype=np.float32)

        population_z_ncs = np.empty((len(x_values), self.population), dtype=np.float32)

        for index, iteration in tqdm(
            enumerate(x_values),
            "Plotting generations",
            len(x_values),
            unit=" plot points",
        ):
            if iteration == 0:
                iteration = 1
            with open(
                self.save_path / "logs" / f"generation_{iteration}.pkl", "rb"
            ) as pkl_file:
                stats = pkl.load(pkl_file)

            genome_raw_stats = stats["genome"]
            population_raw_stats = stats["population"]

            genomes_non_coding_proportion = np.array(
                [genome["Non coding proportion"] for genome in genome_raw_stats],
                dtype=np.float32,
            )
            genomes_non_coding_proportion_mean = np.mean(genomes_non_coding_proportion)
            genomes_non_coding_proportion_var = (
                np.var(genomes_non_coding_proportion)
                * len(genomes_non_coding_proportion)
                / (len(genomes_non_coding_proportion) - 1)
            )
            genomes_non_coding_proportion_means[index] = (
                genomes_non_coding_proportion_mean
            )
            genomes_non_coding_proportion_vars[index] = (
                genomes_non_coding_proportion_var
            )

            genomes_nc_lengths = np.array(
                [genome["Non coding length list"] for genome in genome_raw_stats],
                dtype=np.ndarray,
            )
            genomes_nc_lengths_mean = np.mean(genomes_nc_lengths, axis=0)
            genomes_nc_lengths_var = (
                np.var(genomes_nc_lengths, axis=0)
                * len(genomes_nc_lengths)
                / (len(genomes_nc_lengths) - 1)
            )
            genomes_nc_lengths_means[index] = genomes_nc_lengths_mean
            genomes_nc_lengths_vars[index] = genomes_nc_lengths_var

            population_z_ncs[index] = population_raw_stats["z_nc_array"]

        save_dir = self.save_path / "plots"
        graphics.plot_simulation(
            x_values,
            genomes_non_coding_proportion_means,
            genomes_non_coding_proportion_vars,
            save_dir,
            "Non coding proportion",
        )

        for stat_list_mean, stat_list_var, name in zip(
            (genomes_nc_lengths_means, population_z_ncs),
            (genomes_nc_lengths_vars, None),
            (
                "Genomes non coding lengths",
                "Population z_nc",
            ),
        ):
            for quantile in (0, 0.1, 0.25, 0.33, 0.5, 0.66, 0.75, 0.9, 1):
                index = int((g - 1) * quantile)
                mean_list = stat_list_mean[:, index]
                var_list = (
                    stat_list_var[:, index] if stat_list_var is not None else None
                )
                graphics.plot_simulation(
                    x_values,
                    mean_list,
                    var_list,
                    save_dir,
                    f"{name} - {quantile * 100}th percentile",
                )
        if not only_plot:
            save_mutation_info = self.save_path / "mutation_info"
            save_mutation_info.mkdir(parents=True, exist_ok=True)
            for mutation in self.mutations:
                mutation.stats.compute()
                with open(
                    save_mutation_info / f"{str(mutation)}.json", "w", encoding="utf8"
                ) as json_file:
                    json.dump(mutation.stats.d_stats, json_file, indent=2)
            self.plot_mutation_info()

        print("Simulation plots saved ! ;-)")

        # print("\n\n")
        # print("Plotting generation plots...")
        # # threads = []
        # with concurrent.futures.ThreadPoolExecutor(max_workers=50) as executor:
        #     futures = [
        #         executor.submit(self.plot_one_gen, generation)
        #         for generation in x_values
        #     ]
        #     for _ in tqdm(
        #         concurrent.futures.as_completed(futures),
        #         desc="Plotting generations",
        #         total=len(x_values),
        #         unit=" plot points",
        #     ):
        #         pass

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

    def mutation_is_deleterious_parallel(
        self,
        mutation_event: Mutation,
        genome_index: int,
    ) -> bool:
        mutation_event.genome = self.genomes[genome_index]
        if mutation_event.is_neutral():
            mutation_event.apply()
            return False
        return True

    def process_chunk(
        self,
        mutation_events: list[Mutation],
        genome_indices: list[int],
        living_genomes: npt.NDArray[np.bool_],
        dead_genomes_queue: mp.Queue,
        genomes_changed_queue: mp.Queue,
        process_queue: mp.Queue,
    ) -> None:
        living_genomes_indices = living_genomes.copy()
        genomes_changed = set()
        # start = perf_counter()
        for mutation_event, genome_index in zip(mutation_events, genome_indices):
            if living_genomes[genome_index]:
                if self.mutation_is_deleterious_parallel(mutation_event, genome_index):
                    living_genomes_indices[genome_index] = False
                else:
                    genomes_changed.add(genome_index)
        # end = perf_counter()
        # print(f"Process {mp.current_process().name} - Elapsed time: {end - start} s - {len(genome_indices)/ (end - start)} genomes/s")

        dead_genomes_queue.put(living_genomes_indices)
        genomes_changed_queue.put(genomes_changed)
        process_queue.put("DONE")

    def generation_step_parallel(
        self,
        generation: int,
        genomes_changed: set[int],
        genomes_lengths: npt.NDArray[np.int_],
        biases_genomes: npt.NDArray[np.float_],
        total_bases_number: int,
    ) -> tuple[set[int], npt.NDArray[np.int_], npt.NDArray[np.float_], int]:
        # All individuals are alive at the beginning of a generation step.
        living_genomes = np.ones(self.population, dtype=bool)

        # list that will store all the genomes affected by a neutral deletion that changed their structure.
        structure_change_genome = set()

        # The lengths are only computed for genomes that were changed by a mutation.
        delta = 0
        for genome_changed in genomes_changed:
            new_length = self.genomes[genome_changed].length
            delta += new_length - genomes_lengths[genome_changed]
            genomes_lengths[genome_changed] = new_length
        del genomes_changed

        total_bases_number += delta

        # The normalized length are only computed for genomes that were changed by a mutation.
        biases_genomes = genomes_lengths / total_bases_number

        # Determine the total number of event with a binomial law over all the bases.
        mutation_number = rd.binomialvariate(
            total_bases_number, p=self.total_mutation_rate
        )

        # Determine which mutations happens with a biased (over mutation rate) choice with replacement.
        mutation_events = np.random.choice(
            self.mutations, size=mutation_number, p=self.biases_mutation
        )

        # Determine whiwh genomes are affected with a biased (over the genome length) choice with replacement.
        mutant_genomes = np.random.choice(
            range(self.population), size=mutation_number, p=biases_genomes
        )
        # print(f"Mutation number: {mutation_number}")
        # Parallel computation
        chunk_size = mutation_number // self.num_workers
        dead_genomes_queue = mp.Queue()
        genome_changed_queue = mp.Queue()
        process_queue = mp.Queue()

        processes = []
        for process_index in range(self.num_workers):
            start = process_index * chunk_size

            if process_index == self.num_workers - 1:
                stop = mutation_number
            else:
                stop = (process_index + 1) * chunk_size

            process = mp.Process(
                target=self.process_chunk,
                args=(
                    mutation_events[start:stop],
                    mutant_genomes[start:stop],
                    living_genomes,
                    dead_genomes_queue,
                    genome_changed_queue,
                    process_queue,
                ),
            )
            processes.append(process)
            process.start()
        del mutation_events
        del mutant_genomes

        counter = 0
        while counter < self.num_workers:
            if not process_queue.empty():
                process_queue.get()
                counter += 1

        dead_genomes_queue_size = dead_genomes_queue.qsize()
        for _ in range(dead_genomes_queue_size):
            dead_genomes_indices = ~dead_genomes_queue.get()
            living_genomes[dead_genomes_indices] = False
        del dead_genomes_indices

        genomes_changed_queue_size = genome_changed_queue.qsize()
        for _ in range(genomes_changed_queue_size):
            changed_genomes = genome_changed_queue.get()
            structure_change_genome.update(changed_genomes)
        del changed_genomes

        for process in processes:
            process.join()

        if not living_genomes.any():
            raise RuntimeError(
                f"Generation {generation} - All individuals are dead.\n"
                f"Last checkpoint at generation: {generation - ((generation - 1) % self.plot_point)}"
            )

        if generation % self.plot_point == 0 or generation == 1:
            if generation == 1:
                for genome in self.genomes:
                    genome.compute_stats()
            else:
                # Statistics only needs to be compute if structure has changed.
                for genome_index in structure_change_genome:
                    self.genomes[genome_index].compute_stats()  # type: ignore

            genomes_stats = [
                genome.stats.d_stats for genome in self.genomes[living_genomes]
            ]

            living_percentage = living_genomes.sum() / self.population * 100

            z_nc_list = sorted([genome.z_nc for genome in self.genomes[living_genomes]])
            z_nc_min = z_nc_list[0]
            z_nc_max = z_nc_list[-1]
            z_nc_median = z_nc_list[len(z_nc_list) // 2]

            population_stats = {
                "Living percentage": living_percentage,
                "z_nc min": z_nc_min,
                "z_nc max": z_nc_max,
                "z_nc median": z_nc_median,
            }

            graphics.save_checkpoint(
                self.save_path, genomes_stats, population_stats, generation
            )

        # Wright-Fisher model: random draw with replacement of individuals. Population size is constant.
        parents_indices = rd.choices(
            range(len(self.genomes[living_genomes])), k=self.population
        )

        # Create the set to map the parents with a structure change to their son (if any)
        next_generation_structure_change = set()
        for son_index, parent_index in enumerate(parents_indices):
            if parent_index in structure_change_genome:
                next_generation_structure_change.add(son_index)
        del structure_change_genome

        ####
        # for name, size in sorted(((name, sys.getsizeof(value))
        #                           for name, value in list(locals().items())),
        #                           key= lambda x: -x[1])[:]:
        #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        # print('\n\n')
        # for name, size in sorted(((name, sys.getsizeof(value))
        #                           for name, value in list(globals().items())),
        #                           key= lambda x: -x[1])[:]:
        #     print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

        # As several individuals maybe clones, must ensure every instance is independant, a copy.
        if self.blend:
            self.vec_blend_genomes(self.genomes[living_genomes][parents_indices])
        self.genomes = self.vec_clone(self.genomes[living_genomes][parents_indices])
        return (
            next_generation_structure_change,
            genomes_lengths,
            biases_genomes,
            total_bases_number,
        )
