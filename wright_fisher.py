import json
from copy import deepcopy
import pickle as pkl
from collections import Counter
from pathlib import Path
from time import perf_counter
from multiprocessing import Process
import traceback
from typing import Any
import numpy as np

import graphics
from genome import Genome
from mutations import Mutation
from utils import str_to_int, QUANTILES
from simulation import Simulation


class WrightFisher(Simulation):
    def __init__(
        self,
        config: dict[str, Any],
        load_file: Path = Path(""),
        plot_in_time: bool = False,
        overwrite: bool = False,
        only_plot: bool = False,
    ):
        super().__init__(
            config,
            load_file,
            plot_in_time=plot_in_time,
            overwrite=overwrite,
            only_plot=only_plot,
        )

        self.mutant_parent_indices_counter = Counter()

        self.uniform_mutation: bool = self.check_uniform_mutation()
        self.mutation_predraw_size: int = int(1e7)  # memory overhead very low.

        self.mutations_applied: list[Mutation] = self.draw_mutations()

        self.plot_process = Process(target=lambda: None)
        # self.chunk_number = cpu_count()

    def check_uniform_mutation(self) -> bool:
        """Check if the mutation distribution is uniform.

        Returns:
            bool: True if the mutation distribution is uniform, False otherwise.
        """
        first_rate = self.mutation_rates[0]
        for rate in self.mutation_rates:
            if rate != first_rate:
                return False
        return True

    def draw_mutations(self) -> list[Mutation]:
        """Draw a large number of mutation to avoid drawing them at each generation.

        Returns:
            list[Mutation]: list of reference to Mutation objects.
        """
        if self.uniform_mutation:
            # numpy.random.Generator.choice without setting p is much faster than setting p.
            return list(
                self.rng.choice(self.mutations, size=self.mutation_predraw_size)
            )
        return list(
            self.rng.choice(
                self.mutations, size=self.mutation_predraw_size, p=self.biases_mutation
            )
        )

    def run(
        self,
        only_plot: bool = False,
    ):
        """Main function of the Wright_Fisher simulation, iterates through generations.

        Args:
            only_plot (bool, optional): if True, skip all the processing part, and only calls plot_simulation function. Defaults to False.
        """
        if not only_plot:
            if self.checkpointing:
                print("Loading from checkpoint...")
                livings = self.load_from_checkpoint(
                    self.simulation_config["Generation Checkpoint"]
                )
                print("Checkpoint loaded.")
            else:
                for genome in self.genomes:
                    genome.compute_stats()
                livings: np.ndarray[Any, np.dtype[np.bool_]] = np.ones(
                    self.population, dtype=bool
                )

            if self.plot_in_time:
                self.plot_process.start()

            time_perfs = []
            sum_time_perfs = 0
            main_start_time = perf_counter()

            try:  # Ensure that results will be saved even if an error occurs, including KeyBoardInterrupt.
                for generation in range(1, self.generations + 1):
                    start_time = perf_counter()
                    livings = self.generation_step(generation, livings)
                    end_time = perf_counter()
                    time_perfs.append(end_time - start_time)
                    if generation % self.plot_point == 0:
                        sum_time_perfs = self.print_time_perfs(
                            time_perfs, sum_time_perfs, generation
                        )
                        time_perfs = []

            except KeyboardInterrupt:
                print("*" * 50)
                print(
                    "KeyboardInterrupt while main loop execution, exiting main loop and continuing execution, next KeyboardInterrupt will stop the program"
                )
                print("*" * 50)
                generation -= 1
            except (IndexError, ValueError, ZeroDivisionError) as err:
                print("*" * 50)
                print("Unexpected error, trying to finish simulation...")
                print("Error:")
                print(err)
                traceback.print_exc()
                print("*" * 50)
                generation -= 1
            except RuntimeError as err:
                print("*" * 50)
                print(err)
                print("Trying to plot simulation...")
                print("*" * 50)
                generation -= 1

            if self.plot_process.is_alive():
                self.plot_process.join()
            self.save_checkpoint(generation, livings)
            self.save_population("final")

            print(f"Generation {generation} - End of simulation")
            print(f"Total time: {self.format_time(perf_counter() - main_start_time)}")
            self.plot_simulation(generation)
        else:
            self.plot_simulation(self.generations, ylim=20000)

    def print_time_perfs(
        self,
        time_perfs: list[float],
        sum_time_perfs: float,
        generation: int,
    ) -> float:
        """Computes and prints the time performances of the simulation.

        Args:
            time_perfs (list[float]): List of the time performances of the last generations.
            sum_time_perfs (float): Sum of the time performances of all generations.
            generation (int): Current generation.

        Returns:
            float: the updated sum_time_perfs.
        """
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
            average_time_perf = sum_time_perfs / generation
        except ZeroDivisionError:
            average_time_perf = sum_time_perfs

        print(
            f"\nGeneration {generation}"
            f" - Mean elapsed time by generation: {average_time_perf:.4f} s/generation"
            f" - Last {generation_elapsed} generations: {average_time_perf_over_last_gens:.4f} s/generation"
        )
        print(
            f"{self.format_time(sum_time_perfs)} elapsed since the beginning of the simulation "
            f"- \033[1mEstimated remaining time: {self.format_time(average_time_perf * (self.generations - generation))}\033[0m"
        )
        return sum_time_perfs

    def generation_step(
        self,
        generation: int,
        livings: np.ndarray[Any, np.dtype[np.bool_]],
    ) -> np.ndarray[Any, np.dtype[np.bool_]]:
        """Computes one generation of the Wright-Fisher simulation.

        Args:
            generation (int): Generation number.
            livings (np.ndarray[Any, np.dtype[np.bool_]]): Mask of the living individuals.

        Raises:
            RuntimeError: All individuals died.

        Returns:
            np.ndarray[Any, np.dtype[np.bool_]]: living genome mask.
        """
        if not livings.all():
            self.prepare_parents(livings)
        mutation_per_genome = self.compute_mutation_number()

        livings = np.ones(len(self.genomes), dtype=bool)

        changed_counter = self.iterate_genomes(
            self.genomes, mutation_per_genome, livings
        )

        if not livings.any():
            raise RuntimeError(f"All individuals died on generation {generation}")

        if generation % self.plot_point == 0 or generation == 1:
            self.logging_and_plotting(livings, changed_counter, generation)

        return livings

    def prepare_parents(
        self,
        livings: np.ndarray[Any, np.dtype[np.bool_]],
    ) -> None:
        """Generate new population with living genomes and deepcopy parents that are duplicates.

        Args:
            livings (np.ndarray[Any, np.dtype[np.bool_]]): Living genomes mask
        """
        living_genomes: np.ndarray[Any, Any] = self.genomes[livings]
        double_number: int = self.population - livings.sum()
        clones: np.ndarray[Any, Any] = self.rng.choice(
            living_genomes, size=double_number
        )  # TODO: try to predraw the indices without introducing a bias.
        clone_parents: np.ndarray[Any, Any] = np.array(
            [genome.clone() for genome in clones]
        )
        self.genomes = np.concatenate((clone_parents, living_genomes))

    def compute_mutation_number(
        self,
    ) -> dict[int, np.ndarray]:
        """Computes the number of mutations to apply to each genome.

        Returns:
            dict[int, np.ndarray]: A dictionnary associating genomes and their mutations to apply if any.
        """
        # lengths = self.vec_get_genome_length(self.genomes)
        lengths = np.array([genome.length for genome in self.genomes])
        total_bases_number = lengths.sum()
        genomes_biases = lengths / total_bases_number
        mutation_number = self.rng.binomial(
            total_bases_number, self.total_mutation_rate
        )
        mutant_parent_indices = self.rng.choice(
            self.population, size=mutation_number, p=genomes_biases
        )

        # unique_mutant_parent_indices, counts = np.unique(
        #     mutant_parent_indices, return_counts=True
        # )

        # mutations = list(
        #     self.rng.choice(
        #         self.mutations, size=mutation_number, p=self.biases_mutation
        #     )
        # )
        # mutation_per_genome = {
        #     mutant_parent_index: np.array([mutations.pop() for _ in range(count)])
        #     for mutant_parent_index, count in zip(unique_mutant_parent_indices, counts)
        # }

        if len(self.mutations_applied) < mutation_number:
            self.mutations_applied = self.draw_mutations()

        # mutation_per_genome = {
        #     mutant_parent_index: np.array(
        #         [self.mutations_applied.pop() for _ in range(count)]
        #     )
        #     for mutant_parent_index, count in zip(unique_mutant_parent_indices, counts)
        # }

        # This seems faster (only a few seconds on the test used), TODO: check with other tests.
        self.mutant_parent_indices_counter.clear()
        self.mutant_parent_indices_counter.update(mutant_parent_indices)
        mutation_per_genome: dict[int, np.ndarray] = {
            mutant_parent_index: np.array(
                [self.mutations_applied.pop() for _ in range(mutation_applied)]
            )
            for mutant_parent_index, mutation_applied in self.mutant_parent_indices_counter.items()
        }
        return mutation_per_genome

    def iterate_genomes(
        self,
        parents: np.ndarray[Any, Any],
        mutation_per_genome: dict[int, np.ndarray],
        livings: np.ndarray[Any, np.dtype[np.bool_]],
    ) -> int:
        """Iterates through genomes.

        Args:
            parents (np.ndarray[Any, Any]): Parents genomes to be mutate and replicate.
            mutation_per_genome (dict[int, np.ndarray]): Dictionnary associating genomes and their mutations to apply if any.
            livings (np.ndarray[Any, np.dtype[np.bool_]]): Mask of living genomes.

        Returns:
            float: The count of genomes that changed.
        """
        changed_counter = 0
        for genome_index in range(len(parents)):
            structure_changed_at_least_once = False
            if genome_index in mutation_per_genome:
                for mutation in mutation_per_genome[genome_index]:
                    dead, structure_changed = self.mutation_is_deleterious(
                        mutation, self.genomes[genome_index]
                    )
                    if dead:
                        livings[genome_index] = False
                        break
                    if structure_changed:
                        structure_changed_at_least_once = True
                        if self.homogeneous:
                            self.genomes[genome_index].blend()
            if livings[genome_index] and structure_changed_at_least_once:
                changed_counter += 1
        return changed_counter

    def logging_and_plotting(
        self,
        livings: np.ndarray[Any, np.dtype[np.bool_]],
        changed_counter: int,
        generation: int,
    ) -> None:
        # z_nc_array = self.vec_get_genome_z_nc(self.genomes[livings])
        z_nc_array = np.array([genome.z_nc for genome in self.genomes[livings]])
        z_nc_array.sort()
        quantile_idx = np.array(
            [int(quantile * (len(z_nc_array) - 1)) for quantile in QUANTILES],
            dtype=int,
        )
        z_nc_array = z_nc_array[quantile_idx]
        # genomes_stats = self.vec_compute_genomes_stats(self.genomes[livings])
        genomes_stats = np.array(
            [self.compute_genomes_stats(genome) for genome in self.genomes[livings]]
        )

        living_proportion = livings.sum() / self.population
        change_proportion = changed_counter / self.population
        population_stats = {
            "Living child proportion": living_proportion,
            "Changed proportion": change_proportion,
            "z_nc_array": z_nc_array,
        }
        graphics.save_checkpoint(
            self.save_path / "logs", genomes_stats, population_stats, generation
        )

        if self.plot_in_time:
            self.plot_process.join()
            self.plot_process = Process(
                target=self.plot_simulation,
                args=(generation, True),
            )
            self.plot_process.start()

    def plot_simulation(
        self,
        generation: int,
        on_execution: bool = False,
        ylim: int = -1,
    ):
        g = str_to_int(self.genome_config["g"])
        x_values = np.array(
            [generation for generation in range(0, generation + 1, self.plot_point)],
            dtype=int,
        )

        genomes_non_coding_proportion_means = np.empty(len(x_values), dtype=np.float32)
        genomes_non_coding_proportion_vars = np.empty(len(x_values), dtype=np.float32)

        genomes_nc_lengths_means = np.empty((len(x_values), g), dtype=np.float32)
        genomes_nc_lengths_vars = np.empty((len(x_values), g), dtype=np.float32)

        population_z_ncs = np.empty((len(x_values), len(QUANTILES)), dtype=np.float32)

        living_child_proportions = np.empty(len(x_values), dtype=np.float32)

        changed_proporions = np.empty(len(x_values), dtype=np.float32)

        for index, generation in enumerate(x_values):
            if generation == 0:
                generation = 1
            with open(
                self.save_path / "logs" / f"generation_{generation}.pkl", "rb"
            ) as pkl_file:
                try:
                    stats = pkl.load(pkl_file)
                except EOFError:
                    print(
                        f"EOFError while loading generation {generation}, last generation could be corrupted."
                    )
                    break

            genome_raw_stats = stats["genome"]
            population_raw_stats = stats["population"]

            genomes_non_coding_proportion = np.array(
                [genome["Non coding proportion"] for genome in genome_raw_stats],
                dtype=np.float32,
            )
            genomes_non_coding_proportion_mean = np.mean(genomes_non_coding_proportion)
            if self.population > 1:
                genomes_non_coding_proportion_var = (
                    np.var(genomes_non_coding_proportion)
                    * len(genomes_non_coding_proportion)
                    / (len(genomes_non_coding_proportion) - 1)
                )
            else:
                genomes_non_coding_proportion_var = 0
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
            try:
                genomes_nc_lengths_var = (
                    np.var(genomes_nc_lengths, axis=0)
                    * len(genomes_nc_lengths)
                    / (len(genomes_nc_lengths) - 1)
                )
            except ZeroDivisionError:
                genomes_nc_lengths_var = np.zeros(g, dtype=np.float32)
            genomes_nc_lengths_means[index] = genomes_nc_lengths_mean
            genomes_nc_lengths_vars[index] = genomes_nc_lengths_var

            population_z_ncs[index] = population_raw_stats["z_nc_array"]

            living_child_proportions[index] = population_raw_stats[
                "Living child proportion"
            ]

            changed_proporions[index] = population_raw_stats["Changed proportion"]

        save_dir = self.save_path / "plots"
        graphics.plot_simulation(
            x_values,
            genomes_non_coding_proportion_means,
            genomes_non_coding_proportion_vars,
            save_dir,
            "Non coding proportion",
            ylim=-1,
        )

        graphics.plot_simulation(
            x_values,
            living_child_proportions,
            None,
            save_dir,
            "Living child proportion",
            ylim=-1,
        )

        graphics.plot_simulation(
            x_values,
            changed_proporions,
            None,
            save_dir,
            "Changed genome proportion",
            ylim=-1,
        )

        for quantile in QUANTILES:
            index = int((g - 1) * quantile)
            mean_list = genomes_nc_lengths_means[:, index]
            var_list = genomes_nc_lengths_vars[:, index]
            graphics.plot_simulation(
                x_values,
                mean_list,
                var_list,
                save_dir,
                f"Genomes non coding lengths - {quantile * 100}th percentile",
                ylim=ylim / g,
            )
        for idx_quantile, quantile in enumerate(QUANTILES):
            try:
                mean_list = population_z_ncs[:, idx_quantile]
                graphics.plot_simulation(
                    x_values,
                    mean_list,
                    None,
                    save_dir,
                    f"Population z_nc - {quantile * 100}th percentile",
                    ylim=ylim,
                )
            except IndexError:
                print(
                    f"IndexError while plotting population z_nc - {quantile * 100}th percentile"
                )
        if not on_execution:
            save_mutation_info = self.save_path / "mutation_info"
            save_mutation_info.mkdir(parents=True, exist_ok=True)
            for mutation in self.mutations:
                mutation.stats.compute()
                with open(
                    save_mutation_info / f"{str(mutation)}.json", "w", encoding="utf8"
                ) as json_file:
                    json.dump(mutation.stats.d_stats, json_file, indent=2)
            self.plot_mutation_info()
        # print("\n\n")
        # print("Plotting generation plots...")
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
        return generation

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
