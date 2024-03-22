import json
import pickle as pkl
from collections import Counter
from pathlib import Path
from time import perf_counter
from multiprocessing import Process
import traceback
from typing import Any
import numpy as np

import graphics
from utils import str_to_int, QUANTILES
from simulation import Simulation

PARALLELIZE = False


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
        self.rng = np.random.default_rng()
        self.mutant_parent_indices_counter = Counter()
        # self.chunk_number = cpu_count()

    def run(
        self,
        only_plot: bool = False,
    ):
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
                livings = np.ones(self.population, dtype=bool)

            main_start_time = perf_counter()

            time_perfs = []
            sum_time_perfs = 0
            plot_process = Process(
                target=lambda: None,
            )
            if self.plot_in_time:
                plot_process.start()
            try:
                for generation in range(1, self.generations + 1):
                    start_time = perf_counter()
                    livings, plot_process = self.generation_step(
                        generation, livings, plot_process
                    )

                    end_time = perf_counter()
                    time_perfs.append(end_time - start_time)
                    if generation % self.plot_point == 0:
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
                            f"{self.format_time(end_time - main_start_time)} elapsed since the beginning of the simulation "
                            f"- \033[1mEstimated remaining time: {self.format_time(average_time_perf * (self.generations - generation))}\033[0m"
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

            if plot_process.is_alive():
                plot_process.join()
            self.save_checkpoint(generation, livings)

            print(f"Generation {generation} - End of simulation")
            print(f"Total time: {self.format_time(perf_counter() - main_start_time)}")
            self.save_population("final")
            self.plot_simulation(generation)
        else:
            self.plot_simulation(self.generations, ylim=20000)

    def generation_step(
        self,
        generation: int,
        livings: np.ndarray[Any, np.dtype[np.bool_]],
        plot_process: Process,
    ):
        parents = self.rng.choice(self.genomes[livings], size=self.population)

        lengths = self.vec_get_genome_length(parents)
        total_bases_number = lengths.sum()
        genomes_biases = lengths / total_bases_number
        mutation_number = self.rng.binomial(
            total_bases_number, self.total_mutation_rate
        )

        mutant_parent_indices = self.rng.choice(
            len(parents), size=mutation_number, p=genomes_biases
        )
        self.mutant_parent_indices_counter.update(mutant_parent_indices)
        mutation_per_genome: dict[int, np.ndarray] = {
            mutant_parent_index: self.rng.choice(self.mutations, size=mutation_applied)
            for mutant_parent_index, mutation_applied in self.mutant_parent_indices_counter.items()
        }

        self.genomes = self.vec_clone(parents)
        livings = np.ones(len(parents), dtype=bool)
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

        if not livings.any():
            raise RuntimeError(f"All individuals died on generation {generation}")

        ## LOGGING AND PLOTTING
        if generation % self.plot_point == 0 or generation == 1:
            z_nc_array = self.vec_get_genome_z_nc(self.genomes[livings])
            z_nc_array.sort()
            quantile_idx = np.array(
                [int(quantile * (len(z_nc_array) - 1)) for quantile in QUANTILES],
                dtype=int,
            )
            z_nc_array = z_nc_array[quantile_idx]
            genomes_stats = self.vec_compute_genomes_stats(self.genomes[livings])

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
                plot_process.join()
                plot_process = Process(
                    target=self.plot_simulation,
                    args=(generation, True),
                )
                plot_process.start()
        self.mutant_parent_indices_counter.clear()

        return livings, plot_process

    def plot_simulation(
        self,
        generation: int,
        on_execution: bool = False,
        ylim: int = -1,
    ):
        g = str_to_int(self.genome_config["g"])
        x_values = np.array(
            [generation for generation in range(0, generation + 1, self.plot_point)],
            dtype=np.int64,
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
