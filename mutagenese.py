import math as m
from configparser import ConfigParser

from tqdm import tqdm
import json

import graphics
import mutations
from experiment import Experiment
from genome import Genome
from utils import MUTATIONS, str_to_int, L_M


class Mutagenese(Experiment):
    def __init__(self, config: ConfigParser):
        super().__init__(config)
        self.variable = self.mutagenese_config["Variable"]
        self.l_m = int(self.mutations_config["l_m"])
        self.homogeneous = self.genome_config["Homogeneous"]
        self.orientation = self.genome_config["Orientation"]

        self.power_min = int(m.log10(float(self.mutagenese_config["From"])))
        self.power_max = int(m.log10(float(self.mutagenese_config["To"])))
        self.power_step = int(self.mutagenese_config["Step"])

        self.mutation_names = self.mutations_config["Mutation types"]
        self.mutation_types = [
            MUTATIONS[mutation_type] for mutation_type in self.mutation_names
        ]

        self.experiment_repetitions = str_to_int(self.mutagenese_config["Iterations"])

        self.results = {mutation: {} for mutation in self.mutation_names}

    def run(
        self,
        only_plot: bool = False,
    ):
        if not only_plot:
            if self.variable == "No variable":
                genome = self.prepare_mutagenese(0)
                for mutation, name in zip(self.mutation_types, self.mutation_names):
                    print(f"Mutation type: {name}")
                    if name in L_M:
                        self.results[name][888] = self.loop(mutation(self.l_m, genome))
                    else:
                        self.results[name][888] = self.loop(mutation(genome))
                del genome  # genome can be very large

            else:
                for power in range(self.power_min, self.power_max + 1, self.power_step):
                    print(f"Experience for {self.variable} = 10^{power}")
                    genome = self.prepare_mutagenese(10**power)
                    for mutation, name in zip(self.mutation_types, self.mutation_names):
                        print(f"Mutation type: {name}")
                        if name in L_M:
                            self.results[name][power] = self.loop(
                                mutation(self.l_m, genome)
                            )
                        else:
                            self.results[name][power] = self.loop(mutation(genome))
                    del genome  # genome can be very large

            graphics.save_stats(self.save_path, self.results)

        if self.variable != "No variable":
            self.plot_mutagenese()

    def prepare_mutagenese(
        self,
        value: int,
    ) -> Genome:
        if self.variable == "g":
            g = value
        else:
            g = str_to_int(self.genome_config["g"])

        if self.variable == "z_c":
            z_c = value
        else:
            z_c_factor = str_to_int(self.genome_config["z_c_factor"])
            if self.genome_config["z_c_auto"]:
                z_c = z_c_factor * g
            else:
                z_c = str_to_int(self.genome_config["z_c"])

        if self.variable == "z_nc":
            z_nc = value
        else:
            z_nc_factor = str_to_int(self.genome_config["z_nc_factor"])
            if self.genome_config["z_nc_auto"]:
                z_nc = z_nc_factor * g
            else:
                z_nc = str_to_int(self.genome_config["z_nc"])

        return Genome(g, z_c, z_nc, self.homogeneous, self.orientation)  # type: ignore

    def loop(
        self,
        mutation: mutations.Mutation,
    ) -> dict[str, float]:
        for _ in tqdm(
            range(self.experiment_repetitions),
            "Experiment progress... ",
            self.experiment_repetitions,
        ):
            if mutation.is_neutral():
                mutation.apply(virtually=True)
        mutation.stats.compute(mutation.theory())
        return mutation.stats.d_stats

    def plot_mutagenese(self):
        x_value = [
            10**power
            for power in range(self.power_min, self.power_max + 1, self.power_step)
        ]
        for mutation_type in self.mutation_names:
            save_dir = self.save_path / "stats" / mutation_type

            neutral_proportions = []
            neutral_stds = []
            theoretical_proportions = []
            length_means = []
            length_stds = []

            for power in range(self.power_min, self.power_max + 1, self.power_step):

                with open(
                    save_dir / f"{power}.json", "r", encoding="utf8"
                ) as json_file:
                    d_stats = json.load(json_file)

                neutral_proportions.append(d_stats["Neutral mutations proportion"])
                neutral_stds.append(
                    d_stats[
                        "Neutral mutations standard deviation of proportion estimator"
                    ]
                )
                theoretical_proportions.append(d_stats["Neutral probability theory"])
                length_means.append(d_stats["Length mean"])
                length_stds.append(
                    d_stats["Length standard deviation of mean estimator"]
                )

            graphics.plot_mutagenese(
                x_value,
                neutral_proportions,
                neutral_stds,
                save_dir,
                f"Neutral {mutation_type.lower()} proportion",
                self.variable,
                theoretical_proportions,
            )

            graphics.plot_mutagenese(
                x_value,
                length_means,
                length_stds,
                save_dir,
                f"{mutation_type} length mean",
                self.variable,
            )
