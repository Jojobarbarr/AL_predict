import json
from pathlib import Path

import matplotlib.pyplot as plt
import pickle as pkl
import numpy.typing as npt
import numpy as np
from stats import GenomeStatistics

## MUTAGENESE
def save_stats(save_dir: Path, results: dict[str, dict]) -> None:
    for mutation, result_by_pow in results.items():
        save_dir_specific = save_dir / mutation
        save_dir_specific.mkdir(parents=True, exist_ok=True)
        for power, d_stats in result_by_pow.items():
            with open(save_dir_specific / f"{power}.json", "w", encoding="utf8") as json_file:
                json.dump(d_stats, json_file, indent=2)

def plot_mutagenese(x_value: list[float], y_value: list[float], y_std: list[float], save_path: Path, name: str, 
                    variable: str, theoreticals: list[float]=[]):
        
        plt.clf()
        plt.plot(x_value, y_value, marker='o', label="Estimation")
        if len(theoreticals) > 0:
            plt.plot(x_value, theoreticals, marker='o', label="Theoretical values") 
        plt.errorbar(x_value, y_value, y_std, linestyle='None', marker='o', label="Standard Deviation")
        plt.title(f"{name} for different values of {variable}")
        plt.xlabel(variable)
        plt.xscale("log")
        plt.ylabel(f"{name}")
        # plt.ylim(ylimits)
        plt.yscale("log")
        plt.legend()
        plt.savefig(save_path / f"{name.lower().replace(' ', '_')}_{variable}.jpg")


## SIMULATION 
def save_checkpoint(save_dir: Path, genome_stats: npt.NDArray[dict[str, float]], population_stats: dict[str, float], generation: int):
    save_dir.mkdir(parents=True, exist_ok=True)
    all_stats = {"genome": genome_stats, "population": population_stats}
    if generation == 0:
        generation += 1
    with open(save_dir / f"generation_{generation}.pkl", "wb") as pkl_file:
        pkl.dump(all_stats, pkl_file, protocol=pkl.HIGHEST_PROTOCOL)

def plot_simulation(x_value: npt.NDArray[np.int_], y_value: npt.NDArray[np.float_], std_values: npt.NDArray[np.float_] | None,
                    save_path: Path, name: str):
    save_path.mkdir(parents=True, exist_ok=True)
    plt.clf()
    plt.plot(x_value, y_value, marker='o', label="Estimation")
    # plt.errorbar(x_value, y_value, std_values, linestyle='None', marker='o')
    plt.title(f"{name} over generations")
    plt.xlabel("Generation")
    plt.ylabel(f"{name}")
    plt.legend()
    plt.savefig(save_path / f"{name.lower().replace(' ', '_')}.jpg")

def plot_generation(statistics, generation, min, max, ymax, xlabel, name, save_path: Path):
    save_path.mkdir(parents=True, exist_ok=True)
    save_path_fixed = save_path / "fixed"
    save_path_fixed.mkdir(parents=True, exist_ok=True)
    plt.clf()
    plt.hist(statistics, bins=50)
    plt.title(f"{name} for generation {generation}")
    plt.xlabel(f"{xlabel}")
    plt.ylabel("Count")
    plt.savefig(save_path / f"{name.lower().replace(' ', '_')}_{generation}.jpg")

    if min != 0 or max != 0:
        plt.clf()
        plt.hist(statistics, bins=50, range=(min, max))
        plt.title(f"{name} for generation {generation}")
        plt.xlabel(f"{xlabel}")
        plt.ylabel("Count")
        plt.ylim(0, ymax)
        plt.savefig(save_path_fixed / f"{name.lower().replace(' ', '_')}_{generation}.jpg")
