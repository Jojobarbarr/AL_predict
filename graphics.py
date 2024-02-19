import json
from pathlib import Path

import matplotlib.pyplot as plt


def save_stats(save_dir: Path, results: dict[str, dict]) -> None:
    for mutation, result_by_pow in results.items():
        save_dir_specific = save_dir / mutation
        save_dir_specific.mkdir(parents=True, exist_ok=True)
        for power, d_stats in result_by_pow.items():
            with open(save_dir_specific / f"{power}.json", "w", encoding="utf8") as json_file:
                json.dump(d_stats, json_file, indent=2)

def plot_mutagenese(x_value: list[float], y_value: list[float], y_std: list[float], save_path: Path, name: str, 
                    variable: str, ylimits: tuple[float | int, float | int], theoreticals: list[float]=[]):
        
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