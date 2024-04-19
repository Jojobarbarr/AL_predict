import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json


folder_name = "-2_2"

folders = (
    Path(f"results/simulation/{folder_name}/_plots"),
    Path(f"results/simulation/{folder_name}_model/_plots"),
    # Path("results/simulation/0_0_from_top/_plots"),
)
labels = (
    "Wild",
    "Assuming homogeneity and single directionnality",
    "Wild from top",
)
colors = (
    "red",
    "green",
    "tomato",
)

population = 256
iterative_model_folder = Path(f"results/simulation/{folder_name}/_iterative_model")


figure_dir = Path(f"results/figures/{folder_name}")
figure_dir.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    x_values = []
    nc_proportions = []
    livings = []
    for folder_index, folder in enumerate(folders):
        x_values.append(np.load(folder / "x_values.npy"))
        nc_proportions.append(np.load(folder / "non_coding_proportion.npy"))
        livings.append(np.load(folder / "livings.npy"))

    max_index = (
        min([len(x_values[folder_index]) for folder_index in range(len(folders))]) - 1
    )
    min_generation = x_values[0][max_index]

    with open(
        iterative_model_folder / "config.json", "r", encoding="utf8"
    ) as json_file:
        config = json.load(json_file)
        iterations = config["Iterations"]
        time_acceleration = config["Time acceleration"]

    x_iterative_model = np.array(
        [x * time_acceleration for x in range(0, iterations + 1)]
    )
    x_iterative_model = x_iterative_model[
        x_iterative_model - time_acceleration <= min_generation
    ]
    nc_proportions_iterative_model = np.load(
        iterative_model_folder / "nc_proportions.npy", allow_pickle=True
    )[: len(x_iterative_model)]
    nc_proportions_iterative_model[1:][nc_proportions_iterative_model[1:] == 0] = np.nan
    nc_proportions_iterative_model_constant_Ne = np.load(
        iterative_model_folder / "nc_proportions_constant_Ne.npy",
        allow_pickle=True,
    )[: len(x_iterative_model)]
    Nes_iterative_model = np.load(
        iterative_model_folder / "Nes.npy", allow_pickle=True
    )[: len(x_iterative_model)]
    livings_iterative_model = Nes_iterative_model / population

    plt.clf()
    plt.figure(figsize=(7, 5), dpi=150)
    plt.plot(x_iterative_model, nc_proportions_iterative_model, label="Iterative model")
    plt.plot(
        x_iterative_model,
        nc_proportions_iterative_model_constant_Ne,
        label="Iterative model (constant Ne)",
    )

    for index in range(len(folders)):
        plt.plot(
            x_values[index][:max_index],
            nc_proportions[index][:max_index],
            label=labels[index],
            color=colors[index],
        )
    plt.xlabel("Generation")
    plt.ylabel("Non-coding proportion")
    plt.title("Non-coding proportion over time")
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(figure_dir / "non_coding_proportions.jpg")

    plt.clf()
    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(x_iterative_model, livings_iterative_model, label="Iterative model")
    for index in range(len(folders)):
        plt.plot(
            x_values[index][:max_index],
            livings[index][:max_index],
            label=labels[index],
            color=colors[index],
        )
    plt.xlabel("Generation")
    plt.ylabel("Living child proportion")
    plt.title("Living child proportion over time")
    plt.grid()
    plt.legend()
    # plt.show()
    plt.savefig(figure_dir / "living_proportions.jpg")
