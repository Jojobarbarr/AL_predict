import json
from argparse import ArgumentParser
from decimal import Decimal, getcontext, InvalidOperation, DivisionByZero
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import bisect

from utils import str_to_int

getcontext().prec = 40


def update(
    n: Decimal,
    g: Decimal,
    z_c: Decimal,
    z_nc: Decimal,
    mu: Decimal,
    l_m: Decimal,
    variable_n_e: bool,
) -> tuple[Decimal, int, int, Decimal]:
    """Updates the different parameters according to the current state of the genome

    Args:
        n (Decimal): Population size
        g (Decimal): Number of segments
        z_c (Decimal): Number of coding bases
        z_nc (Decimal): Number of non-coding bases
        mu (Decimal): Mutation rate
        l_m (Decimal): maximum size of Indels
        variable_n_e (bool): If True, the effective population size will be updated

    Returns:
        tuple(Decimal, int, int, Decimal): Updated parameters (L, alpha, segment_length, n_e)
    """
    l = z_c + z_nc
    alpha = int(z_nc / g)
    segment_length = int(l / g)
    if variable_n_e:
        n *= robustness(mu, l, z_nc, g, l_m)
    return l, alpha, segment_length, n


## Probability to be neutral
# Deletion
def vdel(
    z_nc: Decimal,
    g: Decimal,
    l: Decimal,
) -> Decimal:
    """Computes the probability to be neutral for a deletion

    Args:
        z_nc (Decimal): Number of non-coding bases
        g (Decimal): Number of segments
        l (Decimal): Genome length

    Returns:
        Decimal: Probability to be neutral for deletion
    """
    return Decimal((z_nc + g) * z_nc / (2 * g * l * l))


# Duplication
def vdupl(
    z_nc: Decimal,
    g: Decimal,
    l: Decimal,
) -> Decimal:
    """Computes the probability to be neutral for a duplication

    Args:
        z_nc (Decimal): Number of non-coding bases
        g (Decimal): Number of segments
        l (Decimal): Genome length

    Returns:
        Decimal: Probability to be neutral for duplication
    """
    return Decimal((z_nc + g) * (l - g) / (2 * g * l * l))


# Inversion
def vinv(
    z_nc: Decimal,
    g: Decimal,
    l: Decimal,
) -> Decimal:
    """Computes the probability to be neutral for an inversion

    Args:
        z_nc (Decimal): Number of non-coding bases
        g (Decimal): Number of segments
        l (Decimal): Genome length

    Returns:
        Decimal: Probability to be neutral for inversion
    """
    return Decimal((z_nc + g) * (z_nc + g - 1) / (l * (l - 1)))


# Point Mutation
def vpm(
    z_nc: Decimal,
    l: Decimal,
) -> Decimal:
    """Computes the probability to be neutral for a point mutation

    Args:
        z_nc (Decimal): Number of non-coding bases
        l (Decimal): Genome length

    Returns:
        Decimal: Probability to be neutral for point mutation
    """
    return Decimal(z_nc / l)


# Small insertion
def vindel_plus(
    z_nc: Decimal,
    g: Decimal,
    l: Decimal,
) -> Decimal:
    """Computes the probability to be neutral for a small insertion

    Args:
        z_nc (Decimal): Number of non-coding bases
        g (Decimal): Number of segments
        l (Decimal): Genome length

    Returns:
        Decimal: Probability to be neutral for small insertion
    """
    return Decimal((z_nc + g) / l)


# Small deletion
def vindel_moins(
    z_nc: Decimal,
    g: Decimal,
    l: Decimal,
    l_m: Decimal,
) -> Decimal:
    """Computes the probability to be neutral for a small deletion

    Args:
        z_nc (Decimal): Number of non-coding bases
        g (Decimal): Number of segments
        l (Decimal): Genome length
        l_m (Decimal): Maximum size of Indels

    Returns:
        Decimal: Probability to be neutral for small deletion
    """
    return Decimal(g / l * (z_nc / g - (l_m - 1) / 2))


## Robustness (effective fitness)
def robustness(
    mu: Decimal,
    l: Decimal,
    z_nc: Decimal,
    g: Decimal,
    l_m: Decimal,
) -> Decimal:
    """Computes the robustness (effective fitness) of a genome

    Args:
        mu (Decimal): Mutation rate
        l (Decimal): Genome length
        z_nc (Decimal): Number of non-coding bases
        g (Decimal): Number of segments
        l_m (Decimal): Maximum size of Indels

    Returns:
        Decimal: The robustness of this genome
    """
    product = Decimal(0)

    neutr_by_pm = Decimal((1 - mu * (1 - vpm(z_nc, l))) ** l)
    neutr_by_indel_plus = Decimal((1 - mu * (1 - vindel_plus(z_nc, g, l))) ** l)
    neutr_by_indel_moins = Decimal((1 - mu * (1 - vindel_moins(z_nc, g, l, l_m))) ** l)
    neutr_by_del = Decimal((1 - mu * (1 - vdel(z_nc, g, l))) ** l)
    neutr_by_dupl = Decimal((1 - mu * (1 - vdupl(z_nc, g, l))) ** l)
    neutr_by_inv = Decimal((1 - mu * (1 - vinv(z_nc, g, l))) ** l)

    product = (
        neutr_by_pm
        * neutr_by_indel_plus
        * neutr_by_indel_moins
        * neutr_by_del
        * neutr_by_dupl
        * neutr_by_inv
    )
    return product


# Ratio of robustness
def ratio(
    x: Decimal,
    mu: Decimal,
    l: Decimal,
    z_nc: Decimal,
    g: Decimal,
    l_m: Decimal,
) -> Decimal:
    """Computes the ratio of robustness between the wild-type and the mutant.

    Args:
        x (Decimal): the number of non-coding bases that will be added or removed
        mu (Decimal): Mutation rate
        l (Decimal): Genome length
        z_nc (Decimal): Number of non-coding bases
        g (Decimal): Number of segments
        l_m (Decimal): Maximum size of Indels

    Returns:
        Decimal: The ratio of robustness
    """
    return robustness(mu, l, z_nc, g, l_m) / robustness(mu, l + x, z_nc + x, g, l_m)


def pfix_z_nc(
    x: Decimal,
    mu: Decimal,
    z_c: Decimal,
    z_nc: Decimal,
    g: Decimal,
    n_e: Decimal,
    l_m: Decimal,
) -> Decimal:
    """_summary_

    Args:
        x (Decimal): the number of non-coding bases that will be added or removed
        mu (Decimal): Mutation rate
        z_c (Decimal): Number of coding bases
        z_nc (Decimal): Number of non-coding bases
        g (Decimal): Number of segments
        n_e (Decimal): Effective population size
        l_m (Decimal): Maximum size of Indels

    Returns:
        Decimal: Probability of fixation of a mutation
    """
    l = Decimal(z_c + z_nc)
    r = ratio(x, mu, l, z_nc, g, l_m)
    # A = (1 - r) / (1 - r ** (2 * n_e))  # diploid
    p_fix = (1 - (r * r)) / (1 - r ** (2 * n_e))  # haploid
    return Decimal(p_fix)


def deltadel_fix_z_nc(
    g: Decimal,
    l: Decimal,
    alpha: int,
    pre_computed_table_p_fix: np.ndarray,
) -> Decimal:
    """Computes the mean length of deletions that are fixed

    Args:
        g (Decimal): Number of segments
        l (Decimal): Genome length
        alpha (int): Number of non-coding bases per segment
        pre_computed_table_p_fix (np.ndarray): list of the probability of fixation of a deletion
        according to the number of bases deleted

    Returns:
        Decimal: The mean length of deletions that are fixed
    """
    factor = Decimal(g / (l * l))
    table_p_fix_local = np.array(
        [
            (alpha + 1 - Decimal(k)) * Decimal(k) * pre_computed_table_p_fix[k - 1]
            for k in range(1, alpha + 1)
        ],
        dtype=Decimal,
    )
    return factor * Decimal(table_p_fix_local.sum())


def deltadupl_fix_z_nc(
    g: Decimal,
    l: Decimal,
    alpha: int,
    segment_length: int,
    pre_computed_table_p_fix: np.ndarray,
) -> Decimal:
    """Computes the mean length of duplications that are fixed

    Args:
        g (Decimal): Number of segments
        l (Decimal): Genome length
        alpha (int): Number of non-coding bases per segment
        segment_length (int): Length of a segment
        (coding segment length + non coding segment length)
        pre_computed_table_p_fix (np.ndarray): list of the probability of fixation of a deletion
        according to the number of bases added

    Returns:
        Decimal: The mean length of duplications that are fixed
    """
    factor = Decimal((g * g * (alpha + 1)) / (l * l * l))

    table_p_fix_local = np.array(
        [
            (segment_length - Decimal(k)) * Decimal(k) * pre_computed_table_p_fix[k - 1]
            for k in range(1, segment_length)
        ],
        dtype=Decimal,
    )
    return factor * Decimal(table_p_fix_local.sum())


def delta_indelplus_fix_z_nc(
    g: Decimal,
    z_nc: Decimal,
    l_m: int,
    l: Decimal,
    pre_computed_table_p_fix: np.ndarray,
) -> Decimal:
    """Computes the mean length of insertions that are fixed

    Args:
        g (Decimal): Number of segments
        z_nc (Decimal): Number of non-coding bases
        l_m (int): Maximum size of Indels
        l (Decimal): Genome length
        pre_computed_table_p_fix (np.ndarray): list of the probability of fixation of a deletion
        according to the number of bases added

    Returns:
        Decimal: The mean length of insertions that are fixed
    """
    factor = Decimal((z_nc + g) / (l_m * l))
    return factor * Decimal(
        np.array(
            [(Decimal(k + 1)) * pre_computed_table_p_fix[k] for k in range(0, l_m)],
            dtype=Decimal,
        ).sum()
    )


def delta_indelmoins_fix_z_nc(
    g: Decimal,
    l_m: int,
    l: Decimal,
    alpha: int,
    pre_computed_table_p_fix: np.ndarray,
) -> Decimal:
    """Computes the mean length of deletions that are fixed

    Args:
        g (Decimal): Number of segments
        l_m (int): Maximum size of Indels
        l (Decimal): Genome length
        alpha (int): Number of non-coding bases per segment
        pre_computed_table_p_fix (np.ndarray): list of the probability of fixation of a deletion
        according to the number of bases added

    Returns:
        Decimal: The mean length of deletions that are fixed
    """
    factor = Decimal(g / (l_m * l))

    first_part = (
        Decimal(max(alpha - (l_m - 1), 0))
        * np.array(
            [(Decimal(k + 1)) * pre_computed_table_p_fix[k] for k in range(0, l_m)],
            dtype=Decimal,
        ).sum()
    )

    steps = min(l_m - 1, alpha)
    second_part = Decimal(
        np.array(
            [
                np.array(
                    [Decimal(k + 1) * pre_computed_table_p_fix[k] for k in range(0, s)],
                    dtype=Decimal,
                ).sum()
                for s in range(1, steps + 1)
            ],
            dtype=Decimal,
        ).sum()
    )

    return factor * Decimal((first_part + second_part))


def table_p_fix(
    g: Decimal,
    z_c: Decimal,
    z_nc: Decimal,
    n_e: Decimal,
    mu: Decimal,
    l_m: int,
    segment_length: int,
    alpha: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre computes the probability of fixation of a mutation for each possible number of
    bases added or removed

    Args:
        g (Decimal): Number of segments
        z_c (Decimal): Number of coding bases
        z_nc (Decimal): Number of non-coding bases
        n_e (Decimal): Effective population size
        mu (Decimal): Mutation rate
        l_m (int): Maximum size of Indels
        segment_length (int): Number of bases in a segment
        (coding segment length + non coding segment length)
        alpha (int): Number of non-coding bases per segment

    Returns:
        tuple[np.ndarray, np.ndarray]: Two arrays containing the probability of fixation
        of a mutation for each possible number of bases added or removed
    """
    p_fix_positive_k = np.array(
        [
            pfix_z_nc(Decimal(k), mu, z_c, z_nc, g, n_e, Decimal(l_m))
            for k in range(1, max(segment_length, l_m + 1))
        ],
        dtype=Decimal,
    )
    p_fix_negative_k = np.array(
        [
            pfix_z_nc(Decimal(-k), mu, z_c, z_nc, g, n_e, Decimal(l_m))
            for k in range(1, max(alpha + 1, l_m + 1))
        ],
        dtype=Decimal,
    )
    return p_fix_positive_k, p_fix_negative_k


def biais_z_nc(
    g: Decimal,
    z_c: Decimal,
    z_nc: Decimal,
    n: Decimal,
    mu: Decimal,
    l_m: int,
    variable_n_e: bool,
) -> Decimal:
    """Computes the bias between bases added and bases removed

    Args:
        g (Decimal): Number of segments
        z_c (Decimal): Number of coding bases
        z_nc (Decimal): Number of non-coding bases
        n (Decimal): Population size
        mu (Decimal): Mutation rate
        l_m (int): Maximum size of Indels
        variable_n_e (bool): If True, the effective population size will be updated

    Returns:
        Decimal: The bias between bases added and bases removed
    """
    l, alpha, segment_length, n_e = update(
        n, g, z_c, z_nc, mu, Decimal(l_m), variable_n_e
    )
    # print(f"n_e: n_ee}")

    pre_computed_table_p_fix_positive_k, pre_computed_table_p_fix_negative_k = (
        table_p_fix(g, z_c, z_nc, n_e, mu, l_m, segment_length, alpha)
    )
    biais_moins = deltadel_fix_z_nc(
        g, l, alpha, pre_computed_table_p_fix_negative_k
    ) + delta_indelmoins_fix_z_nc(g, l_m, l, alpha, pre_computed_table_p_fix_negative_k)
    biais_plus = deltadupl_fix_z_nc(
        g, l, alpha, segment_length, pre_computed_table_p_fix_positive_k
    ) + delta_indelplus_fix_z_nc(g, z_nc, l_m, l, pre_computed_table_p_fix_positive_k)
    return biais_moins / biais_plus


def iterate(
    g: Decimal,
    z_c: Decimal,
    z_nc: Decimal,
    n: Decimal,
    mu: Decimal,
    l_m: int,
    iterations: int,
    time_acceleration: int,
    variable_n_e: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Applies the model iteratively as an implicit sequence.
    The model outputs the number of bases added or removed, which is used to update z_nc
    and all variables linked to it.
    If variable_n_e is True, the effect on effective population size is taken into account.

    Args:
        g (Decimal): Number of segments
        z_c (Decimal): Number of coding bases
        z_nc (Decimal): Number of non-coding bases
        n (Decimal): Population size
        mu (Decimal): Mutation rate
        l_m (int): Maximum size of Indels
        iterations (int): Number of iterations
        time_acceleration (int): Factor between generations and iterations
        variable_n_e (bool, optional): If True, effective population size will be variable.
        Defaults to True.

    Returns:
        tuple[np.ndarray, np.ndarray]: (array of non coding proportion, array of
        effective population size)
    """
    l, alpha, segment_length, n_e = update(
        n, g, z_c, z_nc, mu, Decimal(l_m), variable_n_e
    )
    print(f"Initial living proportion: {n_e / n}")
    progress_point = iterations // 20
    nc_proportions = np.zeros(iterations + 1, dtype=Decimal)
    n_es = np.zeros(iterations + 1, dtype=Decimal)
    nc_proportions[0] = z_nc / l
    n_es[0] = n_e

    start_time = perf_counter()
    try:
        for iteration in range(1, iterations + 1):
            # Computes the probabilities og fixation for all possible values of k
            pre_computed_table_p_fix_positive_k, pre_computed_table_p_fix_negative_k = (
                table_p_fix(g, z_c, z_nc, n_e, mu, l_m, segment_length, alpha)
            )

            # Computes the contribution of each mutation to change in genome size
            delta_indelmoins: Decimal = delta_indelmoins_fix_z_nc(
                g, l_m, l, alpha, pre_computed_table_p_fix_negative_k
            )
            delta_del: Decimal = deltadel_fix_z_nc(
                g, l, alpha, pre_computed_table_p_fix_negative_k
            )

            delta_indelplus: Decimal = delta_indelplus_fix_z_nc(
                g, z_nc, l_m, l, pre_computed_table_p_fix_positive_k
            )
            delta_dupl: Decimal = deltadupl_fix_z_nc(
                g, l, alpha, segment_length, pre_computed_table_p_fix_positive_k
            )

            bases_deleted: Decimal = mu * l * n * (delta_indelmoins + delta_del)
            bases_inserted: Decimal = mu * l * n * (delta_indelplus + delta_dupl)

            delta: Decimal = time_acceleration * (bases_inserted - bases_deleted)

            # Update the genome
            z_nc += delta
            l, alpha, segment_length, n_e = update(
                n, g, z_c, z_nc, mu, Decimal(l_m), variable_n_e
            )

            # Logs
            nc_proportions[iteration] = z_nc / l
            n_es[iteration] = n_e

            # Prints progression
            if iteration % progress_point == 0:
                checkpoint_time = perf_counter() - start_time
                remaining_time = (iterations - iteration) * (
                    checkpoint_time / iteration
                )
                print(
                    f"Iteration {iteration}/{iterations} - {checkpoint_time:.2f}s elapsed"
                    f" - Remaining about {remaining_time:.2f}s"
                )
    except KeyboardInterrupt:
        print(
            f"Keyboard interruption at iteration {iteration}, exiting the loop and plotting and "
            f"saving results. Another Keyboard interruption stops the execution"
        )
        l, alpha, segment_length, n_e = update(
            n, g, z_c, z_nc, mu, Decimal(l_m), variable_n_e
        )
    except (InvalidOperation, DivisionByZero) as err:
        print(
            "Error during computation, exiting the loop but continuing execution."
            f"Error: {err}"
        )

    print(f"Iterations: {iteration}")
    print(f"Final non-coding proportion: {z_nc / l} (z_nc = {z_nc})")
    return nc_proportions, n_es


def bisect_target(
    x: float,
    g: Decimal,
    z_c: Decimal,
    n: Decimal,
    mu: Decimal,
    l_m: int,
    variable_n_e: bool,
) -> float:
    """Function to bisect to find the equilibrium between the number of non-coding bases

    Args:
        x (float): Number of non-coding bases
        g (Decimal): Number of segments
        z_c (Decimal): Number of coding bases
        n (Decimal): Population size
        mu (Decimal): Mutation rate
        l_m (int): Maximum size of Indels
        variable_n_e (bool): If True, the effective population size will be updated.

    Returns:
        float: The difference between the current bias and 1 (equilibrium)
    """
    result = float(
        biais_z_nc(
            g=g,
            z_c=z_c,
            z_nc=Decimal(x),
            n=n,
            mu=mu,
            l_m=l_m,
            variable_n_e=variable_n_e,
        )
        - 1
    )
    return result


def find_z_nc(
    g: Decimal,
    z_c: Decimal,
    n_e: Decimal,
    mu: Decimal,
    l_m: int,
    variable_n_e=True,
) -> None:
    """Perform a bisection to find the equilibrium between the number of non-coding bases
    and the number of coding bases

    Args:
        g (Decimal): Number of segments
        z_c (Decimal): Number of coding bases
        n_e (Decimal): Population effective size
        mu (Decimal): Mutation rate
        l_m (int): Maximum size of Indels
        variable_n_e (bool, optional): If True, population effective size is variable.
        Defaults to True.
    """
    max_proportion_a_priori: Decimal = Decimal(
        0.9
    )  # Can make the bisection converge or diverge...
    up_start: Decimal = max_proportion_a_priori * z_c / (1 - max_proportion_a_priori)
    z_nc = bisect(
        bisect_target,
        0,
        int(up_start),
        args=(g, z_c, n_e, mu, l_m, variable_n_e),
    )
    assert isinstance(z_nc, float)
    print(f"Final proportion: {Decimal(z_nc) / (z_c + Decimal(z_nc))} (z_nc = {z_nc})")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("config", type=Path, help="Path to the config file")
    parser.add_argument(
        "-i", "--iterations", type=int, required=True, help="Number of iterations"
    )
    parser.add_argument(
        "-t",
        "--time_acceleration",
        type=int,
        default=1,
        help="Time acceleration factor (1 iteration = time_acceleration generations)",
    )
    args = parser.parse_args()

    # Initialisation
    with open(args.config, "r", encoding="utf8") as json_file:
        config = json.load(json_file)

    result_dir = Path(config["Paths"]["Save"])

    g_init: int = str_to_int(config["Genome"]["g"])

    if config["Genome"]["z_c_auto"]:
        beta_init: int = str_to_int(config["Genome"]["beta"])
        z_c_init: int = beta_init * g_init
    else:
        z_c_init: int = str_to_int(config["Genome"]["z_c"])
        if z_c_init % g_init != 0:
            raise ValueError(
                f"z_c must be a multiple of g ({g_init}). You gave {z_c_init}."
            )
        beta_init: int = z_c_init // g_init

    n_init: int = str_to_int(config["Simulation"]["Population size"])
    mu_init: float = float(config["Mutation rates"]["Point mutation"])
    l_m_init: int = str_to_int(config["Mutations"]["l_m"])

    if config["Genome"]["z_nc_auto"]:
        alpha_init: int = str_to_int(config["Genome"]["alpha"])
        z_nc_init: int = alpha_init * g_init
    else:
        z_nc_init: int = str_to_int(config["Genome"]["z_nc"])
        if z_nc_init % g_init != 0:
            raise ValueError(
                f"z_c must be a multiple of g ({g_init}). You gave {z_nc_init}."
            )
        alpha_init: int = z_nc_init // g_init

    result_dir /= "_iterative_model"
    result_dir.mkdir(parents=True, exist_ok=True)

    # ## Computes the bias for the initial values
    # biais = biais_z_nc(
    #     g=Decimal(g_init),
    #     z_c=Decimal(z_c_init),
    #     z_nc=Decimal(z_nc_init),
    #     N=Decimal(n_init),
    #     mu=Decimal(mu_init),
    #     l_m=l_m_init,
    #     variable_n_e=True,
    # )
    # biais_constant_n_e = biais_z_nc(
    #     g=Decimal(g_init),
    #     z_c=Decimal(z_c_init),
    #     z_nc=Decimal(z_nc_init),
    #     N=Decimal(n_init),
    #     mu=Decimal(mu_init),
    #     l_m=l_m_init,
    #     variable_n_e=False,
    # )
    # print(f"Current biais for z_nc = {z_nc_init}: {biais}")
    # print(f"Current biais for z_nc = {z_nc_init}: {biais_constant_n_eee} (Constantn_ee)")

    # ## Finds the equilibrium by bisection
    # find_z_nc(
    #     g=Decimal(g_init),
    #     z_c=Decimal(z_c_init),
    #     n_e=Decimal(n_init),
    #     mu=Decimal(mu_init),
    #     l_m=l_m_init,
    #     variable_n_e=True,
    # )
    # find_z_nc(
    #     g=Decimal(g_init),
    #     z_c=Decimal(z_c_init),
    #     n_e=Decimal(n_init),
    #     mu=Decimal(mu_init),
    #     l_m=l_m_init,
    #     variable_n_e=False,
    # )

    ## Applies iteratively the model as an implicit sequence
    with open(result_dir / "config.json", "w", encoding="utf8") as json_file:
        json.dump(
            {
                "Iterations": args.iterations,
                "Time acceleration": args.time_acceleration,
            },
            json_file,
        )

    # Applies the iterative model with a variable n_e
    print(
        f"g: {g_init}, z_c: {z_c_init}, z_nc: {z_nc_init}, N: {n_init}, mu: {mu_init}"
    )
    results: tuple[np.ndarray, np.ndarray] = iterate(
        g=Decimal(g_init),
        z_c=Decimal(z_c_init),
        z_nc=Decimal(z_nc_init),
        n=Decimal(n_init),
        mu=Decimal(mu_init),
        l_m=l_m_init,
        iterations=args.iterations,
        time_acceleration=args.time_acceleration,
    )
    # Save results
    nc_proportions_final = results[0]
    n_es_final = results[1]
    np.save(result_dir / "nc_proportions.npy", nc_proportions_final)
    np.save(result_dir / "Nes.npy", n_es_final)

    plt.plot(nc_proportions_final)
    plt.title("Non-coding proportion")
    plt.xlabel("Iteration")
    plt.ylabel("Proportion")
    plt.show()
    plt.savefig(result_dir / "nc_proportions.png")

    # Applies the iterative model with a constant n_e
    print(
        f"g: {g_init}, z_c: {z_c_init}, z_nc: {z_nc_init}, N: {n_init}, mu: {mu_init}"
    )
    results: tuple[np.ndarray, np.ndarray] = iterate(
        g=Decimal(g_init),
        z_c=Decimal(z_c_init),
        z_nc=Decimal(z_nc_init),
        n=Decimal(n_init),
        mu=Decimal(mu_init),
        l_m=l_m_init,
        iterations=args.iterations,
        time_acceleration=args.time_acceleration,
        variable_n_e=False,
    )
    # Save results
    nc_proportions_final = results[0]
    np.save(result_dir / "nc_proportions_constant_Ne.npy", nc_proportions_final)

    plt.plot(nc_proportions_final)
    plt.title("Non-coding proportion")
    plt.xlabel("Iteration")
    plt.ylabel("Proportion")
    plt.show()
    plt.savefig(result_dir / "nc_proportions_constant_Ne.png")
