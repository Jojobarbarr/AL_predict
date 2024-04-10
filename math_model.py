import numpy as np
from time import perf_counter
import matplotlib.pyplot as plt
from scipy.optimize import bisect
from decimal import Decimal, getcontext

getcontext().prec = 40


def update(
    N,
    g,
    z_c,
    z_nc,
    mu,
    l_m,
):
    L = z_c + z_nc
    alpha = int(z_nc / g)
    segment_length = int(L / g)
    N *= R(mu, L, z_nc, g, l_m)
    return L, alpha, segment_length, N


# On calcul la probabilité d'être neutre pour chaque type de mutation :
# On en aura besoin pour calculer la robustesse (résistance aux mutations) de chaque génome
# Délétion
def vdel(z_nc, g, L):
    return Decimal((z_nc + g) * z_nc / (2 * g * L * L))


# DuplicationS
def vdupl(z_nc, g, L):
    return Decimal((z_nc + g) * (L - g) / (2 * g * L * L))


# Inversion
def vinv(z_nc, g, L):
    return Decimal((z_nc + g) * (z_nc + g - 1) / (L * (L - 1)))


# Mutation ponctuelle
def vpm(z_nc, L):
    return Decimal(z_nc / L)


# Petite insertion
def vindel_plus(z_nc, g, L):
    return Decimal((z_nc + g) / L)


# Petite délétion
def vindel_moins(z_nc, g, L, l_m):
    return Decimal(g / L * (z_nc / g - (Decimal(l_m) - 1) / 2))


# Calcul de la robustesse
# Probabilité qu'une réplication (avec 0 ou 1 mutation)
def R(mu, L, z_nc, g, l_m):  # same as mean fitness of possible descendants
    s = Decimal(0)

    neutr_by_pm = Decimal((1 - mu * (1 - vpm(z_nc, L))) ** (L))
    neutr_by_indel_plus = Decimal((1 - mu * (1 - vindel_plus(z_nc, g, L))) ** (L))
    neutr_by_indel_moins = Decimal(
        (1 - mu * (1 - vindel_moins(z_nc, g, L, l_m))) ** (L)
    )
    neutr_by_del = Decimal((1 - mu * (1 - vdel(z_nc, g, L))) ** (L))
    neutr_by_dupl = Decimal((1 - mu * (1 - vdupl(z_nc, g, L))) ** (L))
    neutr_by_inv = Decimal((1 - mu * (1 - vinv(z_nc, g, L))) ** (L))

    s = (
        neutr_by_pm
        * neutr_by_indel_plus
        * neutr_by_indel_moins
        * neutr_by_del
        * neutr_by_dupl
        * neutr_by_inv
    )
    return s


# On calcul le ratio des robustesses avant et après une mutation
# Ça nous permet de regarder l'avantage ou désavantage sélectif d'une mutation
def ratio(x, mu, L, z_nc, g, l_m):
    return R(mu, L, z_nc, g, l_m) / R(mu, L + x, z_nc + x, g, l_m)


# Robustness WT over robustness mutant


# Version avec la taille de non codant au lieu de la proportion
def pfix_z_nc(x, mu, z_c, z_nc, g, Ne, l_m):
    L = Decimal(z_c + z_nc)
    R = ratio(x, mu, L, z_nc, g, l_m)
    # A = (1 - R) / (1 - R ** (2 * Ne))  # diploid
    A = (1 - (R * R)) / (1 - R ** (2 * Ne))  # haploid
    return Decimal(A)


# On calcul la taille moyenne des délétions fixées
def deltadel_fix_z_nc(g, L, alpha, pre_computed_tablePfix):
    factor = Decimal(g / (L * L))
    tablePfix = [
        (alpha + 1 - Decimal(k)) * Decimal(k) * pre_computed_tablePfix[k - 1]
        for k in range(1, alpha + 1)
    ]
    # this dictionnary will be used to store the data we already computed
    return factor * Decimal(np.sum(tablePfix))


# On calcul la taille moyenne des duplications fixées
def deltadupl_fix_z_nc(g, L, alpha, segment_length, pre_computed_tablePfix):
    factor = Decimal((g * g * (alpha + 1)) / (L * L * L))

    tablePfix = [
        (segment_length - Decimal(k)) * Decimal(k) * pre_computed_tablePfix[k - 1]
        for k in range(1, segment_length)
    ]
    return factor * Decimal(np.sum(tablePfix))


# On calcul la taille moyenne des petites insertions fixées (Indels)
def delta_indelplus_fix_z_nc(g, z_nc, l_m, L, pre_computed_tablePfix):
    factor = Decimal((z_nc + g) / (l_m * L))
    # this dictionnary will be used to store the data we already computed
    return factor * Decimal(
        np.sum([(Decimal(k + 1)) * pre_computed_tablePfix[k] for k in range(0, l_m)])
    )


# On calcul la taille moyenne des petites deletions fixées (Indels)
def delta_indelmoins_fix_z_nc(g, l_m, L, alpha, pre_computed_tablePfix):
    factor = Decimal(g / (l_m * L))
    # this dictionnary will be used to store the data we already computed

    first_part = Decimal(max(alpha - (l_m - 1), 0)) * np.sum(
        [(Decimal(k + 1)) * pre_computed_tablePfix[k] for k in range(0, l_m)]
    )

    steps = min(l_m - 1, alpha)
    second_part = Decimal(
        np.sum(
            [
                np.sum(
                    [Decimal(k + 1) * pre_computed_tablePfix[k] for k in range(0, s)]
                )
                for s in range(1, steps + 1)
            ]
        )
    )

    return factor * Decimal((first_part + second_part))


def tablePfix(g, z_c, z_nc, Ne, mu, l_m, segment_length, alpha):
    Pfix_positive_k = [
        pfix_z_nc(Decimal(k), mu, z_c, z_nc, g, Ne, l_m)
        for k in range(1, max(segment_length, l_m + 1))
    ]
    Pfix_negative_k = [
        pfix_z_nc(Decimal(-k), mu, z_c, z_nc, g, Ne, l_m)
        for k in range(1, max(alpha + 1, l_m + 1))
    ]
    return Pfix_positive_k, Pfix_negative_k


# On calcul le rapport des tailles moyennes de duplications et délétions fixées:
# On est à l'équilibre pour le biais 1
def biais_z_nc(g, z_c, z_nc, N, mu, l_m):
    g = Decimal(g)
    z_c = Decimal(z_c)
    z_nc = Decimal(z_nc)
    N = Decimal(N)
    mu = Decimal(mu)

    L, alpha, segment_length, Ne = update(N, g, z_c, z_nc, mu, l_m)
    # print(f"Ne: {Ne}")

    pre_computed_tablePfix_positive_k, pre_computed_tablePfix_negative_k = tablePfix(
        g, z_c, z_nc, Ne, mu, l_m, segment_length, alpha
    )
    biais_moins = deltadel_fix_z_nc(
        g, L, alpha, pre_computed_tablePfix_negative_k
    ) + delta_indelmoins_fix_z_nc(g, l_m, L, alpha, pre_computed_tablePfix_negative_k)
    biais_plus = deltadupl_fix_z_nc(
        g, L, alpha, segment_length, pre_computed_tablePfix_positive_k
    ) + delta_indelplus_fix_z_nc(g, z_nc, l_m, L, pre_computed_tablePfix_positive_k)
    return biais_moins / biais_plus


def iterate(g, z_c, z_nc, N, mu, l_m, iterations, time_acceleration):
    g = Decimal(g)
    z_c = Decimal(z_c)
    z_nc = Decimal(z_nc)
    N = Decimal(N)
    mu = Decimal(mu)

    L, alpha, segment_length, Ne = update(N, g, z_c, z_nc, mu, l_m)
    print(f"Initial living proportion: {Ne / N}")
    progress_point = iterations // 20
    nc_proportions = []
    Nes = []

    start_time = perf_counter()
    try:
        for iteration in range(iterations + 1):
            pre_computed_tablePfix_positive_k, pre_computed_tablePfix_negative_k = (
                tablePfix(g, z_c, z_nc, Ne, mu, l_m, segment_length, alpha)
            )
            delta_indelmoins = delta_indelmoins_fix_z_nc(
                g, l_m, L, alpha, pre_computed_tablePfix_negative_k
            )
            delta_del = deltadel_fix_z_nc(
                g, L, alpha, pre_computed_tablePfix_negative_k
            )
            bases_deleted = mu * L * N * (delta_indelmoins + delta_del)

            delta_indelplus = delta_indelplus_fix_z_nc(
                g, z_nc, l_m, L, pre_computed_tablePfix_positive_k
            )
            delta_dupl = deltadupl_fix_z_nc(
                g, L, alpha, segment_length, pre_computed_tablePfix_positive_k
            )
            bases_inserted = mu * L * N * (delta_indelplus + delta_dupl)

            delta = time_acceleration * (bases_inserted - bases_deleted)

            z_nc += delta
            L, alpha, segment_length, Ne = update(N, g, z_c, z_nc, mu, l_m)
            nc_proportions.append(z_nc / L)
            Nes.append(Ne)
            if iteration != 0 and iteration % progress_point == 0:
                checkpoint_time = perf_counter() - start_time
                print(
                    f"Iteration {iteration}/{iterations} - {checkpoint_time:.2f}s elapsed - Remaining about {(iterations - iteration) * (checkpoint_time / iteration):.2f}s"
                )
    except KeyboardInterrupt:
        L, alpha, segment_length, Ne = update(N, g, z_c, z_nc, mu, l_m)
    print(f"Iterations: {iteration}")
    print(f"Final non-coding proportion: {z_nc / L} (z_nc = {z_nc})")
    plt.clf()
    plt.plot(nc_proportions)
    plt.title("Non-coding proportion over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Non-coding proportion")
    plt.show()
    plt.clf()
    plt.plot(Nes)
    plt.title("Effective population size over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Ne")
    plt.show()
    plt.clf()
    plt.plot([Ne / N for Ne in Nes])
    plt.title("Living proportion over iterations")
    plt.xlabel("Iteration")
    plt.ylabel("Living proportion")
    plt.show()
    plt.clf()
    plt.plot([Ne / N for Ne in Nes], nc_proportions)
    plt.title("Living proportion over non-coding proportion")
    plt.xlabel("Living proportion")
    plt.ylabel("Non-coding proportion")
    plt.show()


def funcTargetN(
    x,
    g,
    z_c,
    N,
    mu,
    l_m,
):
    result = float(biais_z_nc(g=g, z_c=z_c, z_nc=x, N=N, mu=mu, l_m=l_m) - 1)
    return result


def find_z_nc(
    g,
    z_c,
    Ne,
    mu,
    l_m,
):
    max_proportion = 0.99
    max = max_proportion * z_c / (1 - max_proportion)
    z_nc = bisect(funcTargetN, 0, int(max), args=(g, z_c, Ne, mu, l_m))
    print(f"Final proportion: {z_nc / (z_c + z_nc)} (z_nc = {z_nc})")


if __name__ == "__main__":
    # start = perf_counter()
    # for _ in range(1000):
    #     biais_z_nc(z_c=100, z_nc=100, g=10, Ne=100, mu=0.001, l_m=10)
    # end = perf_counter()
    # print(biais_z_nc(z_c=100, z_nc=100, g=10, Ne=100, mu=0.001, l_m=10))
    # print(biais_z_nc(z_c=1000, z_nc=100, g=100, Ne=100, mu=0.0001, l_m=10))
    # print(biais_z_nc(z_c=1000, z_nc=1000, g=100, Ne=100, mu=0.0001, l_m=10))
    # print(biais_z_nc(z_c=1000, z_nc=1000, g=100, Ne=100, mu=0.0001, l_m=1000))
    # print(f"Time: {end - start}")

    g = 10
    beta = 10
    z_c = beta * g
    N = 64
    mu = 0.00001
    l_m = 10

    nc_proportion = 0
    z_nc = nc_proportion * z_c / (1 - nc_proportion)
    print(
        f"Current biais for z_nc = {z_nc}: {biais_z_nc(g=g, z_c=z_c, z_nc=z_nc, N=N, mu=mu, l_m=l_m)}"
    )

    find_z_nc(g=g, z_c=z_c, Ne=N, mu=mu, l_m=l_m)

    iterate(
        g=g,
        z_c=z_c,
        z_nc=z_nc,
        N=N,
        mu=mu,
        l_m=l_m,
        iterations=1000,
        time_acceleration=500,
    )
