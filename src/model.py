# import thrird-party libraries
import torch
import numpy as np

# import local libraries
from src.utils import adjency_matrix, degree_matrix, sqrt_matrix, initialize_matrices_normal, initialize_matrices_xavier

def P_C0_KL(X: torch.Tensor, kmax: int) -> torch.Tensor:
    """for even iteration Dykstra algorithm

    Args:
        X (torch.Tensor): P
        kmax (int): maximum number of matching

    Returns:
        torch.Tensor: result of P_C0_KL
    """
    denominator = torch.sum(X, dim=1)  # sum_j X_ij
    numerator = torch.clamp(
        denominator, min=1, max=kmax
    )  # max(1, min(kmax, sum_j X_ij))
    ratio = numerator / denominator
    diag = torch.diag(ratio)  # diag(numerator / denominator)
    return torch.mm(diag, X)  # diag(numerator / denominator) @ X


def P_C1_KL(X: torch.Tensor) -> torch.Tensor:
    """for odd iteration Dykstra algorithm

    Args:
        X (torch.Tensor): P

    Returns:
        torch.Tensor: result of P_C1_KL
    """
    return torch.mm(
        X, torch.diag(1.0 / torch.sum(X, dim=0))
    )  # X @ diag(1 / sum_i X_ij)


def dykstra_operateur(
    P: torch.Tensor, kmax: int, tau: int, nb_iter_dykstra: int
) -> torch.Tensor:
    """compute Dykstra algorithm

    Args:
        P (torch.Tensor): Matrix P
        kmax (int): maximum number of matching
        tau (int): Dykstra algorithm parameter
        nb_iter_dykstra (int): number of iteration

    Returns:
        torch.Tensor: result of Dykstra algorithm
    """

    P_curr = torch.exp(P / tau)
    Q_prev = torch.ones_like(P)
    Q_curr = torch.ones_like(P)

    for t in range(nb_iter_dykstra):
        if t % 2 == 0:
            P_next = P_C0_KL(P_curr * Q_prev, kmax)
        else:
            P_next = P_C1_KL(P_curr * Q_prev)

        temp = P_curr * Q_prev / P_next
        Q_prev = Q_curr
        Q_curr = temp
        P_curr = P_next

    return P_curr

def compute_cost(
    samples: torch.Tensor,
    eta_t: torch.Tensor,
    sigma_t: torch.Tensor,
    L1: torch.Tensor,
    L2: torch.Tensor,
    tau: int,
    kmax: int,
    nb_iter_dykstra: int,
):
    """_summary_

    Args:
        samples (torch.Tensor): samples
        eta_t (torch.Tensor): parameter eta
        sigma_t (torch.Tensor): parameter sigma
        L1 (torch.Tensor): Laplacian matrix of graph 1
        L2 (torch.Tensor): Laplacian matrix of graph 2
        tau (int): parameter of Dykstra algorithm
        kmax (int): maximum number of matching
        nb_iter_dykstra (int): number of iteration of Dykstra algorithm

    Returns:
        _type_: Wasserstein distance of S samples
    """
    cost = 0
    for s in range(samples.shape[0]):
        P = dykstra_operateur(
            eta_t + sigma_t * samples[s, ...], kmax, tau, nb_iter_dykstra
        )
        # print('P', P)
        L2_prime = torch.mm(torch.mm(P, L2), P.T)
        # print(A)
        # compute_pseudo_inverse_laplacian(L1)
        # compute_pseudo_inverse_laplacian(A)
        L1_inv = torch.pinverse(L1)
        L1_inv_sqrt = sqrt_matrix(L1_inv)
        L2_inv = torch.pinverse(L2_prime, rcond=1e-1)

        L1L2L1 = torch.mm(torch.mm(L1_inv_sqrt, L2_inv), L1_inv_sqrt)
        # print(L2_inv)

        W = torch.trace(L1_inv + L2_inv) - 2 * torch.trace(sqrt_matrix(L1L2L1))
        cost += W
    # print(W)
    cost /= samples.shape[0]
    return cost, P


def draw_samples(S: int, V1: int, V2: int) -> torch.Tensor:
    """Draw samples from a unit normal distribution

    Args:
        S (int): Number of samples
        V1 (int): Number of nodes in graph 1
        V2 (int): Number of nodes in graph 2

    Returns:
        torch.Tensor: samples
    """
    return torch.randn(S, V1, V2, dtype=torch.float64)


def compute_Wasserstein_distance(
    G1, G2, cfg, with_plot=False, with_scheduler=False, with_xavier=False
) -> tuple[torch.Tensor, torch.Tensor, list]:
    """_summary_

    Args:
        G1 (_type_): Graph 1
        G2 (_type_): Graph 2
        cfg (_type_): Configuration for the algorithm parameters
        with_plot (bool, optional): display training cost. Defaults to False.
        with_scheduler (bool, optional): use scheduler of not. Defaults to False.
        with_xavier (bool, optional): use Xavier initilization. Defaults to False.

    Returns:
        _type_: eta, sigma and list of cost
    """
    # let V1 and V2 the number of nodes in G1 and G2
    V1 = len(G1.nodes())
    V2 = len(G2.nodes())

    # compute adjency matrix and degree matrix
    if not with_xavier:
        eta_0, sigma_0 = initialize_matrices_normal(V1, V2)
    else:
        eta_0, sigma_0 = initialize_matrices_xavier(V1, V2)

    A1 = adjency_matrix(G1)
    D1 = degree_matrix(G1)
    A2 = adjency_matrix(G2)
    D2 = degree_matrix(G2)

    # comptue the laplacian matrix
    L1 = torch.Tensor(np.subtract(D1, A1)).to(torch.float64)
    L2 = torch.Tensor(np.subtract(D2, A2)).to(torch.float64)

    optimizer = torch.optim.Adam([eta_0, sigma_0], lr=cfg["gamma"], amsgrad=True)
    #
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[700], gamma=0.5
    )
    list_cost = []
    list_matrix_P = []
    for t in range(cfg["nb_iter_grad"]):
        # print(t)
        optimizer.zero_grad()
        samples = draw_samples(cfg["S"], V1, V2)
        # print(samples.type())
        # print(L1.type())

        cost, P = compute_cost(
            samples,
            eta_0,
            sigma_0,
            L1,
            L2,
            cfg["tau"],
            cfg["kmax"],
            cfg["nb_iter_dykstra"],
        )
        list_matrix_P.append(P)
        cost.backward()
        optimizer.step()
        if with_scheduler:
            scheduler.step()
        list_cost.append(cost.item())

        if (t % 100 == 0) and with_plot:
            print(f"iteration {t}    the Wasserstein distance is {cost.item()}")

    return eta_0, sigma_0, list_cost, list_matrix_P