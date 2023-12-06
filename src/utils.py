# Import third party libraries
import numpy as np
import networkx as nx
import torch
import matplotlib.pyplot as plt

# create of function that return adjency matrix
def adjency_matrix(G: nx.Graph) -> np.ndarray:
    """Create adjency matrix from graph

    Args:
        G (nx.Graph): Any graph

    Returns:
        np.ndarray: adjency matrix
    """
    n = len(G.nodes())
    A = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if (i, j) in G.edges():
                A[i][j] = 1
    return np.array(A)


# create a function that return the degree matrix
def degree_matrix(G: nx.Graph) -> np.ndarray:
    """create a function that return the degree matrix

    Args:
        G (nx.Graph): any graph

    Returns:
        np.ndarray: degree matrix
    """
    n = len(G.nodes())
    D = [[0 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        D[i][i] = G.degree(i)
    return np.array(D)


def initialize_matrices_xavier(V1: int, V2: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stand way of initializing eta and sigma in Deep Learning

    Args:
        V1 (int): Number of nodes in graph 1
        V2 (int): Number of nodes in graph 2

    Returns:
        tuple[torch.Tensor, torch.Tensor]: initialized eta and sigma
    """
    eta = torch.empty(V1, V2, requires_grad=True, dtype=torch.float64)
    sigma = torch.empty(V1, V2, requires_grad=True, dtype=torch.float64)

    # Initialize with Xavier uniform distribution
    torch.nn.init.xavier_uniform_(eta)
    torch.nn.init.xavier_uniform_(sigma)

    return eta, sigma


def initialize_matrices_normal(V1: int, V2: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Stand way of initializing eta and sigma in Deep Learning

    Args:
        V1 (int): Number of nodes in graph 1
        V2 (int): Number of nodes in graph 2

    Returns:
        tuple[torch.Tensor, torch.Tensor]: initialized eta and sigma
    """
    eta = torch.randn(V1, V2, requires_grad=True, dtype=torch.float64)
    sigma = torch.randn(V1, V2, requires_grad=True, dtype=torch.float64)
    return eta, sigma

def sqrt_matrix(laplacian_matrix: torch.Tensor) -> torch.Tensor:
    """Compute square root of a matrix with none full rank

    Args:
        laplacian_matrix (torch.Tensor): Laplacian matrix

    Returns:
        torch.Tensor: Square root of the matrix
    """
    eigenvalues, eigenvectors = torch.linalg.eigh(laplacian_matrix)
    # Clamp negative eigenvalues to zero
    sqrt_eigenvalues = torch.sqrt(eigenvalues.clamp(min=0))
    laplacian_sqrt_matrix = torch.mm(
        torch.mm(eigenvectors, torch.diag(sqrt_eigenvalues)), eigenvectors.T
    )
    return laplacian_sqrt_matrix


def create_random_connected_graph(n):
    G = nx.erdos_renyi_graph(n, 0.5)
    while not nx.is_connected(G):
        G = nx.erdos_renyi_graph(n, 0.5)
    return G


def add_random_edges(G, n):
    # start by dooin a copy of the graph
    G_prime = G.copy()
    for i in range(n):
        while True:
            u = np.random.randint(0, len(G_prime.nodes()))
            v = np.random.randint(0, len(G_prime.nodes()))
            if u != v or (u, v) not in G_prime.edges():
                G_prime.add_edge(u, v)
                break
    return G_prime


def add_random_nodes(G, n):
    G_prime = G.copy()
    for _ in range(n):
        G_prime.add_node(len(G_prime.nodes()))
        # create a random edge to existing node
        u = np.random.randint(0, len(G_prime.nodes()) - 1)
        G_prime.add_edge(u, len(G_prime.nodes()) - 1)
    return G_prime


def create_graph_type_cycle(n_cycles, nodes_per_cycle, with_plot=False):
    # Create an empty graph
    G_common_edge_corrected = nx.Graph()

    # Add cycles and connect them with a shared edge
    for i in range(n_cycles):
        # Create a cycle graph with 'nodes_per_cycle' nodes
        cycle_nodes = list(
            range(
                i * (nodes_per_cycle - 1), i * (nodes_per_cycle - 1) + nodes_per_cycle
            )
        )
        cycle_graph = nx.cycle_graph(cycle_nodes)

        # Merge the current cycle graph with the main graph
        G_common_edge_corrected = nx.compose(G_common_edge_corrected, cycle_graph)

    # Since each cycle graph is created with one overlapping node from the previous cycle
    # this automatically creates a shared edge between consecutive cycles.
    if with_plot:
        # Draw the graph
        pos_common_edge_corrected = nx.spring_layout(G_common_edge_corrected)
        nx.draw(G_common_edge_corrected, pos_common_edge_corrected, with_labels=True)

        # Show the plo
        plt.show()
    return G_common_edge_corrected


def create_tree(branching_factor, depth, with_plot=False):
    # Create a directed graph
    G = nx.DiGraph()

    # Add root node
    G.add_node(0)

    # This nested function will be used to add nodes and edges to the graph
    def add_nodes_edges(parent_node, current_depth):
        if current_depth < depth:
            for i in range(branching_factor):
                child_node = len(G.nodes)
                G.add_node(child_node)
                G.add_edge(parent_node, child_node)
                add_nodes_edges(child_node, current_depth + 1)

    # Start adding nodes from the root
    add_nodes_edges(0, 0)

    if with_plot:
        # Draw the graph
        nx.draw(G, with_labels=True, arrows=False)

        # Show the graph
        plt.show()
    return G