# import standard libraries
import os

# import third-party libraries
import imageio
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


def visualize_graphs_with_assignment(
    G1,
    G2,
    fixed_positions_G1,
    fixed_positions_G2,
    P,
    with_plot=None,
    node_labels_G1=None,
    node_labels_G2=None,
):
    """This function visualizes two graphs with node assignments between them.

    Args:
        G1 (_type_): Graph 1
        G2 (_type_): Graph 2
        fixed_positions_G1 (_type_): What are the position if fixed
        fixed_positions_G2 (_type_): What are the position if fixed
        P (_type_): list of P
        with_plot (_type_, optional): Plot or not. Defaults to None.
        node_labels_G1 (_type_, optional): add labels. Defaults to None.
        node_labels_G2 (_type_, optional): add labels. Defaults to None.
    """
    P = P.detach().numpy().copy()

    # Find the best assignment from nodes in G2 to G1 using argmax of matrix P
    assignments = np.argmax(P, axis=0)

    # Generate a color map for G1 nodes
    colors = plt.cm.rainbow(np.linspace(0, 1, G1.number_of_nodes()))

    # Create a color map for G2 nodes based on the assignments to G1 nodes
    color_map_G2 = ["grey"] * G2.number_of_nodes()

    for idx, node in enumerate(G2.nodes()):
        # If a node in G2 is assigned to a node in G1, use the same color and position
        if idx < len(assignments):
            assigned_node = assignments[idx]
            color_map_G2[node] = colors[assigned_node]

    # Draw G1 with the color map and fixed positions
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    nx.draw(
        G1,
        pos=fixed_positions_G1,
        with_labels=True,
        node_color=colors,
        labels=node_labels_G1,
    )
    plt.title("Graph G1")

    # Draw G2 with the color map and positions based on G1
    plt.subplot(122)
    nx.draw(
        G2,
        pos=fixed_positions_G2,
        with_labels=True,
        node_color=color_map_G2,
        labels=node_labels_G2,
    )
    plt.title("Graph G2 with Node Assignments")

    if with_plot is not None:
        plt.show()


def create_video(
    G1,
    G2,
    P_list,
    node_labels_G1=None,
    node_labels_G2=None,
    filename="graph_evolution.mp4",
    fixed_positions=None,
    fixed_positions_G2=None,
):
    """This function create a video from batch of plt plots

    Args:
        G1 (_type_): Grpah 1
        G2 (_type_): Grpah 2
        P_list (_type_): List matrices P
        node_labels_G1 (_type_, optional): add labels. Defaults to None.
        node_labels_G2 (_type_, optional): add labels. Defaults to None.
        filename (str, optional): output. Defaults to 'graph_evolution.mp4'.
        fixed_positions (_type_, optional): want to fix position during training. Defaults to None.
    """
    image_paths = []
    if fixed_positions is None:
        fixed_positions_G1 = nx.spring_layout(G1)
        if fixed_positions_G2 is None:
            fixed_positions_G2 = nx.spring_layout(G2)
    for i, P in enumerate(P_list):
        if i % 20 == 0:
            visualize_graphs_with_assignment(
                G1, G2, fixed_positions_G1, fixed_positions_G2, P
            )

            # Save each plot as an image
            image_path = f"/tmp/graph_{i}.png"
            plt.savefig(image_path)
            image_paths.append(image_path)
            plt.close()  # Close the figure to free memory

    with imageio.get_writer(filename, mode="I") as writer:
        for image_path in image_paths:
            image = imageio.imread(image_path)
            writer.append_data(image)

    # Optionally, remove the images after creating the video
    for image_path in image_paths:
        os.remove(image_path)


def draw_graphs_with_edge_color(list_G, with_different_colors=None):
    """
    Draw multiple graphs in a 2x2 subplot layout without labels, coloring the nodes and edges of the first graph in blue
    in the first three subplots and using default colors for the last one.

    Parameters:
    - G3: A list of graph objects.
    """
    # Define the spring layout for all graphs
    positions = [nx.spring_layout(G) for G in list_G]

    plt.figure(figsize=(6, 12))
    fig, axs = plt.subplots(2, 2, figsize=(10, 6))

    # For the first three graphs, draw nodes and edges that are present in the first graph in blue
    for i, graph in enumerate(list_G[:3]):
        if with_different_colors is not None:
            node_color = ["blue" if node in list_G[0].nodes else "C0" for node in graph]
            edge_color = [
                "blue" if edge in list_G[0].edges else "black" for edge in graph.edges
            ]
            nx.draw(
                graph,
                pos=positions[i],
                with_labels=False,
                ax=axs[i // 2, i % 2],
                node_color=node_color,
                edge_color=edge_color,
            )
        else:
            nx.draw(graph, pos=positions[i], with_labels=False, ax=axs[i // 2, i % 2])

        axs[i // 2, i % 2].set_title(f"Graph {i+1}")

    # Draw the last graph with default colors
    nx.draw(list_G[3], pos=positions[3], with_labels=False, ax=axs[1, 1])
    axs[1, 1].set_title("Graph 4")
    plt.tight_layout()
    plt.show()


def visualize_graphs_with_soft_assignment(
    G1,
    G2,
    fixed_positions_G1,
    fixed_positions_G2,
    P,
    with_plot=None,
    node_labels_G1=None,
    node_labels_G2=None,
):
    """Visualize two graphs with soft assignments as pie charts on nodes of G2 and colored nodes in G1.

    Args:
        G1 (networkx.Graph): Graph 1.
        G2 (networkx.Graph): Graph 2.
        fixed_positions_G1 (dict): Node positions for G1.
        fixed_positions_G2 (dict): Node positions for G2.
        P (np.ndarray): Assignment matrix P, with shape (n_G1, n_G2).
        with_plot (bool): If True, show the plot.
        node_labels_G1 (dict): Labels for the nodes of G1.
        node_labels_G2 (dict): Labels for the nodes of G2.
    """
    # detach P from the computation graph and convert to numpy array
    P = P.detach().numpy().copy()
    # Create a figure for the graphs
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Generate colors for each node in G1
    colors = plt.cm.rainbow(np.linspace(0, 1, G1.number_of_nodes()))

    # Draw G1 with fixed positions and colored nodes
    nx.draw_networkx(
        G1,
        pos=fixed_positions_G1,
        ax=ax[0],
        with_labels=True,
        node_color=colors,
        labels=node_labels_G1,
    )
    ax[0].set_title("Graph G1")

    # Draw G2 with fixed positions but no node colors yet (we will add pie charts instead)
    nx.draw_networkx_edges(G2, pos=fixed_positions_G2, ax=ax[1])
    nx.draw_networkx_labels(G2, pos=fixed_positions_G2, labels=node_labels_G2, ax=ax[1])
    ax[1].set_title("Graph G2 with Soft Assignments")

    # Function to add pie charts to nodes in G2
    def add_pie_charts_to_nodes(ax, positions, assignment_matrix):
        for node in positions:
            pie_size = 0.1  # Size of the pie charts
            trans = ax.transData.transform(
                positions[node]
            )  # Data to display coordinates
            trans = fig.transFigure.inverted().transform(
                trans
            )  # Display to figure coordinates
            pie_ax = fig.add_axes(
                [trans[0] - pie_size / 2, trans[1] - pie_size / 2, pie_size, pie_size],
                aspect="equal",
            )
            pie_ax.set_xticks([])
            pie_ax.set_yticks([])
            probabilities = assignment_matrix[:, node]
            pie_ax.pie(probabilities, colors=colors)

    # Add pie charts to nodes in G2
    add_pie_charts_to_nodes(ax[1], fixed_positions_G2, P)

    # Show the plot if requested
    if with_plot is not None:
        plt.show()


def create_video_soft(
    G1,
    G2,
    P_list,
    node_labels_G1=None,
    node_labels_G2=None,
    filename="graph_evolution.mp4",
    fixed_positions=None,
):
    """This function create a video from batch of plt plots

    Args:
        G1 (_type_): Grpah 1
        G2 (_type_): Grpah 2
        P_list (_type_): List matrices P
        node_labels_G1 (_type_, optional): add labels. Defaults to None.
        node_labels_G2 (_type_, optional): add labels. Defaults to None.
        filename (str, optional): output. Defaults to 'graph_evolution.mp4'.
        fixed_positions (_type_, optional): want to fix position during training. Defaults to None.
    """
    image_paths = []
    if fixed_positions is None:
        fixed_positions_G1 = nx.spring_layout(G1)
        fixed_positions_G2 = nx.spring_layout(G2)
    for i, P in enumerate(P_list):
        if i % 20 == 0:
            visualize_graphs_with_soft_assignment(
                G1, G2, fixed_positions_G1, fixed_positions_G2, P
            )

            # Save each plot as an image
            image_path = f"/tmp/graph_{i}.png"
            plt.savefig(image_path)
            image_paths.append(image_path)
            plt.close()  # Close the figure to free memory

    with imageio.get_writer(filename, mode="I") as writer:
        for image_path in image_paths:
            image = imageio.imread(image_path)
            writer.append_data(image)

    # Optionally, remove the images after creating the video
    for image_path in image_paths:
        os.remove(image_path)

    return fixed_positions_G2
