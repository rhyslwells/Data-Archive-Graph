import networkx as nx
import matplotlib.pyplot as plt


def helloworld():
    """Simple function to print 'Hello World'"""
    print("Hello world")

def plot_graph(graph, figsize=(8, 6), label_nodes=False, center_title="", font_size=14, k=0.1):
    """
    Function to plot the graph.
    
    Args:
        graph (networkx.Graph): The graph to plot.
        figsize (tuple): Size of the plot.
        label_nodes (bool): Whether to label the nodes.
        center_title (str): Title for the plot (e.g., center node).
        font_size (int): Font size for node labels.
        k (float): Constant for layout to adjust edge length.
    """
    plt.figure(figsize=figsize)
    pos = nx.spring_layout(graph, k=k, seed=42)  # Reduced k for shorter edges
    nx.draw(graph, pos, node_size=80, edge_color='gray', alpha=0.6)
    
    if label_nodes:
        nx.draw_networkx_labels(graph, pos, font_size=font_size)  # Increased font_size here
    
    plt.title(f'1-hop Neighborhood of: {center_title}')
    plt.axis('off')
    plt.show()

def get_subgraph(G,center_title, depth=1, remove_loops=False):
    """
    Function to get a subgraph around a center node.
    
    Args:
        center_title (str): The center node's title to generate the ego graph.
        depth (int): The radius of the ego graph.
        remove_loops (bool): Whether to remove self-loops from the subgraph.
    
    Returns:
        networkx.Graph: The generated subgraph.
    """
    if center_title not in G:
        print(f"'{center_title}' not found in graph.")
        return None
    
    # Generate ego graph
    sub = nx.ego_graph(G, center_title, radius=depth)
    
    # Remove self-loops if specified
    if remove_loops:
        self_loops = list(nx.selfloop_edges(sub))
        sub.remove_edges_from(self_loops)
    
    print(f"Subgraph has {len(sub)} nodes and {sub.number_of_edges()} edges")
    return sub
