"""
graph_builder.py
Module for building NetworkX graphs from node data.
"""

import networkx as nx
from typing import List, Dict, Any


def build_graph(data: List[Dict[str, Any]], remove_isolated: bool = False) -> nx.DiGraph:
    """
    Build NetworkX directed graph from filtered data.
    
    Args:
        data: List of node dictionaries
        remove_isolated: If True, remove nodes with no connections
        
    Returns:
        NetworkX DiGraph object
    """
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for item in data:
        G.add_node(
            item['normalized_filename'],
            title=item['filename'],
            category=item.get('category', ''),
            tags=item.get('tags', []),
            url=item.get('url', ''),
            label=item['filename']
        )
    
    # Add edges based on outlinks
    for item in data:
        source = item['normalized_filename']
        for target in item.get('outlinks', []):
            if target in G.nodes():
                G.add_edge(source, target)
    
    # Remove isolated nodes if requested
    if remove_isolated:
        isolated_nodes = list(nx.isolates(G))
        G.remove_nodes_from(isolated_nodes)
        if isolated_nodes:
            print(f"Removed {len(isolated_nodes)} isolated nodes")
    
    return G


def get_largest_component(G: nx.DiGraph) -> nx.DiGraph:
    """
    Extract the largest weakly connected component from the graph.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Subgraph containing the largest component
    """
    if G.number_of_nodes() == 0:
        return G
    
    # Get all weakly connected components
    components = list(nx.weakly_connected_components(G))
    
    if len(components) == 1:
        return G
    
    # Find the largest component
    largest_component = max(components, key=len)
    
    print(f"Extracted largest component: {len(largest_component)} of {G.number_of_nodes()} nodes")
    
    return G.subgraph(largest_component).copy()


def filter_by_degree(G: nx.DiGraph, min_degree: int = 1) -> nx.DiGraph:
    """
    Filter graph to keep only nodes with at least min_degree connections.
    
    Args:
        G: NetworkX DiGraph
        min_degree: Minimum total degree (in-degree + out-degree)
        
    Returns:
        Filtered subgraph
    """
    nodes_to_keep = [node for node in G.nodes() if G.degree(node) >= min_degree]
    
    removed = G.number_of_nodes() - len(nodes_to_keep)
    if removed > 0:
        print(f"Removed {removed} nodes with degree < {min_degree}")
    
    return G.subgraph(nodes_to_keep).copy()


def get_ego_graph(G: nx.DiGraph, center_node: str, radius: int = 1) -> nx.DiGraph:
    """
    Get the ego graph (neighborhood) around a center node.
    
    Args:
        G: NetworkX DiGraph
        center_node: Node to center the ego graph on
        radius: Distance from center node to include
        
    Returns:
        Ego subgraph
    """
    if center_node not in G:
        print(f"Warning: '{center_node}' not found in graph")
        return nx.DiGraph()
    
    ego = nx.ego_graph(G, center_node, radius=radius)
    print(f"Ego graph around '{center_node}': {ego.number_of_nodes()} nodes, {ego.number_of_edges()} edges")
    
    return ego


# Example usage
if __name__ == '__main__':
    from data_loader import load_data, filter_by_tag
    
    # Load and filter data
    data_path = '../data/1-categories_snapshot_linked.json'
    raw_data = load_data(data_path)
    filtered_data = filter_by_tag(raw_data, 'analysis')
    
    # Build graph
    print("\n=== Building Full Graph ===")
    G = build_graph(filtered_data)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    
    # Build graph without isolated nodes
    print("\n=== Building Graph (No Isolated Nodes) ===")
    G_no_isolated = build_graph(filtered_data, remove_isolated=True)
    print(f"Nodes: {G_no_isolated.number_of_nodes()}, Edges: {G_no_isolated.number_of_edges()}")
    
    # Get largest component
    print("\n=== Largest Component ===")
    G_largest = get_largest_component(G)
    print(f"Nodes: {G_largest.number_of_nodes()}, Edges: {G_largest.number_of_edges()}")
    
    # Filter by minimum degree
    print("\n=== Filter by Degree >= 2 ===")
    G_filtered = filter_by_degree(G, min_degree=2)
    print(f"Nodes: {G_filtered.number_of_nodes()}, Edges: {G_filtered.number_of_edges()}")