"""
network_stats.py
Module for calculating and displaying network statistics.
"""

import networkx as nx
from typing import Dict, List, Tuple


def calculate_basic_stats(G: nx.DiGraph) -> Dict:
    """
    Calculate basic network statistics.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Dictionary of basic statistics
    """
    stats = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'density': nx.density(G) if G.number_of_nodes() > 0 else 0,
        'is_connected': nx.is_weakly_connected(G),
        'num_components': nx.number_weakly_connected_components(G)
    }
    
    return stats


def calculate_centrality_measures(G: nx.DiGraph, top_n: int = 5) -> Dict:
    """
    Calculate various centrality measures for the graph.
    
    Args:
        G: NetworkX DiGraph
        top_n: Number of top nodes to return for each measure
        
    Returns:
        Dictionary of centrality measures
    """
    centrality = {}
    
    if G.number_of_nodes() == 0:
        return centrality
    
    # Degree centrality
    in_degree = dict(G.in_degree())
    out_degree = dict(G.out_degree())
    total_degree = {node: G.degree(node) for node in G.nodes()}
    
    centrality['top_in_degree'] = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:top_n]
    centrality['top_out_degree'] = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:top_n]
    centrality['top_total_degree'] = sorted(total_degree.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    # PageRank (for directed graphs)
    try:
        pagerank = nx.pagerank(G)
        centrality['top_pagerank'] = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:top_n]
    except:
        centrality['top_pagerank'] = []
    
    # Betweenness centrality (computationally expensive for large graphs)
    if G.number_of_nodes() < 100:
        try:
            betweenness = nx.betweenness_centrality(G)
            centrality['top_betweenness'] = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:top_n]
        except:
            centrality['top_betweenness'] = []
    
    return centrality


def calculate_clustering_stats(G: nx.DiGraph) -> Dict:
    """
    Calculate clustering statistics.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Dictionary of clustering statistics
    """
    stats = {}
    
    if G.number_of_nodes() == 0:
        return stats
    
    # Convert to undirected for clustering coefficient
    G_undirected = G.to_undirected()
    
    try:
        stats['average_clustering'] = nx.average_clustering(G_undirected)
        stats['transitivity'] = nx.transitivity(G_undirected)
    except:
        stats['average_clustering'] = 0
        stats['transitivity'] = 0
    
    return stats


def find_isolated_nodes(G: nx.DiGraph) -> List[str]:
    """
    Find all isolated nodes (nodes with no connections).
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        List of isolated node IDs
    """
    return list(nx.isolates(G))


def print_statistics(G: nx.DiGraph, detailed: bool = True) -> None:
    """
    Print comprehensive network statistics.
    
    Args:
        G: NetworkX DiGraph
        detailed: If True, print detailed statistics including centrality measures
    """
    print("\n" + "="*60)
    print("NETWORK STATISTICS")
    print("="*60)
    
    # Basic stats
    basic_stats = calculate_basic_stats(G)
    print(f"\n📊 Basic Statistics:")
    print(f"  Nodes: {basic_stats['num_nodes']}")
    print(f"  Edges: {basic_stats['num_edges']}")
    print(f"  Density: {basic_stats['density']:.4f}")
    print(f"  Weakly Connected: {'Yes' if basic_stats['is_connected'] else 'No'}")
    print(f"  Number of Components: {basic_stats['num_components']}")
    
    # Isolated nodes
    isolated = find_isolated_nodes(G)
    if isolated:
        print(f"\n⚠️  Isolated Nodes: {len(isolated)}")
        if len(isolated) <= 10:
            print(f"  {', '.join(isolated)}")
    
    if not detailed or G.number_of_nodes() == 0:
        return
    
    # Centrality measures
    print(f"\n🎯 Centrality Measures:")
    centrality = calculate_centrality_measures(G, top_n=5)
    
    if centrality.get('top_in_degree'):
        print(f"\n  Top 5 by In-Degree (Most Referenced):")
        for node, degree in centrality['top_in_degree']:
            print(f"    {node}: {degree}")
    
    if centrality.get('top_out_degree'):
        print(f"\n  Top 5 by Out-Degree (Most References):")
        for node, degree in centrality['top_out_degree']:
            print(f"    {node}: {degree}")
    
    if centrality.get('top_pagerank'):
        print(f"\n  Top 5 by PageRank:")
        for node, score in centrality['top_pagerank']:
            print(f"    {node}: {score:.4f}")
    
    if centrality.get('top_betweenness'):
        print(f"\n  Top 5 by Betweenness Centrality:")
        for node, score in centrality['top_betweenness']:
            print(f"    {node}: {score:.4f}")
    
    # Clustering stats
    clustering_stats = calculate_clustering_stats(G)
    if clustering_stats:
        print(f"\n🔗 Clustering Statistics:")
        print(f"  Average Clustering Coefficient: {clustering_stats.get('average_clustering', 0):.4f}")
        print(f"  Transitivity: {clustering_stats.get('transitivity', 0):.4f}")
    
    print("\n" + "="*60)


def export_statistics_to_dict(G: nx.DiGraph) -> Dict:
    """
    Export all statistics as a dictionary (useful for saving to JSON).
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        Dictionary containing all statistics
    """
    stats = {
        'basic': calculate_basic_stats(G),
        'centrality': calculate_centrality_measures(G),
        'clustering': calculate_clustering_stats(G),
        'isolated_nodes': find_isolated_nodes(G)
    }
    
    return stats


# Example usage
if __name__ == '__main__':
    from data_loader import load_data, filter_by_tag
    from graph_builder import build_graph
    
    # Load and build graph
    data_path = '../data/1-categories_snapshot_linked.json'
    raw_data = load_data(data_path)
    filtered_data = filter_by_tag(raw_data, 'analysis')
    G = build_graph(filtered_data)
    
    # Print statistics
    print_statistics(G, detailed=True)
    
    # Export statistics
    stats = export_statistics_to_dict(G)
    print(f"\nExported statistics: {list(stats.keys())}")