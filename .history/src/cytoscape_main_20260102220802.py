"""
main_cytoscape.py
Main workflow script for Cytoscape visualizations using shared modules.
"""

from data_loader import (
    load_data, 
    filter_by_tag, 
    filter_by_titles,
    filter_by_category
)
from graph_builder import build_graph, get_largest_component
from network_stats import print_statistics
from cytoscape_visualizer import run_cytoscape_visualization


def cytoscape_workflow_by_tag(
    data_path: str,
    tag: str,
    port: int = 8050,
    remove_isolated: bool = True
):
    """
    Complete workflow: filter by tag and create Cytoscape visualization.
    
    Args:
        data_path: Path to JSON data file
        tag: Tag to filter by
        port: Port to run the server on
        remove_isolated: Remove isolated nodes
    """
    print(f"\n{'='*60}")
    print(f"CYTOSCAPE WORKFLOW: Filter by Tag '{tag}'")
    print(f"{'='*60}")
    
    # Load data
    print(f"\n📂 Loading data from {data_path}...")
    raw_data = load_data(data_path)
    print(f"   Total nodes: {len(raw_data)}")
    
    # Filter by tag
    print(f"\n🔍 Filtering by tag '{tag}'...")
    filtered_data = filter_by_tag(raw_data, tag)
    print(f"   Found {len(filtered_data)} nodes")
    
    if len(filtered_data) == 0:
        print(f"⚠️  No nodes found with tag '{tag}'")
        return
    
    # Build graph
    print(f"\n🏗️  Building graph (remove_isolated={remove_isolated})...")
    G = build_graph(filtered_data, remove_isolated=remove_isolated)
    
    # Print statistics
    print_statistics(G, detailed=True)
    
    # Run Cytoscape visualization
    run_cytoscape_visualization(
        G,
        app_title=f'Network Analysis - {tag.title()} Tag',
        port=port,
        debug=True
    )


def cytoscape_workflow_by_titles(
    data_path: str,
    titles: list,
    port: int = 8050
):
    """
    Complete workflow: filter by specific titles and create Cytoscape visualization.
    
    Args:
        data_path: Path to JSON data file
        titles: List of normalized filenames to include
        port: Port to run the server on
    """
    print(f"\n{'='*60}")
    print(f"CYTOSCAPE WORKFLOW: Filter by Titles")
    print(f"{'='*60}")
    
    # Load data
    print(f"\n📂 Loading data from {data_path}...")
    raw_data = load_data(data_path)
    
    # Filter by titles
    print(f"\n🔍 Filtering by {len(titles)} titles...")
    filtered_data = filter_by_titles(raw_data, titles)
    print(f"   Found {len(filtered_data)} matching nodes")
    
    if len(filtered_data) == 0:
        print(f"⚠️  No nodes found matching provided titles")
        return
    
    # Build graph
    print(f"\n🏗️  Building graph...")
    G = build_graph(filtered_data, remove_isolated=False)
    
    # Print statistics
    print_statistics(G, detailed=True)
    
    # Run Cytoscape visualization
    run_cytoscape_visualization(
        G,
        app_title='Network Analysis - Custom Nodes',
        port=port,
        debug=True
    )


def cytoscape_workflow_largest_component(
    data_path: str,
    tag: str,
    port: int = 8050
):
    """
    Workflow: Extract and visualize the largest connected component with Cytoscape.
    
    Args:
        data_path: Path to JSON data file
        tag: Tag to filter by
        port: Port to run the server on
    """
    print(f"\n{'='*60}")
    print(f"CYTOSCAPE WORKFLOW: Largest Component for Tag '{tag}'")
    print(f"{'='*60}")
    
    # Load and filter
    raw_data = load_data(data_path)
    filtered_data = filter_by_tag(raw_data, tag)
    
    # Build graph
    G = build_graph(filtered_data, remove_isolated=True)
    
    # Extract largest component
    print(f"\n🔍 Extracting largest connected component...")
    G_largest = get_largest_component(G)
    
    # Print statistics
    print_statistics(G_largest, detailed=True)
    
    # Run Cytoscape visualization
    run_cytoscape_visualization(
        G_largest,
        app_title=f'Network Analysis - Largest Component ({tag})',
        port=port,
        debug=True
    )


def cytoscape_workflow_by_category(
    data_path: str,
    category: str,
    port: int = 8050
):
    """
    Workflow: Filter by category and create Cytoscape visualization.
    
    Args:
        data_path: Path to JSON data file
        category: Category to filter by (e.g., 'ML', 'CS')
        port: Port to run the server on
    """
    print(f"\n{'='*60}")
    print(f"CYTOSCAPE WORKFLOW: Filter by Category '{category}'")
    print(f"{'='*60}")
    
    # Load data
    raw_data = load_data(data_path)
    
    # Filter by category
    print(f"\n🔍 Filtering by category '{category}'...")
    filtered_data = filter_by_category(raw_data, category)
    print(f"   Found {len(filtered_data)} nodes")
    
    if len(filtered_data) == 0:
        print(f"⚠️  No nodes found in category '{category}'")
        return
    
    # Build graph
    print(f"\n🏗️  Building graph (removing isolated nodes)...")
    G = build_graph(filtered_data, remove_isolated=True)
    
    # Print statistics
    print_statistics(G, detailed=True)
    
    # Run Cytoscape visualization
    run_cytoscape_visualization(
        G,
        app_title=f'Network Analysis - {category} Category',
        port=port,
        debug=True
    )


def main():
    """Main execution with example workflows."""
    
    data_path = '../data/1-categories_snapshot_linked.json'
    
    print("\n" + "="*60)
    print("CYTOSCAPE NETWORK VISUALIZATION TOOLKIT")
    print("="*60)
    print("\nAvailable workflows:")
    print("  1. Filter by tag")
    print("  2. Filter by specific titles")
    print("  3. Largest connected component")
    print("  4. Filter by category")
    print("\nChoose a workflow to run (or modify this script):")
    print("="*60)
    
    # Example 1: Filter by tag (default)
    print("\n🔹 Running: Filtering by 'analysis' tag")
    cytoscape_workflow_by_tag(
        data_path,
        'analysis',
        port=8050,
        remove_isolated=True
    )
    
    # Uncomment to run other workflows:
    
    # Example 2: Filter by specific titles
    # specific_titles = ['data_analysis', 'eda', 'data_visualisation', 'querying']
    # cytoscape_workflow_by_titles(data_path, specific_titles, port=8050)
    
    # Example 3: Largest component only
    # cytoscape_workflow_largest_component(data_path, 'analysis', port=8050)
    
    # Example 4: Filter by category
    # cytoscape_workflow_by_category(data_path, 'ML', port=8050)


if __name__ == '__main__':
    # You can run individual workflows or the complete main()
    
    # Option 1: Run default workflow
    main()
    
    # Option 2: Run custom workflow
    # data_path = '../data/1-categories_snapshot_linked.json'
    # cytoscape_workflow_by_tag(data_path, 'analysis', port=8050)