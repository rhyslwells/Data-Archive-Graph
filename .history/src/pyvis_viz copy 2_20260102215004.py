"""
main.py
Main workflow script demonstrating how to use all modules together.
"""

from data_loader import (
    load_data, 
    filter_by_tag, 
    filter_by_titles,
    filter_by_category,
    print_data_summary
)
from graph_builder import (
    build_graph,
    get_largest_component,
    filter_by_degree
)
from network_stats import print_statistics
from pyvis_visualizer import create_pyvis_network, create_minimal_network


def workflow_by_tag(data_path: str, tag: str, output_prefix: str = 'network'):
    """
    Complete workflow: filter by tag and create visualization.
    
    Args:
        data_path: Path to JSON data file
        tag: Tag to filter by
        output_prefix: Prefix for output files
    """
    print(f"\n{'='*60}")
    print(f"WORKFLOW: Filter by Tag '{tag}'")
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
    
    # Build graph without isolated nodes
    print(f"\n🏗️  Building graph (removing isolated nodes)...")
    G = build_graph(filtered_data, remove_isolated=True)
    
    # Print statistics
    print_statistics(G, detailed=True)
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    output_file = f'{output_prefix}_{tag}.html'
    create_pyvis_network(G, output_file=output_file, physics_enabled=True)
    
    print(f"\n✅ Complete! Open {output_file} in your browser.")


def workflow_by_titles(data_path: str, titles: list, output_file: str = 'network_custom.html'):
    """
    Complete workflow: filter by specific titles and create visualization.
    
    Args:
        data_path: Path to JSON data file
        titles: List of normalized filenames to include
        output_file: Output HTML filename
    """
    print(f"\n{'='*60}")
    print(f"WORKFLOW: Filter by Titles")
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
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    create_pyvis_network(G, output_file=output_file, physics_enabled=True)
    
    print(f"\n✅ Complete! Open {output_file} in your browser.")


def workflow_largest_component(data_path: str, tag: str, output_file: str = 'network_largest.html'):
    """
    Workflow: Extract and visualize the largest connected component.
    
    Args:
        data_path: Path to JSON data file
        tag: Tag to filter by
        output_file: Output HTML filename
    """
    print(f"\n{'='*60}")
    print(f"WORKFLOW: Largest Component for Tag '{tag}'")
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
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    create_pyvis_network(G_largest, output_file=output_file, physics_enabled=True)
    
    print(f"\n✅ Complete! Open {output_file} in your browser.")


def workflow_by_category(data_path: str, category: str, output_prefix: str = 'network'):
    """
    Workflow: Filter by category and create visualization.
    
    Args:
        data_path: Path to JSON data file
        category: Category to filter by (e.g., 'ML', 'CS')
        output_prefix: Prefix for output files
    """
    print(f"\n{'='*60}")
    print(f"WORKFLOW: Filter by Category '{category}'")
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
    
    # Create visualization
    print(f"\n🎨 Creating visualization...")
    output_file = f'{output_prefix}_category_{category.lower()}.html'
    create_pyvis_network(G, output_file=output_file, physics_enabled=True)
    
    print(f"\n✅ Complete! Open {output_file} in your browser.")


def main():
    """Main execution with example workflows."""
    
    data_path = '../data/1-categories_snapshot_linked.json'
    
    print("\n" + "="*60)
    print("NETWORK ANALYSIS TOOLKIT")
    print("="*60)
    
    # Example 1: Filter by tag
    print("\n\n🔹 Example 1: Filtering by 'analysis' tag")
    workflow_by_tag(data_path, 'analysis', output_prefix='network_analysis')
    
    # Example 2: Filter by specific titles
    print("\n\n🔹 Example 2: Filtering by specific titles")
    specific_titles = ['data_analysis', 'eda', 'data_visualisation', 'querying']
    workflow_by_titles(data_path, specific_titles, output_file='network_custom_nodes.html')
    
    # Example 3: Largest component only
    print("\n\n🔹 Example 3: Largest connected component")
    workflow_largest_component(data_path, 'analysis', output_file='network_largest_component.html')
    
    # Example 4: Filter by category
    # print("\n\n🔹 Example 4: Filtering by 'ML' category")
    # workflow_by_category(data_path, 'ML', output_prefix='network')
    
    print("\n" + "="*60)
    print("ALL WORKFLOWS COMPLETE!")
    print("="*60)
    print("\n📝 Summary of outputs:")
    print("  - network_analysis_analysis.html")
    print("  - network_custom_nodes.html")
    print("  - network_largest_component.html")
    print("\n💡 Open these files in your browser to explore the networks!")


if __name__ == '__main__':
    # You can run individual workflows or the complete main()
    
    # Option 1: Run all examples
    main()
    
    # Option 2: Run single workflow
    # data_path = '../data/1-categories_snapshot_linked.json'
    # workflow_by_tag(data_path, 'analysis', output_prefix='my_network')