import json
import networkx as nx
from pyvis.network import Network

def load_data(filepath):
    """Load JSON data from file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def filter_by_tag(data, tag):
    """Filter nodes that have the specified tag."""
    return [item for item in data if tag in item.get('tags', [])]

def build_graph(data):
    """Build NetworkX graph from filtered data."""
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
    
    return G

def create_pyvis_network(G, output_file='network_graph.html'):
    """Create an interactive PyVis network visualization."""
    
    # Initialize PyVis network
    net = Network(
        height='800px',
        width='100%',
        bgcolor='#ffffff',
        font_color='black',
        directed=True,
        notebook=False
    )
    
    # Configure physics for better layout
    net.set_options("""
    {
      "physics": {
        "barnesHut": {
          "gravitationalConstant": -30000,
          "centralGravity": 0.3,
          "springLength": 150,
          "springConstant": 0.04,
          "damping": 0.09,
          "avoidOverlap": 0.5
        },
        "maxVelocity": 50,
        "minVelocity": 0.1,
        "timestep": 0.5
      },
      "interaction": {
        "hover": true,
        "tooltipDelay": 100,
        "navigationButtons": true,
        "keyboard": true
      },
      "nodes": {
        "font": {
          "size": 14
        }
      },
      "edges": {
        "arrows": {
          "to": {
            "enabled": true,
            "scaleFactor": 0.5
          }
        },
        "smooth": {
          "type": "continuous"
        }
      }
    }
    """)
    
    # Color mapping for categories
    category_colors = {
        'ML': '#0074D9',
        'default': '#7FDBFF'
    }
    
    # Add nodes to PyVis network
    for node, attrs in G.nodes(data=True):
        category = attrs.get('category', 'default')
        color = category_colors.get(category, category_colors['default'])
        
        # Create hover title with details
        hover_title = f"""
        <b>{attrs.get('label', node)}</b><br>
        Category: {category}<br>
        Tags: {', '.join(attrs.get('tags', []))}<br>
        URL: {attrs.get('url', 'N/A')}
        """
        
        net.add_node(
            node,
            label=attrs.get('label', node),
            title=hover_title,
            color=color,
            size=25,
            font={'size': 12}
        )
    
    # Add edges to PyVis network
    for source, target in G.edges():
        net.add_edge(source, target, color='#CCCCCC', width=2)
    
    # Generate and save the HTML file
    net.show(output_file)
    print(f"Network visualization saved to {output_file}")
    print(f"Open the file in your browser to view the interactive graph")
    
    return net

def generate_statistics(G):
    """Generate network statistics."""
    print("\n=== Network Statistics ===")
    print(f"Number of nodes: {G.number_of_nodes()}")
    print(f"Number of edges: {G.number_of_edges()}")
    print(f"Density: {nx.density(G):.4f}")
    
    if G.number_of_nodes() > 0:
        # Degree centrality
        in_degree = dict(G.in_degree())
        out_degree = dict(G.out_degree())
        
        print(f"\nTop 5 nodes by in-degree (most referenced):")
        sorted_in = sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5]
        for node, degree in sorted_in:
            print(f"  {node}: {degree}")
        
        print(f"\nTop 5 nodes by out-degree (most references):")
        sorted_out = sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5]
        for node, degree in sorted_out:
            print(f"  {node}: {degree}")
        
        # Weakly connected components
        if not nx.is_weakly_connected(G):
            num_components = nx.number_weakly_connected_components(G)
            print(f"\nNumber of weakly connected components: {num_components}")

def main():
    """Main execution function."""
    # Load and process data
    data_path = '../data/1-categories_snapshot_linked.json'
    print(f"Loading data from {data_path}...")
    raw_data = load_data(data_path)
    
    # Filter by tag
    tag_to_filter = 'analysis'
    print(f"Filtering nodes with tag: '{tag_to_filter}'...")
    filtered_data = filter_by_tag(raw_data, tag_to_filter)
    print(f"Found {len(filtered_data)} nodes with the '{tag_to_filter}' tag")
    
    # Build graph
    print("Building graph...")
    G = build_graph(filtered_data)
    
    # Generate statistics
    generate_statistics(G)
    
    # Create PyVis visualization
    print("\nCreating interactive visualization...")
    output_file = f'network_analysis_{tag_to_filter}.html'
    create_pyvis_network(G, output_file)
    
    print("\n=== Instructions ===")
    print(f"1. Open {output_file} in your web browser")
    print("2. Hover over nodes to see details")
    print("3. Click and drag nodes to rearrange")
    print("4. Use mouse wheel to zoom")
    print("5. Use navigation buttons in the top-left corner")
    print("6. Click on a node to highlight its connections")

if __name__ == '__main__':
    main()