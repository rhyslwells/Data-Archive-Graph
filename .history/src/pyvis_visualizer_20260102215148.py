"""
pyvis_visualizer.py
Module for creating interactive network visualizations using PyVis.
"""

import networkx as nx
from pyvis.network import Network
from typing import Dict, Optional


def create_pyvis_network(
    G: nx.DiGraph,
    output_file: str = 'network_graph.html',
    height: str = '800px',
    width: str = '100%',
    bgcolor: str = '#ffffff',
    font_color: str = 'black',
    physics_enabled: bool = True,
    category_colors: Optional[Dict[str, str]] = None
) -> Network:
    """
    Create an interactive PyVis network visualization with enhanced styling.
    
    Args:
        G: NetworkX DiGraph
        output_file: Output HTML filename
        height: Height of the visualization
        width: Width of the visualization
        bgcolor: Background color
        font_color: Font color
        physics_enabled: Enable/disable physics simulation
        category_colors: Dictionary mapping categories to colors
        
    Returns:
        PyVis Network object
    """
    
    # Initialize PyVis network
    net = Network(
        height=height,
        width=width,
        bgcolor=bgcolor,
        font_color=font_color,
        directed=True,
        notebook=False
    )
    
    # Default color mapping
    if category_colors is None:
        category_colors = {
            'ML': '#3498db',      # Blue
            'CS': '#e74c3c',      # Red
            'Math': '#2ecc71',    # Green
            'Stats': '#f39c12',   # Orange
            'default': '#95a5a6'  # Gray
        }
    
    # Configure physics for better layout
    if physics_enabled:
        net.set_options("""
        {
          "physics": {
            "barnesHut": {
              "gravitationalConstant": -50000,
              "centralGravity": 0.3,
              "springLength": 200,
              "springConstant": 0.04,
              "damping": 0.09,
              "avoidOverlap": 0.8
            },
            "maxVelocity": 50,
            "minVelocity": 0.1,
            "timestep": 0.5,
            "stabilization": {
              "enabled": true,
              "iterations": 200
            }
          },
          "interaction": {
            "hover": true,
            "tooltipDelay": 100,
            "navigationButtons": true,
            "keyboard": {
              "enabled": true
            },
            "zoomView": true,
            "dragView": true
          },
          "nodes": {
            "font": {
              "size": 16,
              "face": "Tahoma"
            },
            "borderWidth": 2,
            "borderWidthSelected": 4
          },
          "edges": {
            "arrows": {
              "to": {
                "enabled": true,
                "scaleFactor": 0.8
              }
            },
            "smooth": {
              "enabled": true,
              "type": "continuous"
            },
            "color": {
              "inherit": false
            }
          }
        }
        """)
    else:
        net.toggle_physics(False)
    
    # Calculate node sizes based on degree
    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        max_degree = max(degrees.values()) if degrees else 1
        min_degree = min(degrees.values()) if degrees else 0
        
        # Normalize sizes between 20 and 60
        def get_node_size(degree):
            if max_degree == min_degree:
                return 30
            normalized = (degree - min_degree) / (max_degree - min_degree)
            return 20 + normalized * 40
    else:
        def get_node_size(degree):
            return 30
    
    # Add nodes to PyVis network
    for node, attrs in G.nodes(data=True):
        category = attrs.get('category', 'default')
        color = category_colors.get(category, category_colors['default'])
        degree = G.degree(node)
        
        # Create detailed hover title
        tags_list = attrs.get('tags', [])
        tags_str = ', '.join(tags_list) if tags_list else 'None'
        
        hover_title = f"""
        <div style='font-family: Arial; padding: 10px;'>
            <b style='font-size: 16px;'>{attrs.get('label', node)}</b><br><br>
            <b>Category:</b> {category}<br>
            <b>Tags:</b> {tags_str}<br>
            <b>Connections:</b> {degree}<br>
            <b>In-degree:</b> {G.in_degree(node)}<br>
            <b>Out-degree:</b> {G.out_degree(node)}<br>
            <b>ID:</b> {node}<br>
        </div>
        """
        
        # Add node with dynamic sizing
        net.add_node(
            node,
            label=attrs.get('label', node),
            title=hover_title,
            color=color,
            size=get_node_size(degree),
            font={'size': 14, 'face': 'Tahoma'},
            borderWidth=2
        )
    
    # Add edges with varying thickness based on weight (if applicable)
    for source, target in G.edges():
        net.add_edge(
            source,
            target,
            color={'color': '#CCCCCC', 'highlight': '#000000'},
            width=2,
            arrows={'to': {'enabled': True, 'scaleFactor': 0.8}}
        )
    
    # Save the graph
    net.save_graph(output_file)
    print(f"✅ Network visualization saved to: {output_file}")
    
    return net


def create_minimal_network(
    G: nx.DiGraph,
    output_file: str = 'network_minimal.html',
    layout: str = 'hierarchical'
) -> Network:
    """
    Create a minimal, static network visualization without physics.
    
    Args:
        G: NetworkX DiGraph
        output_file: Output HTML filename
        layout: Layout algorithm ('hierarchical', 'random', etc.)
        
    Returns:
        PyVis Network object
    """
    net = Network(
        height='800px',
        width='100%',
        bgcolor='#ffffff',
        directed=True,
        notebook=False
    )
    
    # Disable physics for static layout
    if layout == 'hierarchical':
        net.set_options("""
        {
          "layout": {
            "hierarchical": {
              "enabled": true,
              "direction": "UD",
              "sortMethod": "directed"
            }
          },
          "physics": {
            "enabled": false
          },
          "interaction": {
            "hover": true,
            "navigationButtons": true
          }
        }
        """)
    else:
        net.toggle_physics(False)
    
    # Add nodes and edges
    for node, attrs in G.nodes(data=True):
        net.add_node(node, label=attrs.get('label', node), title=attrs.get('label', node))
    
    for source, target in G.edges():
        net.add_edge(source, target)
    
    net.save_graph(output_file)
    print(f"✅ Minimal network saved to: {output_file}")
    
    return net


# Example usage
if __name__ == '__main__':
    from data_loader import load_data, filter_by_tag
    from graph_builder import build_graph
    
    # Load and build graph
    data_path = '../data/1-categories_snapshot_linked.json'
    raw_data = load_data(data_path)
    filtered_data = filter_by_tag(raw_data, 'analysis')
    
    # Build graph without isolated nodes
    G = build_graph(filtered_data, remove_isolated=True)
    
    print(f"\nCreating visualization for {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Create enhanced visualization
    create_pyvis_network(
        G,
        output_file='network_enhanced.html',
        physics_enabled=True
    )
    
    # Create minimal visualization
    create_minimal_network(
        G,
        output_file='network_minimal.html',
        layout='hierarchical'
    )
    
    print("\n✨ Open the HTML files in your browser to view the visualizations!")