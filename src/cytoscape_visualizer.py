"""
cytoscape_visualizer.py
Module for creating interactive network visualizations using Dash Cytoscape.
"""

import networkx as nx
from dash import Dash, html, Input, Output, callback
import dash_cytoscape as cyto
from typing import Dict, Optional


def graph_to_cytoscape_elements(G: nx.DiGraph) -> list:
    """
    Convert NetworkX graph to Cytoscape elements format.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        List of Cytoscape elements (nodes and edges)
    """
    elements = []
    
    # Add nodes
    for node in G.nodes(data=True):
        node_id, node_data = node
        elements.append({
            'data': {
                'id': node_id,
                'label': node_data.get('title', node_id),
                'category': node_data.get('category', ''),
                'tags': ', '.join(node_data.get('tags', [])),
                'url': node_data.get('url', ''),
                'degree': G.degree(node_id),
                'in_degree': G.in_degree(node_id),
                'out_degree': G.out_degree(node_id)
            }
        })
    
    # Add edges
    for edge in G.edges():
        elements.append({
            'data': {
                'source': edge[0],
                'target': edge[1],
                'id': f"{edge[0]}-{edge[1]}"
            }
        })
    
    return elements


def create_cytoscape_stylesheet(category_colors: Optional[Dict[str, str]] = None) -> list:
    """
    Create Cytoscape stylesheet with enhanced styling.
    
    Args:
        category_colors: Dictionary mapping categories to colors
        
    Returns:
        List of style dictionaries
    """
    if category_colors is None:
        category_colors = {
            'ML': '#3498db',
            'CS': '#e74c3c',
            'Math': '#2ecc71',
            'Stats': '#f39c12',
            'default': '#95a5a6'
        }
    
    # Base stylesheet
    stylesheet = [
        {
            'selector': 'node',
            'style': {
                'content': 'data(label)',
                'text-valign': 'center',
                'text-halign': 'center',
                'background-color': '#95a5a6',
                'color': '#fff',
                'font-size': '12px',
                'width': 'mapData(degree, 0, 10, 40, 80)',
                'height': 'mapData(degree, 0, 10, 40, 80)',
                'text-wrap': 'wrap',
                'text-max-width': '100px',
                'border-width': 2,
                'border-color': '#34495e'
            }
        },
        {
            'selector': 'edge',
            'style': {
                'width': 2,
                'line-color': '#bdc3c7',
                'target-arrow-color': '#bdc3c7',
                'target-arrow-shape': 'triangle',
                'curve-style': 'bezier',
                'arrow-scale': 1.5
            }
        },
        {
            'selector': ':selected',
            'style': {
                'background-color': '#e74c3c',
                'line-color': '#e74c3c',
                'target-arrow-color': '#e74c3c',
                'border-width': 4,
                'border-color': '#c0392b'
            }
        },
        {
            'selector': 'node:active',
            'style': {
                'overlay-color': '#e74c3c',
                'overlay-padding': 10,
                'overlay-opacity': 0.3
            }
        }
    ]
    
    # Add category-specific colors
    for category, color in category_colors.items():
        if category != 'default':
            stylesheet.append({
                'selector': f'node[category = "{category}"]',
                'style': {
                    'background-color': color
                }
            })
    
    return stylesheet


def create_cytoscape_app(
    G: nx.DiGraph,
    app_title: str = 'Network Analysis - Cytoscape',
    port: int = 8050,
    category_colors: Optional[Dict[str, str]] = None
) -> Dash:
    """
    Create a Dash Cytoscape application for network visualization.
    
    Args:
        G: NetworkX DiGraph
        app_title: Title for the web application
        port: Port to run the server on
        category_colors: Dictionary mapping categories to colors
        
    Returns:
        Dash app instance
    """
    # Convert graph to Cytoscape format
    elements = graph_to_cytoscape_elements(G)
    stylesheet = create_cytoscape_stylesheet(category_colors)
    
    # Create Dash app
    app = Dash(__name__)
    
    app.layout = html.Div([
        html.Div([
            html.H2(app_title, style={'textAlign': 'center', 'marginBottom': '10px'}),
            html.P(
                f'Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}',
                style={'textAlign': 'center', 'color': '#7f8c8d', 'marginBottom': '20px'}
            ),
            
            # Layout buttons
            html.Div([
                html.Label('Layout: ', style={'marginRight': '10px', 'fontWeight': 'bold'}),
                html.Button('Cose', id='btn-cose', n_clicks=0, 
                           style={'margin': '5px', 'padding': '8px 16px', 'cursor': 'pointer'}),
                html.Button('Circle', id='btn-circle', n_clicks=0,
                           style={'margin': '5px', 'padding': '8px 16px', 'cursor': 'pointer'}),
                html.Button('Grid', id='btn-grid', n_clicks=0,
                           style={'margin': '5px', 'padding': '8px 16px', 'cursor': 'pointer'}),
                html.Button('Breadthfirst', id='btn-breadth', n_clicks=0,
                           style={'margin': '5px', 'padding': '8px 16px', 'cursor': 'pointer'}),
                html.Button('Concentric', id='btn-concentric', n_clicks=0,
                           style={'margin': '5px', 'padding': '8px 16px', 'cursor': 'pointer'}),
            ], style={'textAlign': 'center', 'marginBottom': '20px'}),
        ]),
        
        # Cytoscape component
        cyto.Cytoscape(
            id='cytoscape-graph',
            elements=elements,
            stylesheet=stylesheet,
            style={'width': '100%', 'height': '700px', 'border': '1px solid #ddd'},
            layout={'name': 'cose', 'animate': True, 'animationDuration': 500}
        ),
        
        # Node information panel
        html.Div(id='node-info', style={
            'marginTop': '20px',
            'padding': '20px',
            'backgroundColor': '#ecf0f1',
            'borderRadius': '5px',
            'minHeight': '100px'
        })
    ], style={'padding': '20px', 'fontFamily': 'Arial, sans-serif'})
    
    @callback(
        Output('cytoscape-graph', 'layout'),
        Input('btn-cose', 'n_clicks'),
        Input('btn-circle', 'n_clicks'),
        Input('btn-grid', 'n_clicks'),
        Input('btn-breadth', 'n_clicks'),
        Input('btn-concentric', 'n_clicks')
    )
    def update_layout(cose_clicks, circle_clicks, grid_clicks, breadth_clicks, concentric_clicks):
        """Update graph layout based on button clicks."""
        from dash import ctx
        
        if not ctx.triggered:
            return {'name': 'cose', 'animate': True}
        
        button_id = ctx.triggered[0]['prop_id'].split('.')[0]
        
        layouts = {
            'btn-cose': {
                'name': 'cose',
                'animate': True,
                'nodeRepulsion': 400000,
                'idealEdgeLength': 100,
                'edgeElasticity': 100,
                'nestingFactor': 5,
                'gravity': 80,
                'numIter': 1000,
                'initialTemp': 200,
                'coolingFactor': 0.95,
                'minTemp': 1.0
            },
            'btn-circle': {'name': 'circle', 'animate': True},
            'btn-grid': {'name': 'grid', 'animate': True},
            'btn-breadth': {
                'name': 'breadthfirst',
                'animate': True,
                'directed': True,
                'spacingFactor': 1.5
            },
            'btn-concentric': {
                'name': 'concentric',
                'animate': True,
                'minNodeSpacing': 100
            }
        }
        
        return layouts.get(button_id, {'name': 'cose', 'animate': True})
    
    @callback(
        Output('node-info', 'children'),
        Input('cytoscape-graph', 'tapNodeData')
    )
    def display_node_info(data):
        """Display information about the selected node."""
        if data:
            return html.Div([
                html.H3(data['label'], style={'color': '#2c3e50', 'marginBottom': '15px'}),
                html.Div([
                    html.Strong('ID: '), html.Span(data['id']),
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong('Category: '), html.Span(data.get('category', 'N/A')),
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong('Tags: '), html.Span(data.get('tags', 'None')),
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong('Total Connections: '), html.Span(str(data.get('degree', 0))),
                ], style={'marginBottom': '8px'}),
                html.Div([
                    html.Strong('In-degree: '), html.Span(str(data.get('in_degree', 0))),
                    html.Span(' | ', style={'margin': '0 10px'}),
                    html.Strong('Out-degree: '), html.Span(str(data.get('out_degree', 0))),
                ], style={'marginBottom': '15px'}),
                html.A(
                    'View on GitHub',
                    href=data.get('url', '#'),
                    target='_blank',
                    style={
                        'display': 'inline-block',
                        'padding': '8px 16px',
                        'backgroundColor': '#3498db',
                        'color': 'white',
                        'textDecoration': 'none',
                        'borderRadius': '4px'
                    }
                ) if data.get('url') else None
            ])
        return html.Div([
            html.P('👆 Click on a node to see details', style={
                'textAlign': 'center',
                'color': '#7f8c8d',
                'fontSize': '16px',
                'paddingTop': '20px'
            })
        ])
    
    return app


def run_cytoscape_visualization(
    G: nx.DiGraph,
    app_title: str = 'Network Analysis - Cytoscape',
    port: int = 8050,
    debug: bool = True
):
    """
    Run the Cytoscape visualization app.
    
    Args:
        G: NetworkX DiGraph
        app_title: Title for the application
        port: Port to run the server on
        debug: Enable debug mode
    """
    app = create_cytoscape_app(G, app_title=app_title, port=port)
    
    print(f"\n{'='*60}")
    print(f"🚀 Starting Cytoscape visualization server...")
    print(f"{'='*60}")
    print(f"📊 Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"🌐 Open your browser and navigate to: http://localhost:{port}")
    print(f"{'='*60}\n")
    
    app.run(debug=debug, port=port)


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
    
    print(f"\nVisualization for {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Run the Cytoscape app
    run_cytoscape_visualization(
        G,
        app_title='Network Analysis - Analysis Tag',
        port=8050,
        debug=True
    )