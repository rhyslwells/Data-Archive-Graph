import json
import networkx as nx
from dash import Dash, html, Input, Output, callback
import dash_cytoscape as cyto

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
            url=item.get('url', '')
        )
    
    # Add edges based on outlinks
    for item in data:
        source = item['normalized_filename']
        for target in item.get('outlinks', []):
            if target in G.nodes():
                G.add_edge(source, target)
    
    return G

def graph_to_cytoscape_elements(G):
    """Convert NetworkX graph to Cytoscape elements format."""
    elements = []
    
    # Add nodes
    for node in G.nodes(data=True):
        node_id, node_data = node
        elements.append({
            'data': {
                'id': node_id,
                'label': node_data.get('title', node_id),
                'category': node_data.get('category', ''),
                'url': node_data.get('url', '')
            }
        })
    
    # Add edges
    for edge in G.edges():
        elements.append({
            'data': {
                'source': edge[0],
                'target': edge[1]
            }
        })
    
    return elements

# Load and process data
data_path = 'data/1-categories_snapshot_linked.json'
raw_data = load_data(data_path)
filtered_data = filter_by_tag(raw_data, 'analysis')
G = build_graph(filtered_data)

print(f"Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

# Convert to Cytoscape format
elements = graph_to_cytoscape_elements(G)

# Define stylesheet
stylesheet = [
    {
        'selector': 'node',
        'style': {
            'content': 'data(label)',
            'text-valign': 'center',
            'text-halign': 'center',
            'background-color': '#0074D9',
            'color': '#fff',
            'font-size': '10px',
            'width': '60px',
            'height': '60px',
            'text-wrap': 'wrap',
            'text-max-width': '80px'
        }
    },
    {
        'selector': 'edge',
        'style': {
            'width': 2,
            'line-color': '#ccc',
            'target-arrow-color': '#ccc',
            'target-arrow-shape': 'triangle',
            'curve-style': 'bezier',
            'arrow-scale': 1
        }
    },
    {
        'selector': ':selected',
        'style': {
            'background-color': '#FF4136',
            'line-color': '#FF4136',
            'target-arrow-color': '#FF4136',
            'source-arrow-color': '#FF4136'
        }
    }
]

# Create Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        html.H2('Network Analysis - "analysis" Tag', style={'textAlign': 'center'}),
        html.P(f'Nodes: {G.number_of_nodes()} | Edges: {G.number_of_edges()}', 
               style={'textAlign': 'center'}),
        html.Div([
            html.Label('Layout: '),
            html.Button('Cose', id='btn-cose', n_clicks=0, style={'margin': '5px'}),
            html.Button('Circle', id='btn-circle', n_clicks=0, style={'margin': '5px'}),
            html.Button('Grid', id='btn-grid', n_clicks=0, style={'margin': '5px'}),
            html.Button('Breadthfirst', id='btn-breadth', n_clicks=0, style={'margin': '5px'}),
        ], style={'textAlign': 'center', 'marginBottom': '10px'})
    ]),
    
    cyto.Cytoscape(
        id='cytoscape-graph',
        elements=elements,
        stylesheet=stylesheet,
        style={'width': '100%', 'height': '800px'},
        layout={'name': 'cose', 'animate': True}
    ),
    
    html.Div(id='node-info', style={'marginTop': '20px', 'padding': '10px'})
])

@callback(
    Output('cytoscape-graph', 'layout'),
    Input('btn-cose', 'n_clicks'),
    Input('btn-circle', 'n_clicks'),
    Input('btn-grid', 'n_clicks'),
    Input('btn-breadth', 'n_clicks')
)
def update_layout(cose_clicks, circle_clicks, grid_clicks, breadth_clicks):
    """Update graph layout based on button clicks."""
    from dash import ctx
    
    if not ctx.triggered:
        return {'name': 'cose', 'animate': True}
    
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    
    layouts = {
        'btn-cose': {'name': 'cose', 'animate': True, 'nodeRepulsion': 400000},
        'btn-circle': {'name': 'circle', 'animate': True},
        'btn-grid': {'name': 'grid', 'animate': True},
        'btn-breadth': {'name': 'breadthfirst', 'animate': True, 'directed': True}
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
            html.H4(f"Selected Node: {data['label']}"),
            html.P(f"ID: {data['id']}"),
            html.P(f"Category: {data.get('category', 'N/A')}"),
            html.A('View on GitHub', href=data.get('url', '#'), target='_blank')
        ])
    return html.P('Click on a node to see details')

if __name__ == '__main__':
    app.run(debug=True, port=8050)