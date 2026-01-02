import networkx as nx
import matplotlib.pyplot as plt

Lets use 
data\1-categories_snapshot_linked.json

data in the json is of the for
  [{
    "category": "ML",
    "filename": "Dimension Table",
    "sha": "bbb5113bd5572ebf7dff382b9f0aedd78fb65799",
    "url": "https://github.com/rhyslwells/Data-Archive/blob/main/content/categories/machine-learning/Dimension%20Table.md",
    "text": "A dimension table is a key component of a [[Star Schema]] or snowflake schema in a data warehouse. It provides descriptive attributes (or dimensions) related to the [[Facts]] stored in a fact table.\n\nThey provide the context and descriptive information necessary for analyzing the quantitative data stored in fact tables (e.g., product names, customer demographics, time periods).\n\n1. **Descriptive Attributes**: Dimension tables contain qualitative data that describe the entities involved in the business process. For example, a product dimension table might include attributes such as product name, category, brand, and manufacturer.\n\n2. **Primary Key**: Each dimension table has a primary key that uniquely identifies each record in the table. This primary key is used as a foreign key in the [[Fact Table]] to establish relationships between the two.\n\n3. **Hierarchies**: Dimension tables often include hierarchies that allow for data to be analyzed at different levels of granularity. For example, a time dimension might include attributes for year, quarter, month, and day, allowing users to drill down or roll up in their analysis.\n\n4. **Smaller Size**: Compared to fact tables, dimension tables are typically smaller in size, as they contain descriptive data rather than large volumes of transactional data.\n\n5. **Static Data**: Dimension tables usually contain relatively static data that does not change frequently, such as product details or customer information. However, they can be updated as needed to reflect changes in the business.\n\n6. **Support for Filtering and Grouping**: Dimension tables enable users to filter and group data in reports and analyses. For example, users can analyze sales data by different dimensions such as time, geography, or product category.\n\nExamples\n  - **TimeDimension**: Contains information about the time period.\n    - Columns: `DateKey`, `Year`, `Quarter`, `Month`, `Day`\n  - **ProductDimension**: Contains product details.\n    - Columns: `ProductKey`, `ProductName`, `ProductCategory`\n  - **RegionDimension**: Contains regional information.\n    - Columns: `RegionKey`, `RegionName`, `Country`\n\n\n\n\n\n[[Dimension Table]]\n   **Tags**:,",
    "aliases": [],
    "date modified": "27-09-2025",
    "tags": [
      "database",
      "modeling"
    ],
    "normalized_filename": "dimension_table",
    "outlinks": [
      "dimension_table",
      "star_schema",
      "facts",
      "fact_table"
    ],
    "inlinks": [
      "components_of_the_database",
      "dimension_table",
      "dimensional_modelling",
      "fact_table",
      "slowly_changing_dimension"
    ]
  },
  {
    "category": "ML",
    "filename": "Dimensional Modelling",
    "sha": "530438c7fbcf3a55a18b222299ff8ac3e14b8543",
    "url": "https://github.com/rhyslwells/Data-Archive/blob/main/content/categories/machine-learning/Dimensional%20Modelling.md",
    "text": "Dimensional modeling is a design technique used in [[Data Warehouse]]used to structure data for efficient ==retrieval== and analysis. It is particularly well-suited for organizing data in a way that supports complex [[Querying]] and reporting, making it easier for business users to understand and interact with the data. \n\nDimensional modeling is a technique in building data warehouses and is associated with methodologies like the ==Kimball== approach, which emphasizes the use of [[Star Schema]] and the importance of understanding business processes and user requirements.\n\nKey Concepts in Dimensional Modeling\n- [[Fact Table]] & [[Facts]]\n- [[Dimension Table]]\n- [[Grain]]\n\nBenefits of Dimensional Modeling: [[Performance Dimensions]]\n\n\n\n\n\n[[Dimensional Modelling]]\n   **Tags**:,",
    "aliases": [],
    "date modified": "27-09-2025",
    "tags": [
      "database",
      "modeling"
    ],
    "normalized_filename": "dimensional_modelling",
    "outlinks": [
      "dimension_table",
      "facts",
      "dimensional_modelling",
      "querying",
      "star_schema",
      "grain",
      "fact_table",
      "performance_dimensions",
      "data_warehouse"
    ],
    "inlinks": [
      "dimensional_modelling",
      "granularity"
    ]
  },
  {
    "category": "ML",
    "filename": "Dimensionality Reduction",
    "sha": "919376586da61a07765859bb55b98aa278377082",
    "url": "https://github.com/rhyslwells/Data-Archive/blob/main/content/categories/machine-learning/Dimensionality%20Reduction.md",
    "text": "Dimensionality reduction is a step in the [[Preprocessing]] phase of machine learning that helps simplify models, enhance interpretability, and improve computational efficiency.\n\nIts a technique used to reduce the number of input variables (features) in a dataset while retaining as much information as possible. This process is essential for several reasons:\n\n1. **Improves Model Performance**: Reducing the number of features can help improve the performance of machine learning models by minimizing overfitting and reducing noise.\n\n2. **Enhances Visualization**: It allows for easier [[Data Visualisation]] of high-dimensional data by projecting it into lower dimensions (e.g., 2D or 3D).\n\n3. **Reduces Computational Cost**: Fewer features mean less computational power and time required for training models.\n\n### Common Techniques\n- **Principal Component Analysis ([[Principal Component Analysis]])**: A statistical method that transforms the data into a new coordinate system, where the greatest variance by any projection lies on the first coordinate ==(principal component/orthogonal components )==, the second greatest variance on the second coordinate, and so on.\n\n- **t-Distributed Stochastic Neighbor Embedding ([[t-SNE]])**: A technique particularly well-suited for visualizing high-dimensional data by reducing it to two or three dimensions while preserving the local structure of the data. t-SNE is a non-linear technique used for visualization and dimensionality reduction by preserving pairwise similarities between data points, making it suitable for exploring high-dimensional data.\n\n- [[Linear Discriminant Analysis]] method used for both classification and dimensionality reduction, which finds a linear combination of features that best separates two or more classes.\n\n### [[Curse of dimensionality]]",
    "aliases": [],
    "date modified": "27-09-2025",
    "tags": [
      "process",
      "visualization"
    ],
    "normalized_filename": "dimensionality_reduction",
    "outlinks": [
      "t-sne",
      "data_visualisation",
      "preprocessing",
      "principal_component_analysis",
      "linear_discriminant_analysis",
      "curse_of_dimensionality"
    ],
    "inlinks": [
      "addressing_multicollinearity",
      "curse_of_dimensionality",
      "data_reduction",
      "data_selection_in_ml",
      "ds_&_ml_portal",
      "dynamic_time_warping",
      "evaluate_embedding_methods",
      "factor_analysis",
      "feature_engineering",
      "feature_extraction",
      "feature_selection",
      "latent_dirichlet_allocation",
      "learning_styles",
      "machine_learning_algorithms",
      "manifold_learning",
      "preprocessing",
      "principal_component_analysis",
      "t-sne",
      "umap",
      "unsupervised_learning",
      "vector_embedding"
    ]
  },
  {
    "category": "ML",
    "filename": "Dimensions",
    "sha": "be6b1b471c9d1073b4df2b30d874c8e45395f80a",
    "url": "https://github.com/rhyslwells/Data-Archive/blob/main/content/categories/machine-learning/Dimensions.md",
    "text": "Dimensions are the categorical buckets that can be used to segment, filter, or group—such as sales amount region, city, product, color, and distribution channel. \n\nTraditionally known from [[OLAP|OLAP]]cubes with Bus Matrixes, and [Dimensional Modeling](Dimensional%20Modelling.md). \n\nThey provide context to the [[Facts]].",
    "aliases": [],
    "date modified": "27-09-2025",
    "tags": [
      "modeling"
    ],
    "normalized_filename": "dimensions",
    "outlinks": [
      "olap",
      "facts"
    ],
    "inlinks": [
      "granularity"
    ]
  },
  {
    "category": "ML",
    "filename": "Distributions in Decision Tree Leaves",
    "sha": "65780c0f52cc38f72c6efb1642b7f7031474fbaa",
    "url": "https://github.com/rhyslwells/Data-Archive/blob/main/content/categories/machine-learning/Distributions%20in%20Decision%20Tree%20Leaves.md",
    "text": "In decision trees, each leaf node typically contains the outcome for all training examples that fall into that leaf. This outcome can be represented in two ways:\n\n#### 1. Single Predicted Value\n\n* Classification: The leaf predicts the majority class (e.g., if 60% of samples are class A, predict A).\n* Regression: The leaf predicts the mean of the target values in that leaf.\n\n#### 2. Distribution of Target Values\n\nInstead of storing just a single prediction, the leaf can store the distribution of the target variable:\n\n* Classification: Store class probabilities.\n  Example: If a leaf has 100 samples -> 60 class 1, 40 class 2 -> Distribution = {class 1: 0.6, class 2: 0.4}.\n* Regression: Store a histogram or density estimate of the continuous values rather than just their mean.\n\n#### Why use distributions?\n\n* Enables probabilistic predictions (e.g., predict class probabilities, not just hard labels).\n* Provides uncertainty estimates, useful for Bayesian methods and risk-sensitive decisions.\n* Improves interpretability by showing the variability in outcomes for samples reaching the same leaf.\n\nThis method is commonly applied in:\n\n* Random Forests and Gradient Boosted Trees (which use class probabilities for classification).\n* Probabilistic decision tree models for uncertainty-aware predictions.",
    "aliases": [],
    "date modified": "27-09-2025",
    "tags": [
      "classifier",
      "explainability"
    ],
    "normalized_filename": "distributions_in_decision_tree_leaves",
    "outlinks": [],
    "inlinks": [
      "decision_tree"
    ]
  }]

Building a subgraph of notes with the tags "analysis"
I want one script for cytoscape and one for pyviz

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

# %% [markdown]
# Explore features of Dash Cytoscape:
# https://dash.plotly.com/cytoscape
# As a way to visualize and explore the network data.
# Once visuals are created, we can use Dash with Render to create a web application to host the investigations.
# 
# https://towardsdatascience.com/making-network-graphs-interactive-with-python-and-pyvis-b754c22c270/
# 
# https://github.com/AlrasheedA/st-link-analysis?tab=readme-ov-file
# 
# 

# %%
from dash import Dash, html
import dash_cytoscape as cyto

app = Dash()

app.layout = html.Div([
    cyto.Cytoscape(
        id='cytoscape-elements-boolean',
        layout={'name': 'preset'},
        style={'width': '100%', 'height': '400px'},
        elements=[
            {
                'data': {'id': 'one', 'label': 'Locked'},
                'position': {'x': 75, 'y': 75},
                'locked': True
            },
            {
                'data': {'id': 'two', 'label': 'Selected'},
                'position': {'x': 75, 'y': 200},
                'selected': True
            },
            {
                'data': {'id': 'three', 'label': 'Not Selectable'},
                'position': {'x': 200, 'y': 75},
                'selectable': False
            },
            {
                'data': {'id': 'four', 'label': 'Not grabbable'},
                'position': {'x': 200, 'y': 200},
                'grabbable': False
            },
            {'data': {'source': 'one', 'target': 'two'}},
            {'data': {'source': 'two', 'target': 'three'}},
            {'data': {'source': 'three', 'target': 'four'}},
            {'data': {'source': 'two', 'target': 'four'}},
        ]
    )
])

if __name__ == '__main__':
    app.run(debug=True)


# %%
"""
Original Demo: http://js.cytoscape.org/demos/cose-layout/

Note: This implementation looks different from the original implementation,
although the input parameters are exactly the same.
"""
import requests
import json

from dash import Dash, html, Input, Output, callback
import dash_cytoscape as cyto

def load_json(st):
    if 'http' in st:
        return requests.get(st).json()
    else:
        with open(st, 'rb') as f:
            x = json.load(f)
        return x

app = Dash()
server = app.server

# Load Data
elements = load_json('https://js.cytoscape.org/demos/colajs-graph/data.json')
stylesheet = load_json('https://js.cytoscape.org/demos/colajs-graph/cy-style.json')

styles = {
    'container': {
        'position': 'fixed',
        'display': 'flex',
        'flex-direction': 'column',
        'height': '100%',
        'width': '100%'
    },
    'cy-container': {
        'flex': '1',
        'position': 'relative'
    },
    'cytoscape': {
        'position': 'absolute',
        'width': '100%',
        'height': '100%',
        'z-index': 999
    }
}

# App
app.layout = html.Div(style=styles['container'], children=[
    html.Div([
        html.Button("Responsive Toggle", id='toggle-button'),
        html.Div(id='toggle-text')
    ]),
    html.Div(className='cy-container', style=styles['cy-container'], children=[
        cyto.Cytoscape(
            id='cytoscape-responsive-layout',
            elements=elements,
            stylesheet=stylesheet,
            style=styles['cytoscape'],
            layout={
                'name': 'cose',
                'idealEdgeLength': 100,
                'nodeOverlap': 20,
                'refresh': 20,
                'fit': True,
                'padding': 30,
                'randomize': False,
                'componentSpacing': 100,
                'nodeRepulsion': 400000,
                'edgeElasticity': 100,
                'nestingFactor': 5,
                'gravity': 80,
                'numIter': 1000,
                'initialTemp': 200,
                'coolingFactor': 0.95,
                'minTemp': 1.0
            },
            responsive=True
        )
    ])
])

@callback(Output('cytoscape', 'responsive'), Input('toggle-button', 'n_clicks'))
def toggle_responsive(n_clicks):
    n_clicks = 2 if n_clicks is None else n_clicks
    toggle_on = n_clicks % 2 == 0
    return toggle_on

@callback(Output('toggle-text', 'children'), Input('cytoscape', 'responsive'))
def update_toggle_text(responsive):
    return '\t' + 'Responsive ' + ('On' if responsive else 'Off')

if __name__ == '__main__':
    app.run(debug=True)

# %%



