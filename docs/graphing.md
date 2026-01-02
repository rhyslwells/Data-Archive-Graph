data_loader.py (shared)
    ↓
graph_builder.py (shared)
    ↓
network_stats.py (shared)
    ↓
├── pyvis_visualizer.py → main.py
└── cytoscape_visualizer.py → main_cytoscape.py


Why PyVis for Streamlit:
Native Integration: Streamlit has streamlit-agraph and can embed PyVis HTML directly with components.html()
Simpler Architecture: PyVis generates standalone HTML files that are easy to embed
No Server Conflicts: Dash Cytoscape runs its own server which conflicts with Streamlit's server
Better Performance: Static HTML rendering is faster than running a nested Dash app