def helloworld():
    print("Hello world")


# G = nx.DiGraph()

# # Add nodes and edges (ensure we fetch outlinks and other necessary attributes correctly)
# for title, note in vault.items():
#     G.add_node(title)  # No tags added since not directly available in the note dictionary
#     outlinks = note.get("outlinks", [])  # Fetch outlinks from an external source if not in the note dictionary
#     for outlink in outlinks:
#         if outlink in vault:  # Only add edge if target exists
#             G.add_edge(title, outlink)

# print(f"\nGraph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")