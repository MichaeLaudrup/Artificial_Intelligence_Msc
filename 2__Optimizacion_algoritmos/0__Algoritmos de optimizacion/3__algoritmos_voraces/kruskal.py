import networkx as nx
import matplotlib.pyplot as plt

graph = nx.Graph()

# Agregar aristas con pesos según la imagen
edges = [
    ('A', 'B', 3),
    ('A', 'C', 6),
    ('A', 'E', 9),
    ('B', 'C', 2),
    ('B', 'D', 2),
    ('B', 'E', 9),
    ('C', 'D', 2),
    ('C', 'F', 9),
    ('D', 'F', 8),
    ('D', 'G', 8),
    ('E', 'G', 8),
    ('F', 'G', 7),
    ('F', 'H', 4),
    ('G', 'I', 9),
    ('G', 'J', 18),
    ('H', 'I', 1),
    ('H', 'J', 3),
    ('I', 'J', 10)
]

# Coste temporal n · log n
sorted_edges = sorted(graph.edges(data=True), key=lambda x: x[2]['weight'])