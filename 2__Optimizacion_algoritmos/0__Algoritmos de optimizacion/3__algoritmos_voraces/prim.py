# Obtener el arbol de recubrimiento minimo utilizando el algoritmo de Prim


# obtener el arbol de recubrimiento minimo utilizando el algoritmo de kruskal
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

# Agregar al grafo
graph.add_weighted_edges_from(edges)

# Dibujar el grafo
pos = nx.spring_layout(graph, seed=42)  # Distribución de nodos
nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=600, font_weight='bold')
nx.draw_networkx_edge_labels(graph, pos, edge_labels={(u, v): d for u, v, d in graph.edges(data='weight')})

import random
def prim_algorithm(graph):
    visited = list()
    candidates = list(graph.nodes)
    current_node = random.choice(candidates)
    visited.append(current_node)

    while(len(visited) != len(candidates)):
        neighbors = list(graph.neighbors(current_node))
        best_w = graph.get_edge_data(current_node, neighbors[0])['weight']
        best_neighbor_node = neighbors[0]

        for neighbor_node in neighbors[1:]:
            neighbor_w = graph.get_edge_data(current_node, neighbor_node)['weight']
            if (neighbor_w < best_w) and (neighbor_node not in visited):
                best_w = neighbor_w
                best_neighbor_node = neighbor_node
        current_node = best_neighbor_node
        print(visited)
        visited.append(current_node)

prim_algorithm(graph)

