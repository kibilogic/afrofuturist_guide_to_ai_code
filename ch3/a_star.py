import heapq
import matplotlib.pyplot as plt
import networkx as nx
import time

# Weighted maze (neighbor, cost)
maze = {
    'A': [('B', 2), ('C', 5)],
    'B': [('D', 4), ('E', 1)],
    'C': [('F', 2)],
    'D': [],
    'E': [('G', 3)],
    'F': [],
    'G': []
}

start_node = 'A'
goal_node = 'G'

# Layout 
pos = {
    'A': (0, 3),
    'B': (-1, 2),
    'C': (1, 2),
    'D': (-2, 1),
    'E': (0, 1),
    'F': (2, 1),
    'G': (0, 0)
}

# Build graph from maze 
G = nx.DiGraph()
for node, neighbors in maze.items():
    for neighbor, weight in neighbors:
        G.add_edge(node, neighbor, weight=weight)

# h(n) - estimated cost to goal
def heuristic(node):
    return abs(pos[node][1] - pos[goal_node][1])

def draw_graph(current_path, visited, title):
    plt.clf()
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw(G, pos, with_labels=True, node_color='lightgrey', edge_color='gray',
            node_size=2000, font_color='white', font_weight='bold')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    nx.draw_networkx_nodes(G, pos, nodelist=visited, node_color='orange')
    if current_path:
        path_edges = list(zip(current_path, current_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='blue', width=3)
        nx.draw_networkx_nodes(G, pos, nodelist=[current_path[-1]], node_color='blue')
    plt.title(title)
    plt.pause(1)

# A* with weights
def a_star_weighted(maze, start, goal):
    print("\n--- A* STRATEGIC ADVANCE WITH WEIGHTS ---")
    frontier = []
    heapq.heappush(frontier, (0, [start]))
    visited = set()
    g_cost = {start: 0}
    step = 0

    while frontier:
        f, path = heapq.heappop(frontier)
        node = path[-1]
        step += 1
        visited.add(node)

        print(f"Step {step}: Campaign at {node}")
        print(f"   Path so far: {path}")
        print(f"   g(n): {g_cost[node]}, h(n): {heuristic(node)}, f(n): {f}")

        draw_graph(path, visited, f"A* Weighted: {node}")

        if node == goal:
            print(f"\nVictory! Final A* Path: {path} with total cost: {g_cost[node]}")
            return path

        for neighbor, weight in maze.get(node, []):
            tentative_g = g_cost[node] + weight
            if neighbor not in g_cost or tentative_g < g_cost[neighbor]:
                g_cost[neighbor] = tentative_g
                f = tentative_g + heuristic(neighbor)
                new_path = path + [neighbor]
                heapq.heappush(frontier, (f, new_path))
                print(f"   Added {neighbor} to campaign with cost {weight}, f(n) = {f}")

    print("Failed to reach the goal.")
    return None

# Visualization
plt.ion()
plt.figure(figsize=(8, 6))

final_path = a_star_weighted(maze, start_node, goal_node)

plt.ioff()
plt.show()

print(f"\nHannibal's strategic position: {final_path}")

