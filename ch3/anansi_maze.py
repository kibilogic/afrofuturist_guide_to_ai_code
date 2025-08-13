import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import time

# Anansi's maze represented as an adjacency list
maze = {
    'A': ['B', 'C'],    # From the starting chamber, two paths diverge
    'B': ['D', 'E'],    # The left branch offers two choices
    'C': ['F'],         # The right branch leads to a single chamber
    'D': [],            # Dead end - no escape here
    'E': ['G'],         # This chamber holds the key to freedom
    'F': [],            # Another dead end
    'G': []             # Victory! The exit chamber
}

start_node = 'A'
goal_node = 'G'

# Build graph from maze
G = nx.DiGraph()
for node, neighbors in maze.items():
    for neighbor in neighbors:
        G.add_edge(node, neighbor)

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

def draw_graph(current_path, visited, title):
    plt.clf()
    nx.draw(G, pos, with_labels=True,
            node_color='lightgrey', edge_color='grey',
            node_size=2000, font_weight='bold', font_color='white')

    # Highlight visited
    nx.draw_networkx_nodes(G, pos, nodelist=visited, node_color='orange')

    # Highlight current path
    if current_path:
        edges = list(zip(current_path, current_path[1:]))
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color='red', width=3)
        nx.draw_networkx_nodes(G, pos, nodelist=[current_path[-1]], node_color='red')

    plt.title(title, fontsize=14)
    plt.pause(1)

# Breadth-First Search
def bfs(maze, start, goal):
    print("\n--- BREADTH-FIRST SEARCH (BFS) ---")
    visited = set()
    queue = deque([[start]])
    step = 0

    while queue:
        path = queue.popleft()
        node = path[-1]
        step += 1
        print(f"Step {step}: Exploring from node {node}")
        neighbors = maze.get(node, [])
        print(f"   Neighbors: {neighbors}")

        if node == goal:
            print(f"Anansi escaped! BFS Path: {path}")
            return path

        if node not in visited:
            visited.add(node)
            for neighbor in neighbors:
                if neighbor not in visited:
                    print(f"   Added {neighbor} to queue")
                    queue.append(path + [neighbor])
        draw_graph(path, visited, f"BFS Visiting: {node}")

    print("No path found.")
    return None


# Depth-First Search
def dfs(maze, start, goal):
    print("\n--- DEPTH-FIRST SEARCH (DFS) ---")
    visited = set()
    stack = [[start]]
    step = 0

    while stack:
        path = stack.pop()
        node = path[-1]
        step += 1
        print(f"Step {step}: Exploring from node {node}")
        neighbors = maze.get(node, [])
        print(f"   Neighbors: {neighbors}")

        if node == goal:
            print(f"Anansi escaped! DFS Path: {path}")
            return path

        if node not in visited:
            visited.add(node)
            for neighbor in reversed(neighbors):
                if neighbor not in visited:
                    print(f"   Added {neighbor} to stack")
                    stack.append(path + [neighbor])
        draw_graph(path, visited, f"DFS Visiting: {node}")

    print("No path found.")
    return None


# Visualization
plt.ion()
plt.figure(figsize=(8, 6))

bfs_path = bfs(maze, start_node, goal_node)
time.sleep(2)
dfs_path = dfs(maze, start_node, goal_node)

plt.ioff()
plt.show()

# Final Path 
print(dfs_path or bfs_path)

