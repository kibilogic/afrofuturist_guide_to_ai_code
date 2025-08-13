import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

# Gaming Website Structure 
gaming_web = {
    'home': ['games', 'esports', 'streaming', 'community'],
    'games': ['fortnite', 'minecraft', 'valorant', 'amongus', 'roblox'],
    'esports': ['tournaments', 'teams', 'rankings', 'live_matches'],
    'streaming': ['twitch', 'youtube', 'tiktok', 'discord'],
    'community': ['forums', 'memes', 'fan_art', 'guides'],
    'fortnite': ['battle_royale', 'creative', 'save_world'],
    'minecraft': ['survival', 'creative', 'multiplayer', 'mods'],
    'valorant': ['ranked', 'agents', 'maps', 'weapons'],
    'amongus': ['online', 'local', 'mods'],
    'roblox': ['games', 'avatar', 'robux'],
    'tournaments': ['championship', 'prizes'],
    'teams': ['fnatic', 'cloud9', 'tsm'],
    'rankings': ['leaderboard'],
    'live_matches': ['schedule'],
    'twitch': ['streamers', 'clips'],
    'youtube': ['gamers', 'tutorials'],
    'tiktok': ['gaming_clips'],
    'discord': ['servers', 'bots'],
    'forums': ['discussions'],
    'memes': ['viral', 'trending'],
    'fan_art': ['contests'],
    'guides': ['tutorials', 'tips'],
    # Terminal nodes (no outgoing links)
    'battle_royale': [], 'creative': [], 'save_world': [],
    'survival': [], 'multiplayer': [], 'mods': [],
    'ranked': [], 'agents': [''], 'maps': [], 'weapons': [],
    'online': [], 'local': [], 'avatar': [], 'robux': [],
    'championship': [], 'prizes': [], 'fnatic': [], 'cloud9': [], 'tsm': [],
    'leaderboard': [], 'schedule': [], 'streamers': [], 'clips': [],
    'gamers': [], 'tutorials': [], 'gaming_clips': [], 'servers': [], 'bots': [],
    'discussions': [], 'viral': [], 'trending': [], 'contests': [], 'tips': []
}

print("""
GAMING WEB CRAWLER SIMULATION
==================================================
Starting from the homepage...
Watch as our crawler explores the gaming universe!
""")

# Build graph 
G = nx.DiGraph(gaming_web)
pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

# BFS setup
start = 'home'
visited_order, queue, visited, step = [], deque([start]), {start}, 0

print(f"Starting crawl from: {start}\n")

while queue:
    current = queue.popleft()
    visited_order.append(current)
    step += 1

    for neighbor in gaming_web.get(current, []):
        if neighbor not in visited:
            visited.add(neighbor)
            queue.append(neighbor)

print(f"\nCrawl complete! Visited {len(visited_order)} pages total!\n" + "="*50)

# Visualization
plt.figure(figsize=(16, 12))
plt.style.use('dark_background')

node_colors_map = {
    'home': '#FF6B35', 'games': '#4ECDC4', 'esports': '#FFEAA7',
    'streaming': '#45B7D1', 'community': '#96CEB4', 'visited': '#DDA0DD',
    'default': '#E8E8E8'
}

def get_color(node):
    if node == 'home': return node_colors_map['home']
    if node in gaming_web['games']: return node_colors_map['games']
    if node in gaming_web['esports']: return node_colors_map['esports']
    if node in gaming_web['streaming']: return node_colors_map['streaming']
    if node in gaming_web['community']: return node_colors_map['community']
    if node in visited_order: return node_colors_map['visited']
    return node_colors_map['default']

nx.draw(G, pos, with_labels=True, node_color=[get_color(node) for node in G.nodes()],
        node_size=1500, font_size=8, font_weight='bold', font_color='white',
        arrows=True, arrowsize=15, arrowstyle='->', edge_color='#555555', width=1.5)

plt.title("GAMING WEB CRAWLER: BFS Exploration",
          fontsize=16, fontweight='bold', color='white', pad=20)

legend_elements = [
    plt.Line2D(
        [0], [0],
        marker='o',
        color='w',
        markerfacecolor=node_colors_map[key],
        markersize=10,
        label=label
    )
    for key, label in zip(
        ['home', 'games', 'streaming', 'community', 'esports', 'visited'],
        ['Homepage', 'Games', 'Streaming', 'Community', 'Esports', 'Other Pages']
    )
]

plt.legend(
    handles=legend_elements,
    loc='upper left',
    bbox_to_anchor=(1.02, 1)
)

stats_text = (
    f"CRAWLER STATS \n"
    f"Total pages crawled: {len(visited_order)}\n"
    f"Games discovered: {len([p for p in visited_order if p in gaming_web['games']])}\n"
    f"Streaming platforms: {len([p for p in visited_order if p in ['twitch', 'youtube', 'tiktok']])}"
)

plt.text(
    0.02, 0.98,
    stats_text,
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    bbox=dict(boxstyle='round', facecolor='#333333', alpha=0.8),
    color='white'
)

plt.tight_layout()
plt.show()

print("\nFun Facts about this crawl:")
print(f"- Most popular section: Games (with {len(gaming_web['games'])} awesome titles)")
print(f"- Total connections mapped: {len(G.edges())} links")

deepest = max(
    len(gaming_web[game]) if isinstance(gaming_web[game], list) else 0
    for game in gaming_web
)


print(f"- Deepest path: {deepest} levels deep")
print("\nReal web crawlers work the same way but with millions of pages")
print("They help search engines like Google index the entire internet")
