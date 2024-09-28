import pygame
import random
import networkx as nx
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from crewai import Agent
import time

# Initialize Pygame
pygame.init()

# Set up display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Campus Tour Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
YELLOW = (255, 255, 0)

# Load GPT-2 model and tokenizer for interactions
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=100, num_return_sequences=1)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# Define performance data and agent statuses
agent_status = {'BI_Agent_A': 'Available', 'BI_Agent_B': 'Available'}
performance_data = {'CI_Success': 0, 'CI_Failures': 0, 'BI_Guided': 0, 'BI_OOS_Violations': 0}
oos_durations = {}  # To track when BI agents went OOS and when they should come back

# Threshold for maximum OOS duration (in seconds)
OOS_THRESHOLD = 5  # Maximum OOS duration in seconds

# Graph representing campus and building navigation
G = nx.Graph()

# Node positions
positions = {
    1: (400, 300), 2: (600, 400), 3: (200, 400), 4: (400, 500),
    5: (400, 400), 6: (700, 500), 7: (600, 300), 8: (600, 200),
    9: (700, 100), 10: (500, 200), 11: (300, 300), 12: (100, 500),
    13: (300, 200), 14: (100, 300), 15: (200, 100), 16: (300, 100),
    17: (500, 400), 18: (400, 200), 19: (400, 100), 20: (500, 500),
    21: (300, 50)
}

# Campus graph edges
edges = [
    (1, 2), (1, 5), (1, 4), (1, 7), (2, 4), (2, 6), (3, 4), (3, 11),
    (3, 12), (4, 5), (5, 11), (5, 17), (5, 18), (6, 7), (7, 8),
    (7, 10), (8, 9), (10, 17), (10, 18), (11, 13), (11, 14), (12, 14),
    (13, 15), (14, 16), (15, 16), (16, 19), (19, 21), (19, 20), (20, 17)
]

# Add nodes and edges to the graph
G.add_nodes_from(positions.keys())
G.add_edges_from(edges)

# Agent class for movement and visualization in Pygame
class AgentSprite(pygame.sprite.Sprite):
    def __init__(self, name, color, position):
        super().__init__()
        self.name = name
        self.image = pygame.Surface((20, 20))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.center = position
        self.path = []
        self.speed = 2
        self.moving = False
        self.inside_building = False

    def set_path(self, path):
        self.path = path
        self.moving = True

    def update(self):
        if self.path and self.moving:
            target_node = self.path[0]
            target_position = positions[target_node]
            dx, dy = target_position[0] - self.rect.centerx, target_position[1] - self.rect.centery
            dist = (dx ** 2 + dy ** 2) ** 0.5
            if dist < self.speed:
                self.rect.center = target_position
                self.path.pop(0)
                if not self.path:
                    self.moving = False
            else:
                self.rect.centerx += dx / dist * self.speed
                self.rect.centery += dy / dist * self.speed

# CrewAI agent definition for CI
ci_agent = Agent(role="CI Agent", goal="Escort visitors", backstory="Navigate visitors to the host.")

# CI agent sprite
ci_agent_sprite = AgentSprite("CI Agent", BLUE, positions[1])

# All sprites
all_sprites = pygame.sprite.Group(ci_agent_sprite)

# Simulating BI agent response (either providing a path or going OOS)
def bi_agent_response(bi_agent_name, building_name):
    if random.random() < 0.3:
        duration = random.randint(1, OOS_THRESHOLD)  # Random OOS duration but limited by the threshold
        oos_durations[bi_agent_name] = time.time() + duration  # Track the return time
        agent_status[bi_agent_name] = 'OOS'
        print(f"BI {bi_agent_name} is Out of Service for {duration} seconds.")
        update_performance('BI Agent', 'BI_OOS_Violations')
        return False
    else:
        print(f"BI {bi_agent_name} provides the path for {building_name}.")
        update_performance('BI Agent', 'BI_Guided')
        return True

# Visitor logic (handling multiple visitors with random hosts)
visitor_count = 0
number_of_visitors = 5  # Generate 5 visitors
building_targets = list(positions.keys())  # All nodes are possible targets

# Randomly assign target buildings for visitors
visitors = [(f"Visitor {i+1}", random.choice(building_targets)) for i in range(number_of_visitors)]
current_visitor = None

def assign_new_visitor():
    global current_visitor, visitor_count
    if visitor_count < len(visitors):
        current_visitor, target_building = visitors[visitor_count]
        print(f"{current_visitor} needs to be escorted to building {target_building}.")
        if not bi_agent_response("BI_Agent_A", target_building):
            print(f"CI agent failed to guide {current_visitor}.")
            update_performance("CI Agent", "CI_Failures")
        else:
            shortest_path = nx.dijkstra_path(G, 1, target_building)
            ci_agent_sprite.set_path(shortest_path)
            update_performance("CI Agent", "CI_Success")
        visitor_count += 1
    else:
        print("All visitors have been escorted.")

# Performance tracking
def update_performance(agent_type, metric):
    performance_data[metric] += 1
    print(f"Performance Update - {agent_type}: {metric} = {performance_data[metric]}")

# Visitor and agent interaction simulation
def simulate_visitor_movement():
    global current_visitor
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()

    # Update BI agent status based on OOS timing
    current_time = time.time()
    for agent, return_time in oos_durations.items():
        if current_time >= return_time and agent_status[agent] == 'OOS':
            print(f"{agent} is now Available again.")
            agent_status[agent] = 'Available'

    # Assign new visitor if CI agent is not moving
    if not ci_agent_sprite.moving and visitor_count < len(visitors):
        assign_new_visitor()

    screen.fill(WHITE)

    # Draw the campus graph with OOS status visualization
    for node, pos in positions.items():
        node_color = BLACK
        if node == 19 and agent_status['BI_Agent_A'] == 'OOS':  # For example, building 19 managed by BI_Agent_A
            node_color = RED  # Color change to red when OOS
        pygame.draw.circle(screen, node_color, pos, 10)
        font = pygame.font.Font(None, 24)
        text = font.render(str(node), True, BLACK)
        screen.blit(text, (pos[0] - 10, pos[1] - 20))

    for edge in edges:
        pygame.draw.line(screen, BLACK, positions[edge[0]], positions[edge[1]], 3)

    all_sprites.update()
    all_sprites.draw(screen)

    pygame.display.flip()

# Main simulation loop
def run_simulation():
    clock = pygame.time.Clock()
    while True:
        simulate_visitor_movement()
        clock.tick(30)

# Run the simulation
run_simulation()
