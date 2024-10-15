import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from simple_rooms import create_simple_room, Agent, create_internal_walls

# Simulation parameters
room_size = (10, 10)
num_rewards = 5
num_steps = 1000
agent_radius = 0.5
agent_start_pos = (5, 5)
num_internal_walls = 5  # Increased for more interesting environments

# Create internal walls
internal_walls = create_internal_walls(num_internal_walls, room_size[0])

# Create the environment
room = create_simple_room(room_size, num_rewards, internal_walls)
agent = Agent(agent_start_pos, agent_radius)

# Set up the plot
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_xlim(0, room_size[0])
ax.set_ylim(0, room_size[1])
ax.set_aspect('equal')

# Plot walls (boundary walls in black, internal walls in red)
boundary_walls = [(wall.start_pos, wall.end_pos) for wall in room.walls[:4]]
internal_walls = [(wall.start_pos, wall.end_pos) for wall in room.walls[4:]]
ax.add_collection(LineCollection(boundary_walls, color='black', linewidth=2))
ax.add_collection(LineCollection(internal_walls, color='red', linewidth=2))

# Plot agent
agent_circle = Circle(agent.position, agent.radius, fc='blue', ec='black')
ax.add_patch(agent_circle)

# Plot rewards
reward_circles = [Circle(reward.position, 0.2, fc='gold', ec='orange') for reward in room.objects]
for circle in reward_circles:
    ax.add_patch(circle)

# Text for displaying total reward
reward_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top')

total_reward = 0

def update():
    global total_reward, agent, room, agent_circle, reward_circles, reward_text
    
    # Random action (you would replace this with your RL agent's policy)
    action = np.random.uniform(-0.01, 0.01, 2)
    
    # Move the agent
    agent.move(action)
    
    # Handle wall collisions
    agent.handle_collision(room)
    
    # Update agent position in plot
    agent_circle.center = agent.position
    
    # Check for reward collection and update rewards
    for reward, circle in zip(room.objects.copy(), reward_circles):
        if not reward.collected:
            collected_reward = reward.check_collection(agent.position, agent.radius)
            if collected_reward > 0:
                total_reward += collected_reward
                circle.set_visible(False)
                room.objects.remove(reward)
    
    # Update reward text
    reward_text.set_text(f'Total Reward: {total_reward:.2f}')
    
    # Return a list of artists that we need to redraw
    return [agent_circle] + reward_circles + [reward_text]

plt.title('RL Agent in Room with Improved Internal Walls')

# Run the simulation
for step in range(num_steps):
    update()
    plt.pause(0.05)  # Pause to create animation effect

plt.show()

print(f"\nSimulation ended. Total reward collected: {total_reward}")
print(f"Remaining rewards: {len([obj for obj in room.objects if not obj.collected])}")
