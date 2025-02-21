import numpy as np
import matplotlib.pyplot as plt
import copy as cp
import os
import imageio.v2 as imageio
from dataclasses import dataclass
from typing import List, Literal

# Create directory for output figures
os.makedirs('figs', exist_ok=True)

@dataclass
class Agent:
    """Class representing an agent (rabbit or fox) in the simulation"""
    type: Literal['rabbit', 'fox']
    x: float
    y: float

class PredatorPreySimulation:
    def __init__(self, params=None):
        # Default parameters
        self.params = {
            'carrying_cap': 500.0,
            'rabbits_init': 100,
            'moveradius_r': 0.03,
            'death_r': 1.0,
            'repro_r': 0.1,
            'foxes_init': 30,
            'moveradius_f': 0.07,
            'death_f': 0.1,
            'repro_f': 0.5,
            'hunt_radius': 0.04
        }
        
        # Update parameters if provided
        if params:
            self.params.update(params)
            
        # Initialize outcome variables
        self.rabbit_pop = []
        self.fox_pop = []
        self.agents = []
        self.t = 0
        
    def initialize(self):
        """Initialize the simulation with rabbits and foxes"""
        self.agents = []
        
        # Create initial rabbits
        for _ in range(self.params['rabbits_init']):
            self.agents.append(Agent(
                type='rabbit',
                x=np.random.random(),
                y=np.random.random()
            ))

        # Create initial foxes
        for _ in range(self.params['foxes_init']):
            self.agents.append(Agent(
                type='fox',
                x=np.random.random(),
                y=np.random.random()
            ))

    def update_one_agent(self):
        """Update state for one randomly chosen agent"""
        if not self.agents:
            return

        ag = self.agents[np.random.randint(len(self.agents))]
        
        # Movement
        move_radius = self.params['moveradius_r'] if ag.type == 'rabbit' else self.params['moveradius_f']
        ag.x += np.random.uniform(-move_radius, move_radius)
        ag.y += np.random.uniform(-move_radius, move_radius)
        
        # Keep agents within bounds
        ag.x = np.clip(ag.x, 0, 1)
        ag.y = np.clip(ag.y, 0, 1)

        # Find nearby agents of opposite type
        neighbors = [
            ag2 for ag2 in self.agents 
            if (ag2.type != ag.type) and 
            ((ag.x - ag2.x)**2 + (ag.y - ag2.y)**2 < self.params['hunt_radius']**2)
        ]

        if ag.type == 'rabbit':
            # Rabbit death from predation
            if neighbors and np.random.random() < self.params['death_r']:
                self.agents.remove(ag)
                return
            
            # Rabbit reproduction
            rabbit_pop_count = len([a for a in self.agents if a.type == "rabbit"])
            if np.random.random() < self.params['repro_r'] * (1 - rabbit_pop_count / self.params['carrying_cap']):
                self.agents.append(cp.copy(ag))
        
        else:  # Fox behavior
            # Fox death from starvation
            if not neighbors and np.random.random() < self.params['death_f']:
                self.agents.remove(ag)
                return
            
            # Fox reproduction after successful hunt
            if neighbors and np.random.random() < self.params['repro_f']:
                self.agents.append(cp.copy(ag))

    def update_one_time_step(self):
        """Update all agents once"""
        for _ in range(len(self.agents)):
            self.update_one_agent()

    def observe(self):
        """Record current state and create visualization"""
        # Count populations
        foxes = [ag for ag in self.agents if ag.type == 'fox']
        rabbits = [ag for ag in self.agents if ag.type == 'rabbit']
        
        self.fox_pop.append(len(foxes))
        self.rabbit_pop.append(len(rabbits))

        # Create visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 10))
        
        ax1.set_title(f'Time {self.t}', fontsize=16)

        # Plot agents
        if rabbits:
            ax1.plot([ag.x for ag in rabbits], [ag.y for ag in rabbits], '.', color="blue", label="Rabbits")
        if foxes:
            ax1.plot([ag.x for ag in foxes], [ag.y for ag in foxes], 'o', color="red", label="Foxes")
        
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect("equal")
        ax1.legend()

        # Plot population history
        ax2.plot(range(self.t + 1), self.fox_pop, color="red", label="Foxes")
        ax2.plot(range(self.t + 1), self.rabbit_pop, color="blue", label="Rabbits")
        ax2.set_xlabel("Time")
        ax2.set_ylabel("Population")
        ax2.legend()
        
        plt.savefig(f'figs/{self.t}.png', bbox_inches='tight')
        plt.close()

    def run(self, time_steps=500, seed=None):
        """Run the simulation for specified number of time steps"""
        if seed is not None:
            np.random.seed(seed)
            
        self.initialize()
        self.observe()

        for self.t in range(1, time_steps + 1):
            print(f"Time step: {self.t}")
            self.update_one_time_step()
            self.observe()

    def create_animation(self, fps=25):
        """Create GIF animation from saved figures"""
        images = [imageio.imread(f'figs/{t}.png') 
                 for t in range(len(self.rabbit_pop))]
        imageio.mimsave('predator_prey.gif', images, fps=fps)

def main():
    # Create and run simulation
    sim = PredatorPreySimulation()
    sim.run(time_steps=500, seed=2025)
    sim.create_animation()

if __name__ == "__main__":
    main() 