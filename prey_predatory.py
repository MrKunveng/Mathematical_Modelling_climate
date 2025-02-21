import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Literal
import os
import imageio.v2 as imageio

@dataclass
class Config:
    """Configuration parameters for the simulation"""
    # Rabbit parameters
    carrying_cap: float = 500.0
    rabbits_init: int = 100
    moveradius_r: float = 0.03
    death_r: float = 1.0
    repro_r: float = 0.1

    # Fox parameters
    foxes_init: int = 30
    moveradius_f: float = 0.07
    death_f: float = 0.1
    repro_f: float = 0.5

    # Simulation parameters
    hunt_radius: float = 0.04
    total_steps: int = 501
    random_seed: int = 2025

class Agent:
    """Agent class representing either a fox or rabbit"""
    def __init__(self, agent_type: Literal['rabbit', 'fox'], x: float, y: float):
        self.type = agent_type
        self.x = x
        self.y = y

    def move(self, move_radius: float) -> None:
        """Move the agent randomly within the specified radius"""
        self.x = np.clip(self.x + np.random.uniform(-move_radius, move_radius), 0, 1)
        self.y = np.clip(self.y + np.random.uniform(-move_radius, move_radius), 0, 1)

    def distance_to(self, other: 'Agent') -> float:
        """Calculate distance to another agent"""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)

class PredatorPreySimulation:
    def __init__(self, config: Config):
        self.config = config
        self.agents: List[Agent] = []
        self.rabbit_pop: List[int] = []
        self.fox_pop: List[int] = []
        self.time_step = 0
        
        # Create directory for figures
        os.makedirs('figs', exist_ok=True)
        
        # Set random seed
        np.random.seed(config.random_seed)

    def initialize(self) -> None:
        """Initialize the simulation with rabbits and foxes"""
        self.agents.clear()
        self.rabbit_pop.clear()
        self.fox_pop.clear()

        # Initialize rabbits
        for _ in range(self.config.rabbits_init):
            self.agents.append(Agent('rabbit', np.random.random(), np.random.random()))

        # Initialize foxes
        for _ in range(self.config.foxes_init):
            self.agents.append(Agent('fox', np.random.random(), np.random.random()))

    def update_one_agent(self) -> None:
        """Update one randomly selected agent"""
        if not self.agents:
            return

        idx = np.random.randint(len(self.agents))
        ag = self.agents[idx]
        
        move_radius = self.config.moveradius_r if ag.type == 'rabbit' else self.config.moveradius_f
        ag.move(move_radius)

        # Find neighbors efficiently
        neighbors = [ag2 for ag2 in self.agents 
                    if ag2 != ag and ag2.type != ag.type 
                    and ag.distance_to(ag2) < self.config.hunt_radius]

        if ag.type == 'rabbit':
            self._handle_rabbit_update(ag, neighbors)
        else:
            self._handle_fox_update(ag, neighbors)

    def _handle_rabbit_update(self, rabbit: Agent, neighbors: List[Agent]) -> None:
        """Handle rabbit reproduction and death"""
        if neighbors and np.random.random() < self.config.death_r:
            if rabbit in self.agents:  # Check if rabbit still exists
                self.agents.remove(rabbit)
            return

        rabbit_count = sum(1 for a in self.agents if a.type == 'rabbit')
        if np.random.random() < self.config.repro_r * (1 - rabbit_count / self.config.carrying_cap):
            # Create new rabbit at slightly offset position
            new_x = np.clip(rabbit.x + np.random.uniform(-0.01, 0.01), 0, 1)
            new_y = np.clip(rabbit.y + np.random.uniform(-0.01, 0.01), 0, 1)
            self.agents.append(Agent('rabbit', new_x, new_y))

    def _handle_fox_update(self, fox: Agent, neighbors: List[Agent]) -> None:
        """Handle fox reproduction and death"""
        if not neighbors and np.random.random() < self.config.death_f:
            if fox in self.agents:  # Check if fox still exists
                self.agents.remove(fox)
            return
        elif neighbors and np.random.random() < self.config.repro_f:
            # Create new fox at slightly offset position
            new_x = np.clip(fox.x + np.random.uniform(-0.01, 0.01), 0, 1)
            new_y = np.clip(fox.y + np.random.uniform(-0.01, 0.01), 0, 1)
            self.agents.append(Agent('fox', new_x, new_y))

    def update_one_time_step(self) -> None:
        """Update all agents once"""
        for _ in range(len(self.agents)):
            self.update_one_agent()

    def observe(self) -> None:
        """Record and visualize the current state"""
        foxes = [ag for ag in self.agents if ag.type == 'fox']
        rabbits = [ag for ag in self.agents if ag.type == 'rabbit']
        
        self.fox_pop.append(len(foxes))
        self.rabbit_pop.append(len(rabbits))

        fig, (ax_map, ax_pop) = plt.subplots(2, 1, figsize=(8, 10))
        ax_map.set_title(f'Time {self.time_step}', fontsize=16)
        
        # Plot agents
        if rabbits:
            ax_map.plot([ag.x for ag in rabbits], [ag.y for ag in rabbits], 
                       '.', color="blue", label="Rabbits", markersize=10)
        if foxes:
            ax_map.plot([ag.x for ag in foxes], [ag.y for ag in foxes], 
                       'o', color="red", label="Foxes", markersize=8)
            
        ax_map.set_xlim(0, 1)
        ax_map.set_ylim(0, 1)
        ax_map.set_aspect("equal")
        ax_map.legend()
        ax_map.grid(True)

        # Plot population trends
        ax_pop.plot(range(self.time_step + 1), self.fox_pop, color="red", label="Foxes")
        ax_pop.plot(range(self.time_step + 1), self.rabbit_pop, color="blue", label="Rabbits")
        ax_pop.legend(loc="upper right")
        ax_pop.set_xlim(0, self.config.total_steps)
        ax_pop.set_ylim(0, self.config.carrying_cap + 50)
        ax_pop.set_xlabel("Time Steps")
        ax_pop.set_ylabel("Population")
        ax_pop.grid(True)

        plt.tight_layout()
        plt.savefig(f'figs/{self.time_step}.png', bbox_inches='tight', pad_inches=0.1)
        plt.close()

    def run(self) -> None:
        """Run the complete simulation"""
        self.initialize()
        self.observe()

        for self.time_step in range(1, self.config.total_steps):
            print(f"Step {self.time_step}")
            self.update_one_time_step()
            self.observe()

        self._create_animation()

    def _create_animation(self) -> None:
        """Create animation from saved figures"""
        images = [imageio.imread(f'figs/{t}.png') 
                 for t in range(self.config.total_steps)]
        imageio.mimsave('predator_prey.gif', images, fps=25)

def main():
    config = Config()
    simulation = PredatorPreySimulation(config)
    simulation.run()

if __name__ == "__main__":
    main()
