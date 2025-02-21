import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy as cp

#################################
#####  PARAMETERS   #####
#################################

carrying_cap = 500.0    # carrying capacity of rabbits
rabbits_init = 100      # initial rabbit population
moveradius_r = 0.03     # magnitude of movement of rabbits
death_r = 1.0           # death rate of rabbits when facing foxes 
repro_r = 0.1           # reproduction rate of rabbits

foxes_init = 30         # initial fox population
moveradius_f = 0.07     # magnitude of movement of foxes
death_f = 0.1           # death rate of foxes with no food
repro_f = 0.5           # reproduction rate of foxes

hunt_radius = 0.04      # radius for collision detection

#################################
#####  AGENT CLASS   #####
#################################

class Agent:
    def __init__(self, agent_type, x=None, y=None):
        self.type = agent_type
        self.x = np.random.random() if x is None else x
        self.y = np.random.random() if y is None else y
    
    def move(self, move_radius):
        self.x = np.clip(self.x + np.random.uniform(-move_radius, move_radius), 0, 1)
        self.y = np.clip(self.y + np.random.uniform(-move_radius, move_radius), 0, 1)
    
    def distance_to(self, other_agent):
        return (self.x - other_agent.x)**2 + (self.y - other_agent.y)**2

#################################
#####  SIMULATION CLASS   #####
#################################

class Simulation:
    def __init__(self):
        self.all_rabbit_x = []
        self.all_rabbit_y = []
        self.all_fox_x = []
        self.all_fox_y = []
        self.rabbit_pop = []
        self.fox_pop = []
        self.agents = []
        
    def initialize(self):
        self.agents = []
        self.agents.extend([Agent('rabbit') for _ in range(rabbits_init)])
        self.agents.extend([Agent('fox') for _ in range(foxes_init)])
    
    def observe(self):
        rabbits = [ag for ag in self.agents if ag.type == 'rabbit']
        foxes = [ag for ag in self.agents if ag.type == 'fox']
        
        self.all_rabbit_x.append([ag.x for ag in rabbits])
        self.all_rabbit_y.append([ag.y for ag in rabbits])
        self.all_fox_x.append([ag.x for ag in foxes])
        self.all_fox_y.append([ag.y for ag in foxes])
        
        self.rabbit_pop.append(len(rabbits))
        self.fox_pop.append(len(foxes))
    
    def run(self, T):
        self.initialize()
        self.observe()  # Initial state
        
        for t in range(1, T):
            print(f"Simulating time step {t}/{T-1}")
            for _ in range(len(self.agents)):
                self.update_one_agent()
            self.observe()

    def update_one_agent(self):
        if not self.agents:
            return

        ag = self.agents[np.random.randint(len(self.agents))]
        move_radius = moveradius_r if ag.type == 'rabbit' else moveradius_f
        
        # Movement
        ag.move(move_radius)
        
        # Collision detection - more efficient now
        neighbors = [ag2 for ag2 in self.agents 
                    if ag2.type != ag.type and ag.distance_to(ag2) < hunt_radius**2]

        # Handle rabbit behavior
        if ag.type == 'rabbit':
            if neighbors and np.random.random() < death_r:
                self.agents.remove(ag)
                return
            
            current_rabbit_population = sum(1 for a in self.agents if a.type == "rabbit")
            reproduction_chance = repro_r * (1 - current_rabbit_population/carrying_cap)
            
            if np.random.random() < reproduction_chance:
                self.agents.append(cp.copy(ag))
        
        # Handle fox behavior
        else:  # ag.type == 'fox'
            if not neighbors:
                if np.random.random() < death_f:
                    self.agents.remove(ag)
                    return
            elif np.random.random() < repro_f:
                self.agents.append(cp.copy(ag))

#################################
#####  SIMULATION & ANIMATION #####
#################################

class Visualizer:
    def __init__(self, simulation):
        self.sim = simulation
        
    def create_animation(self, T, interval=50):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
        
        # Initialize plots
        scat_r = ax1.scatter([], [], c='blue', marker='.', label='Rabbits')
        scat_f = ax1.scatter([], [], c='red', marker='o', label='Foxes')
        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.set_aspect('equal')
        ax1.legend()

        line_r, = ax2.plot([], [], 'b-', label='Rabbits')
        line_f, = ax2.plot([], [], 'r-', label='Foxes')
        ax2.set_xlim(0, T)
        ax2.set_ylim(0, max(max(self.sim.rabbit_pop), max(self.sim.fox_pop)) * 1.1)
        ax2.legend()
        
        def animate(frame):
            scat_r.set_offsets(np.c_[self.sim.all_rabbit_x[frame], 
                                   self.sim.all_rabbit_y[frame]])
            scat_f.set_offsets(np.c_[self.sim.all_fox_x[frame], 
                                   self.sim.all_fox_y[frame]])
            
            line_r.set_data(range(frame+1), self.sim.rabbit_pop[:frame+1])
            line_f.set_data(range(frame+1), self.sim.fox_pop[:frame+1])
            
            ax1.set_title(f'Time: {frame}')
            return scat_r, scat_f, line_r, line_f

        ani = animation.FuncAnimation(fig, animate, frames=T, 
                                    interval=interval, blit=True)
        return ani, fig


if __name__ == "__main__":
    np.random.seed(2025)
    sim = Simulation()
    T = 501
    sim.run(T)
    
    viz = Visualizer(sim)
    ani, fig = viz.create_animation(T)
    
    # Save video (requires ffmpeg)
    #ani.save('predator_prey_simulation.mp4', writer='ffmpeg', fps=15)
    
    # Or show the final population plot
    plt.figure()
    plt.plot(sim.rabbit_pop, 'b-', label='Rabbits')
    plt.plot(sim.fox_pop, 'r-', label='Foxes')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.legend()
    plt.show()