from sim_class import Simulation

# Initialize the simulation with a specified number of agents
sim = Simulation(num_agents=1)  # For one robot

# Example action: Move joints with specific velocities
velocity_x = 0.1
velocity_y = 0.1     
velocity_z = 0.1
drop_command = 1
actions = [[velocity_x, velocity_y, velocity_z, drop_command]]

# Run the simulation for a specified number of steps
state = sim.run(actions, num_steps=1000)
print(state)
