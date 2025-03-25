import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from mesa.datacollection import DataCollector
from mesa.visualization import SolaraViz, make_plot_component, make_space_component



class HeroAgent(mesa.Agent):
    """A hero Agent"""
    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        # Create the agent's variable and set the initial values.
        self.wealth = 1

    def step(self):
        self.move()
      

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

class Wumpus(mesa.Agent):
 
    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

     




class Gold(mesa.Agent):
  
    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)



class Pit(mesa.Agent):
    """A hero Agent"""
    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

     


class Breeze(mesa.Agent):
    """A hero Agent"""
    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)


  
class Stench(mesa.Agent):
    """A hero Agent"""
    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)

        # Create the agent's variable and set the initial values.
        self.wealth = 1

class WompusWorld(mesa.Model):
    """A model with some number of agents."""
    def __init__(self, totalAgents=100, width=20, height=20):
        super().__init__()
        self.total_agents = totalAgents
        self.grid = mesa.space.MultiGrid(width, height, True)
        
        
        # create one hero agent
        agent = HeroAgent(self)
        self.grid.place_agent(agent,(0,0)) #place agent at 0,0
    
        # # Create agents
        # for i in range(self.total_agents):
        #     agent = HeroAgent(self)

        #     # Add the agent to a random grid cell
        #     x = self.random.randrange(self.grid.width)
        #     y = self.random.randrange(self.grid.height)
        #     self.grid.place_agent(agent, (x, y))

        self.running = True

       

    def step(self):
        """Advance the model by one step."""
        #self.datacollector.collect(self)
        self.agents.shuffle_do("step")


model_params = {
    "totalAgents": {
        "type": "SliderInt",
        "value": 3,
        "label": "Number of agents:",
        "min": 1,
        "max": 10,
        "step": 1,
    },
    "width": {
        "type": "SliderInt",
        "value": 20,
        "label": "Width:",
        "min": 10,
        "max": 100,
        "step": 10,
    },
    "height": {
        "type": "SliderInt",
        "value": 20,
        "label": "Height:",
        "min": 10,
        "max": 100,
        "step": 10,
    },
}

#modify this function to change output on grid
def agent_portrayal(agent):
    size = 10
    color = "tab:red"

    if agent.wealth > 3:
        size = 80
        color = "tab:blue"
    elif agent.wealth > 2:
        size = 50
        color = "tab:green"
    elif agent.wealth > 1:
        size = 20
        color = "tab:orange"
    return {"size": size, "color": color}

money_model = WompusWorld(3, 4, 4)

SpaceGraph = make_space_component(agent_portrayal)


page = SolaraViz(
    money_model,
    components=[SpaceGraph],
    model_params=model_params,
    name="Money Model"
)
# This is required to render the visualization in the Jupyter notebook
page