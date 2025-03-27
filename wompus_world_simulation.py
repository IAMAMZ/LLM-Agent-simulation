import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline


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
    def __init__(self, gold_count=1,pit_count=2,wumpus_count=3, width=20, height=20):
        super().__init__()
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.gold_count = gold_count
        self.wumpus_count = wumpus_count
        self.pit_count = pit_count
        
        # create one hero agent
        agent = HeroAgent(self)
        self.grid.place_agent(agent,(0,0)) #place agent at 0,0
    
        # randomly put gold a
        for i in range(self.gold_count):
            agent = Gold(self)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
        
        for i in range(self.pit_count):
            agent = Pit(self)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
            #place breeze next to the pits
            neighbours = self.grid.get_neighborhood(
                                (x,y),
                                moore=False,
                                include_center=False)
            for pos in neighbours:
                breeze = Breeze(self)
                self.grid.place_agent(breeze,pos)
            

        for i in range(self.wumpus_count):
            agent = Wumpus(self)

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
            #place stench next to wumpus
            neighbours = self.grid.get_neighborhood(
                                (x,y),
                                moore=False,
                                include_center=False)
            for pos in neighbours:
                breeze = Stench(self)
                self.grid.place_agent(breeze,pos)
    
        self.running = True

       

    def step(self):
        """Advance the model by one step."""
        #self.datacollector.collect(self)
        self.agents.shuffle_do("step")


model_params = {
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
     "gold_count": {
        "type": "SliderInt",
        "value": 20,
        "label": "Count of Gold:",
        "min": 10,
        "max": 100,
        "step": 10,
    },
     "wumpus_count": {
        "type": "SliderInt",
        "value": 20,
        "label": "Count of Wumpus:",
        "min": 10,
        "max": 100,
        "step": 10,
    },
     "pit_count": {
        "type": "SliderInt",
        "value": 20,
        "label": "Count of Pit",
        "min": 10,
        "max": 100,
        "step": 10,
    },
    
}


def agent_portrayal(agent):
    portrayal = {"size": 40, "color": "gray", "shape": "circle"}
    
    if isinstance(agent, HeroAgent):
        portrayal.update({"color": "blue", "size": 60, "shape": "rect"})
    elif isinstance(agent, Gold):
        portrayal.update({"color": "gold", "shape": "star"})
    elif isinstance(agent, Wumpus):
        portrayal.update({"color": "red", "shape": "triangle"})
    elif isinstance(agent, Pit):
        portrayal.update({"color": "black", "shape": "hexagon"})
    elif isinstance(agent, Breeze):
        portrayal.update({"color": "lightblue", "size": 20, "shape": "circle"})
    elif isinstance(agent, Stench):
        portrayal.update({"color": "darkgreen", "size": 20, "shape": "triangle"})
        
    return portrayal

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