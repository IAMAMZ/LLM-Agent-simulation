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
        self.pathHistory = [] # this is to store the history of the hero agent


    def step(self):
        current_step = self.model.steps
        print(f"\n=== STEP {current_step} ===")
        print(f"Hero at position: {self.pos}")
        
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for cellmate in cellmates:
            if isinstance(cellmate, Gold):
                print("!!! FOUND GOLD - YOU WIN !!!")
                self.model.running = False
            elif isinstance(cellmate, Pit):
                print("!!! FELL IN PIT - GAME OVER !!!")
                self.model.running = False
            elif isinstance(cellmate, Wumpus) and not cellmate.isDead:
                print("!!! ENCOUNTERED LIVE WUMPUS - GAME OVER !!!")
                self.model.running = False
            elif isinstance(cellmate, Breeze):
                print("You feel a breeze...")
            elif isinstance(cellmate, Stench):
                print("You smell something terrible...")
        
        self.move()
      

    def move(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        perceptions = []
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        for cellmate in cellmates:
            if isinstance(cellmate, Breeze):
                perceptions.append("breeze")
            elif isinstance(cellmate, Stench):
                perceptions.append("stench")
        prompt = f"""You are in a {self.model.grid.width}x{self.model.grid.height} grid. 
                    Current position: {self.pos}
                    Perceptions: {', '.join(perceptions) if perceptions else 'none'}
                    Possible moves: right, left, up, down
                    you must make a move

                    Format your response EXACTLY like this:
                    Reasoning: [Your analysis here]
                    Command: [ONLY one of: right/left/up/down]"""
                    
        ans = self.model.PromptModel(
                f"Agent {self.unique_id} in {self.model.grid.width}x{self.model.grid.height} Wumpus World",
                "History: " + " ".join(self.pathHistory[-5:]),  # Last 5 entries
                prompt
            )
        print(f"AI suggests: {ans}")   
        # Parse command from response
        command = self.parse_command(ans)
            
         # Execute command
        self.execute_command(command)
            
         # Update history
        self.pathHistory.append(f"AI Decision: {ans} | Executed: {command}")

    def parse_command(self, response):
        response = response.lower()
        commands = ["right", "left", "up", "down", "shoot"]
        
        # Look for command patterns
        for cmd in commands:
            if f"command: {cmd}" in response:
                return cmd
        # in case there are duplicate commands return first command
        for word in response.split():
            if word in commands:
                return word
        return "stay"  # stay if no matching command

    def execute_command(self, command):
        x, y = self.pos
        width = self.model.grid.width
        height = self.model.grid.height
        
        move_map = {
            "right": ((x+1) % width, y),
            "left": ((x-1) % width, y),
            "up": (x, (y+1) % height),
            "down": (x, (y-1) % height),
            "stay": (x, y)
        }
        
        new_pos = move_map.get(command, (x, y))
        print(f"Moving from {(x,y)} → {new_pos}")
        self.model.grid.move_agent(self, new_pos)
        
    
class Wumpus(mesa.Agent):
 
    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)
        self.isDead = False
    
    def kill_wompus(self):
        self.isDead = True

     




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
    def __init__(self, gold_count=1,pit_count=2,wumpus_count=3, width=5, height=5):
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

            x=0
            y = 0
            while (x==0 and y==0):
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))
        
        for i in range(self.pit_count):
            agent = Pit(self)
            # Add the agent to a random grid cell
            x=0
            y = 0
            while (x==0 and y==0):
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
            x=0
            y = 0
            while (x==0 and y==0):
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
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(i, torch.cuda.get_device_properties(i))

        torch.random.manual_seed(0)

        model_path = "microsoft/Phi-4-mini-instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cuda:0", # "cpu" or "auto" or "cuda:0" for cuda device 0, 1, 2, 3 etc. if you have multiple GPUs
            torch_dtype="auto",
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        #gotta have a tokenizer for each model otherwise the token mappings won't match
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )


       
    def PromptModel(self, context, memorystream, prompt):
        
        #lower temperature generally more predictable results, you can experiment with this
        generation_args = {
            "max_new_tokens": 200,
            "return_full_text": False,
            "temperature": 0.2,
            "do_sample": False,
        }

        llmprompt = prompt
        
        messages = [
            {"role": "system", "content": context},
            {"role": "system", "content": memorystream},
            {"role": "user", "content": llmprompt},
        ]

        #time1 = int(round(time.time() * 1000))

        output = self.pipe(messages, **generation_args)
        return output[0]['generated_text']

        #time2 = int(round(time.time() * 1000))
        #print("Generation time: " + str(time2 - time1))
        #self.datacollector.collect(self)
    def step(self):
        """Advance the model by one step."""
        #self.datacollector.collect(self)
        self.agents.shuffle_do("step")
    def add_log(self, message):
        step = self.schedule.steps
        self.log.append(f"Step {step}: {message}")
        print(f"Step {step}: {message}") 


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

def parse_command(self, response):
    response = response.lower()
    commands = ["right", "left", "up", "down", "shoot"]
    
    # Look for command patterns
    for cmd in commands:
        if f"command: {cmd}" in response:
            return cmd
    # Fallback: find first matching word
    for word in response.split():
        if word in commands:
            return word
    return "stay"  # Default if no valid command
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

page