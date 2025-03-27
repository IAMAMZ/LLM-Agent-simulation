import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import re 
from mesa.datacollection import DataCollector
from mesa.visualization import SolaraViz, make_plot_component, make_space_component
import sys
import datetime

def log_message(*args, **kwargs):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = ' '.join(map(str, args))
    formatted_message = f"{timestamp} - {message}"
    print(formatted_message, **kwargs)
    with open("wumpus_log.txt", "a", encoding="utf-8", newline="\n") as f:
        f.write(formatted_message + "\n")

class HeroAgent(mesa.Agent):
    """A hero Agent"""
    def __init__(self, model):
        super().__init__(model)
        
        self.position_history = []
        self.visited_cells = set([(0,0)]) 
        self.previous_action = None 
        self.arrows = 5 # agent has 5 arrows

  
    def step(self):
        current_step = self.model.steps 
        log_message(f"\n=== STEP {current_step} ===")
        log_message(f"Hero at position: {self.pos}")
        self.visited_cells.add(self.pos) 

        # check what agents are in current cell,
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        current_perceptions = [] # store what you precieve in text for llm
        game_over = False
        for cellmate in cellmates:
            if isinstance(cellmate, Gold):
                log_message("!!! FOUND GOLD - YOU WIN !!!")
                self.model.running = False
                game_over = True
                break
            elif isinstance(cellmate, Pit):
                log_message("!!! FELL IN PIT - GAME OVER !!!")
                self.model.running = False
                game_over = True
                break
            elif isinstance(cellmate, Wumpus) and not cellmate.isDead:
                log_message("!!! ENCOUNTERED LIVE WUMPUS - GAME OVER !!!")
                self.model.running = False
                game_over = True
                break
            elif isinstance(cellmate, Breeze):
                log_message("You feel a breeze...")
                if "breeze" not in current_perceptions: current_perceptions.append("breeze")
            elif isinstance(cellmate, Stench):
                log_message("You smell something terrible...")
                if "stench" not in current_perceptions: current_perceptions.append("stench")

        if game_over:
            return 

        self.decide_move(current_perceptions)


    def decide_move(self, perceptions):
        
        self.position_history.append(str(self.pos) +" "  ''.join(perceptions))       
        potential_moves = self.get_potential_moves()
        neighbor_info = []
        for move_dir, next_pos in potential_moves.items():
            status = "unvisited"
            if next_pos in self.visited_cells:
                status = "visited"
            neighbor_info.append(f"{move_dir} to {next_pos} ({status})")

        prompt = f"""You are an agent exploring a {self.model.grid.width}x{self.model.grid.height} grid world (coordinates from (0,0) to ({self.model.grid.width-1},{self.model.grid.height-1})).
                    Goal: Find the Gold. Avoid Pits and the Wumpus.
                    Rules:
                    - A Breeze indicates a Pit in an adjacent square (up, down, left, or right).
                    - A Stench indicates a Wumpus in an adjacent square (up, down, left, or right).
                    - Moving into a Pit or Wumpus square ends the game.
                    - when you encounter a breeze move towards a safer place 
                    - when you encounter a smell move towareds a safe place or shoot arrows

                    Current State:
                    - Position: {self.pos}
                    - Perceptions at current position: {', '.join(perceptions) if perceptions else 'none'}
                    - Cells already visited: {sorted(list(self.visited_cells))}
                    - Recent positions (last 5): {self.position_history[-5:]}
                    - Arrows left: {str(self.arrows)}

                    Possible Moves from {self.pos}:
                    - {', '.join(neighbor_info)}
                    - 'shoot <direction>' is also a possible command if you suspect a Wumpus.

                    Task: Analyze the situation and choose the next action. Prioritize exploring safe, unvisited squares. Avoid moving back to the immediately previous square ({self.position_history[-2] if len(self.position_history)>1 else 'N/A'}) unless there's a strong reason (e.g., all other options seem dangerous or are visited). If you suspect a Wumpus, consider shooting. Explain your reasoning clearly.

                    Format your response EXACTLY like this:
                    Reasoning: [Your detailed analysis of perceptions, visited cells, potential risks/rewards of neighbors, and strategic choice]
                    Command: [ONLY one of: right | left | up | down | shoot right | shoot left | shoot up | shoot down]""" 
      
        # call the llm
        context_msg = f"Agent {self.unique_id} in {self.model.grid.width}x{self.model.grid.height} Wumpus World. Step {self.model.steps}."
        memory_msg = f"Visited cells: {sorted(list(self.visited_cells))}" 

        ans = self.model.PromptModel(
            context_msg,
            memory_msg, 
            prompt
        )
        log_message(f"LLM Raw Response:\n{ans}")

    
        command = self.parse_command(ans)
      

        self.execute_command(command)
  

    def get_potential_moves(self):
        """ Calculates the coordinates for possible moves (right, left, up, down). """
        x, y = self.pos
        width = self.model.grid.width
        height = self.model.grid.height
        # Uses modulo for grid wrapping
        potential = {
            "right": ((x + 1) % width, y),
            "left": ((x - 1 + width) % width, y), # Ensure positive index before modulo
            "up": (x, (y + 1) % height),
            "down": (x, (y - 1 + height) % height), # Ensure positive index before modulo
        }
        return potential

    def parse_command(self, response):
        """
        Parses the command from the LLM response using regex first,
        then falls back to finding the first command word.
        """
        response = response.strip() 

        # user regex to find command
        match = re.search(r"Command:\s*(right|left|up|down|shoot)\b", response, re.IGNORECASE)
        if match:
            command = match.group(1).lower() 
            # check if command is valid
            if command in ["right", "left", "up", "down", "shoot"]:
                log_message(f"Parsed command via regex: {command}")
                return command

        # fallback: find the first occurrence of a command word

        log_message("Warning: Could not parse command using 'Command:' format. Falling back to first keyword.")
        words = response.lower().split()
        for word in words:
            if word in ["right", "left", "up", "down", "shoot"]:
                log_message(f"Parsed command via fallback: {word}")
                return word

        # stay if nothing is found, but put warning
        log_message("Warning: No valid command found in response. Defaulting to 'stay'.")
        return "stay"

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
        
        if command not in move_map:
            self.shootWompus(command)
        else:
            new_pos = move_map.get(command, (x, y))
            self.visited_cells.add(new_pos)
            log_message(f"Moving from {(x,y)} to {new_pos}")
            self.model.grid.move_agent(self, new_pos)
        
    def shootWompus(self,direction):
        # get all neighbours
        possible_neighboorCoordinate_dic = self.get_potential_moves()
        cellToKill = None
        if direction=="shoot right":
            cellToKill="right"
        elif direction == "shoot left":
            cellToKill = "left"
        elif direction == "shoot up":
            cellToKill = "up"
        elif direction == "shoot down":
            cellToKill = "down"
        
        # get coordinates of cell to kill
        if cellToKill==None:
            return
        kill_coordinates = possible_neighboorCoordinate_dic[cellToKill]
        target_agents = self.model.grid.get_cell_list_contents([kill_coordinates])
        for a in target_agents:
            if isinstance(a,Wumpus):
                a.kill_wompus()
                log_message(f"wompus at {a.pos} eliminated") 
                self.arrows-=1


class Wumpus(mesa.Agent): 
    def __init__(self, model):
        # Pass the parameters to the parent class.
        super().__init__(model)
        self.isDead = False
    
    def kill_wompus(self):
        self.isDead = True

class Gold(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)

class Pit(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)

class Breeze(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)

class Stench(mesa.Agent):
    def __init__(self, model):
        super().__init__(model)

class WumpusWorld(mesa.Model):
    """A model with some number of agents."""
    def __init__(self, gold_count=1,pit_count=1,wumpus_count=1, width=8, height=8):
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
                log_message(i, torch.cuda.get_device_properties(i))

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
            "max_new_tokens": 500,
            "return_full_text": False,
            "temperature": 0.0,
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
        #log_message("Generation time: " + str(time2 - time1))
        #self.datacollector.collect(self)
    def step(self):
        """Advance the model by one step."""
        #self.datacollector.collect(self)
        self.agents.shuffle_do("step")
    def add_log(self, message):
        step = self.steps
        self.log.append(f"Step {step}: {message}")
        log_message(f"Step {step}: {message}") 


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
        portrayal["color"] = "cyan"
        portrayal["marker"] = "*"
        portrayal["zorder"] = 3
        portrayal["size"] = 40
    elif isinstance(agent, Gold):
        portrayal["color"] = "gold"
        portrayal["marker"] = "P"
        portrayal["zorder"] = 2
        portrayal["size"] = 10
    elif isinstance(agent, Wumpus):
        portrayal["color"] = "red"
        portrayal["marker"] = "s"
        portrayal["zorder"] = 2
        portrayal["size"] = 30
    elif isinstance(agent, Pit):
        portrayal.update({"color": "black", "shape": "hexagon"})
    elif isinstance(agent, Breeze):
        portrayal.update({"color": "lightblue", "size": 30, "shape": "circle"})
    elif isinstance(agent, Stench):
        portrayal.update({"color": "darkgreen", "size": 30, "shape": "triangle"})
        
    return portrayal

wumpus_model = WumpusWorld(3, 4, 4)

SpaceGraph = make_space_component(agent_portrayal)


page = SolaraViz(
    wumpus_model,
    components=[SpaceGraph],
    model_params=model_params,
    name="Money Model"
)

page