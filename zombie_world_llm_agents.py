import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa.datacollection import DataCollector
from mesa.visualization import SolaraViz, make_plot_component, make_space_component
import random
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import datetime

def log_message(*args, **kwargs):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    message = ' '.join(map(str, args))
    formatted_message = f"{timestamp} - {message}"
    print(formatted_message, **kwargs)
    with open("zombie_log.txt", "a", encoding="utf-8", newline="\n") as f:
        f.write(formatted_message + "\n")
    
def compute_humans(model):
    count = 0
    for agent in model.agents:
        if not agent.isZombie and not agent.isDead:
            count += 1
    return count

def compute_zombies(model):
    count = 0
    for agent in model.agents:
        if agent.isZombie and not agent.isDead:
            count += 1
    return count

def compute_experienced_humans(model):
    """Count how many agents are experienced humans (not zombies, not dead, isExperienced=True)."""
    count = 0
    for agent in model.agents:
        if (agent.isExperienced) and (not agent.isZombie) and (not agent.isDead):
            count += 1
    return count

def compute_mutant_zombies(model):
    count = 0
    for agent in model.agents:
        if agent.isZombie and agent.isMutant and (not agent.isDead):
            count += 1
    return count

def compute_deaths(model):
    count = 0
    for agent in model.agents:
        if agent.isDead:
            count += 1
    return count


class OutbreakAgent(mesa.Agent):

    def __init__(self, model):
        super().__init__(model)
        self.shots_left = 15
        self.isZombie=False
        self.isDead = False
        self.isExperienced=False # humans learn so with experiance add ability to kill more zombies and have more defence, not 50% chance of infected
        self.isMutant=False # after a while, some zombies get a mutation that makes them more infectious
        self.previous_action = None 
        self.chatHistory = [] # this is the chat history of the agent
    

    def talkwith(self,otherAgentId,otherMsg):
        print(self.chatHistory)
        experienced = "an Experienced" if self.isExperienced else "New to the environment"
        """This is invoked by other agents in the model when they want to talk to you"""

        prompt = f"""You are {experienced} human in a zombie apocylypse, you have met another human with id ${otherAgentId} and they tell you ${otherMsg} respond:"""

        context_msg = f"Agent {self.unique_id} in {self.model.grid.width}x{self.model.grid.height} zombie world. Step {self.model.steps}."
        memory_msg = f"Chat history: {"".join(self.chatHistory)}" 

        ans = self.model.PromptModel(
            context_msg,
            memory_msg, 
            prompt
        )

        self.chatHistory.append("You said" + ans) # add answer to answer history

        return ans


    def step(self):
        current_step = self.model.steps 
        # check what agents are in current cell
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        current_perceptions = [] 
        if self.isDead:
            return
        if not self.isZombie:
            for cellmate in cellmates:
                if cellmate == self:  # Skip if the cellmate is the agent itself
                    continue
                probability_to_talk = 0.1
                if not cellmate.isZombie and not self.isDead:
                    # ask it a question 
                    if random.random() < probability_to_talk:
                        print(f"chat history from action item  {self.chatHistory} \n")
                        experienced = "an Experienced" if self.isExperienced else "New to the environment"
                        log_message(f"\n === STEP {current_step} === \n")
                        log_message(f"chat between {self.unique_id} and {cellmate.unique_id} at {self.pos} ")
                        prompt = f"""You are {experienced} human in a zombie apocylypse, you have met the other human named ${cellmate.unique_id} start a conversation with them based on your message history and previous conversations:"""

                        context_msg = f"Agent {self.unique_id} in {self.model.grid.width}x{self.model.grid.height} zombie world. Step {self.model.steps}."
                        memory_msg = f"Chat history: {"".join(self.chatHistory)}" 

                        question = self.model.PromptModel(
                            context_msg,
                            memory_msg, 
                            prompt
                        )
                        log_message("Question by: " + str(self.unique_id)+ " " + question)
                        ans = cellmate.talkwith(str(self.unique_id),question)
                        print(cellmate.unique_id)
                        log_message("Answer by: "+ str(cellmate.unique_id) + " " + ans)

                        # append the other agent response to chat history
                        self.chatHistory.append(f"Agent {cellmate.unique_id} response: {ans}")
                        print(self.chatHistory)
                        possible_steps = self.model.grid.get_neighborhood(
                        self.pos,
                        moore=True,
                        include_center=False)
                        new_position = self.random.choice(possible_steps)
                        self.model.grid.move_agent(self, new_position)
                        self.move_intelligently(current_perceptions)
            else:
                self.move_randomly()

    def move_randomly(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)

    
    def infect_cellmates(self):
        if self.isDead:
            return
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if self.isZombie: # if the agnet is zombie then they can make others zombie
            if len(cellmates)>1:
                for mate in cellmates:
                    if not mate.isZombie: # if the mate is not zombie, then depending on expereince add defence
                        if mate.isMutant:
                            infectSuccess = random.randint(0,1)
                            if mate.isExperienced:
                                infectSuccess = random.randint(0,3) # 25% chance of infection since they know how to protect themselves
                            if infectSuccess==1:
                                mate.isZombie = True
                        else: # if it's mutant then 75% chance of non experienced humans and 50% chance of experinced humn
                            infectSuccess = random.randint(0,3)
                            if mate.isExperienced:
                                if infectSuccess%2 ==0: # if it's even 50% chance
                                    mate.isZombie = True # spread it ut 50% chance
                                    mate.isMutant = True # mutants spread the mutated virus
                                    return # we are done 
                            else: # if the neightbour is not experienced human
                                if infectSuccess!=2: #75% chance of it not equaling 2 therefore for not expreienced we can infect 75%
                                    mate.isZombie = True
                                    mate.isMutant=True
                                    return


        else: # it is human then they can shoort down their mates and kill them
            if len(cellmates)>1:
                for mate in cellmates:
                    if self.shots_left>0:
                        ShotSuccess = random.randint(0,1)
                        if mate.isExperienced:
                            ShotSuccess=1 # if mate is expereined then kill with one shot
                        if ShotSuccess==1 and mate.isZombie:
                            mate.isDead=True
                            # pick up ammo from the dead zombie
                            self.shots_left+=mate.shots_left

                        self.shots_left-=1

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

    def move_intelligently(self, perceptions):

        prompt = f"""You are an agent exploring a {self.model.grid.width}x{self.model.grid.height} grid world (coordinates from (0,0) to ({self.model.grid.width-1},{self.model.grid.height-1})).
                    Goal: survive zombie world
                    Current State:
                    - Position: {self.pos}
                   
                    - Shots left: {str(self.shots_left)}

                    Possible Moves from {self.pos}:
               

                    Format your response EXACTLY like this:
                    Reasoning: [Your detailed analysis of perceptions, visited cells, potential risks/rewards of neighbors, and strategic choice]
                    Command: [ONLY one of: right | left | up | down | shoot right | shoot left | shoot up | shoot down]""" 
        

        self.move_randomly() # move randomly for now, will change it
        
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

        #log_message("Warning: Could not parse command using 'Command:' format. Falling back to first keyword.")
        words = response.lower().split()
        for word in words:
            if word in ["right", "left", "up", "down", "shoot"]:
                log_message(f"Parsed command via fallback: {word}")
                return word

        # stay if nothing is found, but put warning
        #log_message("Warning: No valid command found in response. Defaulting to 'stay'.")
        return "stay"

class OutBreakModel(mesa.Model):
    """A model with some number of agents."""
    def __init__(self, totalAgents=10, width=20, height=20):
        super().__init__()
        self.total_agents = totalAgents
        self.grid = mesa.space.MultiGrid(width, height, True)
        self.datacollector = mesa.DataCollector(
                 model_reporters={
                "Human Count": compute_humans,
                "Zombie Count": compute_zombies,
                "Experienced Human Count": compute_experienced_humans,
                "Mutant Zombie Count": compute_mutant_zombies,
                "Deaths": compute_deaths
                    }
            )

        # Create agents
        for i in range(self.total_agents):
            agent = OutbreakAgent(self)
            if i>0.9 * self.total_agents:
                agent.isZombie=True

            # Add the agent to a random grid cell
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.running = True
        self.datacollector.collect(self)
        
        if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    log_message(i, torch.cuda.get_device_properties(i))

        torch.random.manual_seed(0)

        model_path = "microsoft/Phi-4-mini-instruct"

        self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",  # Change this line
                torch_dtype="auto",
                trust_remote_code=True,
                # You might combine this with load_in_4bit=True or load_in_8bit=True
                load_in_4bit=True
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
            "max_new_tokens": 50,
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



        output = self.pipe(messages, **generation_args)
        return output[0]['generated_text']


    def step(self):
        """Advance the model by one step."""
        self.datacollector.collect(self)
        self.agents.shuffle_do("step")


model_params = {
    "totalAgents": {
        "type": "SliderInt",
        "value": 50,
        "label": "Number of agents:",
        "min": 10,
        "max": 100,
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
    if agent.isDead:
        return {"size": 0, "color": "gray"}
    portrayal = {
        "size": 50,
        "color": "tab:green"
    }
    if agent.isExperienced:
        portrayal["color"] = "tab:blue"
        portrayal["size"] = 80
    if agent.isZombie and not agent.isMutant:
        portrayal["color"] = "tab:red"
        portrayal["size"] = 80
    
    if agent.isZombie and agent.isMutant:
        portrayal["color"] = "tab:orange"
        portrayal["size"] = 80

    
    
    return portrayal

outBreak_model = OutBreakModel(10, 10, 10)

SpaceGraph = make_space_component(agent_portrayal)
HumansPlot = make_plot_component("Human Count")
ZombiePlot = make_plot_component("Zombie Count")
ExperiencedPlot = make_plot_component("Experienced Human Count")
MutantZombieCountPlot = make_plot_component("Mutant Zombie Count")
DeathCountPlot = make_plot_component("Deaths")


# Run simulaiton for csv
model = OutBreakModel(totalAgents=4, width=10, height=10)
num_steps = 500
for i in range(num_steps):
    model.step()


results_df = model.datacollector.get_model_vars_dataframe()

# Save to CSV
results_df.to_csv("simulation_results.csv", index=False)
print("Simulation results saved to simulation_results.csv")

page = SolaraViz(
    outBreak_model,
    components=[SpaceGraph, HumansPlot,ZombiePlot,ExperiencedPlot,MutantZombieCountPlot,DeathCountPlot],
    model_params=model_params,
    name="Money Model"
)
# This is required to render the visualization in the Jupyter notebook
page