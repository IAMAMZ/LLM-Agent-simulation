import mesa
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mesa.datacollection import DataCollector
from mesa.visualization import SolaraViz, make_plot_component, make_space_component
import random
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import datetime
import re


"""Helper functions """

def log_message(*args, **kwargs):
    timestamp = datetime.datetime.now().strftime("%m-%d %H:%M:%S.%f")[:-3]
    message = ' '.join(map(str, args))
    formatted_message = f"[{timestamp}] {message}"
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
        self.isZombie = False
        self.isDead = False
        self.isExperienced = False
        self.isMutant = False
        self.previous_action = None
        self.chatHistory = []

    def talkwith(self, otherAgentId, otherMsg):
        experienced = "an Experienced" if self.isExperienced else "New to the environment"
        prompt = f"""You are {experienced} human in a zombie apocylypse, you have met another human with id ${otherAgentId} and they tell you ${otherMsg} respond:"""

        context_msg = f"Agent {self.unique_id} in {self.model.grid.width}x{self.model.grid.height} zombie world. Step {self.model.steps}."
        memory_msg = f"Chat history: {"".join(self.chatHistory)}" 

        ans = self.model.PromptModel(
            context_msg,
            memory_msg, 
            prompt
        )

        self.chatHistory.append(f"You said {ans}")
        log_message(f"Agent {self.unique_id} chat history updated: {ans[:50]}...")
        return ans


    def step(self):
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if self.isDead:
            return
            
        if not self.isZombie:
            for cellmate in cellmates:
                if cellmate == self:
                    continue
                if not cellmate.isZombie and not self.isDead:
                        log_message(f"\n=== STEP {current_step} ===")
                        log_message(f"Agent {self.unique_id} initiating chat with {cellmate.unique_id} at {self.pos}")
                        experienced = "an Experienced" if self.isExperienced else "New to the environment"
                        prompt = f"""You are {experienced} human in a zombie apocylypse, you have met the other human named ${cellmate.unique_id} start a conversation with them based on your message history and previous conversations be very breif:"""
                        context_msg = f"Agent {self.unique_id} in {self.model.grid.width}x{self.model.grid.height} zombie world. Step {self.model.steps}."
                        memory_msg = f"Chat history: {"".join(self.chatHistory)}" 

                        question = self.model.PromptModel(
                            context_msg,
                            memory_msg, 
                            prompt
                        )
                        log_message(f"Agent {self.unique_id} question: {question[:100]}...")
                        ans = cellmate.talkwith(str(self.unique_id), question)
                        log_message(f"Agent {cellmate.unique_id} response: {ans[:100]}...")
                        self.chatHistory.append(f"Agent {cellmate.unique_id} response: {ans}")
                        self.handle_death_and_disease_cellmates()
            self.move_intelligently()
        else:
            self.handle_death_and_disease_cellmates()
            self.move_randomly()

    def move_randomly(self):
        possible_steps = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False)
        new_position = self.random.choice(possible_steps)
        self.model.grid.move_agent(self, new_position)
        
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
        log_message(f"Agent {self.unique_id} moving from {(x,y)} to {new_pos}")
        self.model.grid.move_agent(self, new_pos)

    def handle_death_and_disease_cellmates(self):
        if self.isDead:
            return
        cellmates = self.model.grid.get_cell_list_contents([self.pos])
        if self.isZombie:
            if len(cellmates)>1:
                for mate in cellmates:
                    if not mate.isZombie:
                        if mate.isMutant:
                            infectSuccess = random.randint(0,1)
                            if mate.isExperienced:
                                infectSuccess = random.randint(0,3)
                            if infectSuccess==1:
                                mate.isZombie = True
                                log_message(f"Agent {mate.unique_id} infected by mutant zombie {self.unique_id}")
                        else:
                            infectSuccess = random.randint(0,3)
                            if mate.isExperienced:
                                if infectSuccess%2 ==0:
                                    mate.isZombie = True
                                    mate.isMutant = True
                                    log_message(f"Agent {mate.unique_id} infected by {self.unique_id} (mutant conversion)")
                                    return
                            else:
                                if infectSuccess!=2:
                                    mate.isZombie = True
                                    mate.isMutant = True
                                    log_message(f"Agent {mate.unique_id} infected by {self.unique_id} (new mutant)")
                                    return
        else:
            if len(cellmates)>1:
                for mate in cellmates:
                    if self.shots_left>0:
                        ShotSuccess = random.randint(0,1)
                        if mate.isExperienced:
                            ShotSuccess=1
                        if ShotSuccess==1 and mate.isZombie:
                            mate.isDead=True
                            self.shots_left += mate.shots_left
                            log_message(f"Agent {self.unique_id} killed zombie {mate.unique_id} (shots left: {self.shots_left})")
                        self.shots_left -= 1

    def get_potential_moves(self):
        x, y = self.pos
        width = self.model.grid.width
        height = self.model.grid.height
        potential = {
            "right": ((x + 1) % width, y),
            "left": ((x - 1 + width) % width, y),
            "up": (x, (y + 1) % height),
            "down": (x, (y - 1 + height) % height),
        }
        return potential

    def move_intelligently(self):
        log_message(f"\n=== STEP {current_step} ===")
        potential_moves = self.get_potential_moves()
        prompt = f"""You are an agent exploring a {self.model.grid.width}x{self.model.grid.height} grid world (coordinates from (0,0) to ({self.model.grid.width-1},{self.model.grid.height-1})).
                    Goal: survive zombie world and collaborate with humans. Avoid filler words, and share your knowledge based on your chat history. act intelligently you are short on time don't have time to chat too much
                    Current State:
                    - Position: {self.pos}
                    - Shots left: {str(self.shots_left)}
                    Possible Moves from {self.pos}: {potential_moves}
                    base your logic on chat history: {self.chatHistory} and interacting with other agetns
                    Format your response EXACTLY like this:
                    Reasoning: [Your brief and logical analysis of perceptions, visited cells, potential risks/rewards of neighbors, and strategic choice]
                    Command: [ONLY one of: right | left | up | down | shoot right | shoot left | shoot up | shoot down]""" 

        context_msg = f"Agent {self.unique_id} in {self.model.grid.width}x{self.model.grid.height} Zombie World. Step {self.model.steps}."
        memory_msg = f"" 

        ans = self.model.PromptModel(
            context_msg,
            memory_msg, 
            prompt
        )
        log_message(f"Agent {self.unique_id} LLM response:\n{'-'*40}\n{ans}\n{'-'*40}")

        command = self.parse_command(ans)
        self.execute_command(command)
        
    def parse_command(self, response):
        response = response.strip()
        match = re.search(r"Command:\s*(right|left|up|down|shoot)\b", response, re.IGNORECASE)
        if match:
            command = match.group(1).lower()
            if command in ["right", "left", "up", "down", "shoot"]:
                return command

        words = response.lower().split()
        for word in words:
            if word in ["right", "left", "up", "down", "shoot"]:
                return word

        return "stay"


class OutBreakModel(mesa.Model):
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

        for i in range(self.total_agents):
            agent = OutbreakAgent(self)
            if i > (0.5 * self.total_agents):
                agent.isZombie = True
            x = self.random.randrange(self.grid.width)
            y = self.random.randrange(self.grid.height)
            self.grid.place_agent(agent, (x, y))

        self.running = True
        self.datacollector.collect(self)
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                log_message(f"CUDA device {i}: {torch.cuda.get_device_properties(i)}")

        torch.random.manual_seed(0)

        model_path = "microsoft/Phi-4-mini-instruct"
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype="auto",
            trust_remote_code=True,
            load_in_4bit=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )

    def PromptModel(self, context, memorystream, prompt):
        generation_args = {
            "max_new_tokens": 100,
            "return_full_text": False,
            "temperature": 0.2,
            "do_sample": False,
        }

        messages = [
            {"role": "system", "content": context},
            {"role": "system", "content": memorystream},
            {"role": "user", "content": prompt},
        ]

        output = self.pipe(messages, **generation_args)
        return output[0]['generated_text']

    def step(self):
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



outBreak_model = OutBreakModel(14, 4, 4)



SpaceGraph = make_space_component(agent_portrayal)
HumansPlot = make_plot_component("Human Count")
ZombiePlot = make_plot_component("Zombie Count")
ExperiencedPlot = make_plot_component("Experienced Human Count")
MutantZombieCountPlot = make_plot_component("Mutant Zombie Count")
DeathCountPlot = make_plot_component("Deaths")



page = SolaraViz(
    outBreak_model,
    components=[SpaceGraph, HumansPlot, ZombiePlot, ExperiencedPlot, MutantZombieCountPlot, DeathCountPlot],
    model_params=model_params,
    name="Zombie Outbreak Simulation"
)

page