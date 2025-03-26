from mesa.experimental.cell_space import CellAgent, FixedAgent
import math
from mesa import Model
from mesa.datacollection import DataCollector
from mesa.experimental.cell_space import OrthogonalVonNeumannGrid
from mesa.experimental.devs import ABMSimulator
from mesa.visualization import (
    Slider,
    SolaraViz,
    make_plot_component,
    make_space_component,
)


"""Helper method to find specified cell within a grid"""
def find_cell(collection,x,y):
    foundCell = None
    for element in collection:
        if element.coordinate[0] == x and element.coordinate[1] == y:
            foundCell = element
            break
            
    return foundCell


class Wumpus(FixedAgent):

    @property
    def is_dead(self):
        """Whether the grass patch is fully grown."""
        return self._is_dead
    def __init__(
      self,model      
    ):
        super().__init__(model)
        _is_dead = False




class Pit(FixedAgent):
      
    def __init__(
      self,model      
    ):
        super().__init__(model)

class Gold(FixedAgent):
    def __init__(
      self,model      
    ):
        super().__init__(model)

class Hero(CellAgent):
    def __init__(
      self,model,cell=None     
    ):
        super().__init__(model)
        self.cell=cell
    
    def step(self):
        self.move()
    
    def move(self):
        neighbors = self.cell.neighborhood
        
        self.cell = neighbors.select_random_cell()






class WumpusWorld(Model):
    """
    A model of Wompus world
    """

    description = (
        "A model for simulating the wumpus world from AI textbook"
    )

    def __init__(
        self,
        width=20,
        height=20,
        initial_gold=1,
        initial_pit=1,
        initial_wumpus=1,
        seed=None,
        simulator: ABMSimulator = None,
    ):
        """Create a new wompus world model with the given parameters.

        Args:
            height: Height of the grid
            width: Width of the grid
         
            simulator: ABMSimulator instance for event scheduling
        """
        super().__init__(seed=seed)
        self.simulator = simulator
        self.simulator.setup(self)

        # Initialize model parameters
        self.height = height
        self.width = width
     

        # Create grid using experimental cell space
        self.grid = OrthogonalVonNeumannGrid(
            [self.height, self.width],
            torus=True,
            capacity=math.inf,
            random=self.random,
        )

    

        # Create Hero:
        Hero.create_agents(
            self,
            1,
            cell=find_cell(self.grid.all_cells.cells, 0,0),
        )
   
  


        # Collect initial data
        self.running = True
        self.datacollector.collect(self)

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

    def step(self):
        """Execute one step of the model."""
        # First activate all sheep, then all wolves, both in random order
        self.agents_by_type[Hero].shuffle_do("step")


        # Collect data
        self.datacollector.collect(self)


def portrayal(agent):
    if agent is None:
        return

    portrayal = {
        "size": 25,
    }

    if isinstance(agent, Hero):
        portrayal["color"] = "tab:red"
        portrayal["marker"] = "o"
        portrayal["zorder"] = 2


    return portrayal


model_params = {
    "seed": {
        "type": "InputText",
        "value": 42,
        "label": "Random Seed",
    },
    "grass": {
        "type": "Select",
        "value": True,
        "values": [True, False],
        "label": "grass regrowth enabled?",
    },
    "grass_regrowth_time": Slider("Grass Regrowth Time", 20, 1, 50),
    "initial_sheep": Slider("Initial Sheep Population", 100, 10, 300),
    "sheep_reproduce": Slider("Sheep Reproduction Rate", 0.04, 0.01, 1.0, 0.01),
    "initial_wolves": Slider("Initial Wolf Population", 10, 5, 100),
    "wolf_reproduce": Slider(
        "Wolf Reproduction Rate",
        0.05,
        0.01,
        1.0,
        0.01,
    ),
    "wolf_gain_from_food": Slider("Wolf Gain From Food Rate", 20, 1, 50),
    "sheep_gain_from_food": Slider("Sheep Gain From Food", 4, 1, 10),
}


def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def post_process_lines(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))


space_component = make_space_component(
    portrayal, draw_grid=False, post_process=post_process_space
)
lineplot_component = make_plot_component(
    {"Wolves": "tab:orange", "Sheep": "tab:cyan", "Grass": "tab:green"},
    post_process=post_process_lines,
)

simulator = ABMSimulator()
model = WumpusWorld(simulator=simulator)

page = SolaraViz(
    model,
    components=[space_component, lineplot_component],
    model_params=model_params,
    name="Wolf Sheep",
    simulator=simulator,
)
page  
