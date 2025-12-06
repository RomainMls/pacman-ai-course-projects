import numpy as np
import random
import torch

from pacman_module.pacman import runGame
from pacman_module.ghostAgents import SmartyGhost

from architecture import PacmanNetwork
from pacmanagent import PacmanAgent


SEED = 42
random.seed(SEED)
np.random.seed(SEED)

path_to_saved_model = "pacman_model.pth"

# Feel free to add code here depending on your implementation
D = 339
hiddens = [256, 256]
model = PacmanNetwork(D, hiddens)
model.load_state_dict(torch.load(path_to_saved_model, map_location="cpu"))
model.eval()

pacman_agent = PacmanAgent(model)

score, elapsed_time, nodes = runGame(
    layout_name="test_layout",
    pacman=pacman_agent,
    ghosts=[SmartyGhost(1)],
    beliefstateagent=None,
    displayGraphics=True,
    expout=0.0,
    hiddenGhosts=False,
)

print(f"Score: {score}")
print(f"Computation time: {elapsed_time}")
