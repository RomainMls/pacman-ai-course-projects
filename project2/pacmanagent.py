from pacman_module.game import Agent

from data import state_to_tensor
from train import PacmanDataset
import math
import torch

# containing the implementation of an agent whose decisions are based on the predictions of your model

class PacmanAgent(Agent):
    def __init__(self, model):
        """
        Initialize the neural network Pacman agent.

        Arguments:
            model: The trained neural network model.
        """
        super().__init__()

        self.model = model.eval()

    def get_action(self, state):
        """
        Return the action chosen by the neural network given the
        current state.

        Arguments:
            state: a GameState object
        """
        
        actionList = ["North", "South", "East", "West", "Stop"]
                
        x = state_to_tensor(state).unsqueeze(0)
        with torch.no_grad():
            pred = self.model(x)
            
        logits = pred[0]
        max = [-math.inf, 0]
        i = 0
        for logit in logits:
            if logit.item() > max[0]:
                max[0] = logit
                max[1] = i
            i += 1
        
        actionId = max[1]
        
        return actionList[actionId]
