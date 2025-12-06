import pickle

import torch
from torch.utils.data import Dataset
from  pacman_module.pacman import *
from pacman_module.game import *

# containing the handling of the dataset

def state_to_tensor(state: GameState):
    """
    Build the input of your network.
    We encourage you to do some clever feature engineering here!

    Returns:
        A tensor of features representing the state

    Arguments:
        state: a GameState object
    """
    
    # On construit le vecteur de features comme ceci : 
    # [positionPacmanX, positionPacmanY, 
    # ghost1PosX, ghost1PosY, ghost1DirX, ghost1DirY, ghost1IsScared,
    # gridFoods matrice de booléen en vecteur,
    # gridCapsules matrice de booléen en vecteur,
    # grodWalls matrice de booléen en vecteur
    # ]
    data = []
    gridWalls = state.getWalls()
    width = gridWalls.width
    height = gridWalls.height
    
    pacmanPosition = state.getPacmanPosition()
    data.append(pacmanPosition[0])
    data.append(pacmanPosition[1])
    
    #On a qu'un seul fantome énoncé dit ("a smarty ghost")
    ghost = state.getGhostState(1)
    
    ghostPosition = ghost.getPosition()
    data.append(ghostPosition[0])
    data.append(ghostPosition[1])
    
    legal = state.getLegalPacmanActions()
    data.append(1 if "North" in legal else 0)
    data.append(1 if "South" in legal else 0)
    data.append(1 if "East" in legal else 0)
    data.append(1 if "West" in legal else 0)
    
    distanceX = ghostPosition[0] - pacmanPosition[0]
    distanceY = ghostPosition[1] - pacmanPosition[1]
    data.append(distanceX)
    data.append(distanceY)
    data.append(abs(distanceX) + abs(distanceY))
    data.append((distanceX ** 2 + distanceY ** 2) ** 0.5)
    
    direction = ghost.getDirection() #(0,1) -> North for ex
    directionDict = {
        "North": (0, 1),
        "South": (0,-1),
        "East": (1,0),
        "West": (-1, 0),
        "Stop": (0, 0),
    }
    
    direction = directionDict[direction]
    data.append(direction[0])
    data.append(direction[1])
    
    data.append(ghost.scaredTimer)
    gridFoods = state.getFood()
    
    capsulesList = state.getCapsules()        

    for x in range(-4,5):
        for y in range(-4,5):
            newX = pacmanPosition[0] + x
            newY = pacmanPosition[1] - y
            if(newX >= 0 and newX < width and newY >= 0 and newY < height):
                data.append(1 if gridFoods[newX][newY] else 0)    
                data.append(1 if gridWalls[newX][newY] else 0)
                data.append(1 if ghostPosition[0] == newX and ghostPosition[1] == newY else 0)
                data.append(1 if (newX, newY) in capsulesList else 0)
            else:
                data.append(0)
                data.append(0)
                data.append(0)
                data.append(0)
    return torch.tensor(data, dtype=torch.float32)


class PacmanDataset(Dataset):
    def __init__(self, path):
        """
        Load and transform the pickled dataset into a format suitable
        for training your architecture.

        Arguments:
            path: The file path to the pickled dataset.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)

        self.inputs = []
        self.actions = []

        for s, a in data:
            x = state_to_tensor(s)
            self.inputs.append(x)
            self.actions.append(a)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.actions[idx]
