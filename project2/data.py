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
    data.append(state.getPacmanPosition()[0])
    data.append(state.getPacmanPosition()[1])
    
    #On a qu'un seul fantome énoncé dit ("a smarty ghost")
    ghost = state.getGhostState(1)
    
    position = ghost.getPosition()
    data.append(position[0])
    data.append(position[1])
    
    direction = ghost.getDirection() #(0,1) -> North for ex
    directionDict = {
        "North": (0, 1),
        "South": (0,-1),
        "East": (1,0),
        "West": (-1, 0),
    }
    
    direction = directionDict[direction]
    data.append(direction[0])
    data.append(direction[1])
    
    # isScared = ghost.scaredTimer > 0
    data.append(ghost.scaredTimer > 0)
    
    gridFoods = state.getFood()
    for i in range(0, gridFoods.width):
        for j in range(0, gridFoods.height):
            data.append(gridFoods[i][j])
    
    capsulesList = state.getCapsules()        
    for i in range(0, gridFoods.width):
        for j in range(0, gridFoods.height):
            if (i, j) in capsulesList:
                data.append(1)
            else:
                data.append(0)
        
    gridWalls = state.getWalls()
    for i in range(0, gridWalls.width):
        for j in range(0, gridWalls.height):
            data.append(gridWalls[i][j])
            
    return torch.tensor(data)


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
            print(a)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.actions[idx]

PacmanDataset("datasets/pacman_dataset.pkl")