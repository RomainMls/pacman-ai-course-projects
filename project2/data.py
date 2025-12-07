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
    MAX_DIST = (width ** 2 + height ** 2) ** 0.5
    
    pacmanPosition = state.getPacmanPosition()
    data.append(pacmanPosition[0] / width)
    data.append(pacmanPosition[1] / height)
    
    #On a qu'un seul fantome énoncé dit ("a smarty ghost")
    ghost = state.getGhostState(1)
    
    ghostPosition = ghost.getPosition()
    data.append(ghostPosition[0] / width)
    data.append(ghostPosition[1] / height)
    
    distanceX = ghostPosition[0] - pacmanPosition[0]
    distanceY = ghostPosition[1] - pacmanPosition[1]
    data.append(distanceX / width)
    data.append(distanceY / height)
    data.append((abs(distanceX) + abs(distanceY)) / (width + height))
    data.append((distanceX ** 2 + distanceY ** 2) ** 0.5 / ((width ** 2 + height ** 2) ** 0.5))
    ghostDistanceWithWalls = get_distance(gridWalls, pacmanPosition, ghostPosition)
    data.append((ghostDistanceWithWalls / MAX_DIST) if ghostDistanceWithWalls is not None else MAX_DIST / MAX_DIST)
    
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
    # Cul de sac
    data.append(is_dead_end(state, pacmanPosition))
    
    data.append(ghost.scaredTimer)
    
    gridFoods = state.getFood()
    foodList = gridFoods.asList()
    data.append(len(foodList) / (width * height))
    capsulesList = state.getCapsules()  
    
    distancesList = []
    index = -1
    if(len(foodList) > 0):
        for food in foodList:
            # food est un tuple qui donne la position de la food
            d = get_distance(gridWalls, pacmanPosition, food)
            if d is None:
                d = MAX_DIST
            distancesList.append(d)
            
        minDist = min(distancesList)
        index = distancesList.index(min(distancesList))
    else:
        minDist = MAX_DIST
    
    data.append(minDist / MAX_DIST)
    if(index != -1):
        data.append((foodList[index][0] - pacmanPosition[0]) / width)
        data.append((foodList[index][1] - pacmanPosition[1]) / height)
    else:
        data.append(0)
        data.append(0)

    distancesList = []
    index = -1
    if(len(capsulesList) > 0):
        for capsule in capsulesList:
            # capsule est un tuple qui donne la position de la food
            d = get_distance(gridWalls, pacmanPosition, capsule)
            if d is None:
                d = MAX_DIST
            distancesList.append(d)
            
        minDist = min(distancesList)
        index = distancesList.index(min(distancesList))
    else:
        minDist = MAX_DIST
    
    data.append(minDist / MAX_DIST)
    if(index != -1):
        data.append((capsulesList[index][0] - pacmanPosition[0]) / width)
        data.append((capsulesList[index][1] - pacmanPosition[1]) / height)
    else:
        data.append(0)
        data.append(0)
    
    for x in range(-7,8):
        for y in range(-7,8):
            newX = pacmanPosition[0] + x
            newY = pacmanPosition[1] - y
            if(newX >= 0 and newX < width and newY >= 0 and newY < height):
                data.append(1 if gridFoods[newX][newY] else 0)    
                data.append(1 if gridWalls[newX][newY] else 0)
                data.append(1 if ghostPosition[0] == newX and ghostPosition[1] == newY else 0)
                data.append(1 if (newX, newY) in capsulesList else 0)
            else:
                data.append(0)
                data.append(1)
                data.append(0)
                data.append(0)
    return torch.tensor(data, dtype=torch.float32)

def get_distance(walls, start, goal):
    """Returns shortest path distance between start and goal,
    None if unreachable using a bfs algorithm."""
    if start == goal:
        return 0
    fringe = Queue()
    fringe.push((start, 0))
    visited = set()
    while True:
        if fringe.isEmpty():
            return None
        position, dist = fringe.pop()
        for next_legal_pos in Actions.getLegalNeighbors(position, walls):
            if next_legal_pos == goal:
                return dist + 1
            if next_legal_pos not in visited:
                visited.add(next_legal_pos)
                fringe.push((next_legal_pos, dist + 1))

def is_dead_end(state, pos):
    walls = state.getWalls()
    nWalls = 0
    for directionX, directionY in [(0,1),(0,-1),(1,0),(-1,0)]:
        newX, newY = pos[0] + directionX, pos[1] + directionY
        if walls[newX][newY]:
            nWalls += 1
    return 1 if nWalls >= 3 else 0

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
