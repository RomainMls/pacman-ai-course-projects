import numpy as np
import math
from pacman_module.game import Agent, Directions, manhattanDistance

from typing import Tuple, List
from pacman_module.game import Grid
from pacman_module.game import Actions

#J'ai ajouté les types dans les arguments des fonctions pour mieux s'y retrouver
class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def transition_matrix(self, walls: Grid, position: Tuple[int, int]):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (k, l) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (k, l).
        """

        W = walls.width
        H = walls.height
        T = np.zeros((W, H, W,  H))
        # L'idée est simple, on va pour chaque position (i, j) regarder les déplacements possibles du fantome (k, l):
        # il a plus de chance de s'éloigner de pacman vu qu'il veut le fuir 
        # donc on utilise des poids pour donner de plus grandes proba si le déplacement en (k, l) à partir de (i, j) éloigne le fantome de pacman
        # on normalise pour obtenir une proba
        for i in range(0, W):
            for j in range(0, H):
                if(walls[i][j] == True):
                    continue

                neighbors = Actions.getLegalNeighbors((i, j), walls)
                if(len(neighbors) == 0):
                    continue
                distance1 = manhattanDistance((i, j), position)
                weights = []
                Z = 0
                
                for (k, l) in neighbors:
                    distance2 = manhattanDistance((k, l), position)
                    #Plus la peur est grande, plus on favorise l'éloignement de pacman
                    if self.ghost == "fearless":
                        w =  1
                    elif self.ghost == "afraid":
                        if distance1 < distance2:
                            w = 2
                        else:
                            w = 1
                    elif self.ghost == "terrified":
                        if distance1 < distance2:
                            w  = 6
                        else:   
                            w = 1
                    weights.append((k, l, w))
                    Z += w
                    
                for (k, l, w) in weights:
                    T[i][j][k][l] = w / Z
        return T

    def observation_matrix(self, walls: Grid, evidence: float, position: Tuple[int, int]):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """
        
        # Le principe ici est que plus l'evidence observée est proche de la distance réelle entre fantome (i, j) et Pacman, 
        # plus z est proche de np (sa moyenne)
        # Or, la proba P(z) d'une loi bin est max autour de la moyenne
        # donc plus z proche de np, plus P(z) (et donc P(evidence | distance)) est élevée
        # P(e_t | X_t) doit être max quand e_t = vrai distance(X_t) c'est bien ce qu'on obtient
        n = 4
        p = 0.5
        W = walls.width
        H = walls.height
        O = np.zeros((W, H))
        for i in range (0, W):
            for j in range (0, H):
                if(walls[i][j] == True):
                    continue
                distance = manhattanDistance((i, j), position)
                
                # On peut calculer z en l'isolant, par précaution entier car le combinatoire n'accepte que des entiers
                z = int(evidence - distance + n * p)
                if(z <= n and z >= 0):
                    O[i][j] = math.comb(n, z) * (p ** z) * ((1 - p) ** (n-z)) # Proba d'avoir P(Z=z)

        O /= np.sum(O) # Normalisation pour que la somme de la matrice égale 1
        return O
                

    def update(self, walls: Grid, belief: np.ndarray, evidence: float, position: Tuple[int, int]):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """

        T = self.transition_matrix(walls, position)
        O = self.observation_matrix(walls, evidence, position)
        
        W = walls.width
        H = walls.height

        # Initialisation uniforme si aucune croyance
        if(belief is None):
            nbCases = 0
            belief = np.zeros((W, H))
            for i in range(0, W):
                for j in range(0, H):
                    if(walls[i][j] == True):
                        belief[i][j] = 0
                    else:
                        nbCases += 1
            for i in range(0, W):
                for j in range(0, H):
                    if(walls[i][j] == False):
                        belief[i][j] = 1/nbCases
                        
        # 1.Prediction
        prediction = np.zeros((W, H))
        for k in range(0, W):
            for l in range(0, H):

                # On calcule la croyance prédite prediction(k, l) = 
                # somme sur tous les couples (i, j) [P(X_t=(k,l) | X_{t-1}=(i,j)) * b_{t-1}(i, j)]
                # (Slide prediction)
                totalProb = 0
                for i in range(0, W):
                    for j in range(0, H):
                        totalProb += T[i][j][k][l] * belief[i][j]             
                prediction[k][l] = totalProb
        
        # 2.Correction (slide bayes filter)
        b_t = np.zeros((W, H))
        for k in range(0, W):
            for l in range(0, H):
                b_t[k][l] = O[k][l] * prediction[k][l]
        
        b_t /= np.sum(b_t)
        return b_t
                

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls,
                    beliefs[i],
                    evidences[i],
                    position,
                )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()

    def _get_action(self, walls: Grid, beliefs: List[np.ndarray], eaten: List[bool], position: Tuple[int, int]):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        W = walls.width
        H = walls.height
        closestGhost = [None, math.inf, None] #Ghost ID, distance between it and pacman, index of ghost's position
        ghostIDCounter = 0
        for belief in beliefs:
            if(eaten[ghostIDCounter]):
                ghostIDCounter += 1
                continue
            maxProba = 0
            GhostPosition = (0, 0)
            for i in range(0, W):
                for j in range(0, H):
                    if(belief[i][j] > maxProba):
                        maxProba = belief[i][j]
                        GhostPosition = (i, j)
                        
            distance = manhattanDistance(GhostPosition, position)
            if(distance < closestGhost[1]):
                closestGhost[0] = ghostIDCounter
                closestGhost[1] = distance
                closestGhost[2] = GhostPosition
            ghostIDCounter += 1
        
        legalMoves = Actions.getLegalNeighbors(position, walls)

        distanceBefore = math.inf
        next_pos = ()
        if(not legalMoves):
            return Directions.STOP
        
        for legalMove in legalMoves:
            distance = manhattanDistance(closestGhost[2], legalMove)
            if(distance < distanceBefore):
                next_pos = legalMove
                distanceBefore = distance
            
        if(next_pos[0] > position[0]):
            return Directions.EAST
        if(next_pos[0] < position[0]):
            return Directions.WEST
        if(next_pos[1] < position[1]):
            return Directions.SOUTH
        if(next_pos[1] > position[1]):
            return Directions.NORTH
        else:
            return Directions.STOP            

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )
