import math
import random

import numpy as np

from pacman_module.game import Actions, Agent, Directions, manhattanDistance
from pacman_module.util import Queue


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def transition_matrix(self, walls, position):
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

        # L'idée est simple, on va pour chaque position (i, j) regarder les déplacements possibles du fantome (k, l):
        # il a plus de chance de s'éloigner de pacman vu qu'il veut le fuir 
        # donc on utilise des poids pour donner de plus grandes proba si le déplacement en (k, l) à partir de (i, j) éloigne le fantome de pacman
        # on normalise pour obtenir une proba
        
        w = walls.width
        h = walls.height
        t = np.zeros((w, h, w,  h))
        
        for i in range(w):
            for j in range(h):
                if walls[i][j]:
                    continue

                neighbors = Actions.getLegalNeighbors((i, j), walls)
                if len(neighbors) == 0:
                    continue
                distance1 = manhattanDistance((i, j), position)
                weights = []
                Z = 0
                
                for k, l in neighbors:
                    distance2 = manhattanDistance((k, l), position)
                    # Plus la peur est grande, plus on favorise l'éloignement de pacman
                    if self.ghost == "fearless":
                        w = 1
                    elif self.ghost == "afraid":
                        if distance1 < distance2:
                            w = 2
                        else:
                            w = 1
                    elif self.ghost == "terrified":
                        if distance1 < distance2:
                            w = 6
                        else:   
                            w = 1
                    weights.append((k, l, w))
                    Z += w
                    
                for k, l, w in weights:
                    t[i][j][k][l] = w / Z
        return t

    def observation_matrix(self, walls, evidence, position):
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
        
        # Le principe ici est que plus l'evidence observée est proche de la
        # distance réelle entre fantome (i, j) et Pacman, 
        # plus z est proche de np (sa moyenne)
        # Or, la proba P(z) d'une loi bin est max autour de la moyenne
        # Donc plus z proche de np, 
        #      plus P(z) (et donc P(evidence | distance)) est élevée
        # P(e_t | X_t) doit être max quand e_t = vrai distance(X_t)
        # C'est bien ce qu'on obtient
        
        n = 4
        p = 0.5
        w = walls.width
        h = walls.height
        o = np.zeros((w, h))
        
        for i in range(w):
            for j in range(h):
                if walls[i][j]:
                    continue
                
                distance = manhattanDistance((i, j), position)
                
                # On peut calculer z en l'isolant, 
                # par précaution entier car le combinatoire n'accepte que des entiers
                z = int(evidence - distance + n * p)

                if 0 <= z <= n:
                    # Proba d'avoir P(Z=z)
                    o[i][j] = math.comb(n, z) * (p ** z) * ((1 - p) ** (n - z))

        # Normalisation pour que la somme de la matrice égale 1
        o /= np.sum(o)
        return o
                

    def update(self, walls, belief, evidence, position):
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

        t = self.transition_matrix(walls, position)
        o = self.observation_matrix(walls, evidence, position)
        
        w = walls.width
        h = walls.height

        # Initialisation uniforme si aucune croyance
        if belief is None:
            nb_cases = 0
            belief = np.zeros((w, h))
            for i in range(w):
                for j in range(h):
                    if walls[i][j]:
                        belief[i][j] = 0
                    else:
                        nb_cases += 1
            for i in range(w):
                for j in range(h):
                    if not walls[i][j]:
                        belief[i][j] = 1 / nb_cases
                        
        # 1.Prediction
        prediction = np.zeros((w, h))
        for k in range(w):
            for l in range(h):

                # On calcule la croyance prédite prediction(k, l) = 
                # somme sur tous les couples (i, j) [P(X_t=(k,l) | X_{t-1}=(i,j)) * b_{t-1}(i, j)]
                # (Slide prediction)
                total_prob = 0
                for i in range(w):
                    for j in range(h):
                        total_prob += t[i][j][k][l] * belief[i][j]             
                prediction[k][l] = total_prob
        
        # 2.Correction (slide bayes filter)
        b_t = np.zeros((w, h))
        for k in range(w):
            for l in range(h):
                b_t[k][l] = o[k][l] * prediction[k][l]
        
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


    def get_distance(self, walls, start, goal):
        """Returns shortest path distance between start and goal, or None if unreachable using a bfs algorithm."""
        if start == goal:
            return 0

        fringe = Queue()
        fringe.push((start, 0))
        visited = set()

        while True:
            if fringe.isEmpty():
                return None

            (x, y), dist = fringe.pop()
            for nx, ny in Actions.getLegalNeighbors((x, y), walls):
                if (nx, ny) == goal:
                    return dist + 1
                if (nx, ny) not in visited:
                    visited.add((nx, ny))
                    fringe.push(((nx, ny), dist + 1))



    def _get_action(self, walls, beliefs, eaten, position):
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
        closest_ghost = [None, math.inf, None] # Ghost ID, distance, position
        ghost_id_counter = 0
        
        for belief in beliefs:
            if eaten[ghost_id_counter]:
                ghost_id_counter += 1
                continue
            max_proba = 0
            ghost_position = (0, 0)
            
            for i in range(W):
                for j in range(H):
                    if belief[i][j] > max_proba:
                        max_proba = belief[i][j]
                        ghost_position = (i, j)
                        
            distance = self.get_distance(walls, ghost_position, position)
            if distance < closest_ghost[1]:
                closest_ghost[0] = ghost_id_counter
                closest_ghost[1] = distance
                closest_ghost[2] = ghost_position
            ghost_id_counter += 1
        
        legal_moves = Actions.getLegalNeighbors(position, walls)

        if not legal_moves:
            return Directions.STOP

        best_moves = []
        min_dist = float('inf')
        target = closest_ghost[2]

        for move in legal_moves:
            dist = self.get_distance(walls, move, target)
            if dist is not None:
                if dist < min_dist:
                    min_dist = dist
                    best_moves = [move]
                elif dist == min_dist:
                    best_moves.append(move)

        if not best_moves:
            return random.choice(legal_moves)

        if random.random() < 0.05:
            return random.choice(legal_moves)
        else:
            next_pos = random.choice(best_moves)

        x, y = position
        nx, ny = next_pos

        if nx > x:
            return Directions.EAST
        if nx < x:
            return Directions.WEST
        if ny > y:
            return Directions.NORTH
        if ny < y:
            return Directions.SOUTH
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
