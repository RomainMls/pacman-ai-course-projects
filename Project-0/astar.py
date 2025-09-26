from pacman_module.game import Agent, Directions
from pacman_module.util import PriorityQueue
from math import sqrt


def key(state):
    """Returns a key that uniquely identifies a Pacman game state.

    Arguments:
        state: a game state. See API or class `pacman.GameState`.

    Returns:
        A hashable key tuple.
    """

    return (
        state.getPacmanPosition(),
        state.getFood(),
        tuple(state.getCapsules())
    )

def g(n):
    return -n.getScore()

def h(n):
    pacman_pos = n.getPacmanPosition()
    matrix = n.getFood()
    counter = 0
    for i in range(matrix.width):
        for j in range(matrix.height):
            if matrix[i][j]:
                # under-estimate the cost of going there (the higher, the better)
                cost = 10   # reaching is +10
                cost += ((pacman_pos[0] - i) ** 2 + (pacman_pos[1] - j) ** 2) # each step is +1
                if n.getNumFood() == 1:
                    cost += 500     # a winning end is +500

                counter += cost

    return counter

class PacmanAgent(Agent):
    """Pacman agent based on depth-first search (DFS)."""

    def __init__(self):
        super().__init__()

        self.moves = None

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Return:
            A legal move as defined in `game.Directions`.
        """

        if self.moves is None:
            self.moves = self.astar(state)

        if self.moves:
            return self.moves.pop(0)
        else:
            return Directions.STOP

    def astar(self, state):
        """Given a Pacman game state, returns a list of legal moves to solve
        the search layout.

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A list of legal moves.
        """

        path = []
        fringe = PriorityQueue()
        fringe.push((state, path), 0)
        closed = set()

        while True:
            if fringe.isEmpty():
                return []

            priority, (current, path) = fringe.pop()

            if current.isWin():
                return path

            current_key = key(current)

            if current_key in closed:
                continue
            else:
                closed.add(current_key)

            for successor, action in current.generatePacmanSuccessors():
                fringe.push((successor, path + [action]), g(successor)+h(successor))

        return path
