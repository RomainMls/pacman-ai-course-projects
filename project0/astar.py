from pacman_module.game import Agent, Directions
from pacman_module.util import PriorityQueue


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


def g(prev_cost, prev, next):
    cost = prev_cost
    cost += 1 # 1 point per step
    cost += (len(prev.getCapsules()) - len(next.getCapsules())) * 5 # 5 points per capsule eaten
    return cost

def h(n):
    if n.getNumFood() == 0:
        return 0

    pacman_pos = n.getPacmanPosition()
    matrix = n.getFood()

    mostleft = pacman_pos[0]
    mostright = pacman_pos[0]
    mostup = pacman_pos[1]
    mostdown = pacman_pos[1]
    for i in range(matrix.width):
        for j in range(matrix.height):
            if matrix[i][j]:
                if i < mostleft:
                    mostleft = i
                if i > mostright:
                    mostright = i
                if j < mostup:
                    mostup = j
                if j > mostdown:
                    mostdown = j

    leftDistance = pacman_pos[0] - mostleft
    rightDistance = mostright - pacman_pos[0]
    upDistance = pacman_pos[1] - mostup
    downDistance = mostdown - pacman_pos[1]
    if leftDistance < rightDistance:
        horizontalDistance = 2 * leftDistance + rightDistance
    else:
        horizontalDistance = 2 * rightDistance + leftDistance
    if upDistance < downDistance:
        verticalDistance = 2 * upDistance + downDistance
    else:
        verticalDistance = 2 * downDistance + leftDistance

    return verticalDistance + horizontalDistance

class PacmanAgent(Agent):
    """Pacman agent based on A*."""

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
        fringe.push((state, path), h(state))
        closed = set()

        while True:
            if fringe.isEmpty():
                return []

            current_cost, (current, path) = fringe.pop()

            if current.isWin():
                return path

            current_key = key(current)

            if current_key in closed:
                continue
            else:
                closed.add(current_key)

            for successor, action in current.generatePacmanSuccessors():
                fringe.push((successor, path + [action]), g(current_cost, current, successor)+h(successor))

        return path
