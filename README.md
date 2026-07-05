# Authors
 - Malaise Romain
 - Renard Baptiste
 - Schins Martin

# IA Course INFO8006 Projects

This repository gathers three artificial intelligence course projects built around a Pacman environment. Each project focuses on a different AI topic, from classical search algorithms to probabilistic reasoning and neural networks.

It is mainly a student repository: the goal is to keep track of the implementations, experiments, and course deliverables in one clean place.

## Projects

### Project 0 - Search Algorithms

The first project implements search-based Pacman agents. The goal is to solve mazes by planning a sequence of actions that maximizes the score.

Implemented agents include:

- Depth-first search baseline
- Breadth-first search
- Uniform-cost search
- A* search

Example:

```bash
cd project0
python run.py --agent bfs --layout medium
python run.py --agent astar --layout medium
```

### Project 1 - Bayes Filter

The second project focuses on probabilistic inference. Pacman has to locate invisible ghosts using noisy distance measurements, then choose actions based on its belief state.

Main parts:

- transition model
- observation model
- Bayes filter update
- Pacman controller using ghost belief states

Example:

```bash
cd project1
python run.py --ghost afraid --nghosts 1 --layout large_filter --seed 42
```

### Project 2 - Neural Network Agent

The third project uses PyTorch to train a neural network agent. Game states are transformed into feature vectors, then the model predicts Pacman's next action.

Main parts:

- dataset loading and feature extraction
- neural network architecture
- training pipeline
- trained Pacman agent
- CSV submission generation

Example:

```bash
cd project2
python train.py
python run.py
python write_submission.py
```

## Requirements

The projects are written in Python. Depending on the project, useful packages include:

- `numpy`
- `torch`
- `matplotlib`
- `pandas`

Project 0 and Project 1 mostly use the provided Pacman framework. Project 2 requires PyTorch for training and running the neural network model.

## Notes

The `pacman_module/` folders contain the provided game framework used by the course. The code written for the projects is mainly in files such as `bfs.py`, `astar.py`, `bayesfilter.py`, `architecture.py`, `data.py`, `train.py`, and `pacmanagent.py`.

Some files are course material or generated outputs, so this repository should be read as a learning/project archive rather than a polished standalone application.
