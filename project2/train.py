import pickle

import torch
import torch.nn as nn

from architecture import PacmanNetwork
from data import PacmanDataset
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# containing the training loop of your model

class Pipeline(nn.Module):
    def __init__(self, path):
        """
        Initialize your training pipeline.

        Arguments:
            path: The file path to the pickled dataset.
        """
        super().__init__()

        self.path = path
        self.dataset = PacmanDataset(self.path)
        
        
        self.batchSize = 128
        
        self.epochs = 80
        self.show_every = 40
        
        self.train_loader = DataLoader(self.dataset, batch_size=self.batchSize, shuffle=True)
        
        D = len(self.dataset.inputs[0])
        
        self.hiddens = [512, 256, 128]
        
        self.model = PacmanNetwork(D, self.hiddens)

        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate choisit pour que l'entrainement soit stable
        # et pas trop lent et pour pouvoir sortir des minimums locaux
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self):
        print("Beginning of the training of your network...")

        
        actionDict = {
            "North": 0,
            "South": 1,
            "East": 2,
            "West": 3,
            "Stop": 4,
        }
        
        losses = []
        self.model.train()
        for epoch in range(self.epochs):
            for featuresVector, action in self.train_loader:
                actionIdsVect = [actionDict[a] for a in action]
                action = torch.tensor(actionIdsVect, dtype=torch.long)
                
                pred = self.model.forward(featuresVector)
                loss = self.criterion(pred, action)
                
                self.optimizer.zero_grad()
                
                loss.backward()
                
                self.optimizer.step()
                
                losses.append(loss.item())
                
        plt.plot(losses)
        plt.title(f"After epoch {epoch}")
        plt.ylabel("Loss")
        plt.xlabel("Steps")
        plt.xscale("log")
        plt.grid()
        plt.show()
                
        torch.save(self.model.state_dict(), "pacman_model.pth")
        print("Model saved !")


if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()
