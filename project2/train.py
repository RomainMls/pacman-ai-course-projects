import pickle

import torch
import torch.nn as nn

from architecture import PacmanNetwork
from data import PacmanDataset
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt
import math

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
        dataset = PacmanDataset(self.path)
        
        n = len(dataset)
        nTrain = int(0.9 * n)
        nTest = n - nTrain
        
        D = len(dataset.inputs[0])
        
        self.dataset, self.testset = random_split(dataset, [nTrain, nTest])        
        
        self.batchSize = 128
        
        self.epochs = 42
        self.show_every = 40
        
        self.train_loader = DataLoader(self.dataset, batch_size=self.batchSize, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=self.batchSize)
        
        self.hiddens = [256, 256]
        
        self.model = PacmanNetwork(D, self.hiddens)
        print(self.model)
        self.criterion = nn.CrossEntropyLoss()
        
        # Learning rate choisit pour que l'entrainement soit stable
        # et pas trop lent et pour pouvoir sortir des minimums locaux
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=7e-4)

    def train(self):
        print("Beginning of the training of your network...")

        
        actionDict = {
            "North": 0,
            "South": 1,
            "East": 2,
            "West": 3,
        }
        
        losses = []
        self.bestAccuracy = 0
        self.trainingAccuracy = 0
        
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
            
            trainAccuracy = self.evaluate(self.train_loader)
            testAccuracy = self.evaluate(self.test_loader)
            print(f"Epoch {epoch}, TrainAccuracy = {trainAccuracy:.4f} and testAccuracy = {testAccuracy:.4f}")

            if testAccuracy > self.bestAccuracy:
                self.bestAccuracy = testAccuracy
                self.trainingAccuracy = trainAccuracy

        torch.save(self.model.state_dict(), "pacman_model.pth")
        print("Model saved !")
                    
        plt.plot(losses)
        plt.title(f"After epoch {epoch}")
        plt.ylabel("Loss")
        plt.xlabel("Steps")
        plt.xscale("log")
        plt.grid()
        plt.show()
        
    def evaluate(self, loader):
        actionDict = {
            "North": 0,
            "South": 1,
            "East": 2,
            "West": 3,
        }
        
        correct = 0
        total = 0
        
        self.model.eval()
        with torch.no_grad():
            for featuresVectorBatch, actionBatch in loader:
                preds = self.model(featuresVectorBatch)
                actionIdsVect = [actionDict[a] for a in actionBatch]

                for i in range(len(featuresVectorBatch)):
                    logits = preds[i]
                    actionId = torch.argmax(logits).item()
                                    
                    if(actionIdsVect[i]== actionId):
                        correct += 1
                    total += 1
                    i += 1
    
        return correct/total

if __name__ == "__main__":
    pipeline = Pipeline(path="datasets/pacman_dataset.pkl")
    pipeline.train()

    print("Training accuracy:", pipeline.trainingAccuracy)
    print("Testing accuracy:", pipeline.bestAccuracy)
    