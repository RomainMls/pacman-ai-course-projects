import torch
import torch.nn as nn

# containing the implementation of your neural network

class PacmanNetwork(nn.Module):
    """
    Your neural network architecture.
    """
    def __init__(self, D, hiddenDims, activation=nn.ReLU()):
        super().__init__()
        
        dropout = 0.3
        self.mlp = [nn.Linear(D, hiddenDims[0]), activation, nn.Dropout(dropout)]
        
        for i in range(len(hiddenDims) - 1):
            self.mlp += [nn.Linear(hiddenDims[i], hiddenDims[i + 1]), activation, nn.Dropout(dropout)]
        
        # On veut des logits en sortie pas des proba
        # parce que CrossEntropyLoss() se base sur des logits et pas des proba
        output_layer = [nn.Linear(hiddenDims[-1], 4)]
        self.mlp = nn.Sequential(*(self.mlp + output_layer))
        
    def forward(self, x):
        # x est notre vecteur de features
        
        pred = self.mlp(x)
        
        return pred
