import pickle
import torch
import pandas as pd

from data import state_to_tensor
from architecture import PacmanNetwork
import math

# submission.csv containing the action predictions for each game
# state contained in the pacman_test.pkl file.
# The predictions must follow the same order as
# the states appear in pacman_test.pkl


class SubmissionWriter:
    def __init__(self, test_set_path, model_path):
        """
        Initialize the writing of your submission.
        Pay attention that the test set only contains GameState objects,
        it's no longer (GameState, action) pairs.

        Arguments:
            test_set_path: The file path to the pickled test set.
            model_path: The file path to the trained model.
        """
        with open(test_set_path, "rb") as f:
            self.test_set = pickle.load(f)

        D = len(state_to_tensor(self.test_set[0]))
        self.model = PacmanNetwork(D, [256, 256])
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def predict_on_testset(self):
        """
        Generate predictions for the test set.

        !!! Your predicted actions should follow the same order
        as the test set provided.
        """

        actionList = ["North", "South", "East", "West"]

        actions = []
        for state in self.test_set:
            x = state_to_tensor(state).unsqueeze(0)
            with torch.no_grad():
                pred = self.model(x)

            logits = pred[0]
            actionId = torch.argmax(logits).item()

            actions.append(actionList[actionId])
        return actions

    def write_csv(self, actions, file_name="submission"):
        """
        ! Do not modify !

        Write the predicted actions (North, South, ...)
        to a CSV file.

        """
        submission = pd.DataFrame(
            data={
                'ACTION': actions,
            },
            columns=["ACTION"]
        )

        submission.to_csv(file_name + ".csv", index=False)


if __name__ == "__main__":
    writer = SubmissionWriter(
        test_set_path="datasets/pacman_test.pkl",
        model_path="pacman_model.pth"  # change if needed
    )
    predictions = writer.predict_on_testset()
    writer.write_csv(predictions)
