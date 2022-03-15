import numpy as np
import torch

from climatehack import BaseEvaluator
from model import AttentionUNet


class Evaluator(BaseEvaluator):
    def setup(self):
        """Sets up anything required for evaluation.

        In this case, it loads the trained model (in evaluation mode)."""

        #preprocess = ImagesPreprocess()
        
        #self.model = PerceiverCH(preprocess, latent_dim=(32, 128), heads=8, wide_factor=4, latent_count=6)
        self.model = AttentionUNet()
        self.model.load_state_dict(torch.load("curr_model", map_location='cpu'))
        self.model.eval()

    def predict(self, coordinates: np.ndarray, data: np.ndarray) -> np.ndarray:
        """Makes a prediction for the next two hours of satellite imagery.

        Args:
            coordinates (np.ndarray): the OSGB x and y coordinates (2, 128, 128)
            data (np.ndarray): an array of 12 128*128 satellite images (12, 128, 128)

        Returns:
            np.ndarray: an array of 24 64*64 satellite image predictions (24, 64, 64)
        """

        assert coordinates.shape == (2, 128, 128)
        assert data.shape == (12, 128, 128)

        with torch.no_grad():
            prediction = (
                self.model(torch.from_numpy(data).view(1, 12, 128, 128).float())
                .view(24, 64, 64)
                .detach()
                .numpy()
            )

            assert prediction.shape == (24, 64, 64)

            return prediction


def main():
    evaluator = Evaluator()
    evaluator.evaluate()


if __name__ == "__main__":
    main()
