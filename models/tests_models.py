import logging
import unittest
import torch
from torchsummary import summary

from Autoencoder_Upsampling import Autoencoder_Upsampling
from Autoencoder_ConvTranspose import Autoencoder_ConvTranspose
from Autoencoder_Initial import Autoencoder_Initial


logger = logging.getLogger(__name__)


class Tests_Models(unittest.TestCase):
    "Test models"
    @classmethod
    def setUpClass(cls):
        cls.images = torch.randn([64, 3, 32, 32])
        cls.models = [
            Autoencoder_Upsampling,
            Autoencoder_ConvTranspose,
            Autoencoder_Initial,
        ]

    def test_model_output(self):
        "Test how whole models work"
        ground_shape = torch.Size([64, 3, 32, 32])
        for model in self.models:
            model = model()
            try:
                outputs_shape = model(self.images).shape
                self.assertEqual(
                    outputs_shape, ground_shape,
                    'Model {} gives wrong tensor'.format(
                        model.__class__.__name__
                    )
                )
            except RuntimeError:
                logger.error('Model {} does not work'.format(
                    model.__class__.__name__))

    def test_encoder_output(self):
        "Test how encoders work"
        ground_shape = torch.Size([64, 300, 2, 2])
        for model in self.models:
            model = model()
            encoder = model.encoder_layers()
            outputs_shape = encoder(self.images).shape
            print(summary(model, (3, 32, 32)))
            self.assertEqual(
                outputs_shape, ground_shape,
                'Encoder of model {} gives wrong vector'.format(
                    model.__class__.__name__
                )
            )


if __name__ == '__main__':
    unittest.main()
