import logging
import unittest
import torch
from torchmetrics.functional import structural_similarity_index_measure

from MSE import MSE


logger = logging.getLogger(__name__)


class Tests_Metrics(unittest.TestCase):
    "Test metrics"

    def test_mse(self):
        "Test MSE"
        real_image = torch.ones([64, 3, 32, 32])
        fake_correct_image = torch.ones([64, 3, 32, 32])
        fake_wrong_image = torch.zeros([64, 3, 32, 32])
        correct_output = 0
        different_output = 1
        try:
            same_images = MSE(fake_correct_image, real_image)
            self.assertEqual(
                same_images, correct_output,
                'MSE gives wrong answer with the same image'
            )
            opposite_images = MSE(fake_wrong_image, real_image)
            self.assertEqual(
                opposite_images, different_output,
                'MSE gives wrong answer with the opposite image'
            )
        except ValueError:
            logger.error('MSE does not work')

    def test_ssim(self):
        "Test SSIM"
        SSIM = structural_similarity_index_measure
        real_image = torch.ones([64, 3, 32, 32])
        fake_correct_image = torch.ones([64, 3, 32, 32])
        fake_wrong_image = torch.zeros([64, 3, 32, 32])
        correct_output = 1
        different_output = 0
        try:
            same_images = SSIM(fake_correct_image, real_image)
            self.assertEqual(
                same_images, correct_output,
                'SSIM gives wrong answer with the same image'
            )
            opposite_images = SSIM(fake_wrong_image, real_image)
            self.assertEqual(
                opposite_images, different_output,
                'SSIM gives wrong answer with the opposite image'
            )
        except TypeError:
            logger.error('preds and target do not have the same data type')
        except ValueError:
            logger.error(
                'preds and target do not have right shape',
                'or check the parameters'
            )


if __name__ == '__main__':
    unittest.main()
