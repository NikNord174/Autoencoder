import torch.nn as nn

import constants


class Subpixel_Conv(nn.Module):
    def __init__(self):
        super(Subpixel_Conv, self).__init__()
        self.channels = [3, 100, 200, 400]
        self.encoder = self.encoder_layers()
        self.decoder = self.decoder_layers()

    def simple_enc_block(self,
                         input_channels: int = 3,
                         output_channels: int = 3,
                         kernel_size: int = 3,
                         padding: int = 0,
                         maxpool_kernel: int = 2) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU(constants.ReLU_FACTOR),
            nn.MaxPool2d(maxpool_kernel),
            )

    def encoder_layers(self) -> nn.Sequential:
        layers = []
        for i in range(len(self.channels)-1):
            if self.channels[i] != self.channels[-2]:
                layers.append(self.simple_enc_block(
                    input_channels=self.channels[i],
                    output_channels=self.channels[i+1])
                )
            else:
                layers.append(self.simple_enc_block(
                    input_channels=self.channels[i],
                    output_channels=self.channels[i+1],
                    padding=0,
                    maxpool_kernel=2,
                    )
                )
        return nn.Sequential(*layers)

    def decoder_layers(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(400, 768, 1),
            nn.BatchNorm2d(768),
            nn.LeakyReLU(0.2),
            nn.PixelShuffle(16),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
