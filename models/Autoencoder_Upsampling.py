import torch.nn as nn


class Autoencoder_Upsampling(nn.Module):
    def __init__(self):
        super(Autoencoder_Upsampling, self).__init__()
        self.channels = [3, 100, 200, 250, 300]
        self.encoder = self.encoder_layers()
        self.decoder = self.decoder_layers()

    def simple_enc_block(self,
                         input_channels: int = 3,
                         output_channels: int = 3,
                         kernel_size: int = 3,
                         padding: int = 1,
                         maxpool_kernel: int = 2) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
                padding=padding,
            ),
            nn.BatchNorm2d(output_channels),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(0.5),
            nn.MaxPool2d(maxpool_kernel),)

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
                    )
                )
        return nn.Sequential(*layers)

    def simple_dec_block(self,
                         input_channels: int = 3,
                         output_channels: int = 3,
                         kernel_size: int = 1,
                         scale_factor: int = 2,
                         mode: str = 'nearest') -> nn.Sequential:
        return nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode=mode),
            nn.Conv2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernel_size,
            ),
            nn.BatchNorm2d(output_channels),
            nn.Dropout(p=0.2),
            nn.LeakyReLU(0.5)
            )

    def decoder_layers(self) -> nn.Sequential:
        layers = []
        dec_channels = list(reversed(self.channels))
        for i in range(len(dec_channels)-1):
            if dec_channels[i] != dec_channels[-2]:
                layers.append(self.simple_dec_block(
                    input_channels=dec_channels[i],
                    output_channels=dec_channels[i+1])
                )
            else:
                layers.append(self.simple_dec_block(
                    input_channels=dec_channels[i],
                    output_channels=dec_channels[i+1])
                )
        return nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
