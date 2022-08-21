import numpy as np

import torch
import torch.nn as nn


class Autoencoder_Initial(nn.Module):
    def __init__(self):
        super(Autoencoder_Initial, self).__init__()
        self.channels = [3, 100, 200, 250, 300]
        self.hidden_state = 1000
        self.encoder = self.encoder_layers()
        self.flatten = nn.Flatten()
        self.lin_neurons = [300, 2, 2]
        self.enc_neurons = np.prod(self.lin_neurons)
        self.linear = nn.Linear(self.enc_neurons,
                                self.hidden_state)
        self.rev_linear = nn.Linear(self.hidden_state,
                                    self.enc_neurons)
        self.decoder = self.decoder_layers()

    def simple_enc_block(self,
                         input_channels: int = 3,
                         output_channels: int = 3,
                         kernel_size: int = 3,
                         stride: int = 2,
                         padding: int = 1,
                         final_layer: bool = False) -> nn.Sequential:
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(in_channels=input_channels,
                          out_channels=output_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding),
                nn.BatchNorm2d(output_channels),
                nn.Dropout(p=0.2),
                nn.LeakyReLU(0.2)
                )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=input_channels,
                          out_channels=output_channels,
                          kernel_size=kernel_size,
                          stride=stride,
                          padding=padding)
                )

    def encoder_layers(self) -> nn.Sequential:
        layers = []
        for i in range(len(self.channels)-1):
            if self.channels[i] != self.channels[-2]:
                layers.append(self.simple_enc_block(
                    input_channels=self.channels[i],
                    output_channels=self.channels[i+1]
                ))
            else:
                layers.append(self.simple_enc_block(
                    input_channels=self.channels[i],
                    output_channels=self.channels[i+1],
                    final_layer=True
                    ))
        return nn.Sequential(*layers)

    def simple_dec_block(self,
                         input_channels: int = 3,
                         output_channels: int = 3,
                         kernel_size: int = 4,
                         stride: int = 2,
                         padding: int = 1,
                         final_layer: bool = False) -> nn.Sequential:
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=input_channels,
                                   out_channels=output_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding),
                nn.BatchNorm2d(output_channels),
                nn.Dropout(p=0.2),
                nn.LeakyReLU(0.2)
                )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=input_channels,
                                   out_channels=output_channels,
                                   kernel_size=kernel_size,
                                   stride=stride,
                                   padding=padding)
                )

    def decoder_layers(self) -> nn.Sequential:
        layers = []
        dec_channels = list(reversed(self.channels))
        for i in range(len(dec_channels)-1):
            if dec_channels[i] != dec_channels[-2]:
                layers.append(self.simple_dec_block(
                    input_channels=dec_channels[i],
                    output_channels=dec_channels[i+1]
                    ))
            else:
                layers.append(self.simple_dec_block(
                    input_channels=dec_channels[i],
                    output_channels=dec_channels[i+1],
                    final_layer=True
                    ))
        return nn.Sequential(*layers)

    def forward(self, x):
        encoded = self.encoder(x)
        hidden_layer = torch.flatten(encoded, start_dim=1)
        hidden_layer = self.linear(hidden_layer)
        hidden_layer = self.rev_linear(hidden_layer)
        hidden_layer = hidden_layer.view(-1, *self.lin_neurons)
        decoded = self.decoder(hidden_layer)
        return decoded
