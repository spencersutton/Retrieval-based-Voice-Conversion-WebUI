import torch
import torch.nn as nn

from .deepunet import DeepUnet


class E2E(nn.Module):
    def __init__(
        self,
        n_blocks: int,
        n_gru: int,
        kernel_size: tuple[int, int],
        en_de_layers: int = 5,
        inter_layers: int = 4,
        in_channels: int = 1,
        en_out_channels: int = 16,
    ):
        super().__init__()

        self.unet = DeepUnet(
            kernel_size,
            n_blocks,
            en_de_layers,
            inter_layers,
            in_channels,
            en_out_channels,
        )
        self.cnn = nn.Conv2d(en_out_channels, 3, (3, 3), padding=(1, 1))
        if n_gru:
            self.fc = nn.Sequential(
                self.BiGRU(3 * 128, 256, n_gru),
                nn.Linear(512, 360),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )
        else:
            self.fc = nn.Sequential(
                nn.Linear(3 * nn.N_MELS, nn.N_CLASS),
                nn.Dropout(0.25),
                nn.Sigmoid(),
            )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        mel = mel.transpose(-1, -2).unsqueeze(1)
        x = self.cnn(self.unet(mel)).transpose(1, 2).flatten(-2)
        x = self.fc(x)
        return x

    class BiGRU(nn.Module):
        def __init__(
            self,
            input_features: int,
            hidden_features: int,
            num_layers: int,
        ):
            super().__init__()
            self.gru = nn.GRU(
                input_features,
                hidden_features,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.gru(x)[0]
