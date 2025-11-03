import torch.nn as nn
import torch

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels: int,
                 hidden_dim: int,
                 out_channels: int,
                 stride: int = 1):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.InstanceNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                      stride=stride, padding=1, bias=False),
            nn.InstanceNorm1d(hidden_dim),
            nn.ReLU(inplace=True),

            nn.Conv1d(hidden_dim, out_channels, kernel_size=1,
                      stride=1, padding=0, bias=False),
            nn.InstanceNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

        if (out_channels != in_channels) or (stride != 1):
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                          stride=stride, padding=0, bias=False),
                nn.InstanceNorm1d(out_channels),

            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x):
        return self.bottleneck(x) + self.downsample(x)


class FilmLayer(nn.Module):
    def __init__(self, feature_dim: int, meta_embed_dim: int):
        super().__init__()
        self.gamma_linear = nn.Linear(meta_embed_dim, feature_dim)
        self.beta_linear = nn.Linear(meta_embed_dim, feature_dim)
        self.batch1d = nn.InstanceNorm1d(feature_dim)
        self.relu = nn.GELU()

    def forward(self, x, meta_embed):
        gamma = self.gamma_linear(meta_embed).unsqueeze(-1)
        beta = self.beta_linear(meta_embed).unsqueeze(-1)
        return self.relu(
            self.batch1d(gamma * x + beta)
        )


class SpectraEncoder(nn.Module):
    def __init__(self, output_dim: int,
                 hidden_dims: list[int] = [32, 64, 128, 256],
                 num_blocks_per_layer: list[int] = [2, 3, 2, 2],
                 meta_embed_dim: int = 1,
                 ):
        super().__init__()

        self.meta_embed_dim = meta_embed_dim
        in_channels = hidden_dims[0]

        self.init_compress = nn.Sequential(
            nn.Conv1d(2, in_channels, kernel_size=7, stride=2, groups=2, padding=3, bias=False),
            nn.InstanceNorm1d(in_channels),
            nn.GELU(),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        self.layers = nn.ModuleList()
        self.film_layers = nn.ModuleList()

        for i, (out_channels, num_blocks) in enumerate(zip(hidden_dims, num_blocks_per_layer)):
            stride = 1 if i == 0 else 2
            layer = self._make_layer(in_channels, out_channels, num_blocks, stride)
            self.layers.append(layer)
            film_layer = FilmLayer(out_channels, meta_embed_dim)
            self.film_layers.append(
                film_layer
            )
            in_channels = out_channels

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dims[-1], output_dim)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, stride: int):

        layers = []
        layers.append(ResidualBlock1D(in_channels, in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, g_feature: torch.Tensor, spec: torch.Tensor, res_embed: torch.Tensor):
        """
        :param g_feature: torch.Tensor with shape [batch_shape, input_size]
        :param spec: torch.Tensor with shape [batch_shape, input_size]
        x and y data, where y is spectra points, x are g-factors points. Data is given when g is increasing
        :param res_embed: torch.Tensor with shape [batch_shape, meta_embed_dim]
        """
        x = self.init_compress(torch.stack([g_feature, spec], dim=-2))
        for layer, film_layer in zip(self.layers, self.film_layers):
            x = layer(x)
            x = film_layer(x, res_embed)
        x = self.global_avg_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x
