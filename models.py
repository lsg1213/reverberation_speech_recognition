import torchaudio
import torch
from torch.nn import *


# https://github.com/sooftware/conformer/blob/9318418cef5b516cd094c0ce2a06b80309938c70/conformer/convolution.py#L152
class Conv2dSubampling(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super(Conv2dSubampling, self).__init__()
        self.sequential = Sequential(
            Conv2d(in_channels, out_channels, kernel_size=3, stride=2),
            ReLU(),
            Conv2d(out_channels, out_channels, kernel_size=3, stride=2),
            ReLU(),
        )

    def forward(self, inputs, input_lengths):
        inputs = inputs.transpose(-2,-1)
        outputs = self.sequential(inputs.unsqueeze(1))
        batch_size, channels, subsampled_lengths, sumsampled_dim = outputs.size()

        outputs = outputs.permute(0, 2, 1, 3)
        outputs = outputs.contiguous().view(batch_size, subsampled_lengths, channels * sumsampled_dim)

        output_lengths = input_lengths >> 2
        output_lengths -= 1

        return outputs, output_lengths


def _lengths_to_padding_mask(lengths: torch.Tensor, max_length=None) -> torch.Tensor:
    batch_size = lengths.shape[0]
    if max_length is None:
        max_length = int(torch.max(lengths).item())
    padding_mask = torch.arange(max_length, device=lengths.device, dtype=lengths.dtype).expand(
        batch_size, max_length
    ) >= lengths.unsqueeze(1)
    return padding_mask


class Conformer_module(torchaudio.models.Conformer):
    def __init__(self, input_dim: int, num_heads: int, ffn_dim: int, num_layers: int, depthwise_conv_kernel_size: int, dropout: float = 0):
        super().__init__(input_dim, num_heads, ffn_dim, num_layers, depthwise_conv_kernel_size, dropout)
        self.max_length = None

    def forward(self, input: torch.Tensor, lengths: torch.Tensor):
        encoder_padding_mask = _lengths_to_padding_mask(lengths, self.max_length)

        x = input.transpose(0, 1)
        for layer in self.conformer_layers:
            x = layer(x, encoder_padding_mask)
        return x.transpose(0, 1), lengths


class Conformer(Module):
    def __init__(self, config):
        # input_dim: int, num_heads: int, ffn_dim: int, num_layers: int, depthwise_conv_kernel_size: int, dropout: float = 0
        super().__init__()
        self.config = config
        input_dim = 80
        encoder_dim = 144
        dropout = 0.1
        self.conv_subsampling = Conv2dSubampling(1, encoder_dim)
        self.input_linear = Sequential(
            Linear(encoder_dim * (((input_dim - 1) // 2 - 1) // 2), encoder_dim),
            Dropout(p=dropout),
        )
        self.conformer_module = Conformer_module(input_dim=encoder_dim, num_heads=4, ffn_dim=encoder_dim, num_layers=16, depthwise_conv_kernel_size=33, dropout=dropout)
        self.language_model = Language_model(input_dim=encoder_dim, hidden_dim=4096)
        self.fc = Linear(4096, 1000, bias=False)

    def forward(self, inputs, length):
        x, l = self.conv_subsampling(inputs, length)
        x = self.input_linear(x)
        self.conformer_module.max_length = x.shape[1]
        x, l = self.conformer_module(x, l)
        x = self.language_model(x)[0]
        x = self.fc(x)
        x = torch.nn.functional.log_softmax(x, dim=-1)
        return x, l
        

class Language_model(Module):
    def __init__(self, input_dim=144, hidden_dim=4096) -> None:
        super().__init__()
        self.model = LSTM(input_dim, hidden_dim, 3, batch_first=True)

    def forward(self, inputs: torch.Tensor):
        return self.model(inputs)

