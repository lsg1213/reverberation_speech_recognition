import torchaudio


class Conformer(torchaudio.models.Conformer):
    def __init__(self, config):
        # input_dim: int, num_heads: int, ffn_dim: int, num_layers: int, depthwise_conv_kernel_size: int, dropout: float = 0
        super().__init__(input_dim=1, num_heads=4, ffn_dim=144, num_layers=16, depthwise_conv_kernel_size=32, dropout=0.)
        self.config = config
        

