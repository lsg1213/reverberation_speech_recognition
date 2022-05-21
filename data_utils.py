from torchaudio.datasets import LIBRISPEECH


def get_dataset(config, mode):
    if mode == 'train':
        mode = 'train-clean-360'
    elif mode == 'val':
        mode = 'dev-clean'
    elif mode == 'test':
        mode = 'test-clean'
    return LIBRISPEECH(config.datapath, mode, download=True)

    