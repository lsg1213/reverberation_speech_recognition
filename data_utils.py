from torchaudio.datasets import LIBRISPEECH
import torchaudio
import sentencepiece as spm
from torch.nn.functional import one_hot, pad
import torch


import os


def get_dataset(config, mode):
    if mode == 'train':
        mode = 'train-clean-360'
    elif mode == 'val':
        mode = 'dev-clean'
    elif mode == 'test':
        mode = 'test-clean'
    return Librispeech(config, mode, download=True)


class Word_process():
    def __init__(self, prefix='devset') -> None:
        self.prefix = prefix
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(f'{self.prefix}.model')

    def encode(self, sentence: str):
        return self.sp.EncodeAsIds(sentence)

    def decode(self, Ids):
        return self.sp.Decode(Ids)


class Librispeech(LIBRISPEECH):
    def __init__(self, config, url: str = "train-clean-100", folder_in_archive: str = "LibriSpeech", download: bool = False) -> None:
        self.config = config
        super().__init__(config.datapath, url, folder_in_archive, download)
        self.sr = 16000
        self.word = Word_process()

    def __getitem__(self, n: int):
        fileid = self._walker[n]
        speaker_id, chapter_id, utterance_id = fileid.split("-")
        

        file_text = speaker_id + "-" + chapter_id + self._ext_txt
        file_text = os.path.join(self._path, speaker_id, chapter_id, file_text)

        fileid_audio = speaker_id + "-" + chapter_id + "-" + utterance_id
        file_audio = fileid_audio + self._ext_audio
        file_audio = os.path.join(self._path, speaker_id, chapter_id, file_audio)

        # Load audio
        waveform, sample_rate = torchaudio.load(file_audio)
        self.sr = sample_rate

        with open(file_text) as ft:
            for line in ft:
                fileid_text, transcript = line.strip().split(" ", 1)
                if fileid_audio == fileid_text:
                    break
            else:
                # Translation not found
                raise FileNotFoundError("Translation not found for " + fileid_audio)
        label = torch.tensor(self.word.encode(transcript))

        wavelength, labellength = waveform.shape[-1], label.shape[-1]
        waveform = pad(waveform, (0, self.config.max_wave_length - waveform.shape[-1]))
        label = pad(label, (0, self.config.max_label_length - label.shape[-1]))
        return waveform, label, transcript, wavelength, labellength

