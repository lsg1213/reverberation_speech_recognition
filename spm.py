import sentencepiece as spm
from data_utils import get_dataset
from args import get_args


def main(config):
    val_set = get_dataset(config, 'val')

    input_file = 'spm_input.txt'
    with open(input_file, 'w', encoding='utf-8') as f:
        for _, corpus in val_set:
            f.write(f'{corpus}\n')

    vocab_size = 1000
    prefix = 'devset'
    cmd = f'--input={input_file} --model_prefix={prefix} --vocab_size={vocab_size}'
    spm.SentencePieceTrainer.Train(cmd)


if __name__ == '__main__':
    main(get_args())

