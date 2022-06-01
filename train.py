from argparse import ArgumentError
from math import ceil
import os

from args import get_args
from multiprocessing import cpu_count
import json

import torch
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torchaudio
from transformers import get_scheduler

from utils import *
from data_utils import *
from models import *
from callbacks import *
from metric import WER



def iterloop(config, writer, epoch, model, criterion, dataloader, metric, optimizer=None, mode='train', scheduler=None):
    device = get_device()
    losses = []

    transforms = [torchaudio.transforms.MelSpectrogram(config.sr, n_fft=config.win_size, hop_length=config.win_hop, n_mels=80).to(device),
                  torchaudio.transforms.TimeMasking(ceil(config.max_wave_length / config.win_size) * 0.05, iid_masks=True),
                  torchaudio.transforms.FrequencyMasking(27, iid_masks=True)]
    scores = []
    input_scores = []
    output_scores = []
    with tqdm(dataloader) as pbar:
        for inputs in pbar:
            wave, label, transcript, wavelen, labellen = inputs
            wave, label, wavelen, labellen = wave.to(device), label.to(device), wavelen.to(device), labellen.to(device)
            
            wavelen = torch.ceil(wavelen / config.win_hop).type(wavelen.dtype)
            inputs = wave
            for transform in transforms:
                inputs = transform(inputs)

            logits, length = model(inputs.squeeze(1), wavelen)
            loss = criterion(logits.transpose(0,1), label, length, labellen)
            
            if mode == 'train':
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            losses.append(loss.item())
            progress_bar_dict = {'mode': mode, 'loss': np.mean(losses)}

            if mode == 'val':
                score = metric(logits.argmax(-1), transcript)
                scores.append(score)
                progress_bar_dict['WER'] = np.mean(scores)
            
            pbar.set_postfix(progress_bar_dict)

            if scheduler is not None:
                scheduler.step()

    writer.add_scalar(f'{mode}/loss', np.mean(losses), epoch)
    if len(scores) != 0:
        writer.add_scalar(f'{mode}/WER', np.mean(scores), epoch)

    outputs = [np.mean(losses)]
    if len(scores) != 0:
        outputs.append(np.mean(scores))
    return outputs


def get_model(config):
    if config.model == 'conformer':
        model = Conformer(config)
    return model


def main(config):
    os.environ['CUDA_VISIBLE_DEVICES'] = config.gpus
    name = f'{config.task}_' + (config.model if config.model != '' else 'baseline')
    config.name = name + '_' + config.name if config.name != '' else ''
    config.tensorboard_path = os.path.join(config.tensorboard_path, config.name)
    writer = SummaryWriter(config.tensorboard_path)
    savepath = os.path.join('save', config.name)
    device = get_device()
    makedir(config.tensorboard_path)
    makedir(savepath)

    init_epoch = 0
    final_epoch = 0
    
    train_set = get_dataset(config, 'train')
    val_set = get_dataset(config, 'val')
    test_set = get_dataset(config, 'test')

    train_loader = DataLoader(
        train_set,
        shuffle=True,
        batch_size=config.batch_size,
        num_workers=cpu_count() // torch.cuda.device_count()
    )

    val_loader = DataLoader(
        val_set,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=cpu_count() // torch.cuda.device_count()
    )

    test_loader = DataLoader(
        test_set,
        shuffle=False,
        batch_size=config.batch_size,
        num_workers=cpu_count() // torch.cuda.device_count()
    )

    model = get_model(config)

    optimizer = Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.98), eps=1e-9, weight_decay=1e-6)
    scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=4000 * 8, num_training_steps=ceil(len(train_set) / config.batch_size) * config.epoch)

    with open(os.path.join(savepath, 'config.json'), 'w') as f:
        json.dump(vars(config), f)

    callbacks = []

    callbacks.append(EarlyStopping(monitor="val_score", mode="min", patience=config.max_patience, verbose=True))
    callbacks.append(Checkpoint(checkpoint_dir=os.path.join(savepath, 'checkpoint.pt'), monitor='val_score', mode='min', verbose=True))
    criterion = torch.nn.CTCLoss()
    metric = WER(Word_process())

    if config.resume:
        resume = torch.load(os.path.join(savepath, 'checkpoint.pt'))
        model.load_state_dict(resume['model'])
        model = model.to(device)
        optimizer.load_state_dict(resume['optimizer'])
        scheduler.load_state_dict(resume['scheduler'])
        init_epoch = resume['epoch']
        for callback in callbacks:
            state = resume.get(type(callback).__name__)
            if state is not None:
                callback.load_state_dict(state)

    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    for epoch in range(init_epoch, config.epoch):
        print(f'--------------- epoch: {epoch} ---------------')
        model.train()
        train_loss = iterloop(config, writer, epoch, model, criterion, train_loader, metric, optimizer=optimizer, mode='train', scheduler=scheduler)

        model.eval()
        with torch.no_grad():
            val_loss, val_score = iterloop(config, writer, epoch, model, criterion, val_loader, metric, mode='val')

        results = {'train_loss': train_loss, 'val_loss': val_loss, 'val_score': val_score}

        final_epoch += 1
        for callback in callbacks:
            if type(callback).__name__ == 'Checkpoint':
                if torch.cuda.device_count() > 1:
                    model_state = {}
                    for k, v in model.state_dict().items():
                        model_state[k[7:]] = v
                else:
                    model_state = model.state_dict()
                callback.elements.update({
                    'model': model_state,
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch + 1
                })
                for cb in callbacks:
                    if type(cb).__name__ != 'Checkpoint':
                        state = cb.state_dict()
                        callback.elements[type(cb).__name__] = state
            if type(callback).__name__ == 'EarlyStopping':
                tag = callback(results)
                if tag == False:
                    resume = torch.load(os.path.join(savepath, 'best.pt'))
                    model = get_model(config)
                    model.load_state_dict(resume['model'])
                    model = model.to(device)
                    model.eval()
                    with torch.no_grad():
                        test_loss, test_score = iterloop(config, writer, epoch, model, criterion, test_loader, metric, mode='val')
                    writer.add_scalar('test/loss', test_loss, resume['epoch'])
                    writer.add_scalar('test/WER', test_score, resume['epoch'])
                    return
            else:
                callback(results)
        print('---------------------------------------------')
    resume = torch.load(os.path.join(savepath, 'best.pt'))
    model = get_model(config)
    model.load_state_dict(resume['model'])
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        test_loss, test_score = iterloop(config, writer, epoch, model, criterion, test_loader, metric, mode='val')
    writer.add_scalar('test/loss', test_loss, resume['epoch'])
    writer.add_scalar('test/WER', test_score, resume['epoch'])
    

if __name__ == '__main__':
    config = get_args()
    config.win_size = int(config.sr * 0.025)
    config.win_hop = int(config.sr * 0.01)
    main(config)

