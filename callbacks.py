import os

from torch import inf, save
from utils import makedir


class Checkpoint:
    def __init__(self, elements: dict = {}, checkpoint_dir='save', monitor='val_loss', mode='min', verbose=True) -> None:
        self.elements = elements
        self.dir = checkpoint_dir
        self.bestdir = os.path.join('/'.join(checkpoint_dir.split('/')[:-1]), 'best.pt')
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        if mode == 'min':
            self.score = inf
        elif mode == 'max':
            self.score = - inf
        self.patience = 0

    def cmp(self, score):
        if self.mode == 'min':
            if score < self.score:
                return True
            else:
                return False
        elif self.mode == 'max':
            if score > self.score:
                return True
            else:
                return False

    def __call__(self, score: dict):
        score = score.get(self.monitor)
        
        cmp = self.cmp(score)
        self.elements['Checkpoint'] = self.state_dict()
        makedir('/'.join(self.dir.split('/')[:-1]))
        save(self.elements, self.dir)
        if self.verbose:
            print(f'Make checkpoint to {self.dir}')

        if cmp:
            self.score = score
            save(self.elements, self.bestdir)
            if self.verbose:
                print(f'Make best model to {self.bestdir}')

    def state_dict(self):
        return {'monitor': self.monitor, 'mode': self.mode, 'score': self.score}

    def load_state_dict(self, state: dict):
        for k, v in state.items():
            st = getattr(self, k)
            st = v


class EarlyStopping:
    def __init__(self, monitor='val_loss', mode='min', patience=30, verbose=True) -> None:
        self.monitor = monitor
        self.max_patience = patience
        self.mode = mode
        self.verbose = verbose
        if mode == 'min':
            self.score = inf
        elif mode == 'max':
            self.score = - inf
        self.patience = 0

    def cmp(self, score):
        if self.mode == 'min':
            if score < self.score:
                return True
            else:
                return False
        elif self.mode == 'max':
            if score > self.score:
                return True
            else:
                return False

    def __call__(self, score):
        score = score.get(self.monitor)
        
        cmp = self.cmp(score)
        if cmp:
            self.patience = 0
            self.score = score
        else:
            self.patience += 1
            if self.verbose:
                print(f'Early stop patience {self.patience - 1} -> {self.patience}', end='')
                print(f' < {self.max_patience}' if self.patience < self.max_patience else '')

        if self.patience == self.max_patience:
            print('EARLY STOPPING!')
            return False
        return True

    def state_dict(self):
        return {'monitor': self.monitor, 'max_patience': self.max_patience, 'mode': self.mode, 'score': self.score, 'patience': self.patience}

    def load_state_dict(self, state: dict):
        for k, v in state.items():
            st = getattr(self, k)
            st = v
    