import pytorch_lightning as pl
import torch
import torch.nn as nn
from argparse import ArgumentParser
import numpy as np
from nystrom_attention import NystromAttention, Nystromformer # https://github.com/lucidrains/nystrom-attention
import torchvision.models as models
from torch.optim.optimizer import Optimizer


## Lookahead optimizer + CE Loss

class TransMIL(pl.LightningModule):
    def __init__(self, n_classes, n_channels=3):
        super(TransMIL, self).__init__()
        self.L = 512 # 512 node fully connected layer
        self.D = 128 # 128 node attention layer
        self.K = 1   
        self.sqrtN = 12 # size for squaring
        self.n_classes = n_classes
        self.n_channels = n_channels

        resnet50 = models.resnet50(pretrained=True)    
        modules = list(resnet50.children())[:-3]
        self.resnet_extractor = nn.Sequential(
            *modules,
            nn.AdaptiveAvgPool2d(1),
            View((-1, 1024)),
            nn.Linear(1024, self.L)
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, 512))

        self.nystromer1 = Nystromformer(dim = self.L, depth=1)
        self.nystromer2 = Nystromformer(dim = self.L, depth=1)
        self.grp_conv1 = nn.Conv2d(512, 512, 3, padding=1)
        self.grp_conv2 = nn.Conv2d(512, 512, 5, padding=2)
        self.grp_conv3 = nn.Conv2d(512, 512, 7, padding=3)
        self.ln = nn.LayerNorm(self.L)
        if self.n_classes == 2:

            self.fc = nn.Linear(self.L, 1)
        else: 
            self.fc = nn.Linear(self.L, self.n_classes)
        
    def forward(self, x):

        x = x.squeeze(0)
        if self.n_channels==1:
            x = torch.cat((x,x,x), 1)
            
        Hf = self.feature_extractor(x)
        Hf = Hf.squeeze(1) #512 features

        ##################################################
        # Squaring 
        ####################################################
        # ToDo: Testing 
        N = np.square(self.sqrtN)
        while Hf.shape[0] != N:
            if Hf.shape[0] > N: # some of our WSIs have more than N patches, but most have less
                Hf = Hf[:N, :]
            else:
                d = N - Hf.shape[0]
                missing = Hf[:d, :]
                Hf = torch.cat((Hf, missing), 0)
        Hf = Hf.squeeze().unsqueeze(0)
        Hs = torch.cat((self.class_token, Hf), 1)

        ####################################################
        # MSA
        ####################################################
        Hs = self.nystromer1(Hs) + Hs
        ####################################################
        # PPEG
        ####################################################
        Hc, Hf = torch.split(Hs, [1,Hs.shape[1]-1], 1)
        Hf = Hf.view(1, -1, self.sqrtN, self.sqrtN)

        Hf1 = self.grp_conv1(Hf)
        Hf2 = self.grp_conv2(Hf)
        Hf3 = self.grp_conv3(Hf)
        Hfs = Hf1 + Hf2 + Hf3
        
        Hfs = Hfs.view(1, N, -1)
        Hs = torch.cat((Hc, Hfs), 1)

        ####################################################
        # MSA 2
        ####################################################
        Hs = self.nystromer1(Hs) + Hs

        ####################################################
        # MLP 
        ####################################################
        Hc, Hf = torch.split(Hs, [1,Hs.shape[1]-1], 1)
        Hc = Hc.squeeze()
        output = self.fc(self.ln(Hc))
        Y_hat = torch.ge(output, 1/self.n_classes).float() 

        return output, Y_hat, Hc

    def configure_optimizers(self):
        lr = 2e-4
        decay = 1e-5
        optim = Lookahead(torch.optim.AdamW(self.parameters(), lr=lr, decay=decay))
        return optim
        # return torch.optim.Adam(self.parameters(), lr=self.lr)

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        '''
        Reshapes the input according to the shape saved in the view data structure.
        '''
        batch_size = input.size(0)
        shape = (batch_size, *self.shape)
        out = input.view(shape)
        return out

class Lookahead(Optimizer):
    r"""PyTorch implementation of the lookahead wrapper.
    Lookahead Optimizer: https://arxiv.org/abs/1907.08610
    """

    def __init__(self, optimizer, la_steps=5, la_alpha=0.8, pullback_momentum="none"):
        """optimizer: inner optimizer
        la_steps (int): number of lookahead steps
        la_alpha (float): linear interpolation factor. 1.0 recovers the inner optimizer.
        pullback_momentum (str): change to inner optimizer momentum on interpolation update
        """
        self.optimizer = optimizer
        self._la_step = 0  # counter for inner optimizer
        self.la_alpha = la_alpha
        self._total_la_steps = la_steps
        pullback_momentum = pullback_momentum.lower()
        assert pullback_momentum in ["reset", "pullback", "none"]
        self.pullback_momentum = pullback_momentum

        self.state = defaultdict(dict)

        # Cache the current optimizer parameters
        for group in optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['cached_params'] = torch.zeros_like(p.data)
                param_state['cached_params'].copy_(p.data)
                if self.pullback_momentum == "pullback":
                    param_state['cached_mom'] = torch.zeros_like(p.data)

    def __getstate__(self):
        return {
            'state': self.state,
            'optimizer': self.optimizer,
            'la_alpha': self.la_alpha,
            '_la_step': self._la_step,
            '_total_la_steps': self._total_la_steps,
            'pullback_momentum': self.pullback_momentum
        }

    def zero_grad(self):
        self.optimizer.zero_grad()

    def get_la_step(self):
        return self._la_step

    def state_dict(self):
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict)

    def _backup_and_load_cache(self):
        """Useful for performing evaluation on the slow weights (which typically generalize better)
        """
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                param_state['backup_params'] = torch.zeros_like(p.data)
                param_state['backup_params'].copy_(p.data)
                p.data.copy_(param_state['cached_params'])

    def _clear_and_load_backup(self):
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                p.data.copy_(param_state['backup_params'])
                del param_state['backup_params']

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def step(self, closure=None):
        """Performs a single Lookahead optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = self.optimizer.step(closure)
        self._la_step += 1

        if self._la_step >= self._total_la_steps:
            self._la_step = 0
            # Lookahead and cache the current optimizer parameters
            for group in self.optimizer.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    p.data.mul_(self.la_alpha).add_(param_state['cached_params'], alpha=1.0 - self.la_alpha)  # crucial line
                    param_state['cached_params'].copy_(p.data)
                    if self.pullback_momentum == "pullback":
                        internal_momentum = self.optimizer.state[p]["momentum_buffer"]
                        self.optimizer.state[p]["momentum_buffer"] = internal_momentum.mul_(self.la_alpha).add_(
                            1.0 - self.la_alpha, param_state["cached_mom"])
                        param_state["cached_mom"] = self.optimizer.state[p]["momentum_buffer"]
                    elif self.pullback_momentum == "reset":
                        self.optimizer.state[p]["momentum_buffer"] = torch.zeros_like(p.data)

        return loss
