#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

sys.path.insert(0, "..")


# In[2]:


import random
from pprint import pprint

import numpy as np
import scipy.ndimage as ndi
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from torch import nn
from torchmore import combos, flex, layers
from torchvision import transforms

import webdataset as wds

pl.seed_everything(42)


# In[3]:


def collate4ocr(samples):
    """Collate image+sequence samples into batches.

    This returns an image batch and a compressed sequence batch using CTCLoss conventions.
    """
    images, seqs = zip(*samples)
    images = [im.unsqueeze(2) if im.ndimension() == 2 else im for im in images]
    bh, bw, bd = map(max, zip(*[x.shape for x in images]))
    result = torch.zeros((len(images), bh, bw, bd), dtype=torch.float)
    for i, im in enumerate(images):
        if im.dtype == torch.uint8:
            im = im.float() / 255.0
        h, w, d = im.shape
        dy, dx = random.randint(0, bh - h), random.randint(0, bw - w)
        result[i, dy : dy + h, dx : dx + w, :d] = im
    return result, seqs


# In[4]:


bucket = "pipe:curl -s -L http://storage.googleapis.com/nvdata-ocropus-words/"
shards_train = bucket + "uw3-word-{000000..000022}.tar"


def make_loader(spec, num_workers=4, batch_size=8, nshuffle=1000):
    dataset = wds.DataPipeline(
        wds.shardspec(spec),
        wds.split_by_worker,
        wds.detshuffle(100),
        wds.cached_tarfile_to_samples(verbose=True),
        wds.shuffle(nshuffle),
        wds.decode("torchrgb"),
        wds.to_tuple("png;jpg txt"),
        wds.batched(batch_size, collation_fn=collate4ocr),
    )
    loader = wds.WebLoader(
        dataset, num_workers=num_workers, batch_size=None, persistent_workers=True
    )
    loader = loader.with_epoch(nbatches=1000)
    return loader


dl = make_loader(shards_train)
image, info = next(iter(dl))
print(image.shape)
print(info)


# In[5]:


class MaxReduce(nn.Module):
    d: int

    def __init__(self, d: int):
        super().__init__()
        self.d = d

    def forward(self, x):
        return x.max(self.d)[0]


def make_text_model(noutput=1024, shape=(1, 3, 48, 300)):
    """Text recognition model using 2D LSTM and convolutions."""
    model = nn.Sequential(
        *combos.conv2d_block(32, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(48, 3, mp=(2, 1), repeat=2),
        *combos.conv2d_block(64, 3, mp=2, repeat=2),
        *combos.conv2d_block(96, 3, repeat=2),
        flex.Lstm2(100),
        # layers.Fun("lambda x: x.max(2)[0]"),
        MaxReduce(2),
        flex.ConvTranspose1d(400, 1, stride=2),
        flex.Conv1d(100, 3),
        flex.BatchNorm1d(),
        nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(100, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
    )
    flex.shape_inference(model, shape)
    return model


# In[6]:


def pack_for_ctc(seqs):
    """Pack a list of sequences for nn.CTCLoss."""
    allseqs = torch.cat(seqs).long()
    alllens = torch.tensor([len(s) for s in seqs]).long()
    return (allseqs, alllens)


def ctc_decode(probs, sigma=1.0, threshold=0.7, full=False):
    """A simple decoder for CTC-trained OCR recognizers.

    :probs: d x l sequence classification output
    """
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu().numpy()
    if probs.ndim == 3:
        return [ctc_decode(probs[i]) for i in range(len(probs))]
    assert probs.ndim == 2, probs.shape
    d, L = probs.shape
    assert d == 1024
    probs = probs.T
    delta = np.amax(abs(probs.sum(1) - 1))
    assert delta < 1e-4, f"input not normalized ({delta}); did you apply .softmax()?"
    probs = ndi.gaussian_filter(probs, (sigma, 0))
    probs /= probs.sum(1)[:, None]
    labels, n = ndi.label(probs[:, 0] < threshold)
    mask = np.tile(labels[:, None], (1, probs.shape[1]))
    mask[:, 0] = 0
    maxima = ndi.maximum_position(probs, mask, np.arange(1, np.amax(mask) + 1))
    if not full:
        return [c for r, c in sorted(maxima)]
    else:
        return [(r, c, probs[r, c]) for r, c in sorted(maxima)]


class TextModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = make_text_model()
        self.ctc_loss = nn.CTCLoss(zero_infinity=True)

    def forward(self, x):
        return self.model(x)

    def compute_loss(self, outputs, targets):
        targets, tlens = pack_for_ctc(targets)
        b, d, L = outputs.size()
        olens = torch.full((b,), L, dtype=torch.long)
        outputs = outputs.log_softmax(1)
        outputs = layers.reorder(outputs, "BDL", "LBD")
        assert tlens.size(0) == b
        assert tlens.sum() == targets.size(0)
        return self.ctc_loss(outputs.cpu(), targets.cpu(), olens.cpu(), tlens.cpu())

    def training_step(self, batch, batch_nb):
        images, texts = batch
        outputs = self.forward(images)
        if batch_nb % 100 == 0:
            preds = ctc_decode(outputs.softmax(1))
            preds = ["".join([chr(c) for c in p]) for p in preds]
            print("preds", preds)
        seqs = [torch.tensor([ord(c) for c in s]) for s in texts]
        loss = self.compute_loss(outputs.log_softmax(1), seqs)
        self.log("train/loss", loss, on_step=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=1e-4)


# In[ ]:

callbacks = [
    ModelCheckpoint(
        save_weights_only=True, verbose=True, mode="min", monitor="train/loss"
    ),
    LearningRateMonitor("epoch"),
]

text_model = TextModel()
train_loader = make_loader(shards_train)
trainer = pl.Trainer(
    default_root_dir="_checkpoints",
    gpus=[0],
    max_epochs=1000,
    callbacks=callbacks,
    progress_bar_refresh_rate=1,
)
trainer.fit(text_model, train_loader)


# In[ ]:
