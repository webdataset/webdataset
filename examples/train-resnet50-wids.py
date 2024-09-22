#!/usr/bin/env python
# coding: utf-8

import os
import resource
import time
import traceback
from collections import deque
from functools import partial

import numpy as np
import torch
import torchvision.transforms as transforms
import typer
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50

import wids


def enumerate_report(seq, delta, growth=1.0):
    last = 0
    count = 0
    for count, item in enumerate(seq):
        now = time.time()
        if now - last > delta:
            last = now
            yield count, item, True
        else:
            yield count, item, False
        delta *= growth


transform_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

transform_val = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def make_sample(sample, val=False):
    image = sample[".jpg"]
    label = sample[".cls"]
    if val:
        return transform_val(image), label
    else:
        return transform_train(image), label


def make_loader(split, num_workers=4):
    assert split in ["train", "val"], "split must be either 'train' or 'val'"

    if split == "train":
        dataset = wids.ShardListDataset(
            "gs://webdataset/fake-imagenet/imagenet-train.json",
            cache_dir="./_cache",
            keep=True,
        )
        dataset.add_transform(make_sample)
        sampler = wids.DistributedChunkedSampler(dataset, chunksize=1000, shuffle=True)
        sampler.set_epoch(0)
        loader = DataLoader(
            dataset, batch_size=64, num_workers=num_workers, sampler=sampler
        )
    else:
        dataset = wids.ShardListDataset(
            "gs://webdataset/fake-imagenet/imagenet-val.json",
            cache_dir="./_cache",
            keep=True,
        )
        dataset.add_transform(partial(make_sample, val=True))
        loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    return loader


def train_model(trainloader=None, num_epochs=10, model=None, report_every=10):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    losses, accuracies = deque(maxlen=100), deque(maxlen=100)

    for epoch in range(num_epochs):
        for i, data, verbose in enumerate_report(trainloader, report_every):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            pred = outputs.cpu().detach().argmax(dim=1, keepdim=True)
            correct = pred.eq(labels.cpu().view_as(pred)).sum().item()
            accuracy = correct / float(len(labels))

            losses.append(loss.item())
            accuracies.append(accuracy)

            if verbose and len(losses) > 5:
                print(
                    "[%d, %5d] loss: %.5f correct: %.5f"
                    % (epoch + 1, i + 1, np.mean(losses), np.mean(accuracies)),
                )

    print("Finished Training")


os.environ["TRACEBACK"] = "0"

app = typer.Typer()


@app.command()
def main(
    num_epochs: int = typer.Option(5, help="Number of epochs to train for"),
    report_every: float = typer.Option(5.0, help="Report every N seconds"),
    num_workers: int = typer.Option(4, help="Number of workers"),
):
    try:
        loader = make_loader("train", num_workers=num_workers)
        model = resnet50(pretrained=False)
        train_model(
            trainloader=loader,
            num_epochs=num_epochs,
            model=model,
            report_every=report_every,
        )
    except Exception as e:
        traceback.print_exc()


if __name__ == "__main__":
    app()
