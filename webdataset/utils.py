"""Miscellaneous utility functions."""

import re
import importlib
import itertools as itt


def identity(x):
    """Return the argument as is."""
    return x


def safe_eval(s, expr="{}"):
    """Evaluate the given expression more safely."""
    if re.sub("[^A-Za-z0-9_]", "", s) != s:
        raise ValueError(f"safe_eval: illegal characters in: '{s}'")
    return eval(expr.format(s))


def lookup_sym(sym, modules):
    """Look up a symbol in a list of modules."""
    for mname in modules:
        module = importlib.import_module(mname, package="webdataset")
        result = getattr(module, sym, None)
        if result is not None:
            return result
    return None


def repeatedly0(loader, nepochs=999999999, nbatches=999999999999):
    """Repeatedly returns batches from a DataLoader."""
    for epoch in range(nepochs):
        for sample in itt.islice(loader, nbatches):
            yield sample


def guess_batchsize(batch):
    """Guess the batch size by looking at the length of the first element in a tuple."""
    return len(batch[0])


def repeatedly(source, nepochs=None, nbatches=None, nsamples=None, batchsize=guess_batchsize):
    """Repeatedly yield samples from an iterator."""
    epoch = 0
    batch = 0
    total = 0
    while True:
        for sample in source:
            yield sample
            batch += 1
            if nbatches is not None and batch >= nbatches:
                return
            if nsamples is not None:
                total += guess_batchsize(sample)
                if total >= nsamples:
                    return
        epoch += 1
        if nepochs is not None and epoch >= nepochs:
            return
