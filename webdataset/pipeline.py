#%%
import os
import sys
import time
import random
import yaml
from dataclasses import dataclass
from typing import List

import braceexpand

#from .pytorch import IterableDataset
from .pytorch import IterableDataset
from .composable import Composable
from .filters import Curried


def stage(f, *args, **kw):
    return Curried(f)(*args, **kw)

class DataPipeline(IterableDataset):

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.pipeline = []
        for arg in args:
            if arg is None:
                continue
            if isinstance(arg, list):
                self.pipeline.extend(arg)
            else:
                self.pipeline.append(arg)

    def invoke(self, f, *args, **kwargs):
        if isinstance(f, Composable) and len(args) == 1:
            f.source_(*args)
            return f
        if isinstance(f, IterableDataset) and len(args) == 0:
            return iter(f)
        if isinstance(f, list):
            return iter(f)
        if callable(f):
            result = f(*args, **kwargs)
            return result
        raise ValueError(f"{f}: not a valid pipeline stage")


    def __iter__(self):
        source = self.invoke(self.pipeline[0])
        for step in self.pipeline[1:]:
            source = self.invoke(step, source)
        return source

    def stage(self, i):
        return self.pipeline[i]
