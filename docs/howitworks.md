```python
%pylab inline

import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
import webdataset as wds
from itertools import islice
```

    Populating the interactive namespace from numpy and matplotlib


# How it Works

WebDataset is powerful and it may look complex from the outside, but its structure is quite simple: most of
the code consists of functions mapping an input iterator to an output iterator:


```python
def add_noise(source, noise=0.01):
    for inputs, targets in source:
        inputs = inputs + noise * torch.randn_like(inputs)
        yield inputs, targets
```

To write new processing stages, a function like this is all you ever have to write. 
The rest is really bookkeeping: we need to be able
to repeatedly invoke functions like this for every epoch, and we need to chain them together.

To turn a function like that into an `IterableDataset`, and chain it with an existing dataset, you can use the `webdataset.Processor` class:


```python
dataset = ...
noisy_dataset = wds.Processor(dataset, add_noise, noise=0.02)
```

The `webdataset.WebDataset` class is just a wrapper for `Processor` with a default initial processing pipeline and some convenience methods.  Full expanded, the above pipeline can be written as:


```python
url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar"
url = f"pipe:curl -L -s {url} || true"
```


```python
dataset = wds.ShardList(url)
dataset = wds.Processor(dataset, wds.url_opener)
dataset = wds.Processor(dataset, wds.tar_file_expander)
dataset = wds.Processor(dataset, wds.group_by_keys)
dataset = wds.Processor(dataset, wds.shuffle, 100)
dataset = wds.Processor(dataset, wds.decode, wds.imagehandler("torchrgb"))
dataset = wds.Processor(dataset, wds.to_tuple, "png;jpg;jpeg", "json")
noisy_dataset = wds.Processor(dataset, add_noise, noise=0.02)

next(iter(noisy_dataset))[0].shape
```




    torch.Size([3, 683, 1024])



You can mix the shorthands with explicit constructions of processors:


```python
dataset = wds.WebDataset(url).shuffle(100).decode("torchrgb").to_tuple("png;jpg;jpeg", "json")
noisy_dataset = wds.Processor(dataset, add_noise, noise=0.02)

next(iter(noisy_dataset))[0].shape
```




    torch.Size([3, 768, 1024])



`wds.Processor` is just an `IterableDataset` instance; you can use it wherever you might use an `IterableDataset`. That means that all the functionality from the WebDataset library is available with other iterable sources.

Let's start by defining a simple SQL-based `IterableDataset`.


```python
import sqlite3
import pickle
import io
import torch
from torch.utils.data import IterableDataset

class SqlDataset(IterableDataset):
    def __init__(self, dbname):
        self.db = sqlite3.connect(dbname)
        self.db.execute("create table if not exists data (inputs blob, targets blob)")

    def add(self, inputs, targets):
        self.db.execute("insert into data (inputs, targets) values (?, ?)",
                        (wds.torch_dumps(inputs), wds.torch_dumps(targets)))
    def __iter__(self):
        query = "select inputs, targets from data"
        cursor = self.db.execute(query)
        for inputs, targets in cursor:
            yield wds.torch_loads(inputs), wds.torch_loads(targets)
            
    def __len__(self):
        return self.db.execute("select count(*) from data").fetchone()[0]
        
!rm -f test.db
dataset = SqlDataset("test.db")
size=32
for i in range(1000):
    dataset.add(torch.randn(3, size, size), torch.randn(3, size, size))
print(len(dataset), next(iter(dataset))[0].shape)
```

    1000 torch.Size([3, 32, 32])


Now we can chain this `IterableDataset` implementation with `webdataset.Processor`:


```python
dataset = wds.Processor(dataset, wds.shuffle, 100)
dataset = wds.Processor(dataset, wds.batched, 16)
noisy_dataset = wds.Processor(dataset, add_noise, noise=0.02)
print(next(iter(noisy_dataset))[0].shape)
```

    torch.Size([16, 3, 32, 32])

