# Sharding, Parallel I/O, and `DataLoader`

WebDataset datasets are usually split into many shards; this is both to achieve parallel I/O and to shuffle data.


```python
%pylab inline

import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
import webdataset as wds
from itertools import islice
```

    Populating the interactive namespace from numpy and matplotlib


Sets of shards can be given as a list of files, or they can be written using the brace notation, as in `openimages-train-{000000..000554}.tar`. For example, the OpenImages dataset consists of 554 shards, each containing about 1 Gbyte of images. You can open the entire dataset as follows (note the explicit use of both `shardshuffle=True` (for shuffling the shards and the `.shuffle` processor for shuffling samples inline).



```python
url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-{000000..000554}.tar"
url = f"pipe:curl -L -s {url} || true"

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225])

preproc = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

dataset = (
    wds.WebDataset(url, shardshuffle=True)
    .shuffle(100)
    .decode("pil")
    .to_tuple("jpg;png", "json")
    .map_tuple(preproc)
)

x, y = next(iter(dataset))
print(x.shape, str(y)[:50])
```

    torch.Size([3, 224, 224]) [{'ImageID': '19a7594f418fe39e', 'Source': 'xclick


When used with a standard Torch `DataLoader`, this will would perform parallel I/O and preprocessing. However, the recommended way of using `IterableDataset` with `DataLoader` is to do the batching explicitly in the `Dataset`:


```python
batch_size = 20
dataloader = torch.utils.data.DataLoader(dataset.batched(batch_size), num_workers=4, batch_size=None)
images, targets = next(iter(dataloader))
images.shape
```




    torch.Size([20, 3, 224, 224])



# Explicit Dataset Sizes

Ideally, you shouldn't use `len(dataset)` or `len(loader)` at all in your training loop. However, some code may use calls to the `len(.)` function. `WebDataset` generally propagates such calls back through the chain of dataset processors. Generally, `IterableDataset` implementations don't have a size, but you can specify an explicit size using the `length=` argument to `WebDataset`.

You can also use the `ResizedDataset` class to force an `IterableDataset` to have a specific epoch length and (if desired) set a separate nominal epoch length.


```python

```
