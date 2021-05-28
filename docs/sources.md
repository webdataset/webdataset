```python
import webdataset as wds
import braceexpand
from torch.utils.data import IterableDataset
from webdataset import gopen
```

# Local and Remote Storage URLs

WebDataset refers to data sources using file paths or URLs. The following are all valid ways of referring to a data source:


```python
dataset = wds.WebDataset("dataset-000.tar")
dataset = wds.WebDataset("file:dataset-000.tar")
dataset = wds.WebDataset("http://server/dataset-000.tar")
```

An additional way of referring to data is using the `pipe:` scheme, so the following is also equivalent to the above references:


```python
dataset = wds.WebDataset("pipe:cat dataset-000.tar")
```

You can use the same notation for accessing data in cloud storage:


```python
dataset = wds.WebDataset("pipe:gsutil cat gs://somebucket/dataset-000.tar")
```

Note that access to standard web schemas are implemented using `curl`. That is, `http://server/dataset.tar` is internally simply treated like `pipe:curl -s -L 'http://server/dataset.tar'`. The use of `curl` to access Internet protocols actually is more efficient than using the built-in `http` library because it results in asynchronous name resolution and downloads.

File opening is handled by `webdataset.gopen.gopen`. This is a small function that just wraps standard Python file I/O and pipe capabilities.

You can define handlers for new schemes or override implementations for existing schemes by adding entries to `wds.gopen_schemes`:


```python
def gopen_gs(url, mode="rb", bufsize=8192):
    ...

gopen.gopen_schemes["gs"] = gopen_gs 
```

# Standard Input/Output

For the following examples, assume that we have a program called `image-classifier` that takes a WebDataset containing just JPEG files as input and produces a WebDataset containing JPEG files and their corresponding classifications in JSON format:

```Bash
image-classifier input-shard.tar --output=output-shard.tar --model=some-model.pth
```

As a special case, the string "-" refers to standard input (reading) or standard output (writing). This allows code using WebDataset to be used as part of pipes. This is useful, for example, inside Kubernetes containers with limited local storage. Assume that you store shards in Google Cloud and access it with `gsutil`. Using "-", you can simply write:

```Bash
gsutil cat gs://input-bucket/data-000174.tar | image-classifer - -o - | gsutil cp - gs://output-bucket/output-000174.tar
```

It's also useful to create shards on the fly using `tar` and extract the result immediately; this lets you use shard based programs directly for operating on individual files. For example, for the `image-classifier` program above, you can write:

```Bash
tar cf - *.jpg | shard-classifier - -o - | tar xvf - --include '.json'
```

This is the rough equivalent of:

```Bash
for fname in *.jpg; do
   image-classifier $fname > $(basename $fname .jpg).cls
done
```

# Multiple Shards and Mixing Datasets

The `WebDataset` and `ShardList` classes take either a string or a list of strings as an argument. When given a string, the string is expanded using `braceexpand`. Therefore, the following three datasets are equivalent:


```python
dataset = wds.WebDataset(["dataset-000.tar", "dataset-001.tar", "dataset-002.tar", "dataset-003.tar"])
dataset = wds.WebDataset("dataset-{000..003}.tar")
dataset = wds.WebDataset("file:dataset-{000..003}.tar")
```

For complex training problems, you may want to mix multiple datasets, where each dataset consists of multiple shards. A good way is to expand each shard spec individually using `braceexpand` and concatenate the lists. Then you can pass the result list as an argument to `WebDataset`.


```python
urls = (
    list(braceexpand.braceexpand("imagenet-{000000..000146}.tar")) +
    list(braceexpand.braceexpand("openimages-{000000..000547}.tar")) +
    list(braceexpand.braceexpand("custom-images-{000000..000999}.tar"))
)
print(len(urls))
dataset = wds.WebDataset(urls, shardshuffle=True).shuffle(10000).decode("torchrgb")
```

    1695


# Mixing Datsets with a Custom `IterableDataset` Class

For more complex sampling problems, you can also write sample processors. For example, to sample equally from several datasets, you could write something like this (the `Shorthands` and `Composable` base classes just add some convenience methods):


```python
class SampleEqually(IterableDataset, wds.Shorthands, wds.Composable):
    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
    def __iter__(self):
        sources = [iter(ds) for ds in self.datasets]
        while True:
            for source in sources:
                try:
                    yield next(source)
                except StopIteration:
                    return
```

Now we can mix samples from different sources in more complex ways:


```python
dataset1 = wds.WebDataset("imagenet-{000000..000146}.tar", shardshuffle=True).shuffle(1000).decode("torchrgb")
dataset2 = wds.WebDataset("openimages-{000000..000547}.tar", shardshuffle=True).shuffle(1000).decode("torchrgb")
dataset3 = wds.WebDataset("custom-images-{000000..000999}.tar", shardshuffle=True).shuffle(1000).decode("torchrgb")
dataset = SampleEqually([dataset1, dataset2, dataset3]).shuffle(1000)
```
