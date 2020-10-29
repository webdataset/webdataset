[![Test](https://github.com/tmbdev/webdataset/workflows/Test/badge.svg)](https://github.com/tmbdev/webdataset/actions?query=workflow%3ATest)
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/tmbdev/webdataset/?ref=repository-badge)

# WebDataset

WebDataset is a PyTorch Dataset (IterableDataset) implementation providing
efficient access to datasets stored in POSIX tar archives and uses only sequential/streaming
data access. This brings substantial performance advantage in many compute environments, and it
is essential for very large scale training.

While WebDataset scales to very large problems, it also works well with smaller datasets and simplifies
creation, management, and distribution of training data for deep learning.

WebDataset implements standard PyTorch `IterableDataset` interface and works with the PyTorch `DataLoader`.
Access to datasets is as simple as:

```Python
import webdataset as wds

dataset = wds.WebDataset(url).shuffle(1000).decode("torchrgb").to_tuple("jpg;png", "json")
dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=16)

for inputs, outputs in dataloader:
    ...
```

In that code snippet, `url` can refer to a local file, a local HTTP server, a cloud storage object, an object
on an object store, or even the output of arbitrary command pipelines.

WebDataset fulfills a similar function to Tensorflow's TFRecord/tf.Example
classes, but it is much easier to adopt because it does not actually
require any kind of data conversion: data is stored in exactly the same
format inside tar files as it is on disk, and all preprocessing and data
augmentation code remains unchanged.

# Installation and Documentation

    $ pip install webdataset

For the Github version:

    $ pip install git+https://github.com/tmbdev/webdataset.git

Documentation: [ReadTheDocs](http://webdataset.readthedocs.io)

Examples:

- [loading videos](https://github.com/tmbdev/webdataset/blob/master/docs/video-loading-example.ipynb)
- [splitting raw videos into clips for training](https://github.com/tmbdev/webdataset/blob/master/docs/ytsamples-split.ipynb)
- [converting the Falling Things dataset](https://github.com/tmbdev/webdataset/blob/master/docs/falling-things-make-shards.ipynb)

# Introductory Videos

Here are some videos talking about WebDataset and large scale deep learning:

- [Introduction to Large Scale Deep Learning](https://www.youtube.com/watch?v=kNuA2wflygM)
- [Loading Training Data with WebDataset](https://www.youtube.com/watch?v=mTv_ePYeBhs)
- [Creating Datasets in WebDataset Format](https://www.youtube.com/watch?v=v_PacO-3OGQ)
- [Tools for Working with Large Datasets](https://www.youtube.com/watch?v=kIv8zDpRUec)

# Using WebDataset

WebDataset reads dataset that are stored as tar files, with the simple convention that files that belong together and make up a training sample share the same basename. WebDataset can read files from local disk or from any pipe, which allows it to access files using common cloud object stores.


```bash
%%bash
curl -s http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar | tar tf - | sed 10q
```

    e39871fd9fd74f55.jpg
    e39871fd9fd74f55.json
    f18b91585c4d3f3e.jpg
    f18b91585c4d3f3e.json
    ede6e66b2fb59aab.jpg
    ede6e66b2fb59aab.json
    ed600d57fcee4f94.jpg
    ed600d57fcee4f94.json
    ff47e649b23f446d.jpg
    ff47e649b23f446d.json



```python
%pylab inline

import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
import webdataset as wds
from itertools import islice

url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar"
url = f"pipe:curl -L -s {url} || true"
```

    Populating the interactive namespace from numpy and matplotlib


For starters, let's use the `webdataset.Dataset` class to illustrate how the `webdataset` library works.


```python
dataset = wds.WebDataset(url)

for sample in islice(dataset, 0, 3):
    for key, value in sample.items():
        print(key, repr(value)[:50])
    print()
```

    __key__ 'e39871fd9fd74f55'
    jpg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x01
    json b'[{"ImageID": "e39871fd9fd74f55", "Source": "xcli
    
    __key__ 'f18b91585c4d3f3e'
    jpg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00
    json b'[{"ImageID": "f18b91585c4d3f3e", "Source": "acti
    
    __key__ 'ede6e66b2fb59aab'
    jpg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00
    json b'[{"ImageID": "ede6e66b2fb59aab", "Source": "acti
    


There are common processing stages you can add to a dataset to make it a drop-in replacement for any existing dataset. For convenience, common operations are available through a "fluent" interface (as chained method calls).


```python
dataset = (
    wds.WebDataset(url)
    .shuffle(100)
    .decode("rgb")
    .to_tuple("jpg;png", "json")
)

for image, data in islice(dataset, 0, 3):
    print(image.shape, image.dtype, type(data))
```

    (701, 1024, 3) float32 <class 'list'>
    (1024, 768, 3) float32 <class 'list'>
    (1024, 683, 3) float32 <class 'list'>


The `webdataset.Dataset` class has some common operations:

- `shuffle(n)`: shuffle the dataset with a buffer of size `n`; also shuffles shards (see below)
- `decode(decoder, ...)`: automatically decode files (most commonly, you can just specify `"pil"`, `"rgb"`, `"rgb8"`, `"rgbtorch"`, etc.)
- `rename(new="old1;old2", ...)`: rename fields
- `map(f)`: apply `f` to each sample
- `map_dict(key=f, ...)`: apply `f` to its corresponding key
- `map_tuple(f, g, ...)`: apply `f`, `g`, etc. to their corresponding values in the tuple
- `pipe(f)`: `f` should be a function that takes an iterator and returns a new iterator

Stages commonly take a `handler=` argument, which is a function that gets called when there is an exception; you can write whatever function you want, but common functions are:

- `webdataset.ignore_and_stop`
- `webdataset.ignore_and_continue`
- `webdataset.warn_and_stop`
- `webdataset.warn_and_continue`
- `webdataset.reraise_exception`


Here is an example that uses `torchvision` data augmentation the same way you might use it with a `FileDataset`.


```python
def identity(x):
    return x

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
    wds.WebDataset(url)
    .shuffle(100)
    .decode("pil")
    .to_tuple("jpg;png", "json")
    .map_tuple(preproc, identity)
)

for image, data in islice(dataset, 0, 3):
    print(image.shape, image.dtype, type(data))
```

    torch.Size([3, 224, 224]) torch.float32 <class 'list'>
    torch.Size([3, 224, 224]) torch.float32 <class 'list'>
    torch.Size([3, 224, 224]) torch.float32 <class 'list'>


You can find the full PyTorch ImageNet sample code converted to WebDataset at [tmbdev/pytorch-imagenet-wds](http://github.com/tmbdev/pytorch-imagenet-wds)

# How it Works

WebDataset is powerful and it may look complex from the outside, but its structure is quite simple: most of
the code consists of functions mapping an input iterator to an output iterator:

```Python
def add_noise(source, noise=0.01):
    for inputs, targets in source:
        inputs = inputs + noise * torch.randn_like(inputs)
        yield inputs, targets
```

To write new processing stages, a function like this is all you ever have to write. 
The rest is really bookkeeping: we need to be able
to repeatedly invoke functions like this for every epoch, and we need to chain them together.

To turn a function like that into an `IterableDataset`, and chain it with an existing dataset, you can use the `webdataset.Processor` class:

```Python
noisy_dataset = webdataset.Processor(add_noise, noise=0.02)(dataset)
```

The `webdataset.WebDataset` class is just a wrapper for `Processor` with a default initial processing pipeline and some convenience methods.  Full expanded, the above pipeline can be written as:

```Python
dataset = wds.ShardList(url)
dataset = wds.Processor(wds.url_opener)(dataset)
dataset = wds.Processor(wds.tar_file_expander)(dataset)
dataset = wds.Processor(wds.group_by_keys)(dataset)
dataset = wds.Processor(wds.shuffle, 100)(dataset)
dataset = wds.Processor(wds.decode, "torchrgb")(dataset)
noisy_dataset = wds.Processor(wds.augment_sample, noise=0.02)(dataset)
```

`wds.Processor` is just an `IterableDataset` instance; you can use it wherever you might use an `IterableDataset` and mix the two styles freely.

For example, you can reuse WebDataset processors with existing `IterableDataset` implementations, for example if you want shuffling, caching, or batching with them. Let's say you have a class `MySqlIterableDataset` that iterates over samples from an SQL database and you want to shuffle and batch the results. You can write:

```Python
dataset = MySqlIterableDataset(database_connection)
dataset = wds.Processor(wds.shuffle, 100)(dataset)
dataset = wds.Processor(wds.batch, 16)(dataset)
noisy_dataset = wds.Processor(wds.augment_sample, noise=0.02)(dataset)
```

# Sharding and Parallel I/O

WebDataset datasets are usually split into many shards; this is both to achieve parallel I/O and to shuffle data.

Sets of shards can be given as a list of files, or they can be written using the brace notation, as in `openimages-train-{000000..000554}.tar`.

For example, the OpenImages dataset consists of 554 shards, each containing about 1 Gbyte of images. You can open the entire dataset as follows.


```python
url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-{000000..000554}.tar"
url = f"pipe:curl -L -s {url} || true"
dataset = (
    wds.WebDataset(url, shardshuffle=True)
    .shuffle(100)
    .decode("pil")
    .to_tuple("jpg;png", "json")
    .map_tuple(preproc, identity)
)
```

Note the explicit use of both `shardshuffle=True` (for shuffling the shards) and the `.shuffle` processor (for shuffling samples inline).

When used with a standard Torch `DataLoader`, this will now perform parallel I/O and preprocessing.


```python
dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=16)
images, targets = next(iter(dataloader))
images.shape
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    <ipython-input-7-45431386c528> in <module>
          1 dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=16)
    ----> 2 images, targets = next(iter(dataloader))
          3 images.shape


    ~/proj/webdataset/venv/lib/python3.8/site-packages/torch/utils/data/dataloader.py in __next__(self)
        433         if self._sampler_iter is None:
        434             self._reset()
    --> 435         data = self._next_data()
        436         self._num_yielded += 1
        437         if self._dataset_kind == _DatasetKind.Iterable and \


    ~/proj/webdataset/venv/lib/python3.8/site-packages/torch/utils/data/dataloader.py in _next_data(self)
       1083             else:
       1084                 del self._task_info[idx]
    -> 1085                 return self._process_data(data)
       1086 
       1087     def _try_put_index(self):


    ~/proj/webdataset/venv/lib/python3.8/site-packages/torch/utils/data/dataloader.py in _process_data(self, data)
       1109         self._try_put_index()
       1110         if isinstance(data, ExceptionWrapper):
    -> 1111             data.reraise()
       1112         return data
       1113 


    ~/proj/webdataset/venv/lib/python3.8/site-packages/torch/_utils.py in reraise(self)
        426             # have message field
        427             raise self.exc_type(message=msg)
    --> 428         raise self.exc_type(msg)
        429 
        430 


    RuntimeError: Caught RuntimeError in DataLoader worker process 0.
    Original Traceback (most recent call last):
      File "/home/tmb/proj/webdataset/venv/lib/python3.8/site-packages/torch/utils/data/_utils/worker.py", line 198, in _worker_loop
        data = fetcher.fetch(index)
      File "/home/tmb/proj/webdataset/venv/lib/python3.8/site-packages/torch/utils/data/_utils/fetch.py", line 35, in fetch
        return self.collate_fn(data)
      File "/home/tmb/proj/webdataset/venv/lib/python3.8/site-packages/torch/utils/data/_utils/collate.py", line 83, in default_collate
        return [default_collate(samples) for samples in transposed]
      File "/home/tmb/proj/webdataset/venv/lib/python3.8/site-packages/torch/utils/data/_utils/collate.py", line 83, in <listcomp>
        return [default_collate(samples) for samples in transposed]
      File "/home/tmb/proj/webdataset/venv/lib/python3.8/site-packages/torch/utils/data/_utils/collate.py", line 81, in default_collate
        raise RuntimeError('each element in list of batch should be of equal size')
    RuntimeError: each element in list of batch should be of equal size



The recommended way of using `IterableDataset` with `DataLoader` is to do the batching explicitly in the `Dataset`. You can also set a nominal length for a dataset.


```python
url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-{000000..000554}.tar"
url = f"pipe:curl -L -s {url} || true"
bs = 20

dataset = (
    wds.WebDataset(url, length=int(1e9) // bs)
    .shuffle(100)
    .decode("pil")
    .to_tuple("jpg;png", "json")
    .map_tuple(preproc, identity)
    .batched(20)
)

dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=None)
images, targets = next(iter(dataloader))
images.shape
```

The `ResizedDataset` is also helpful for connecting iterable datasets to `DataLoader`: it lets you set both a nominal and an actual epoch size; it will repeatedly iterate through the entire dataset and return data in chunks with the given epoch size.

# Data Decoding

Data decoding is a special kind of transformations of samples. You could simply write a decoding function like this:

```Python
def my_sample_decoder(sample):
    result = dict(__key__=sample["__key__"])
    for key, value in sample.items():
        if key == "png" or key.endswith(".png"):
            result[key] = mageio.imread(io.BytesIO(value))
        elif ...:
            ...
    return result

dataset = wds.Processor(wds.map, my_sample_decoder)(dataset)
```

This gets tedious, though, and it also unnecessarily hardcodes the sample's keys into the processing pipeline. To help with this, there is a helper class that simplifies this kind of code. The primary use of `Decoder` is for decoding compressed image, video, and audio formats, as well as unzipping `.gz` files.

Here is an example of automatically decoding `.png` images with `imread` and using the default `torch_video` and `torch_audio` decoders for video and audio:

```Python
def my_png_decoder(key, value):
    if not key.endswith(".png"):
        return None
    assert isinstance(value, bytes)
    return imageio.imread(io.BytesIO(value))

dataset = wds.Decoder(my_png_decoder, wds.torch_video, wds.torch_audio)(dataset)
```

You can use whatever criteria you like for deciding how to decode values in samples. When used with standard `WebDataset` format files, the keys are the full extensions of the file names inside a `.tar` file. For consistency, it's recommended that you primarily rely on the extensions (e.g., `.png`, `.mp4`) to decide which decoders to use. There is a special helper function that simplifies this:

```Python
def my_decoder(value):
    return imageio.imread(io.BytesIO(value))
    
dataset = wds.Decoder(wds.handle_extension(".png", my_decoder))(dataset)
```

# Utility for "Smaller" Datasets and Desktop Computing

WebDataset is an ideal solution for training on petascale datasets kept on high performance distributed data stores like AIStore, AWS/S3, and Google Cloud. Compared to data center GPU servers, desktop machines have much slower network connections, but training jobs on desktop machines often also use much smaller datasets.

WebDataset also is very useful for such smaller datasets, and it can easily be used for developing and testing on small datasets and then scaling up to large datasets by simply using more shards.

There are several different ways in which WebDataset is useful for desktop usage.

## Direct Copying of Shards

Let's take the OpenImages dataset as an example; it's half a terabyte large. For development and testing, you may not want to download the entire dataset, but you may also not want to use the dataset remotely. With WebDataset, you can just download a small number of shards and use them during development.


```python
!curl -L -s http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar > /tmp/openimages-train-000000.tar
```


```python
dataset = wds.WebDataset("/tmp/openimages-train-000000.tar")
repr(next(iter(dataset)))[:200]
```




    "{'__key__': 'e39871fd9fd74f55', 'jpg': b'\\xff\\xd8\\xff\\xe0\\x00\\x10JFIF\\x00\\x01\\x01\\x01\\x01:\\x01:\\x00\\x00\\xff\\xdb\\x00C\\x00\\x06\\x04\\x05\\x06\\x05\\x04\\x06\\x06\\x05\\x06\\x07\\x07\\x06\\x08\\n\\x10\\n\\n\\t\\t\\n\\x14\\x0e"



Note that the WebDataset class works the same way on local files as it does on remote files. Furthermore, unlike other kinds of dataset formats and archive formats, downloaded datasets are immediately useful and don't need to be unpacked.

## Automatic Shard Caching

Downloading a few shards manually is useful for development and testing. But WebDataset permits us to automate downloading and caching of shards. This is accomplished by giving a `cache_dir` argument to the WebDataset constructor.

Here, we make two passes through the dataset, using the cached version on the second pass.

Note that caching happens in parallel with iterating through the dataset. This means that if you write a WebDataset-based I/O pipeline, training starts immediately; the training job does not have to wait for any shards to download first.

Automatic shard caching is useful for distributing deep learning code, for academic computer labs, and for cloud computing.


```python
!rm -rf ./cache

dataset = wds.WebDataset(url, cache_dir="./cache")

print("=== first pass")

for sample in dataset:
    pass

print("=== second pass")

for i, sample in enumerate(dataset):
    for key, value in sample.items():
        print(key, repr(value)[:50])
    print()
    if i >= 3: break
        
!ls -l ./cache
```

    [caching <webdataset.gopen.Pipe object at 0x7f767465bf40> at ./cache/82d02c03-bb0d-3796-81b8-03deb601ab2b.~3996278~ ]


    === first pass


    [done caching ./cache/82d02c03-bb0d-3796-81b8-03deb601ab2b ]
    [finished ./cache/82d02c03-bb0d-3796-81b8-03deb601ab2b]
    [opening cached ./cache/82d02c03-bb0d-3796-81b8-03deb601ab2b ]


    === second pass
    __key__ 'e39871fd9fd74f55'
    jpg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x01
    json b'[{"ImageID": "e39871fd9fd74f55", "Source": "xcli
    
    __key__ 'f18b91585c4d3f3e'
    jpg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00
    json b'[{"ImageID": "f18b91585c4d3f3e", "Source": "acti
    
    __key__ 'ede6e66b2fb59aab'
    jpg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00
    json b'[{"ImageID": "ede6e66b2fb59aab", "Source": "acti
    
    __key__ 'ed600d57fcee4f94'
    jpg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x01
    json b'[{"ImageID": "ed600d57fcee4f94", "Source": "acti
    
    total 987924
    -rw-rw-r-- 1 tmb tmb 1011630080 Oct 28 22:40 82d02c03-bb0d-3796-81b8-03deb601ab2b


## Automatic Sample Caching

WebDataset also provides a way of caching training samples directly. This works with samples coming from any IterableDataset as input. The cache is stored in an SQLite3 database.

Sample-based caching is implemented by the `DBCache` class. You specify a filename for the database and the maximum number of samples you want to cache. Samples will initially be read from the original IterableDataset, but after either the samples run out or the maximum number of samples has been reached, subsequently, samples will be served from the database cache stored on local disk. The database cache persists between invocations of the job.

Automatic sample caching is useful for developing and testing deep learning jobs, as well as for caching data coming from slow IterableDataset sources, such as network-based database connections or other slower data sources.

You can add or remove the caching stage at will (e.g., to switch between testing and production), without affecting the rest of the code at all.


```python
!rm -rf ./cache.db

dataset = wds.WebDataset(url).compose(wds.DBCache, "./cache.db", 1000)

print("=== first pass")

for sample in dataset:
    pass

print("=== second pass")

for i, sample in enumerate(dataset):
    for key, value in sample.items():
        print(key, repr(value)[:50])
    print()
    if i >= 3: break
        
!ls -l ./cache.db
```

    [DBCache opened ./cache.db size 1000 total 0]
    === first pass
    [DBCache total 0 size 1000 more caching]
    [DBCache finished caching total 1000 (size 1000)]
    === second pass
    [DBCache starting dbiter total 1000 size 1000]
    __key__ 'e39871fd9fd74f55'
    jpg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x01
    json b'[{"ImageID": "e39871fd9fd74f55", "Source": "xcli
    
    __key__ 'f18b91585c4d3f3e'
    jpg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00
    json b'[{"ImageID": "f18b91585c4d3f3e", "Source": "acti
    
    __key__ 'ede6e66b2fb59aab'
    jpg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00
    json b'[{"ImageID": "ede6e66b2fb59aab", "Source": "acti
    
    __key__ 'ed600d57fcee4f94'
    jpg b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x01
    json b'[{"ImageID": "ed600d57fcee4f94", "Source": "acti
    
    -rw-r--r-- 1 tmb tmb 485199872 Oct 28 22:52 ./cache.db


# Creating a WebDataset

## Using `tar`

Since WebDatasets are just regular tar files, you can usually create them by just using the `tar` command. All you have to do is to arrange for any files that should be in the same sample to share the same basename. Many datasets already come that way. For those, you can simply create a WebDataset with

```
$ tar --sort=name -cf dataset.tar dataset/
```

If your dataset has some other directory layout, you may need a different file name in the archive from the name on disk. You can use the `--transform` argument to GNU tar to transform file names. You can also use the `-T` argument to read the files from a text file and embed other options in that text file.

## The `tarp create` Command

The [`tarp`](https://github.com/tmbdev/tarp) command is a little utility for manipulating `tar` archives. Its `create` subcommand makes it particularly simple to construct tar archives from files. The `tarp create` command takes a recipe for building
a tar archive that contains lines of the form:

```
archive-name-1 source-name-1
archive-name-2 source-name-2
...
```

The source name can either be a file, "text:something", or "pipe:something".

## Programmatically in Python

You can also create a WebDataset with library functions in this library:

- `webdataset.TarWriter` takes dictionaries containing key value pairs and writes them to disk
- `webdataset.ShardWriter` takes dictionaries containing key value pairs and writes them to disk as a series of shards

Here is a quick way of converting an existing dataset into a WebDataset; this will store all tensors as Python pickles:

```Python
sink = wds.TarWriter("dest.tar")
dataset = open_my_dataset()
for index, (input, output) in dataset:
    sink.write({
        "__key__": "sample%06d" % index,
        "input.pyd": input,
        "output.pyd": output,
    })
sink.close()
```

Storing data as Python pickles allows most common Python datatypes to be stored, it is lossless, and the format is fast to decode.
However, it is uncompressed and cannot be read by non-Python programs. It's often better to choose other storage formats, e.g.,
taking advantage of common image compression formats.

If you know that the input is an image and the output is an integer class, you can also write something like this:

```Python
sink = wds.TarWriter("dest.tar")
dataset = open_my_dataset()
for index, (input, output) in dataset:
    assert input.ndim == 3 and input.shape[2] == 3
    assert input.dtype = np.float32 and np.amin(input) >= 0 and np.amax(input) <= 1
    assert type(output) == int
    sink.write({
        "__key__": "sample%06d" % index,
        "input.jpg": input,
        "output.cls": output,
    })
sink.close()
```

The `assert` statements in that loop are not necessary, but they document and illustrate the expectations for this
particular dataset. Generally, the ".jpg" encoder can actually encode a wide variety of array types as images. The
".cls" encoder always requires an integer for encoding.

Here is how you can use `TarWriter` for writing a dataset without using an encoder:

```Python
sink = wds.TarWriter("dest.tar", encoder=False)
for basename in basenames:
    with open(f"{basename}.png", "rb") as stream):
        image = stream.read()
    cls = lookup_cls(basename)
    sample = {
        "__key__": basename,
        "input.png": image,
        "target.cls": cls
    }
    sink.write(sample)
sink.close()
```

Since no encoder is used, if you want to be able to read this data with the default decoder, `image` must contain a byte string corresponding to a PNG image (as indicated by the ".png" extension on its dictionary key), and `cls` must contain an integer encoded in ASCII (as indicated by the ".cls" extension on its dictionary key).

# Writing Filters and Offline Augmentation

Webdataset can be used for filters and offline augmentation of datasets. Here is a complete example that pre-augments a shard and extracts class labels.


```python
def extract_class(data):
    # mock implementation
    return 0

def augment_wds(input, output, maxcount=999999999):
    src = (
        wds.WebDataset(input)
        .decode("pil")
        .to_tuple("__key__", "jpg;png", "json")
        .map_tuple(identity, preproc, identity)
    )
    with wds.TarWriter(output) as dst:
        for key, image, data in islice(src, 0, maxcount):
            print(key)
            image = image.numpy().transpose(1, 2, 0)
            image -= amin(image)
            image /= amax(image)
            sample = {
                "__key__": key,
                "png": image,
                "cls": extract_class(data)
            }
            dst.write(sample)
```

Now run the augmentation pipeline:


```python
url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar"
url = f"pipe:curl -L -s {url} || true"
augment_wds(url, "_temp.tar", maxcount=5)
```

    e39871fd9fd74f55
    f18b91585c4d3f3e
    ede6e66b2fb59aab
    ed600d57fcee4f94
    ff47e649b23f446d


To verify that things worked correctly, let's look at the output file:


```bash
%%bash
tar tf _temp.tar
```

    e39871fd9fd74f55.cls
    e39871fd9fd74f55.png
    f18b91585c4d3f3e.cls
    f18b91585c4d3f3e.png
    ede6e66b2fb59aab.cls
    ede6e66b2fb59aab.png
    ed600d57fcee4f94.cls
    ed600d57fcee4f94.png
    ff47e649b23f446d.cls
    ff47e649b23f446d.png


If you want to preprocess the entire OpenImages dataset with a process like this, you can use your favorite job queueing or worflow system.

For example, using Dask, you could process all 554 shards in parallel using code like this:

```Python
shards = braceexpand.braceexpand("{000000..000554}")
inputs = [f"gs://bucket/openimages-{shard}.tar" for shard in shards]
outputs = [f"gs://bucket2/openimages-augmented-{shard}.tar" for shard in shards]
results = [dask.delayed(augment_wds)(args) for args in zip(inputs, outputs)]
dask.compute(*results)
```

Note that the data is streaming from and to Google Cloud Storage buckets, so very little local storage is required on each worker.

For very large scale processing, it's easiest to submit separate jobs to a Kubernetes cluster using the Kubernetes `Job` template, or using a workflow engine like Argo.

# Common Processors

Recall that processors are just functions that take an iterator as an argument and return another iterator (they get turned into an `IterableDataset` by wrapping them with `webdataset.Processor`).

The basic processors used for decoding WebDataset format files are:

- `ShardList` generates an iterator of URLs; you may need to modify this to generate different shard subsets for multinode computation
- `url_opener` opens each URL in turn and returns a bytestream
- `tarfile_expander` takes each bytestream in its input and generates an iterator over the tar files in that stream
- `group_by_keys` groups files together into training samples

In addition to those tar-file related processing functions, there are additional processing functions for common dataset operations:

- `info` -- provide info about training samples from an IterableDataset
- `shuffle` -- shuffle the IterableDataset
- `select` -- select samples from the IterableDataset
- `decode` -- apply decoding functions to each key-value pair in a sample
- `map` -- apply a mapping function to each sample
- `rename` -- rename the keys in key-value pairs in a sample
- `map_dict` -- apply functions to selected key-value pairs
- `to_tuple` -- convert dictionary-based samples to tuple-based samples
- `map_tuple` -- apply functions to the elements of each tuple-based sample
- `batched` -- batch samples together using a collation functions
- `unbatched` -- unbatch batched samples

Use `help(wds.batched)` etc. to get more information on each function.

Whether you prefer `WebDataset` or `Dataset` is a matter of style.

# Splitting Shards across Nodes and Workers

Unlike traditional PyTorch `Dataset` instances, `WebDataset` splits data across nodes at the shard level. This functionality is handled inside the `ShardList` class. To customize splitting data across nodes, you can either write your own `ShardList` function, or you can give the `ShardList` class a `splitter=` argument when you create it. The `splitter` function should select a subset of shards from the given list of shards based on the current node and worker ID.

# Data Sources

The `ShardList` class takes either a string or a list of URLs as an argument. If it is given a string, the string is expanded using the `braceexpand` library. So, the following are equivalent:

```Python
ShardList("dataset-{000..001}.tar")
ShardList(["dataset-000.tar", "dataset-001.tar"])
```

The url strings in a shard list are handled by default by the `webdataset.url_opener` filter. It recognizes three simple kinds of strings: "-", "/path/to/file", and "pipe:command":

- the string "-", referring to stdin
- a UNIX path, opened as a regular file
- a URL-like string with the schema "pipe:"; such URLs are opened with `subprocess.Popen`. For example:
    - `pipe:curl -s -L http://server/file` accesses a file via HTTP
    - `pipe:gsutil cat gs://bucket/file` accesses a file on GCS
    - `pipe:az cp --container bucket --name file --file /dev/stdout` accesses a file on Azure
    - `pipe:ssh host cat file` accesses a file via `ssh`

It might seem at first glance to be "more efficient" to use built-in Python libraries for accessing object stores rather than subprocesses, but efficient object store access from Python really requires spawning a separate process anyway, so this approach to accessing object stores is not only convenient, it also is as efficient as we can make it in Python.

# Related Libraries and Software

The [AIStore](http://github.com/NVIDIA/aistore) server provides an efficient backend for WebDataset; it functions like a combination of web server, content distribution network, P2P network, and distributed file system. Together, AIStore and WebDataset can serve input data from rotational drives distributed across many servers at the speed of local SSDs to many GPUs, at a fraction of the cost. We can easily achieve hundreds of MBytes/s of I/O per GPU even in large, distributed training jobs.

The [tarproc](http://github.com/tmbdev/tarproc) utilities provide command line manipulation and processing of webdatasets and other tar files, including splitting, concatenation, and `xargs`-like functionality.

The [tensorcom](http://github.com/tmbdev/tensorcom/) library provides fast three-tiered I/O; it can be inserted between [AIStore](http://github.com/NVIDIA/aistore) and [WebDataset](http://github.com/tmbdev/webdataset) to permit distributed data augmentation and I/O. It is particularly useful when data augmentation requires more CPU than the GPU server has available.

You can find the full PyTorch ImageNet sample code converted to WebDataset at [tmbdev/pytorch-imagenet-wds](http://github.com/tmbdev/pytorch-imagenet-wds)

