[![Test](https://github.com/tmbdev/webdataset/workflows/Test/badge.svg)](https://github.com/tmbdev/webdataset/actions?query=workflow%3ATest)
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/tmbdev/webdataset/?ref=repository-badge)

# WebDataset

WebDataset is a PyTorch Dataset (IterableDataset) implementation providing
efficient access to datasets stored in POSIX tar archives.

Storing data in POSIX tar archives greatly speeds up I/O operations on
rotational storage and on networked file systems because it permits all
I/O operations to operate as large sequential reads and writes.

WebDataset fulfills a similar function to Tensorflow's TFRecord/tf.Example
classes, but it is much easier to adopt because it does not actually
require any kind of data conversion: data is stored in exactly the same
format inside tar files as it is on disk, and all preprocessing and data
augmentation code remains unchanged.

# Installation and Documentation

```Bash
    $ pip install webdataset
```

For the Github version:

```Bash
    $ pip install git+https://github.com/tmbdev/webdataset.git
```

Documentation: [ReadTheDocs](http://webdataset.readthedocs.io)

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
from torchvision import transforms
import webdataset as wds
from itertools import islice

url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar"
url = f"pipe:curl -L -s {url} || true"
```

    Populating the interactive namespace from numpy and matplotlib


WebDatasets are an implementation of PyTorch `IterableDataset` and fully compatible with PyTorch input pipelines. By default, WebDataset just iterates through the files in a tar file without decoding anything, returning related files in each sample.


```python
dataset = wds.Dataset(url)

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
    wds.Dataset(url)
    .shuffle(100)
    .decode()
    .to_tuple("jpg;png", "json")
)

for image, data in islice(dataset, 0, 3):
    print(image.shape, image.dtype, type(data))
```

    (762, 1024, 3) float32 <class 'list'>
    (768, 1024, 3) float32 <class 'list'>
    (1024, 768, 3) float32 <class 'list'>


Common operations:

- `shuffle(n)`: shuffle the dataset with a buffer of size `n`; also shuffles shards (see below)
- `decode([type])`: automatically decode files; the `type` determines desired outputs for images, video, and audio: `pil`, `rgb`, `rgb8`, `rgbtorch`, etc.
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
    wds.Dataset(url)
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


# Sharding and Parallel I/O

In order to be able to shuffle data better and to process and load data in parallel, it is a good idea to shard it; that is, to split up the dataset into several `.tar` files.

WebDataset uses standard UNIX brace notation for sharded dataset. For example, the OpenImages dataset consists of 554 shards, each containing about 1 Gbyte of images. You can open the entire dataset as follows.


```python
url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-{000000..000554}.tar"
url = f"pipe:curl -L -s {url} || true"
dataset = (
    wds.Dataset(url)
    .shuffle(100)
    .decode("pil")
    .to_tuple("jpg;png", "json")
    .map_tuple(preproc, identity)
)
```

When used with a standard Torch `DataLoader`, this will now perform parallel I/O and preprocessing.


```python
dataloader = torch.utils.data.DataLoader(dataset, num_workers=4, batch_size=16)
images, targets = next(iter(dataloader))
images.shape
```




    torch.Size([16, 3, 224, 224])



The recommended way of using `IterableDataset` with `DataLoader` is to do the batching explicitly in the `Dataset`. In addition, you need to set a nominal length for the `Dataset` in order to avoid warnings from `DataLoader`.


```python
url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-{000000..000554}.tar"
url = f"pipe:curl -L -s {url} || true"
bs = 20

dataset = (
    wds.Dataset(url, length=int(1e9) // bs)
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




    torch.Size([20, 3, 224, 224])



The `ResizedDataset` is also helpful for connecting iterable datasets to `DataLoader`: it lets you set both a nominal and an actual epoch size; it will repeatedly iterate through the entire dataset and return data in chunks with the given epoch size.

The WebDataset library also provides an alternative to `DataLoader` called `MultiDataset`. It distributes `IterableDatasets` across multiple workers and collects the results in a way very similar to `DataLoader`. Unlike `DataLoader`, you don't have to worry about calculating the epoch length, and you can configure the `MultiDataset` using the same interface as a WebDataset. For example, if you want to shuffle samples between the batches returned by individual workers, you can write:

```Python
dataloader = wds.MultiDataset(dataset, workers=4).unbatched().shuffle(1000).batched(128)
```

# Data Decoding

WebDataset stores data in files contained inside `.tar` archives. This allows datasets to be stored in a bit-identical way to the way they are usually stored on disk. In addition, it allows WebDataset to take advantage of existing conventions and facilities for dealing with metadata and compression.

Loading takes place in two steps: first, the binary contents of each file are read into memory, and then the files are decoded. Reading is carried out by the `webdataset.Dataset` class itself. You can decode using any function you like. If you invoke `webdataset.Dataset(...).map(my_decoder)`, then `my_decoder` will simply be called on a dictionary with full extensions as keys and binary vectors as values.

In most cases, howeer, it's more convenient to use the `.decode` method, since it decodes images based on extensions. The `.decode` method takes one argument that specifies how decoding is to take place. That argument is a dictionary consisting of a last-extension string and a corresponding function for decoding a file with that extension. Note that samples in WebDataset are grouped based on the full extension, while decoding takes place based on the last extension. So, `sample.input.png` is represented in the sample with the key of `input.png`, but its last extension is `png`, which identifies it as an image file.

There are a number of automatic decoders built in that already understand many common extensions (recommended formats are in bold face):

- **jpg**, **ppm**, jpeg, img, image, pbm, pgm, png : image
- **txt**, text, transcript                     : string
- **cls**, cls2, class, count, index, inx, id   : integer
- **pyd**, pickle                               : Python pickle (using `pickle.loads`)
- **pth**                                       : Torch pickle (using `torch.load`)
- **json**, jsn                                 : JSON encoded object (using `json.loads`)
- **ten**, tb                                   : fast binary tensor format
- **mp4**, **ogg**, **mjpeg**, avi, mov, h264                       : video (using torchvision `load`)
- **flac**, **mp3**, sox                            : audio (using torchaudio `load`)

You select a set of these by giving a string rather than dictionary as an argument to the `.decode` method. Strings of the form `<tensor-type><image-format><8bit>` are recognized, where `<tensor-type>` can be empty (NumPy), `torch`, or `pil`; `<image-format>` can be `l`, `rgb` (same as empty), or `rgba`, and `<8bit>` can either be empty (floating point values in the range from 0 to 1) or `8` (outputs `uint8` tensors).

Common and recommended arguments for `.decoder` are:

- **pil** - for `torchvision` data augmentation
- **rgb** - for NumPy-based data augmentation, forcing RGB inputs in the range 0..1, in CHW order
- **torchrgb** - for torch-based data augmentation, forcing RGB format in the range 0..1, in CHW order
- **torchrgb8** - for torch-based data augmentation, forcing RGB format using `uint8`, in CHW order
- **l8** - for large grayscale images in HW order


# Splitting Shards across Nodes and Workers

Datasets are generally split across workers and processing nodes by shards. This is handled by `Dataset.shard_fn`. It will in turn call four hook functions in sequences:

```Python
self.reseed_hook()
urls = self.node_selection(urls)   # hook for splitting up shards across nodes
urls = self.shard_selection(urls)  # hook for splitting up shards across workers
urls = self.shard_shuffle(urls)    # hook for shuffling the shards
```

You can put any function in there you like. By default `reseed_hook`, `node_selection` and `shard_shuffle` do nothing, while `shard_selection` uses PyTorch's worker globals for splitting up shards across workers. The `shard_shuffle` function is set to a random shuffle when you use the `.shuffle(...)` method on the `Dataset`; if you want to override that, set it after configuring the `.shuffle` method.

# Data Sources

When creating a dataset with `webdataset.Dataset(url)`, the URL can be:

- the string "-", referring to stdin
- a UNIX path, opened as a regular file
- a URL-like string with the schema "pipe:"; such URLs are opened with `subprocess.Popen`. For example:
    - `pipe:curl -s -L http://server/file` accesses a file via HTTP
    - `pipe:gsutil cat gs://bucket/file` accesses a file on GCS
    - `pipe:az cp --container bucket --name file --file /dev/stdout` accesses a file on Azure
    - `pipe:ssh host cat file` accesses a file via `ssh`
- any other URL-like string with another schema; such URLs are passed to the `objectio` libraries if it is installed

It might seem at first glance to be "more efficient" to use built-in Python libraries for accessing object stores rather than subprocesses, but efficient object store access from Python really requires spawning a separate process anyway, so this approach to accessing object stores is not only convenient, it also is as efficient as we can make it in Python.

# Creating a WebDataset

Since WebDatasets are just regular tar files, you can usually create them by just using the `tar` command. All you have to do is to arrange for any files that should be in the same sample to share the same basename. Many datasets already come that way. For those, you can simply create a WebDataset with

```Bash
$ tar --sort=name -cf dataset.tar dataset/
```

If your dataset has some other directory layout, you can either rearrange the files on disk, or you can use `tar --transform` to get the right kinds of names in your tar file.

You can also create a WebDataset with library functions in this library:

- `webdataset.TarWriter` takes dictionaries containing key value pairs and writes them to disk
- `webdataset.ShardWriter` takes dictionaries containing key value pairs and writes them to disk as a series of shards

Here is how you can use `TarWriter` for writing a dataset:

```Python
sink = wds.TarWriter("dest.tar", encoder=False)
for basename in basenames:
    with open(f"{basename}.png", "rb") as stream):
        image = stream.read()
    cls = lookup_cls(basename)
    sample = {
        "__key__": basename,
        "png": image,
        "cls": cls
    }
    sink.write(sample)
sink.close()
```

# Writing Filters and Offline Augmentation

Webdataset can be used for filters and offline augmentation of datasets. Here is a complete example that pre-augments a shard and extracts class labels.


```python
def extract_class(data):
    # mock implementation
    return 0

def augment_wds(input, output, maxcount=999999999):
    src = (
        wds.Dataset(input)
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

# Related Libraries and Software

The [AIStore](http://github.com/NVIDIA/aistore) server provides an efficient backend for WebDataset; it functions like a combination of web server, content distribution network, P2P network, and distributed file system. Together, AIStore and WebDataset can serve input data from rotational drives distributed across many servers at the speed of local SSDs to many GPUs, at a fraction of the cost. We can easily achieve hundreds of MBytes/s of I/O per GPU even in large, distributed training jobs.

The [tarproc](http://github.com/tmbdev/tarproc) utilities provide command line manipulation and processing of webdatasets and other tar files, including splitting, concatenation, and `xargs`-like functionality.

The [tensorcom](http://github.com/tmbdev/tensorcom/) library provides fast three-tiered I/O; it can be inserted between [AIStore](http://github.com/NVIDIA/aistore) and [WebDataset](http://github.com/tmbdev/webdataset) to permit distributed data augmentation and I/O. It is particularly useful when data augmentation requires more CPU than the GPU server has available.

You can find the full PyTorch ImageNet sample code converted to WebDataset at [tmbdev/pytorch-imagenet-wds](http://github.com/tmbdev/pytorch-imagenet-wds)
