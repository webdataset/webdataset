```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib



```python
import webdataset.dataset as wds
from itertools import islice
import simplejson
import sys, os, subprocess
import PIL
import io
```

# Dataset Formats

We store data in sharded tar files. Sharded tar files are regular tar files with a few conventions:

- all files making up a training sample need to be physically consecutive in the tar file
- all files with a common basename are grouped into a single sample
- tar files can be split into sequentially numbered shards and are referred to by shell-style brace notation (e.g., `imagenet_train-{0000..0147}.tgz`

You can often just tar up dataset with `tar --sorted`. For more complex uses, there are command line tools and even special high performance servers.


```python
!tar -ztvf testdata/imagenet-000000.tgz | head
```

    gzip: warning: GZIP environment variable is deprecated; use an alias or script
    -rw-r--r-- tmb/tmb           3 1969-12-31 16:00 10.cls
    -rw-rw-r-- tmb/tmb       75442 2018-04-16 10:21 10.png
    -rw-r--r-- tmb/tmb           9 1969-12-31 16:00 10.wnid
    -rw-r--r-- tmb/tmb           4 1969-12-31 16:00 10.xml
    -rw-r--r-- tmb/tmb           3 1969-12-31 16:00 12.cls
    -rw-rw-r-- tmb/tmb       80108 2018-04-16 10:21 12.png
    -rw-r--r-- tmb/tmb           9 1969-12-31 16:00 12.wnid
    -rw-r--r-- tmb/tmb           4 1969-12-31 16:00 12.xml
    -rw-r--r-- tmb/tmb           3 1969-12-31 16:00 13.cls
    -rw-rw-r-- tmb/tmb      242280 2018-04-16 10:21 13.png
    tar: write error


Note that a bunch of presharded datasets are available on Google Cloud Storage at:

- gs://lpr-imagenet
- gs://lpr-coco


# Simple Dataset Loading

The `WebDataset` class is a reader for sharded tar files. By default, it reads files from a tar file, groups them into training samples by basename, applies standard file name extension conventions for decoding the images into tensors, and returns a dictionary mapping the extensions to decoded data.


```python
ds = wds.WebDataset("testdata/imagenet-000000.tgz", 47)
for sample in ds:
    image = sample["png"]
    imshow(image)
    break
```


![png](webdataset-demo_files/webdataset-demo_6_0.png)



```python
for key, value in sample.items():
    print(key, "=", repr(value)[:80])
```

    __key__ = '10'
    cls = 304
    png = array([[[0.99215686, 0.99215686, 0.99215686],
            [0.99215686, 0.99215686, 0
    wnid = b'n04380533'
    xml = b'None'


Note how filenames are interpreted; consider a tar file containing the files:

- `dir/base.input.png`
- `dir/base.output.png`

This will be decoded into a Python sample containing two tensors:

    {
        "__key__": "dir/base",
        "input.png": ...decoded png image...,
        "output.png": ...decoded png image...
    }


Observe that everything from the first period (".") in the filename portion to the end becomes the key.

# Field Selection

You can select fields by providing the `extensions=` argument. When provided, the iterator will yield tuples rather than dictionaries. This is convenient for training.

Since files are sometimes stored using different extensions, you can provide alternative names. Below, the first element of the tuple consists of any decoded file with an extension of "png" or "jpg", and the second element consists of the decoded ".cls" file.


```python
ds = wds.WebDataset("testdata/imagenet-000000.tgz", 47,
                    extensions="png;jpg cls")
for sample in ds:
    image = sample[0]
    print(image.shape, image.dtype)
    imshow(image)
    break
```

    (793, 600, 3) float32



![png](webdataset-demo_files/webdataset-demo_11_1.png)


# Decoders

One significant advantage of using tar files as a record sequential format is that there are a lot of conventions and tools for encoding and compressing binary data on file systems. This allows `WebDataset` to provide many useful defaults. For example, we can decode to 8-bit RGB images like this.


```python
ds = wds.WebDataset("testdata/imagenet-000000.tgz", 47, extensions="png;jpg cls", decoder="rgb8")
for sample in ds:
    image = sample[0]
    print(image.shape, image.dtype)
    imshow(image)
    break
```

    (793, 600, 3) uint8



![png](webdataset-demo_files/webdataset-demo_13_1.png)


To decode to PIL images, use `decoder="pil"`.


```python
ds = wds.WebDataset("testdata/imagenet-000000.tgz", 1000, decoder="pil")
for sample in ds:
    break
for k, v in sample.items():
    print(k, repr(v)[:60])
```

    __key__ '10'
    cls 304
    png <PIL.Image.Image image mode=RGB size=600x793 at 0x7F5E9AF4EF
    wnid b'n04380533'
    xml b'None'


Here is an example of grayscale decoding (the default decoder uses PIL/Pillow, and its image specs generally work as decoder strings).


```python
ds = wds.WebDataset("http://storage.googleapis.com/lpr-imagenet/imagenet_train-0050.tgz", 1000000,
                    extensions="jpg;png cls", decoder="l")
for sample in ds:
    image = sample[0]
    imshow(image, cmap=cm.jet)
    break
```


![png](webdataset-demo_files/webdataset-demo_17_0.png)


You can also get raw binary data file contents.


```python
ds = wds.WebDataset("testdata/imagenet-000000.tgz", 1000, decoder=None)
for sample in ds:
    break
for k, v in sample.items():
    print(k, repr(v)[:60])
```

    __key__ '10'
    cls b'304'
    png b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x02X\x00\x00\x
    wnid b'n04380533'
    xml b'None'


# Custom Decoders

You may need to deal with file types that `WebDataset` has no built-in support for. You can easily define your own handlers by defining a dictionary that maps file name extensions to decoding functions.


```python
# use the existing handlers as a base
handlers = dict(wds.default_handlers["rgb"])
# add a special handler for "jpg"
def decode_jpg_and_resize(data):
    return PIL.Image.open(io.BytesIO(data)).resize((128, 128))
handlers["jpg"] = decode_jpg_and_resize

ds = wds.WebDataset("http://storage.googleapis.com/lpr-imagenet/imagenet_train-0050.tgz", 1000000,
                    extensions="jpg;png cls", decoder=handlers)
for sample in ds:
    image = sample[0]
    print(image)
    imshow(image)
    break
```

    <PIL.Image.Image image mode=RGB size=128x128 at 0x7F5E9AF3CA20>



![png](webdataset-demo_files/webdataset-demo_22_1.png)


If you need even more control, you can set the decoder to a function that just maps the sample to another sample.


```python
def mydecoder(sample):
    return {k: len(v) for k, v in sample.items()}

ds = wds.WebDataset("http://storage.googleapis.com/lpr-imagenet/imagenet_train-0050.tgz", 1000000,
                    extensions="jpg;png cls", decoder=mydecoder)
for sample in ds:
    image = sample[0]
    print(sample)
    break
```

    (16444, 3)


# Accessing Web Servers / Object Stores

As the name suggests, `WebDataset` is intended for accessing training data on web servers, cloud storage, etc. To do this, you simply specify a URL as your dataset.


```python
ds = wds.WebDataset("http://storage.googleapis.com/lpr-imagenet/imagenet_train-0100.tgz", 60000,
                    extensions="jpg;png cls")
for sample in ds:
    image = sample[0]
    imshow(image)
    break
```


![png](webdataset-demo_files/webdataset-demo_27_0.png)


Larger datasets are broken up into shards and you refer to shards using shell-style brace notation. For example, the Imagenet dataset is customarily broken up into 147 shards, each of which is 1 Gbyte large. Shuffling and other operations can be carried out at the shard level and parallelized across many nodes.


```python
ds = wds.WebDataset("http://storage.googleapis.com/lpr-imagenet/imagenet_train-{0000..0147}.tgz", 1000000,
                    extensions="jpg;png cls", shuffle=0)
for sample in ds:
    image = sample[0]
    imshow(image)
    break
```


![png](webdataset-demo_files/webdataset-demo_29_0.png)


The default shard opener in `WebDataset` uses command line tools to access remote data; this provides asynchronous I/O and makes it easy to test things. It is probably the best choice for most users. But you can define custom openers that work any way you want: they can manipulate file names, use Python libraries instead of command line programs, etc.


```python
def opener(url):
    print(url, file=sys.stderr)
    cmd = "curl -s 'http://storage.googleapis.com/lpr-imagenet/imagenet_train-{}.tgz'".format(url)
    return subprocess.Popen(cmd, bufsize=1000000, shell=True, stdout=subprocess.PIPE).stdout

ds = wds.WebDataset("{0000..0147}", 1000000,
                    extensions="jpg;png cls", shuffle=100, opener=opener)
for sample in ds:
    image = sample[0]
    imshow(image)
    break
```

    0031



![png](webdataset-demo_files/webdataset-demo_31_1.png)


Note that we used a `shuffle` option here. Any shuffle greater than 0 will shuffle the shards. Shuffles greater than one will additionally shuffle the samples in a buffer of size `shuffle`.

# Associating Extra Data

A common practice in DL datasets is to store large objects separately from smaller data, like class labels. `WebDataset` provides a simple way to associate these with one another. Here, we load the extra data from a JSON file, but it can be stored anywhere you like (sqlite, db, etc.)


```python
extra_data = simplejson.loads(open("testdata/imagenet-extra.json").read())
def associate(key):
    return dict(MY_EXTRA_DATA=extra_data[key])

ds = wds.WebDataset("testdata/imagenet-000000.tgz", 1000000, associate=associate)

for sample in ds:
    print(sample.keys())
    break
```

    dict_keys(['__key__', 'cls', 'png', 'wnid', 'xml', 'MY_EXTRA_DATA'])


Note that as long as you don't require the sample grouping (as in many standard databases), any tar file of your dataset is automatically a `WebDataset`. For example, if you associate classes with images outside the loader, then you can simply extract the JPEG images alone using `WebDataset`. E.g., 


```python
def associate(key):
    return dict(cls="<class for {}>".format(key))

ds = wds.WebDataset("http://storage.googleapis.com/lpr-imagenet/imagenet_train-{0000..0147}.tgz", 1000000,
                    extensions="jpg;png cls", shuffle=1, associate=associate)

for sample in islice(ds, 0, 10):
    print(f"<tensor {sample[0].shape}>\t{sample[1]}")
```

    <tensor (400, 500, 3)>	<class for n04004767_11891>
    <tensor (375, 500, 3)>	<class for n02398521_26211>
    <tensor (500, 333, 3)>	<class for n01694178_8331>
    <tensor (300, 400, 3)>	<class for n01824575_398>
    <tensor (600, 448, 3)>	<class for n03447721_14673>
    <tensor (375, 500, 3)>	<class for n02077923_6151>
    <tensor (400, 523, 3)>	<class for n02791124_5189>
    <tensor (400, 600, 3)>	<class for n04154565_5511>
    <tensor (375, 500, 3)>	<class for n07753592_11182>
    <tensor (333, 500, 3)>	<class for n04270147_972>


# Standard Torch Data Augmentation

Since `WebDataset` can decode images into PIL format, using it with existing `torchvision` augmentation pipelines is easy.


```python
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],                    
                                 std=[0.229, 0.224, 0.225])                     
                                                                                
preproc = transforms.Compose([                                                  
    transforms.RandomResizedCrop(224),                                          
    transforms.RandomHorizontalFlip(),                                          
    transforms.ToTensor(),                                                      
    normalize,                                                                  
]) 

ds = wds.WebDataset("http://storage.googleapis.com/lpr-imagenet/imagenet_train-{0000..0147}.tgz", 1000000,
                    decoder="pil",
                    extensions="jpg;png cls",
                    transforms=[preproc, lambda x: x-1, lambda x:x])
for sample in ds:
    break

image = sample[0]
print(type(image), image.dtype, image.size())
print(sample[1])
```

    <class 'torch.Tensor'> torch.float32 torch.Size([3, 224, 224])
    851


# Binary Tensor Format

The `tenbin` library contains a codec for binary tensors; these can be written to disk with the `.ten` extension, and that extension has handlers in the default `WebDataset`. This format is useful both for quickly saving/loading tensors in files and in tar archive, and for sending them over the network. Data is 64-byte aligned and represented in a way that is simple enough to decode even on a GPU. 


```python
from webdataset import tenbin
a = randn(1733, 345)
a_encoded = tenbin.encode_buffer([a])
print(repr(a_encoded)[:100])
a_decoded = tenbin.decode_buffer(a_encoded)[0]
assert (a==a_decoded).all()
```

    bytearray(b'~TenBin~(\x00\x00\x00\x00\x00\x00\x00f8\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\



```python
!tar tvf testdata/tendata.tar | sed 3q
```

    -rw-r--r-- tmb/tmb       12736 2019-08-07 12:50 000000.ten
    -rw-r--r-- tmb/tmb       12736 2019-08-07 12:50 000001.ten
    -rw-r--r-- tmb/tmb       12736 2019-08-07 12:50 000002.ten
    tar: write error



```python
from importlib import reload
import webdataset.dataset as wds
reload(wds)
ds = wds.WebDataset("testdata/tendata.tar", 100, extensions="ten")
for sample in islice(ds, 0, 10):
    xs, ys = sample[0]
    print(xs.shape, ys.shape)
```

    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)


# Sequential Serialized Containers

Training formats like TFRecord do not contain files, but instead are a sequential record format that contains serialized datastructures in the form of protocol buffers. WebDataset offers an analogous format, replacing the TFRecord container format with ANSI tar files and
the protobuf format with a choice of MessagePack, JSON, Python dump files, or binary tensor files. Specify the `container` option to read a container dataset. The usual rules for key interpretation and automatic decoding are still in effect, so if you want raw binary
data, you need to specify `decoder=None` as well.


```python
!tar tvf testdata/mpdata.tar | sed 3q
```

    -rw-r--r-- tmb/tmb           7 2019-08-07 12:46 000000.mp
    -rw-r--r-- tmb/tmb           7 2019-08-07 12:46 000001.mp
    -rw-r--r-- tmb/tmb           7 2019-08-07 12:46 000002.mp
    tar: write error



```python
from importlib import reload
import webdataset.dataset as wds
reload(wds)
ds = wds.WebDataset("testdata/mpdata.tar", 100, container="mp", decoder=None)
for sample in ds:
    print(sample)
    break
```

    {'x': 0, 'y': 0, '__key__': '000000.mp'}



```python
!tar tvf testdata/tendata.tar | sed 3q
```

    -rw-r--r-- tmb/tmb       12736 2019-08-07 12:50 000000.ten
    -rw-r--r-- tmb/tmb       12736 2019-08-07 12:50 000001.ten
    -rw-r--r-- tmb/tmb       12736 2019-08-07 12:50 000002.ten
    tar: write error



```python
from importlib import reload
import webdataset.dataset as wds
reload(wds)
ds = wds.WebDataset("testdata/tendata.tar", 100, container="ten", decoder=None)
for xs, ys in islice(ds, 0, 10):
    print(xs.shape, ys.shape)
```

    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)
    (28, 28) (28, 28)


# Combination with Dataloader

`WebDataset` is a standard `IterableDataset` and can be used with the regular `DataLoader`.

Here is a complete example of using `WebDataset` with `DataLoader` and multiprocessing for loading.


```python
import torch
from torchvision import transforms

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],                    
                                 std=[0.229, 0.224, 0.225])                     
                                                                                
preproc = transforms.Compose([                                                  
    transforms.RandomResizedCrop(224),                                          
    transforms.RandomHorizontalFlip(),                                          
    transforms.ToTensor(),                                                      
    normalize,                                                                  
]) 

shards = "http://storage.googleapis.com/lpr-imagenet/imagenet_train-{0000..0147}.tgz"
dataset = wds.WebDataset(shards, 1000000,
                         decoder="pil", extensions="jpg;png cls",
                         transforms=[preproc, lambda x: x-1, lambda x:x])

loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=4)
for xs, ys in islice(loader, 0, 10):
    print(xs.size(), ys.size())
```

    torch.Size([16, 3, 224, 224]) torch.Size([16])
    torch.Size([16, 3, 224, 224]) torch.Size([16])
    torch.Size([16, 3, 224, 224]) torch.Size([16])
    torch.Size([16, 3, 224, 224]) torch.Size([16])
    torch.Size([16, 3, 224, 224]) torch.Size([16])
    torch.Size([16, 3, 224, 224]) torch.Size([16])
    torch.Size([16, 3, 224, 224]) torch.Size([16])
    torch.Size([16, 3, 224, 224]) torch.Size([16])
    torch.Size([16, 3, 224, 224]) torch.Size([16])
    torch.Size([16, 3, 224, 224]) torch.Size([16])


Note that parallelization happens at the level of shards, so you should have at least as many shards as workers.


```python
ds = wds.WebDataset("testdata/imagenet-000000.tgz", 1000, decoder=None)
dl = torch.utils.data.DataLoader(ds, num_workers=4)
for sample in dl:
    break
```

    /home/tmb/exp/webdataset/webdataset/dataset.py:666: UserWarning: num_workers 4 > num_shards 1
      warnings.warn(f"num_workers {total} > num_shards {len(self.full_urls)}")



```python

```
