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


# Desktop Usage and Caching

WebDataset is an ideal solution for training on petascale datasets kept on high performance distributed data stores like AIStore, AWS/S3, and Google Cloud. Compared to data center GPU servers, desktop machines have much slower network connections, but training jobs on desktop machines often also use much smaller datasets. WebDataset also is very useful for such smaller datasets, and it can easily be used for developing and testing on small datasets and then scaling up to large datasets by simply using more shards.


Here are different usage scenarios:

| environment | caching strategy |
|-|-|
| cloud training against cloud buckets | use WebDataset directly with cloud URLs |
| on premises training with high performance store (e.g., AIStore) | use WebDataset directly with storage URLs. |
| prototyping, development, testing for large scale training | copy a few shards to local disk OR use automatic shard caching OR use DBCache |
| on premises training with slower object stores/networks | use automatic shard caching or DBCache for entire dataset |
| desktop deep learning, smaller dataset | copy all shards to disk manually OR use automatic shard caching |
| training with IterableDataset sources other than WebDataset | use DBCache |

_The upshot is: you can write a single I/O pipeline that works for both local and remote data, and for both small and large datasets, and you can fine-tune performance and take advantage of local storage by adding the `cache_dir` and `DBCache` options._

Let's look at how these different methods work.

## Direct Copying of Shards

Let's take the OpenImages dataset as an example; it's half a terabyte large. For development and testing, you may not want to download the entire dataset, but you may also not want to use the dataset remotely. With WebDataset, you can just download a small number of shards and use them during development.


```python
!test -f /tmp/openimages-train-000000.tar || curl -L -s http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar > /tmp/openimages-train-000000.tar
```


```python
dataset = wds.WebDataset("/tmp/openimages-train-000000.tar")
repr(next(iter(dataset)))[:200]
```




    "{'__key__': 'e39871fd9fd74f55', 'jpg': b'\\xff\\xd8\\xff\\xe0\\x00\\x10JFIF\\x00\\x01\\x01\\x01\\x01:\\x01:\\x00\\x00\\xff\\xdb\\x00C\\x00\\x06\\x04\\x05\\x06\\x05\\x04\\x06\\x06\\x05\\x06\\x07\\x07\\x06\\x08\\n\\x10\\n\\n\\t\\t\\n\\x14\\x0e"



Note that the WebDataset class works the same way on local files as it does on remote files. Furthermore, unlike other kinds of dataset formats and archive formats, downloaded datasets are immediately useful and don't need to be unpacked.

## Automatic Shard Caching

Downloading a few shards manually is useful for development and testing. But WebDataset permits us to automate downloading and caching of shards. This is accomplished by giving a `cache_dir` argument to the WebDataset constructor. Note that caching happens in parallel with iterating through the dataset. This means that if you write a WebDataset-based I/O pipeline, training starts immediately; the training job does not have to wait for any shards to download first.

Automatic shard caching is useful for distributing deep learning code, for academic computer labs, and for cloud computing.

In this example, we make two passes through the dataset, using the cached version on the second pass.


```python
!rm -rf ./cache

# just using one URL for demonstration
url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar"
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

    [caching <webdataset.gopen.Pipe object at 0x7fe2832feaf0> at ./cache/9fd87fa8-d42e-3be4-a3a6-839de961b98a.~2601956~ ]


    === first pass


    [done caching ./cache/9fd87fa8-d42e-3be4-a3a6-839de961b98a ]
    [finished ./cache/9fd87fa8-d42e-3be4-a3a6-839de961b98a]
    [opening cached ./cache/9fd87fa8-d42e-3be4-a3a6-839de961b98a ]


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
    -rw-rw-r-- 1 tmb tmb 1011630080 Nov  2 09:44 9fd87fa8-d42e-3be4-a3a6-839de961b98a




Using automatic shard caching, you end up with bit-identical copies of the original dataset in the local shard cache. By default, shards are named based on a MD5 checksum of their original URL. If you want to reuse the downloaded cached files, you can override the cache file naming with the `cache_name=` argument to `WebDataset` and `DBCache`.

You can disable shard caching by setting the shard cache directory name to `None`.

## Automatic Sample Caching

WebDataset also provides a way of caching training samples directly. This works with samples coming from any IterableDataset as input. The cache is stored in an SQLite3 database. Sample-based caching is implemented by the `DBCache` class. You specify a filename for the database and the maximum number of samples you want to cache. Samples will initially be read from the original IterableDataset, but after either the samples run out or the maximum number of samples has been reached, subsequently, samples will be served from the database cache stored on local disk. The database cache persists between invocations of the job.

Automatic sample caching is useful for developing and testing deep learning jobs, as well as for caching data coming from slow IterableDataset sources, such as network-based database connections or other slower data sources.


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
    [DBCache total 0 size 1000 more caching]


    === first pass


    [DBCache finished caching total 1000 (size 1000)]
    [DBCache starting dbiter total 1000 size 1000]


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
    
    -rw-r--r-- 1 tmb tmb 485199872 Nov  2 09:44 ./cache.db


You can disable the cache by changing the cache file name to `None`. This makes it easy to enable/disable the cache for testing.

Sample-based caching using `DBCache` gives you more flexibility than shard-based caching: you can cache before or after decoding and before or after data augmentation. However, unlike shard-based caching, the cache won't be considered "complete" until the number of cached samples requested have been cached. The `DBCache` class is primarily useful for testing, and for caching data that comes from `IterableDataset` sources other than `WebDataset`.
