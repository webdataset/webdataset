Convert Coco Dataset to Shards
==============================

Coco data is laid out as a directory containing annotations in the `annotations/` subdirectory and individual image files in the `train2014` and `val2014` subdirectories.

A simple conversion would just consist of a command like `tar cf train2014.tar train2014`. 

WebDataset makes it possible to associate annotations directly with each image, making reading and processing the data much simpler. This means that we need to put the code that associates annotations with images into the conversion script (i.e., here). The code below is fairly inelegant, but it gets the job done.

The output is a tar file containing data like this:

    -r--r--r-- bigdata/bigdata    633 2019-10-30 23:19 000025.captions.json
    -r--r--r-- bigdata/bigdata   5427 2019-10-30 23:19 000025.instances.json
    -r--r--r-- bigdata/bigdata 196370 2019-10-30 23:19 000025.jpg
    -r--r--r-- bigdata/bigdata    662 2019-10-30 23:19 000030.captions.json
    -r--r--r-- bigdata/bigdata   4159 2019-10-30 23:19 000030.instances.json
    -r--r--r-- bigdata/bigdata  71463 2019-10-30 23:19 000030.jpg
    -r--r--r-- bigdata/bigdata    634 2019-10-30 23:19 000034.captions.json
    -r--r--r-- bigdata/bigdata   3063 2019-10-30 23:19 000034.instances.json
    -r--r--r-- bigdata/bigdata 406018 2019-10-30 23:19 000034.jpg
    
After this conversion, data can then directly be accessed like:

    for sample in WebDataset("coco-000000.tar"):
        image, captions, instances = sample["jpg"], sample["captions.json"], sample["instances.json"]
        ...


```python
%cd /mdata/coco-raw
```

    /mdata/coco-raw



```python
%pylab inline
```

    Populating the interactive namespace from numpy and matplotlib



```python
import os
import os.path
import random as pyr
import sys
import tarfile
import argparse
import re
import simplejson
import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib as mpl
import pprint
import tarfile
import time
import io
pp = pprint.PrettyPrinter(indent=4)
def jsp(x): print(simplejson.dumps(x, indent=4, sort_keys=True))
from IPython.display import display
```

# train2014


```python
with open("annotations/instances_train2014.json") as stream:
    instances = simplejson.load(stream)
with open("annotations/captions_train2014.json") as stream:
    captions = simplejson.load(stream)
with open("annotations/COCO_Text.json") as stream:
    cocotext = simplejson.load(stream)
with open("annotations/person_keypoints_train2014.json") as stream:
    person = simplejson.load(stream)
```


```python
all_images = {}
for x in instances["images"]:
    all_images[x["id"]] = dict(__key__=x["id"], info=x, image=x["file_name"], instances=None, captions=None, texts=None, persons=None)
def dappend(k, x):
    global all_images
    entry = all_images[x["image_id"]]
    if entry.get(k) is None: entry[k] = []
    entry[k].append(x)
for x in instances["annotations"]:
    dappend("instances", x)
for x in captions["annotations"]:
    dappend("captions", x)
for x in person["annotations"]:
    dappend("persons", x)
for _, x in cocotext["anns"].items():
    dappend("texts", x)
```


```python
def stats(l): return (len(l), amin(l), median(l), mean(l), amax(l))
```


```python
print(stats([len(x["instances"]) for x in all_images.values() if x["instances"] is not None]))
print(stats([len(x["captions"]) for x in all_images.values() if x["captions"] is not None]))
print(stats([len(x["persons"]) for x in all_images.values() if x["persons"] is not None]))
print(stats([len(x["texts"]) for x in all_images.values() if x["texts"] is not None]))
```

    (82081, 1, 4.0, 7.369634872869483, 93)
    (82783, 5, 5.0, 5.0023917954169335, 7)
    (45174, 1, 2.0, 4.102271217957232, 20)
    (26847, 1, 3.0, 5.433083770998622, 211)



```python
if isinstance(all_images, dict):
    all_images = sorted(list(all_images.items()))
all_images = [x[1] for x in all_images]
```


```python
for k, v in all_images[0].items():
    print("{:9s} {}".format(k, repr(v)[:60]))
```

    __key__   9
    info      {'license': 3, 'file_name': 'COCO_train2014_000000000009.jpg
    image     'COCO_train2014_000000000009.jpg'
    instances [{'segmentation': [[500.49, 473.53, 599.73, 419.6, 612.67, 3
    captions  [{'image_id': 9, 'id': 661611, 'caption': 'Closeup of bins o
    texts     None
    persons   None



```python
# read images directly
def myimread(fname, base="train2014"):
    with open(base+"/"+fname, "rb") as stream:
        return stream.read()
    
# functions for converting data structures to UTF-8 encoded JSON
def cvjson(data):
    return bytes(simplejson.dumps(data, indent=4), encoding="utf-8")

# the fields we pay attention to
names = {
    "info.json": "info",
    "jpg": "image",
    "persons.json": "persons",
    "instances.json": "instances",
    "texts.json": "texts",
    "captions.json": "captions"
}
names = {v:k for k,v in names.items()}

def convert_sample(sample, base="train2014"):
    # use the number as the key
    result = dict(__key__="%06d"%(sample["__key__"],))
    # store the image in JPEG format
    result["jpg"] = myimread(sample["image"], base=base)
    # add whatever meta data we have as separate JSON files
    for k in "persons instances texts captions".split():
        if k in sample and sample[k] is not None:
            result[names[k]] = cvjson(sample[k])
    return result
```


```python
result = convert_sample(all_images[0])
for k, v in result.items():
    print("{:20s} {}".format(k, repr(v)[:60]))
```

    __key__              '000009'
    jpg                  b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00
    instances.json       b'[\n    {\n        "segmentation": [\n            [\n      
    captions.json        b'[\n    {\n        "image_id": 9,\n        "id": 661611,\n 



```python
# FOR TESTING THIS NOTEBOOK ONLY; COMMENT OUT FOR FULL CONVERSION
all_images = all_images[:12000]
```


```python
!rm -rf coco-*.tar
from webdataset import writer
print(len(all_images))
tarf = writer.ShardWriter("_coco-%06d.tar", encoder=False, maxcount=5000, maxsize=3e9)
for sample in all_images:
    result = convert_sample(sample)
    tarf.write(result)
tarf.close()
```

    12000
    # writing _coco-000000.tar 0 0.0 GB 0
    # writing _coco-000001.tar 5000 0.9 GB 5000
    # writing _coco-000002.tar 5000 0.9 GB 10000


Coco Text
=========

Look at Coco Text Maps


```python
figsize(16,8)
l = list(all_images)
pyr.shuffle(l)
for v in l:
    k = v["__key__"]
    if v["texts"] is None: continue
    print(v.keys())
    image = imread("train2014/"+v["image"])
    h, w = image.shape[:2]
    textmap = zeros((h, w), 'f')
    for l in v["texts"]:
        x, y, w, h = [int(r) for r in l["bbox"]]
        textmap[y:y+h, x:x+w] = 1.0
    subplot(121); imshow(image)
    subplot(122); imshow(textmap)
    break
```

    dict_keys(['__key__', 'info', 'image', 'instances', 'captions', 'texts', 'persons'])



![png](coco-to-shards_files/coco-to-shards_17_1.png)


Write a dataset that contains only images with text annotations.


```python
!rm -rf _cocotexts-*.tar
from webdataset import writer
print(len(all_images))
tarf = writer.ShardWriter("_cocotexts-%06d.tar", encoder=False, maxcount=5000, maxsize=3e9)
for sample in all_images:
    if "texts" not in sample: continue
    result = convert_sample(sample)
    tarf.write(result)
tarf.close()
```

    12000
    # writing _cocotexts-000000.tar 0 0.0 GB 0
    # writing _cocotexts-000001.tar 5000 0.9 GB 5000
    # writing _cocotexts-000002.tar 5000 0.9 GB 10000


# val2014

Now repeat the process for the validation set. (We might make this prettier by abstracting some of the functionality into more functions.)


```python
with open("annotations/instances_val2014.json") as stream:
    instances = simplejson.load(stream)
with open("annotations/captions_val2014.json") as stream:
    captions = simplejson.load(stream)
with open("annotations/person_keypoints_val2014.json") as stream:
    person = simplejson.load(stream)
```


```python
all_images = {}
for x in instances["images"]:
    all_images[x["id"]] = dict(__key__=x["id"], info=x, image=x["file_name"], instances=None, captions=None, texts=None, persons=None)
def dappend(k, x):
    global all_images
    entry = all_images[x["image_id"]]
    if entry.get(k) is None: entry[k] = []
    entry[k].append(x)
for x in instances["annotations"]:
    dappend("instances", x)
for x in captions["annotations"]:
    dappend("captions", x)
for x in person["annotations"]:
    dappend("persons", x)

```


```python
print(stats([len(x["instances"]) for x in all_images.values() if x["instances"] is not None]))
print(stats([len(x["captions"]) for x in all_images.values() if x["captions"] is not None]))
print(stats([len(x["persons"]) for x in all_images.values() if x["persons"] is not None]))
```

    (40137, 1, 4.0, 7.271968507860578, 70)
    (40504, 5, 5.0, 5.003308315228126, 7)
    (21634, 1, 2.0, 4.074743459369511, 16)



```python
if isinstance(all_images, dict):
    all_images = sorted(list(all_images.items()))
all_images = [x[1] for x in all_images]
```


```python
result = convert_sample(all_images[0], base="val2014")
for k, v in result.items():
    print("{:20s} {}".format(k, repr(v)[:60]))
```

    __key__              '000042'
    jpg                  b'\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x01\x00H\x00H\x00
    instances.json       b'[\n    {\n        "segmentation": [\n            [\n      
    captions.json        b'[\n    {\n        "image_id": 42,\n        "id": 641613,\n



```python
# FOR TESTING THIS NOTEBOOK ONLY; COMMENT OUT FOR FULL CONVERSION
all_images = all_images[:12000]
```


```python
!rm -rf _coco-val-*.tar
from webdataset import writer
print(len(all_images))
tarf = writer.ShardWriter("_coco-val-%06d.tar", encoder=False, maxcount=5000, maxsize=3e9)
for sample in all_images:
    result = convert_sample(sample, base="val2014")
    tarf.write(result)
tarf.close()
```

    12000
    # writing _coco-val-000000.tar 0 0.0 GB 0
    # writing _coco-val-000001.tar 5000 0.9 GB 5000
    # writing _coco-val-000002.tar 5000 0.9 GB 10000



```python

```
