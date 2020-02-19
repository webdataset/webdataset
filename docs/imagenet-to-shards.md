Convert Imagenet Dataset to Shards
==============================

Imagenet data is laid out as a directory containing annotations in the `ILSVRC2012_devkit_*` subdirectories and individual image files in the `train` and `val` subdirectories.

A simple conversion would just consist of a command like `tar cf train.tar train`. 

WebDataset makes it possible to associate annotations directly with each image, making reading and processing the data much simpler. This means that we need to put the code that associates annotations with images into the conversion script (i.e., here). The code below is fairly inelegant, but it gets the job done.

The output is a tar file containing data like this:

    -r--r--r-- bigdata/bigdata   2 2019-10-30 22:17 n02096437_3246.cls
    -r--r--r-- bigdata/bigdata 16426 2019-10-30 22:17 n02096437_3246.jpg
    -r--r--r-- bigdata/bigdata    62 2019-10-30 22:17 n02096437_3246.json
    -r--r--r-- bigdata/bigdata     3 2019-10-30 22:17 n03240683_4321.cls
    -r--r--r-- bigdata/bigdata 124353 2019-10-30 22:17 n03240683_4321.jpg
    -r--r--r-- bigdata/bigdata    402 2019-10-30 22:17 n03240683_4321.json
    -r--r--r-- bigdata/bigdata      3 2019-10-30 22:17 n02091032_4199.cls
    -r--r--r-- bigdata/bigdata 133184 2019-10-30 22:17 n02091032_4199.jpg
    -r--r--r-- bigdata/bigdata     42 2019-10-30 22:17 n02091032_4199.json
    
After this conversion, data can then directly be accessed like:

    for sample in WebDataset("imagenet-000000.tar"):
        image, annotations, cls = sample["jpg"], sample["json"], sample["cls"]
        ...
        
Or:

    for image, cls in WebDataset("imagenet-000000.tar", extensions="jpg cls"):
        ...


```python
%cd /mdata/imagenet-raw
```

    /mdata/imagenet-raw



```python
import os, sys, glob, os.path, sqlite3
import random as pyr
import re
import PIL.Image
import numpy as np
import io
import xmltodict
import warnings
import simplejson
import itertools as itt
import random

def readfile(path, mode="rb"):
    with open(path, mode) as stream:
        return stream.read()
def writefile(path, data):
    mode = "w" if isinstance(data, str) else "wb"
    with open(path, mode) as stream:
        stream.write(data)
def pilreads(data):
    stream = io.BytesIO(data)
    return np.array(PIL.Image.open(stream))
```


```python
jpegs = sorted(glob.glob("train/*/*.JPEG"))
print(len(jpegs), len(glob.glob("train/*/*.xml")))
```

    1281167 544546



```python
import scipy.io
meta = scipy.io.loadmat("ILSVRC2012_devkit_t12/data/meta.mat")
meta = meta["synsets"]
def scalar(x):
    for i in range(10):
        if isinstance(x, str): break
        try: x = x[0]
        except: break
    return x
wnid2id = {scalar(l[0][1]): int(scalar(l[0][0])) for l in meta}
wnid2cname = {scalar(l[0][1]): str(scalar(l[0][2])) for l in meta}
print(list(wnid2id.items())[:5])
print(list(wnid2cname.items())[:5])
```

    [('n02119789', 1), ('n02100735', 2), ('n02110185', 3), ('n02096294', 4), ('n02102040', 5)]
    [('n02119789', 'kit fox, Vulpes macrotis'), ('n02100735', 'English setter'), ('n02110185', 'Siberian husky'), ('n02096294', 'Australian terrier'), ('n02102040', 'English springer, English springer spaniel')]



```python
mode = "train"
def pathinfo(path):
    global mode
    if mode=="val":
        match = re.search(r"^[a-z]*/([^/]+)/ILSVRC2012_val_(\d+)\.JPEG", path)
    elif mode=="train":
        match = re.search(r"^[a-z]*/([^/]+)/\1_(\d+)\.JPEG", path)
    return match.group(1), int(match.group(2))
print(jpegs[3])
pathinfo(jpegs[3])
```

    train/n01440764/n01440764_10040.JPEG





    ('n01440764', 10040)




```python
def pathkey(path):
    return re.sub('.JPEG$', '', re.sub('.*/', '', path))

pathkey(jpegs[3])
```




    'n01440764_10040'




```python
def pathcls(path):
    return wnid2id[pathinfo(path)[0]]

pathcls(jpegs[3])
```




    449




```python
def jpeginfo(path):
    xmlpath = re.sub(".JPEG$", ".xml", path)
    if not os.path.exists(xmlpath):
        info = {}
    else:
        xml = readfile(xmlpath, "r")
        info = xmltodict.parse(xml)
    folder = pathinfo(path)[0]
    info["cls"] = wnid2id[folder]
    info["cname"] = wnid2cname[folder]
    return info

infos = [jpeginfo(jpegs[i]) for i in range(100)]
infos = list(filter(lambda a: a is not None, infos))
print(simplejson.dumps(infos[0], indent=4))
```

    {
        "cls": 449,
        "cname": "tench, Tinca tinca"
    }



```python
from webdataset import writer
from importlib import reload
reload(writer)

def write_shards(dest, jpegs, maxsize=1e9):
    jpegs = jpegs.copy()
    random.shuffle(jpegs)
    sink = writer.ShardWriter(dest, maxsize=maxsize, encoder=False)
    for i, fname in enumerate(jpegs):
        key = pathkey(fname)
        jpeg = readfile(fname)
        info = jpeginfo(fname)
        cls = pathcls(fname)    
        if info is None: info = dict(cls=cls)
        assert cls == info["cls"]
        json = simplejson.dumps(info)
        if i%1000==0: print(i, key, len(jpeg), json[:50])
        sample = dict(__key__=key,
                      jpg=jpeg,
                      json=json.encode("utf-8"),
                      cls=str(cls).encode("utf-8"))
        sink.write(sample)
    sink.close()
```


```python
write_shards("imagenet_train-%06d.tar", jpegs)
```

    # writing imagenet_train-000000.tar 0 0.0 GB 0
    0 n02398521_89849 153209 {"cls": 167, "cname": "hippopotamus, hippo, river 
    1000 n02113023_5478 119980 {"cls": 197, "cname": "Pembroke, Pembroke Welsh co
    2000 n03709823_4491 20618 {"cls": 818, "cname": "mailbag, postbag"}
    3000 n03710637_28365 113928 {"annotation": {"folder": "n03710637", "filename":
    4000 n01688243_6710 67245 {"cls": 468, "cname": "frilled lizard, Chlamydosau
    5000 n03721384_2038 72796 {"cls": 339, "cname": "marimba, xylophone"}
    6000 n02268853_5633 111716 {"annotation": {"folder": "n02268853", "filename":
    7000 n02277742_4064 97637 {"annotation": {"folder": "n02277742", "filename":
    8000 n03220513_13770 139727 {"cls": 897, "cname": "dome"}
    # writing imagenet_train-000001.tar 8640 1.0 GB 8640
    9000 n02423022_488 110651 {"cls": 12, "cname": "gazelle"}
    10000 n04131690_13611 102261 {"annotation": {"folder": "n04131690", "filename":
    11000 n03388043_19775 250215 {"annotation": {"folder": "n03388043", "filename":
    12000 n03240683_3388 128665 {"annotation": {"folder": "n03240683", "filename":
    13000 n02488291_742 18082 {"annotation": {"folder": "n02488291", "filename":
    14000 n01440764_2866 148521 {"annotation": {"folder": "n01440764", "filename":
    15000 n03877845_20785 134590 {"cls": 685, "cname": "palace"}
    16000 n04443257_13622 66028 {"annotation": {"folder": "n04443257", "filename":
    17000 n02389026_9653 4504 {"cls": 39, "cname": "sorrel"}
    # writing imagenet_train-000002.tar 8655 1.0 GB 17295
    18000 n02974003_4111 86318 {"cls": 563, "cname": "car wheel"}
    19000 n02667093_9530 14924 {"cls": 853, "cname": "abaya"}
    20000 n02788148_47570 184693 {"annotation": {"folder": "n02788148", "filename":
    21000 n09835506_22736 85479 {"cls": 954, "cname": "ballplayer, baseball player
    22000 n01514859_12 147454 {"annotation": {"folder": "n01514859", "filename":
    23000 n02094258_2539 26295 {"annotation": {"folder": "02094258", "filename": 
    24000 n04344873_7202 24285 {"cls": 311, "cname": "studio couch, day bed"}
    25000 n03692522_7503 1657 {"cls": 536, "cname": "loupe, jeweler's loupe"}
    # writing imagenet_train-000003.tar 8680 1.0 GB 25975
    26000 n03627232_10607 36256 {"annotation": {"folder": "n03627232", "filename":
    27000 n02410509_5259 148104 {"annotation": {"folder": "n02410509", "filename":
    28000 n02114855_2262 113822 {"cls": 58, "cname": "coyote, prairie wolf, brush 
    29000 n02747177_5767 158598 {"annotation": {"folder": "n02747177", "filename":
    30000 n01484850_21850 23497 {"annotation": {"folder": "n01484850", "filename":
    31000 n04367480_16416 74347 {"cls": 828, "cname": "swab, swob, mop"}
    32000 n02979186_18472 2798 {"annotation": {"folder": "n02979186", "filename":
    33000 n13133613_9563 52454 {"annotation": {"folder": "n13133613", "filename":
    34000 n04019541_34514 74985 {"cls": 572, "cname": "puck, hockey puck"}
    # writing imagenet_train-000004.tar 8808 1.0 GB 34783
    35000 n02797295_7118 127404 {"cls": 258, "cname": "barrow, garden cart, lawn c
    36000 n01872401_3459 148924 {"annotation": {"folder": "n01872401", "filename":
    37000 n02120079_34823 102434 {"cls": 159, "cname": "Arctic fox, white fox, Alop
    38000 n02268853_9294 136626 {"annotation": {"folder": "n02268853", "filename":
    39000 n02910353_7196 22125 {"cls": 580, "cname": "buckle"}
    40000 n03775546_15847 142862 {"cls": 829, "cname": "mixing bowl"}
    41000 n02342885_7912 91144 {"annotation": {"folder": "n02342885", "filename":
    42000 n01530575_2903 84138 {"cls": 386, "cname": "brambling, Fringilla montif
    43000 n03355925_18442 32582 {"annotation": {"folder": "n03355925", "filename":
    # writing imagenet_train-000005.tar 8739 1.0 GB 43522
    44000 n04310018_3036 144755 {"cls": 263, "cname": "steam locomotive"}
    45000 n02487347_5925 3469 {"annotation": {"folder": "n02487347", "filename":
    46000 n04458633_18992 178161 {"cls": 700, "cname": "totem pole"}
    47000 n03207743_9579 202515 {"annotation": {"folder": "n03207743", "filename":
    48000 n04118776_1963 36330 {"cls": 519, "cname": "rule, ruler"}
    49000 n07754684_2289 308631 {"annotation": {"folder": "n07754684", "filename":
    50000 n03063689_2485 37099 {"cls": 674, "cname": "coffeepot"}
    51000 n02115641_44778 183254 {"annotation": {"folder": "n02115641", "filename":
    52000 n04366367_5863 174311 {"cls": 681, "cname": "suspension bridge"}
    # writing imagenet_train-000006.tar 8758 1.0 GB 52280
    53000 n04162706_37140 103509 {"annotation": {"folder": "n04162706", "filename":
    54000 n07717556_7941 161577 {"cls": 742, "cname": "butternut squash"}
    55000 n01981276_11031 138361 {"annotation": {"folder": "n01981276", "filename":
    56000 n02397096_2032 135927 {"cls": 120, "cname": "warthog"}
    57000 n03825788_13849 2696 {"cls": 915, "cname": "nipple"}
    58000 n01644900_8852 166135 {"annotation": {"folder": "n01644900", "filename":
    59000 n02086240_5767 14180 {"annotation": {"folder": "n02086240", "filename":
    60000 n09421951_8958 25430 {"annotation": {"folder": "n09421951", "filename":
    61000 n02085620_884 173722 {"cls": 173, "cname": "Chihuahua"}
    # writing imagenet_train-000007.tar 8986 1.0 GB 61266
    62000 n13044778_2144 66581 {"cls": 878, "cname": "earthstar"}
    63000 n04286575_9596 6222 {"cls": 593, "cname": "spotlight, spot"}
    64000 n04209239_7360 163382 {"annotation": {"folder": "n04209239", "filename":
    65000 n04251144_13655 25659 {"cls": 507, "cname": "snorkel"}
    66000 n03016953_13671 2806 {"cls": 303, "cname": "chiffonier, commode"}
    67000 n04522168_10215 10298 {"annotation": {"folder": "n04522168", "filename":
    68000 n04525305_14241 139287 {"annotation": {"folder": "n04525305", "filename":
    69000 n03792972_933 12988 {"annotation": {"folder": "n03792972", "filename":
    # writing imagenet_train-000008.tar 8549 1.0 GB 69815
    70000 n02423022_10604 19970 {"cls": 12, "cname": "gazelle"}
    71000 n04371774_1024 30013 {"cls": 569, "cname": "swing"}
    72000 n01882714_4598 162963 {"cls": 213, "cname": "koala, koala bear, kangaroo
    73000 n02412080_8241 176957 {"cls": 81, "cname": "ram, tup"}
    74000 n07836838_22714 81400 {"cls": 953, "cname": "chocolate sauce, chocolate 
    75000 n02259212_4083 130853 {"cls": 637, "cname": "leafhopper"}
    76000 n02025239_3802 164701 {"cls": 433, "cname": "ruddy turnstone, Arenaria i
    77000 n02109525_16473 42043 {"annotation": {"folder": "n02109525", "filename":
    78000 n01669191_9214 134317 {"annotation": {"folder": "n01669191", "filename":
    # writing imagenet_train-000009.tar 8656 1.0 GB 78471
    79000 n01806143_2388 215512 {"annotation": {"folder": "n01806143", "filename":
    80000 n03759954_97678 151072 {"cls": 509, "cname": "microphone, mike"}
    81000 n03729826_25167 158694 {"annotation": {"folder": "n03729826", "filename":
    82000 n03649909_13802 211829 {"cls": 374, "cname": "lawn mower, mower"}
    83000 n02480855_18667 93033 {"cls": 104, "cname": "gorilla, Gorilla gorilla"}
    84000 n02979186_15766 81182 {"annotation": {"folder": "n02979186", "filename":
    85000 n01582220_2532 92580 {"cls": 394, "cname": "magpie"}
    86000 n04039381_23170 65615 {"cls": 860, "cname": "racket, racquet"}
    87000 n01601694_23215 126798 {"cls": 396, "cname": "water ouzel, dipper"}
    # writing imagenet_train-000010.tar 8639 1.0 GB 87110
    88000 n02965783_5588 95718 {"cls": 576, "cname": "car mirror"}
    89000 n03141823_19792 104394 {"annotation": {"folder": "n03141823", "filename":
    90000 n09468604_72275 67413 {"annotation": {"folder": "n09468604", "filename":
    91000 n02002724_18661 99019 {"annotation": {"folder": "n02002724", "filename":
    92000 n01829413_2802 174654 {"annotation": {"folder": "n01829413", "filename":
    93000 n01631663_8061 107331 {"cls": 496, "cname": "eft"}
    94000 n01667778_1517 46833 {"cls": 461, "cname": "terrapin"}
    95000 n04483307_4485 103144 {"annotation": {"folder": "n04483307", "filename":
    # writing imagenet_train-000011.tar 8740 1.0 GB 95850
    96000 n02825657_13820 64833 {"annotation": {"folder": "n02825657", "filename":
    97000 n02206856_1904 125029 {"annotation": {"folder": "n02206856", "filename":
    98000 n04371430_14407 15718 {"cls": 945, "cname": "swimming trunks, bathing tr
    99000 n02808440_33879 83663 {"annotation": {"folder": "n02808440", "filename":
    100000 n03666591_11631 46196 {"annotation": {"folder": "n03666591", "filename":
    101000 n02536864_3047 47594 {"annotation": {"folder": "n02536864", "filename":
    102000 n03400231_19454 81193 {"cls": 671, "cname": "frying pan, frypan, skillet
    103000 n03935335_35078 93293 {"cls": 931, "cname": "piggy bank, penny bank"}
    104000 n03065424_18067 75769 {"annotation": {"folder": "n03065424", "filename":
    # writing imagenet_train-000012.tar 8845 1.0 GB 104695
    105000 n02268853_9093 107078 {"annotation": {"folder": "n02268853", "filename":
    106000 n03841143_24277 95854 {"annotation": {"folder": "n03841143", "filename":
    107000 n03884397_19226 3866 {"cls": 352, "cname": "panpipe, pandean pipe, syri
    108000 n02130308_5787 109972 {"cls": 206, "cname": "cheetah, chetah, Acinonyx j
    109000 n03124170_15749 124723 {"cls": 881, "cname": "cowboy hat, ten-gallon hat"



```python
jpegs = sorted(glob.glob("val/*/*.JPEG"))
print(len(jpegs), len(glob.glob("val/*/*.xml")))
```


```python
mode = "val"
write_shards("imagenet_val-%06d.tgz", jpegs, maxsize=1e11)
```


```python

```
