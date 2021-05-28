# Objectron conversion to WebDataset Format


```python
import re
import os
import os.path
import json
```


```python
assert os.path.exists("objectron-files.txt")
# create with:
# !gsutil ls -r gs://objectron > objectron-files.txt
```

# Parameters


```python
shard_size = 3 # samples
bucket = "tmbdev-objectron"
only_with_anno = False  # only keep samples with annotation
max_shards = 5 # for testing; for production, set to 99999
```

# Creating the File Lists

Read the complete file list and find the video files.


```python
files = set(x.strip() for x in open("objectron-files.txt").readlines())
movs = set(x for x in files if "video.MOV" in x)
len(files), len(movs)
```




    (210448, 20088)



Assemble a list of samples, each sample comprising the video file, its corresponding geometry, and the annotation.


```python
def cleanpath(s):
    return re.sub("gs://objectron/videos/", "", s).lower()

samples = []
for mov in movs:
    base = re.sub("/video.MOV", "", mov)
    geo = base + "/geometry.pbdata"
    anno = re.sub("/videos/", "/annotations/", base) + ".pbdata"
    sample = [f"""{cleanpath(mov)} pipe:gsutil cat {mov}"""]
    sample += [f"""{cleanpath(geo)} pipe:gsutil cat {geo}"""]
    if anno in files and not only_with_anno:
        # fix up the path to be in the same directory
        sample += [f"""{cleanpath(base+"/anno.pbdata")} pipe:gsutil cat {anno}"""]
    samples.append(sample)
print(samples[0])
len(samples)
```

    ['bottle/batch-27/43/video.mov pipe:gsutil cat gs://objectron/videos/bottle/batch-27/43/video.MOV', 'bottle/batch-27/43/geometry.pbdata pipe:gsutil cat gs://objectron/videos/bottle/batch-27/43/geometry.pbdata', 'bottle/batch-27/43/anno.pbdata pipe:gsutil cat gs://objectron/annotations/bottle/batch-27/43.pbdata']





    20088



Split up the complete list of samples into shards of size `shard_size`.


```python
shards = []
for i in range(0, len(samples), shard_size):
    shards.append(samples[i:i+shard_size])
shards = [[x for l in shard for x in l] for shard in shards]
shards = shards[:max_shards]
print(shards[0][:10])
print(len(shards))
```

    ['bottle/batch-27/43/video.mov pipe:gsutil cat gs://objectron/videos/bottle/batch-27/43/video.MOV', 'bottle/batch-27/43/geometry.pbdata pipe:gsutil cat gs://objectron/videos/bottle/batch-27/43/geometry.pbdata', 'bottle/batch-27/43/anno.pbdata pipe:gsutil cat gs://objectron/annotations/bottle/batch-27/43.pbdata', 'laptop/batch-3/16/video.mov pipe:gsutil cat gs://objectron/videos/laptop/batch-3/16/video.MOV', 'laptop/batch-3/16/geometry.pbdata pipe:gsutil cat gs://objectron/videos/laptop/batch-3/16/geometry.pbdata', 'laptop/batch-3/16/anno.pbdata pipe:gsutil cat gs://objectron/annotations/laptop/batch-3/16.pbdata', 'shoe/batch-34/7/video.mov pipe:gsutil cat gs://objectron/videos/shoe/batch-34/7/video.MOV', 'shoe/batch-34/7/geometry.pbdata pipe:gsutil cat gs://objectron/videos/shoe/batch-34/7/geometry.pbdata', 'shoe/batch-34/7/anno.pbdata pipe:gsutil cat gs://objectron/annotations/shoe/batch-34/7.pbdata']
    5



```python
os.system("gsutil rm ")
for i, f in enumerate(shards):
    print(i, end=" ", flush=True)
    with os.popen(f"gsutil cp - gs://{bucket}/objectron-{i:04d}.txt", "w") as stream:
        stream.write("\n".join(f) + "\n")
```

    0 1 2 3 4 

# Creating the Shards

First, a simple function that takes a ".txt" file and creates the corresponding shard.

The core of the task is just handled by a simple shell command.


```python
import os

def makeshard(src):
    output = re.sub(".txt$", ".tar", src)
    assert output != src
    # output creation on GCS is atomic, so if the file exists, we're done
    if os.system(f"gsutil stat {output}") == 0:
        return f"{output}: already exists"
    # create the .tar shard in a fully streaming mode
    cmd = f"gsutil cat {src} | tarp create - -o - | gsutil cp - {output}"
    print(cmd)
    assert 0 == os.system(cmd)
    return f"{output}: OK"
    
makeshard("gs://tmbdev-objectron/objectron-0000.txt")
```




    'gs://tmbdev-objectron/objectron-0000.tar: already exists'



# Parallel Execution

Next, let's parallelize that with Dask.


```python
from dask.distributed import Client
from dask import delayed
import dask
import dask.bag as db
```


```python
client = Client(n_workers=4)
npartitions = 4 # used below
client
```

    /home/tmb/proj/webdataset/venv/lib/python3.8/site-packages/distributed/node.py:151: UserWarning: Port 8787 is already in use.
    Perhaps you already have a cluster running?
    Hosting the HTTP server on port 46693 instead
      warnings.warn(





<table style="border: 2px solid white;">
<tr>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Client</h3>
<ul style="text-align: left; list-style: none; margin: 0; padding: 0;">
  <li><b>Scheduler: </b>tcp://127.0.0.1:44381</li>
  <li><b>Dashboard: </b><a href='http://127.0.0.1:46693/status' target='_blank'>http://127.0.0.1:46693/status</a></li>
</ul>
</td>
<td style="vertical-align: top; border: 0px solid white">
<h3 style="text-align: left;">Cluster</h3>
<ul style="text-align: left; list-style:none; margin: 0; padding: 0;">
  <li><b>Workers: </b>4</li>
  <li><b>Cores: </b>24</li>
  <li><b>Memory: </b>67.46 GB</li>
</ul>
</td>
</tr>
</table>




```python
sources = [s.strip() for s in os.popen(f"gsutil ls gs://{bucket}/objectron-*.txt").readlines()]
```


```python
sources = db.from_sequence(sources, npartitions=npartitions)
results = sources.map(makeshard)
results.compute()
```




    ['gs://tmbdev-objectron/objectron-0000.tar: already exists',
     'gs://tmbdev-objectron/objectron-0001.tar: already exists',
     'gs://tmbdev-objectron/objectron-0002.tar: already exists',
     'gs://tmbdev-objectron/objectron-0003.tar: already exists',
     'gs://tmbdev-objectron/objectron-0004.tar: already exists']



# Running It for Real

Note that if you want to run this for real, you need to:

- change `shard_size` to something like 50-100
- change the bucket
- change `max_shards` to 999999
- set up dask to run actually distributed


```python

```
