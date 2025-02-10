# Dataset Generation

This is a simple example of dataset generation using WebDataset `TarWriter`. Shard are uploaded to a server or to the cloud as they are generated.

Parallel dataset generation with Ray is illustrated at the very end.

This particular notebook generates short text samples using GPT-2. These can be used to generate OCR training data.


```python
# package installs for colab

import sys

if "google.colab" in sys.modules:
    !pip install --quiet webdataset
    !pip install --quiet adapter-transformers
    !pip install --quiet sentencepiece
    !pip install --quiet datasets
```


```python
import uuid
import webdataset as wds
import os

from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformers import pipeline
import textwrap
```


```python
# Parameters
nsamples = 10
ntokens = 100
nshards = 3

```


```python
# text generation with Huggingface and GPT2

tokenizer = GPT2Tokenizer.from_pretrained("gpt2", padding_side="left")
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained("gpt2")
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


def generate(n, prompt=""):
    """Generate n words of text, starting with prompt."""
    global tokenizer, model, generator
    output = generator(
        prompt,
        max_length=n + len(tokenizer.encode(prompt)),
        do_sample=True,
        temperature=0.99,
        top_k=50,
        top_p=0.99,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )[0]
    return output["generated_text"]


text = generate(100).strip()
print()
print(textwrap.fill(text, 64))
```


```python
# function generating an entire shard using TarWriter


def generate_shard(oname, nsamples=10000, ntokens=500, prefix=""):
    """Generate a shard of samples with text.

    Each sample has a "__key__" field and a "txt.gz" field.
    That is, the individual text files are compressed automatically on write.
    They will be automatically decompressed when read.
    """
    with wds.TarWriter(oname) as output:
        for i in range(nsamples):
            text = generate(100).strip()
            key = uuid.uuid4().hex
            text = generate(ntokens)
            sample = {"__key__": key, "txt.gz": text}
            output.write(sample)
            if i % 10 == 0:
                print(f"{i:6d} {prefix}:", repr(text)[:60])


generate_shard("temp.tar", nsamples=10, ntokens=10)
!ls -l temp.tar
!tar tf temp.tar | head -5
```


```python
# We need a couple of simple functions to upload to the cloud.


def cloud_exists(oname):
    """Check whether a file exists in the cloud."""
    # return os.system(f"gsutil stat gs://mybucket/500tokens/{oname}") == 0
    return True


def cloud_upload(oname):
    """Upload a file to the cloud."""
    # assert os.system(f"gsutil cp {oname} gs://mybucket/500tokens/{oname}") == 0
    pass
```


```python
# We can now generate a shard and upload it to the cloud.
# We skip the generation if the file already exists in the cloud.


def generate_and_upload(i):
    """Generate a shard and upload it to the cloud."""
    oname = f"text-{i:06d}.tar"
    if cloud_exists(oname):
        print(f"{oname} already exists, skipping")
        return False
    generate_shard(oname, nsamples=nsamples, ntokens=ntokens, prefix=f"{i:6d} {oname}")
    cloud_upload(oname)
    os.remove(oname)
    return True
```


```python
# For sequential generation, use this

for i in range(nshards):
    generate_and_upload(i)
```


```python
%%script true
# For parallel generation, use this

import ray

@ray.remote(num_cpus=1, num_gpus=1)
def ray_generate_and_upload(i):
    """A Ray remote function that generates a shard and uploads it to the cloud."""
    return generate_and_upload(i)

def generate_shards(nshards=10):
    """Generate a number of shards and upload them to the cloud.
    
    Runs in parallel on a Ray cluster.
    """
    ray.init(address='auto')  # Connect to the Ray cluster
    tasks = [ray_generate_and_upload.remote(i) for i in range(nshards)]
    ray.shutdown()
    return shard_names
```


```python

```
