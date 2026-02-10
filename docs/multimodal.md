# Multi-Modal WebDataset: Loading from Separate Tar Files

## The Problem

Standard WebDataset assumes all modalities for a sample (image, text, embedding, etc.)
are interleaved within the same tar file:

```
shard_0000.tar
  sample_000.jpg
  sample_000.txt
  sample_000.npy
  sample_001.jpg
  sample_001.txt
  sample_001.npy
  ...
```

Many real-world datasets instead store each modality in **separate tar directories**:

```
images/shard_0000.tar         embeddings/shard_0000.tar
  sample_000.jpg                sample_000.emb.npy
  sample_001.jpg                sample_001.emb.npy
  ...                           ...
images/shard_0001.tar         embeddings/shard_0001.tar
  sample_002.jpg                sample_002.emb.npy
  ...                           ...
```

Naively loading from separate pipelines and zipping them risks misalignment:
independent shuffling, splitting across workers/nodes, or missing samples can
cause images to be paired with the wrong embeddings.

## The Solution: `MultiModalWebDataset`

`MultiModalWebDataset` pairs shards across modalities **before** any splitting
or shuffling occurs. Paired shard dicts flow through the pipeline as atomic units,
guaranteeing alignment:

```
PairedShardList -> nodesplitter -> workersplitter -> shard shuffle -> PairedTarExpander -> samples
```

## Quick Start

```python
import webdataset as wds
from torch.utils.data import DataLoader

ds = wds.MultiModalWebDataset(
    modalities={
        "images": "/data/images/shard_{0000..0099}.tar",
        "embeddings": "/data/embeddings/shard_{0000..0099}.tar",
    },
    shardshuffle=100,
)

for sample in ds:
    print(sample["__key__"])      # e.g. "sample_0000_0001"
    print(sample["jpg"])          # raw bytes from images tar
    print(sample["emb.npy"])     # raw bytes from embeddings tar
    break
```

All the same fluent chaining methods from `WebDataset` work:

```python
ds = wds.MultiModalWebDataset(
    modalities={
        "images": "/data/images/shard_{0000..0099}.tar",
        "embeddings": "/data/embeddings/shard_{0000..0099}.tar",
    },
    shardshuffle=100,
    detshuffle=True,
    seed=42,
).decode("rgb").to_tuple("jpg", "emb.npy").batched(32)

for images, embeddings in DataLoader(ds, num_workers=4):
    # images: list of numpy arrays (variable size) or tensor batch (fixed size)
    # embeddings: list/tensor of numpy arrays
    ...
```

## Constructor Parameters

```python
wds.MultiModalWebDataset(
    modalities,           # dict[str, str | list] - modality name -> shard URL pattern
    handler=...,          # exception handler (default: reraise_exception)
    shardshuffle=...,     # int or False - shard shuffle buffer size
    detshuffle=False,     # bool - deterministic shuffling for reproducibility
    nodesplitter=...,     # function for distributed node splitting (default: single_node_only)
    workersplitter=...,   # function for DataLoader worker splitting (default: split_by_worker)
    select_files=None,    # optional predicate to filter files within tars
    rename_files=None,    # optional function to rename files within tars
    empty_check=True,     # raise ValueError if no samples found
    seed=None,            # random seed for shuffling
    missing_key_policy="skip",  # how to handle misaligned keys: "skip", "partial", "error"
)
```

## URL Patterns

Shard URLs support the same formats as `WebDataset`:

```python
# Brace expansion
"images/shard_{0000..0099}.tar"

# Explicit list
["images/shard_0000.tar", "images/shard_0001.tar"]

# Remote URLs via pipe:
"pipe:aws s3 cp s3://bucket/images/shard_{0000..0099}.tar -"

# Any scheme supported by gopen (http, https, gs, etc.)
"https://storage.example.com/images/shard_{0000..0099}.tar"
```

**Requirement:** All modalities must have the **same number of shards**. Shards
are paired by index (the first URL from each modality forms a group, the second
URL from each forms the next group, etc.).

## Handling Misaligned Keys

Samples are matched across modalities by their `__key__` (the filename without
extension). If some samples are missing from a modality, the `missing_key_policy`
parameter controls behavior:

### `"skip"` (default)

Drop any sample not present in **all** modalities. This is the safest option.

```python
# images/shard has keys: a, b, c
# embeddings/shard has keys: a, c
# Result: only a, c are yielded (b is skipped)

ds = wds.MultiModalWebDataset(
    modalities={"images": ..., "embeddings": ...},
    missing_key_policy="skip",  # default
    shardshuffle=False,
)
```

### `"partial"`

Yield samples with whatever modalities are available. Useful when you want
to handle missing data downstream.

```python
# images/shard has keys: a, b, c
# embeddings/shard has keys: a, c
# Result: a (both), b (images only), c (both)

ds = wds.MultiModalWebDataset(
    modalities={"images": ..., "embeddings": ...},
    missing_key_policy="partial",
    shardshuffle=False,
)

for sample in ds:
    has_embedding = "emb.npy" in sample  # False for sample "b"
```

### `"error"`

Raise `ValueError` immediately if any key mismatch is detected. Use this
when your shards should be perfectly aligned and you want to catch data
corruption.

```python
ds = wds.MultiModalWebDataset(
    modalities={"images": ..., "embeddings": ...},
    missing_key_policy="error",
    shardshuffle=False,
)

# Raises ValueError: "key mismatch across modalities: images='b', embeddings='c'"
list(ds)
```

## Three or More Modalities

Works with any number of modalities:

```python
ds = wds.MultiModalWebDataset(
    modalities={
        "images": "/data/images/shard_{0000..0099}.tar",
        "embeddings": "/data/embeddings/shard_{0000..0099}.tar",
        "captions": "/data/captions/shard_{0000..0099}.tar",
    },
    shardshuffle=100,
).decode("rgb")

for sample in ds:
    image = sample["jpg"]         # decoded numpy array
    embedding = sample["emb.npy"] # decoded numpy array
    caption = sample["txt"]       # decoded string
```

## Metadata Fields

Each merged sample contains:

| Key | Description |
|-----|-------------|
| `__key__` | Sample key (shared across modalities) |
| `__url__` | Space-separated URLs of all source tar files |
| `__url_<name>__` | URL of the tar file for a specific modality |

```python
for sample in ds:
    print(sample["__key__"])              # "photo_00000_00000042"
    print(sample["__url_images__"])       # "/data/images/shard_0000.tar"
    print(sample["__url_embeddings__"])   # "/data/embeddings/shard_0000.tar"
```

## Extension Key Collisions

If two modalities produce the same extension key (e.g., both have `.txt` files),
a `ValueError` is raised. To avoid this, use distinct file extensions in each
modality's tar files, or use `rename_files` to disambiguate:

```python
ds = wds.MultiModalWebDataset(
    modalities={
        "captions_en": "/data/en/shard_{0000..0099}.tar",    # contains .txt
        "captions_fr": "/data/fr/shard_{0000..0099}.tar",    # also contains .txt -- collision!
    },
    shardshuffle=False,
)
# Raises: ValueError: extension key 'txt' from modality 'captions_fr' collides...
```

## Distributed Training

`MultiModalWebDataset` supports multi-node and multi-worker training out of the
box, using the same `nodesplitter` and `workersplitter` functions as `WebDataset`.
Since shards are paired before splitting, each worker/node receives complete
shard pairs:

```python
ds = wds.MultiModalWebDataset(
    modalities={...},
    shardshuffle=100,
    detshuffle=True,
    seed=42,
    nodesplitter=wds.split_by_node,    # split shards across DDP ranks
    workersplitter=wds.split_by_worker, # split shards across DataLoader workers
)

loader = DataLoader(ds, num_workers=4, batch_size=None)
```

## Streaming from Cloud Storage (Wasabi/S3/GCS)

For S3-compatible storage (AWS, Wasabi, MinIO, etc.), use `pipe:` URLs:

```python
s3cmd = "AWS_ACCESS_KEY_ID=... AWS_SECRET_ACCESS_KEY=... aws s3 cp --endpoint-url=https://s3.wasabisys.com"
bucket = "s3://my-bucket/dataset"

ds = wds.MultiModalWebDataset(
    modalities={
        "images": [f"pipe:{s3cmd} {bucket}/images/shard_{i:05d}.tar -" for i in range(100)],
        "embeddings": [f"pipe:{s3cmd} {bucket}/embeddings/shard_{i:05d}.tar -" for i in range(100)],
    },
    shardshuffle=100,
)
```

For GCS, use `gs://` URLs directly (requires `gsutil`):

```python
ds = wds.MultiModalWebDataset(
    modalities={
        "images": "gs://my-bucket/images/shard_{0000..0099}.tar",
        "embeddings": "gs://my-bucket/embeddings/shard_{0000..0099}.tar",
    },
    shardshuffle=100,
)
```

## Comparison with Column Store Approach

The older [column store approach](column-store.md) uses `__url__` rewriting to
load additional columns on the fly. `MultiModalWebDataset` improves on this:

| Feature | Column Store | MultiModalWebDataset |
|---------|-------------|---------------------|
| Shard alignment | Manual (must ensure matching order) | Automatic (paired by index) |
| Shuffling safety | Fragile (breaks if shards reordered) | Safe (pairs travel together) |
| Worker splitting | Must match exactly | Handled automatically |
| Missing samples | Crashes on `next()` | Configurable policy |
| API | Custom `compose()` stage | Drop-in class with fluent interface |

## Lower-Level Components

For advanced use cases, the building blocks are available separately:

### `PairedShardList`

Iterable dataset that yields paired URL dicts:

```python
psl = wds.PairedShardList(
    modalities={
        "images": "/data/images/shard_{0000..0099}.tar",
        "embeddings": "/data/embeddings/shard_{0000..0099}.tar",
    },
    seed=42,  # optional shuffling
)

for shard_group in psl:
    print(shard_group)
    # {"urls": {"images": "/data/images/shard_0042.tar", "embeddings": "/data/embeddings/shard_0042.tar"}}
```

### `PairedTarExpander`

Pipeline filter that opens paired tars and merges samples:

```python
ds = wds.DataPipeline(
    wds.PairedShardList(modalities={...}),
    wds.split_by_worker,
    wds.PairedTarExpander(handler=wds.warn_and_continue, missing_key_policy="skip"),
)
```

## Full Example

```python
import webdataset as wds
from torch.utils.data import DataLoader

# Define modalities
modalities = {
    "images": "/data/images/shard_{0000..0113}.tar",
    "embeddings": "/data/embeddings/shard_{0000..0113}.tar",
}

# Create dataset with full pipeline
train_ds = (
    wds.MultiModalWebDataset(
        modalities=modalities,
        shardshuffle=100,
        detshuffle=True,
        seed=42,
        handler=wds.warn_and_continue,
    )
    .decode("rgb")
    .to_tuple("jpg", "emb.npy")
    .shuffle(1000)
    .batched(32, collation_fn=None)  # None for variable-size images
)

# Use with DataLoader
loader = DataLoader(train_ds, num_workers=4, batch_size=None)

for batch in loader:
    images, embeddings = zip(*batch)
    # Process batch...
    break
```
