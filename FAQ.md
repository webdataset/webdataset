
# WebDataset FAQ

This is a Frequently Asked Questions file for WebDataset.  It is
automatically generated from selected WebDataset issues using AI.

Since the entries are generated automatically, not all of them may
be correct.  When in doubt, check the original issue.

------------------------------------------------------------------------------

Issue #350

Q: How does shuffling work in WebDataset, and how can I achieve optimal shuffling?

A: The `.shuffle(...)` method in WebDataset shuffles samples within a shard. To
shuffle shards, you need to use `WebDataset(..., shardshuffle=True, ...)`. For
optimal shuffling, you should shuffle both between shards and within shards.
Here is an example of how to achieve this:

```python
dataset = WebDataset(..., shardshuffle=100).shuffle(...) ... .batched(64)
dataloader = WebLoader(dataset, num_workers=..., ...).unbatched().shuffle(5000).batched(batch_size)
```

This approach ensures that data is shuffled at both the shard level and the
sample level, providing a more randomized dataset for training.

------------------------------------------------------------------------------

Issue #332

Q: How can I ensure deterministic dataloading with WebDataset to cover all images without any being left behind?

A: To achieve deterministic dataloading with WebDataset, you should iterate
through each sample in sequence from beginning to end. This can be done by
setting up your dataset and iterator as follows:

```python
dataset = WebDataset("data-{000000..000999}.tar")
for sample in dataset:
    ...
```

For large-scale parallel inference, consider parallelizing over shards using a
parallel library like Ray. Here’s an example:

```python
def process_shard(input_shard):
    output_shard = ...  # compute output shard name
    src = wds.WebDataset(input_shard).decode("PIL")
    snk = wds.TarWriter(output_shard)
    for sample in src:
        sample["cls"] = classifier(sample["jpg"])
        snk.write(sample)
    src.close()
    snk.close()
    return output_shard

shards = list(braceexpand("data-{000000..000999}.tar"))
results = ray.map(process_shard, shards)
ray.get(results)
```

This approach ensures that each shard is processed deterministically and in
parallel, covering all images without any being left behind.

------------------------------------------------------------------------------

Issue #331

Q: How can I handle gzipped tar files with WIDS when loading the SAM dataset?

A: When using WIDS to load the SAM dataset, you may encounter a
`UnicodeDecodeError` due to the dataset being gzipped. WIDS does not natively
support gzipped tar files for random access. As a workaround, you can re-tar the
dataset without compression. This can be done using the `tarfile` library in
Python to extract and re-tar the files. Alternatively, you can compress
individual files within the tar archive (e.g., `.json.gz` instead of `.json`),
which WIDS can handle. Here is an example of re-tarring the dataset:

```python
import os
import tarfile
from tqdm import tqdm

src_tar_path = "path/to/sa_000000.tar"
src_folder_path = "path/to/src_folder"
tgt_folder_path = "path/to/tgt_folder"
rpath = os.path.relpath(src_tar_path, src_folder_path)
t = tarfile.open(src_tar_path)

fpath = os.path.join(tgt_folder_path, rpath)
os.makedirs(os.path.dirname(fpath), exist_ok=True)
tdev = tarfile.open(fpath, "w")

for idx, member in tqdm(enumerate(t.getmembers())):
    print(idx, member, flush=True)
    tdev.addfile(member, t.extractfile(member.name))

t.close()
tdev.close()
print("Finish")
```

This approach ensures compatibility with WIDS by avoiding the issues associated with gzipped tar files.

------------------------------------------------------------------------------

Issue #329

Q: How can I create a JSON file for WIDS from an off-the-shelf WDS shards dataset?

A: To create a JSON file for WIDS from an existing WDS shards dataset, you can
use the `widsindex` command provided by the webdataset distribution. This
command takes a list of shards or shardspecs and generates an index for them.
Here is a basic example of how to use `widsindex`:

```bash
widsindex shard1.tar shard2.tar shard3.tar > index.json
```

This command will create an `index.json` file that can be used for random access
in WIDS. For more details, you can refer to the example JSON file provided in
the webdataset repository: [imagenet-
train.json](https://storage.googleapis.com/webdataset/fake-imagenet/imagenet-
train.json).

------------------------------------------------------------------------------

Issue #319

Q: How can I handle more elaborate, hierarchical grouping schemes in WebDataset,
such as multi-frame samples or object image datasets with multiple supporting
files?

A: When dealing with complex hierarchical data structures in WebDataset, you can
use a naming scheme with separators like "." to define the hierarchy. For
example, you can name files as `sample_0.frames.0.jpg` and
`sample_0.frames.1.jpg` to represent different frames. However, for more complex
structures, it's often easier to use a flat naming scheme and express the
hierarchy in a JSON file. This way, you can number the files sequentially and
include the structure in the JSON metadata.

```json
{
    "frames": ["000.jpg", "001.jpg", "002.jpg"],
    "timestamps": [10001, 10002, 10003],
    "duration": 3
}
```

Files would be named as:

```
sample_0.000.jpg
sample_0.001.jpg
sample_0.002.jpg
sample_0.json
```

This approach simplifies the file naming and allows you to maintain complex structures within the JSON metadata.

------------------------------------------------------------------------------

Issue #316

Q: How can I handle the error caused by webdataset attempting to collate `npy` files of different `num_frames` lengths?

A: This error occurs because webdataset's default collation function cannot
handle `npy` files with varying `num_frames` lengths. To resolve this, you can
specify a custom collation function that can handle these variations.
Alternatively, you can set `combine_tensors=False` and `combine_scalars=False`
to prevent automatic collation. Here's how you can specify a custom collation
function:

```python
def custom_collate_fn(batch):
    images, texts = zip(*batch)
    return list(images), list(texts)

pipeline.extend([
    wds.select(filter_no_caption_or_no_image),
    wds.decode(handler=log_and_continue),
    wds.rename(image="npy", text="txt"),
    wds.map_dict(image=lambda x: x, text=lambda text: tokenizer(text)[0]),
    wds.to_tuple("image", "text"),
    wds.batched(args.batch_size, partial=not is_train, collation_fn=custom_collate_fn)
])
```

Alternatively, you can disable tensor and scalar combination:

```python
pipeline.extend([
    wds.select(filter_no_caption_or_no_image),
    wds.decode(handler=log_and_continue),
    wds.rename(image="npy", text="txt"),
    wds.map_dict(image=lambda x: x, text=lambda text: tokenizer(text)[0]),
    wds.to_tuple("image", "text"),
    wds.batched(args.batch_size, partial=not is_train, combine_tensors=False, combine_scalars=False)
])
```

------------------------------------------------------------------------------

Issue #314

Q: How can I detect when a tarball has been fully consumed by `tariterators.tar_file_expander`?

A: Detecting when a tarball has been fully consumed by
`tariterators.tar_file_expander` can be challenging, especially with remote
files represented as `io.BytesIO` objects. One approach is to add
`__index_in_shard__` metadata to the samples. When this index is 0, it indicates
that the last shard was fully consumed. Another potential solution is to hook
into the `close` method, which might be facilitated by fixing issue #311. This
would allow you to detect when the tar file is closed, signaling that it has
been fully processed.

```python
# Example of adding __index_in_shard__ metadata
for sample in tar_file_expander(fileobj):
    if sample['__index_in_shard__'] == 0:
        print("Last shard fully consumed")
```

```python
# Example of hooking into the close method (hypothetical)
class CustomTarFileExpander(tariterators.tar_file_expander):
    def close(self):
        super().close()
        print("Tar file fully consumed")
```

------------------------------------------------------------------------------

Issue #307

Q: Is it possible to skip loading certain files when reading a tar file with WebDataset?

A: When using WebDataset to read a tar file, it is not possible to completely
skip loading certain files (e.g., `jpg` files) into memory because WebDataset
operates on a streaming paradigm. This means that all bytes must be read from
the tar file, even if they are not used. However, you can use the `select_files`
argument in the `wds.WebDataset` class or `wds.tarfile_to_samples` to filter out
unwanted files. For more efficient access, consider using the "wids" interface,
which provides indexed access and only reads the necessary data from disk.

```python
import webdataset as wds

# Use select_files to filter out unwanted files
dataset = wds.WebDataset("path/to/dataset.tar", select_files=["*.txt"])

for sample in dataset:
    # Process only the txt files
    print(sample["txt"])
```

If performance is a concern, generating a new dataset with only the required fields might be the best approach.

------------------------------------------------------------------------------

Issue #303

Q: Why does the total number of steps per epoch change when using `num_workers > 0` in DDP training with WebDataset?

A: When using `num_workers > 0` in DDP training with WebDataset, the total
number of steps per epoch can change due to how the dataset is partitioned and
shuffled across multiple workers. To address this, you should remove the
`with_epoch` from the WebDataset and apply it to the WebLoader instead.
Additionally, consider adding cross-worker shuffling to ensure proper data
distribution.

```python
data = wds.WebDataset(self.url, resampled=True).shuffle(1000).map(preprocess_train)
loader = wds.WebLoader(data, pin_memory=True, shuffle=False, batch_size=20, num_workers=2).with_epoch(200)
```

For cross-worker shuffling:

```python
loader = wds.WebLoader(data, pin_memory=True, shuffle=False, batch_size=20, num_workers=2)
loader = loader.unbatched().shuffle(2000).batched(20).with_epoch(200)
```

------------------------------------------------------------------------------

Issue #300

Q: How can I prevent training from blocking when using WebDataset with large shards on the cloud?

A: To prevent training from blocking when using WebDataset with large shards,
you can increase the `num_workers` parameter in your `DataLoader`. This approach
leverages additional workers to handle the download and processing of shards
concurrently, thus minimizing idle time. For example, if downloading takes 25%
of the total processing time for one shard, you can increase `num_workers` to
accommodate this overhead. Additionally, you can adjust the `prefetch_factor` to
ensure that the next shard is downloaded before the current one is exhausted.

```python
dataset = wds.WebDataset(urls, cache_dir=cache_dir, cache_size=cache_size)
dataloader = DataLoader(dataset, num_workers=8, prefetch_factor=4)
```

By setting `verbose=True` in `WebDataset`, you can verify that each worker is
requesting different shards, ensuring efficient data loading.

------------------------------------------------------------------------------

Issue #291

Q: How can I skip a sample if decoding fails in NVIDIA DALI?

A: If you encounter issues with decoding samples (e.g., corrupt images) in
NVIDIA DALI, you can use the `handler` parameter to manage such errors. By
setting the `handler` to `warn_and_continue`, you can skip the problematic
samples without interrupting the data pipeline. This can be applied when
invoking methods like `.map`, `.map_tuple`, or `.decode`.

Example:
```python
import webdataset as wds
from webdataset.handlers import warn_and_continue

ds = (
    wds.WebDataset(url, handler=warn_and_continue, shardshuffle=True, verbose=verbose)
    .map(_mapper, handler=warn_and_continue)
    .to_tuple("jpg", "cls")
    .map_tuple(transform, identity, handler=warn_and_continue)
    .batched(batch_size)
)
```
This setup ensures that any decoding errors are handled gracefully, allowing the
data pipeline to continue processing subsequent samples.

------------------------------------------------------------------------------

Issue #289

Q: Can webdataset support interleaved datasets such as MMC4, where one example may have a text list with several images?

A: Yes, webdataset can support interleaved datasets like MMC4. You can represent
this structure similarly to how you would on a file system. An effective
approach is to use a `.json` file to reference the hierarchical structure,
including the text list and associated images. Then, include the image files
within the same sample. This method allows you to maintain the relationship
between the text and multiple images within a single dataset entry.

Example:
```json
{
  "text": ["text1", "text2"],
  "images": ["image1.jpg", "image2.jpg"]
}
```
Ensure that the images referenced in the `.json` file are included in the same sample directory.

------------------------------------------------------------------------------

Issue #283

Q: How can I provide credentials for reading objects from a private bucket using
`webdataset` with a provider other than AWS or GCS, such as NetApp?

A: To load a `webdataset` from a non-publicly accessible bucket with a provider
like NetApp, you can use the `pipe:` protocol in your URL. This allows you to
specify a shell command to read a shard, including any necessary authentication.
Ensure that the command works from the command line with the provided
credentials. For example, you can create a script that includes your `access key
id` and `secret access key` and use it in the `pipe:` URL.

Example:
```python
url = "pipe:./read_from_netapp.sh"
dataset = webdataset.WebDataset(url)
```

In `read_from_netapp.sh`, include the necessary commands to authenticate and read from your NetApp bucket.

------------------------------------------------------------------------------

Issue #282

Q: Why does the `with_epoch` configuration in WebDataset change behavior when
specifying `batch_size` in the DataLoader for distributed DDP training with
PyTorch Lightning?

A: The `with_epoch` method in WebDataset slices the data stream based on the
number of items, not batches. When you set `with_epoch` to a specific number, it
defines the number of samples per epoch. However, when you specify a
`batch_size` in the DataLoader, the effective number of epochs becomes the
number of batches, as each batch contains multiple samples. To align the epoch
size with your batch size, you need to divide the desired number of samples per
epoch by the batch size. For example, if you want 100,000 samples per epoch and
your batch size is 32, you should set `with_epoch(100_000 // 32)`.

```python
# Example adjustment
self.train_dataset = MixedWebDataset(self.cfg, self.dataset_cfg,
train=True).with_epoch(100_000 // self.cfg.TRAIN.BATCH_SIZE).shuffle(4000)
train_dataloader = torch.utils.data.DataLoader(self.train_dataset,
self.cfg.TRAIN.BATCH_SIZE, drop_last=True,
num_workers=self.cfg.GENERAL.NUM_WORKERS,
prefetch_factor=self.cfg.GENERAL.PREFETCH_FACTOR)
```

This ensures that the epoch size is consistent with the number of batches processed.

------------------------------------------------------------------------------

Issue #280

Q: How do I correctly use the `wds.DataPipeline` to create a dataset and get the expected batch size?

A: When using `wds.DataPipeline` to create a dataset, ensure that you iterate
over the dataset itself, not a loader. The correct pipeline order is crucial for
the expected output. Here is the corrected code snippet:

```python
dataset = wds.DataPipeline(
    wds.SimpleShardList(url),
    wds.shuffle(100),
    wds.split_by_worker,
    wds.tarfile_to_samples(),
    wds.shuffle(1000),
    wds.decode("torchrgb"),
    get_patches,
    wds.shuffle(10000),
    wds.to_tuple("big.jpg", "json"),
    wds.batched(16)
)

batch = next(iter(dataset))
batch[0].shape, batch[1].shape
```

The expected output should be `(torch.Size([16, 256, 256, 3]),
torch.Size([16]))`, reflecting the correct batch size of 16.

------------------------------------------------------------------------------

Issue #278

Q: Why is `with_epoch(N)` needed for multi-node training in WebDataset?

A: `with_epoch(N)` is essential for multi-node training with WebDataset because
it helps manage the concept of epochs in an infinite stream of samples. Without
`with_epoch(N)`, the epoch would continue indefinitely, which is problematic for
training routines that rely on epoch boundaries. WebDataset generates an
infinite stream of samples when resampling is chosen, making `with_epoch(N)` a
useful tool to define epoch boundaries. Unlike the `sampler` in PyTorch's
DataLoader, `set_epoch` is not needed in WebDataset because it uses the
IterableDataset interface, which does not support `set_epoch`.

```python
# Example usage of with_epoch(N) in WebDataset
import webdataset as wds

dataset = wds.WebDataset("data.tar").with_epoch(1000)
for epoch in range(num_epochs):
    for sample in dataset:
        # Training code here
```

This ensures that the dataset is properly segmented into epochs, facilitating multi-node training.

------------------------------------------------------------------------------

Issue #264

Q: How can I return the file name (without the extension) in the training loop using WebDataset?

A: To include the file name (without the extension) in the `metadata` dictionary
within your training loop, you can utilize the `sample["__key__"]` which
contains the stem of the file name. You can modify your pipeline to include a
step that adds this information to the `metadata`. Here is an example of how you
can achieve this:

```python
import webdataset as wds

def add_filename_to_metadata(sample):
    sample["metadata"]["filename"] = sample["__key__"]
    return sample

pipeline = [
    wds.SimpleShardList(input_shards),
    wds.split_by_worker,
    wds.tarfile_to_samples(handler=log_and_continue),
    wds.select(filter_no_caption_or_no_image),
    wds.decode("pilrgb", handler=log_and_continue),
    wds.rename(image="jpg;png;jpeg;webp", text="txt", metadata="json"),
    wds.map(add_filename_to_metadata),
    wds.map_dict(
        image=preprocess_img,
        text=lambda text: tokenizer(text),
        metadata=lambda x: x,
    ),
    wds.to_tuple("image", "text", "metadata"),
    wds.batched(args.batch_size, partial=not is_train),
]
```

This way, the `metadata` dictionary will include the file name without the
extension, accessible via `metadata["filename"]`.

------------------------------------------------------------------------------

Issue #262

Q: Are there plans to support WebP in WebDataset?

A: Yes, WebDataset can support WebP images. You can add a handler to
`webdataset.writer.default_handlers` or override the `encoder` in `TarWriter`.
ImageIO supports WebP natively, so it may be added by default in the future. For
now, you can manually handle WebP images by converting them to bytes and writing
them to the dataset.

```python
import io
import torch
import numpy
from PIL import Image
import webdataset as wds

sink = wds.TarWriter('test.tar')
image_byte = io.BytesIO()
image = Image.fromarray(numpy.uint8(torch.randint(0, 256, (256, 256, 3))))
image.save(image_byte, format='webp')
sink.write({
    '__key__': 'sample001',
    'txt': 'Are you ok?',
    'jpg': torch.rand(256, 256, 3).numpy(),
    'webp': image_byte.getvalue()
})
```

This approach allows you to store WebP images directly in your dataset.

------------------------------------------------------------------------------

Issue #261

Q: Why is my dataset created with `TarWriter` so large, and how can I reduce its size?

A: The large size of your dataset is due to the way tensors are being saved.
When saving individual rows, each tensor points to a large underlying byte array
buffer, resulting in excessive storage usage. To fix this, use
`torch.save(v.clone(), buffer)` to save only the necessary data. Additionally,
saving many small tensors incurs overhead. You can mitigate this by saving your
`.tar` files as `.tar.gz` files, which will be decompressed upon reading.
Another approach is to save batches of data and unbatch them upon reading.

```python
# Use clone to avoid saving large underlying buffers
for k, v in d.items():
    buffer = io.BytesIO()
    torch.save(v.clone(), buffer)
    obj[f"{k}.pth"] = buffer.getvalue()
```

```python
# Save as .tar.gz to reduce overhead
with wds.TarWriter(f"/tmp/dest.tar.gz") as sink:
    # Your saving logic here
```

------------------------------------------------------------------------------

Issue #260

Q: How can I set the epoch length in WebDataset, and why is the `with_epoch()` method name confusing?

A: The `with_epoch()` method in WebDataset is used to set the epoch length
explicitly, which is crucial for distributed training. However, the name
`with_epoch()` can be confusing as it doesn't clearly convey its purpose. A more
descriptive name like `.set_one_epoch_samples()` would make its functionality
clearer. To use this method, you can call it on your dataset object as shown
below:

```python
dataset = WebDataset("path/to/dataset").with_epoch(1000)
```

This sets the epoch length to 1000 samples. Improving the documentation with
more details and examples can help users understand its purpose better.

------------------------------------------------------------------------------

Issue #259

Q: Is there a way to use password-protected tar files during streaming?

A: Yes, you can use password-protected tar files during streaming by
incorporating streaming decoders in your data pipeline. For example, you can use
`curl` to download the encrypted tar files and `gpg` to decrypt them on the fly.
Here is a sample pipeline:

```bash
urls="pipe:curl -L -q https://storage.googleapis.com/bucket/shard-{000000..000999}.tar.encrypted | gpg --decrypt -o - -"
dataset=WebDataset(urls).decode(...) etc.
```

This approach allows you to handle encrypted tar files seamlessly within your data processing workflow.

------------------------------------------------------------------------------

Issue #257

Q: How can I efficiently load a subset of files in a sample to avoid reading unnecessary data?

A: To efficiently load a subset of files in a sample, you can use the
`select_files` option in WebDataset and `tarfiles_to_samples` to specify which
files to extract. This approach helps in avoiding the overhead of reading
unnecessary data. However, note that skipping data by seeking is usually not
significantly faster than reading the data due to the way hard drives work. The
most efficient method is to pre-select the files needed during training and
ensure your tar files contain only those files.

```python
import webdataset as wds

# Example of using select_files to specify which files to extract
dataset = wds.WebDataset("path/to/dataset.tar").select_files("main.jpg", "aux0.jpg", "aux1.jpg")

for sample in dataset:
    main_image = sample["main.jpg"]
    aux_image_0 = sample["aux0.jpg"]
    aux_image_1 = sample["aux1.jpg"]
    # Process the images as needed
```

By pre-selecting the necessary files, you can optimize your data loading process and improve training efficiency.

------------------------------------------------------------------------------

Issue #256

Q: Why does my WebDataset pipeline consume so much memory and crash during training?

A: The high memory consumption in your WebDataset pipeline is likely due to the
shuffle buffer size. WebDataset keeps a certain number of samples in memory for
shuffling, which can lead to high memory usage. For instance, if your
`_SAMPLE_SHUFFLE_SIZE` is set to 5000, it means that 5000 samples are kept in
memory. Reducing the shuffle buffer size can help mitigate this issue. Try
adjusting the `_SHARD_SHUFFLE_SIZE` and `_SAMPLE_SHUFFLE_SIZE` parameters to
smaller values and see if it reduces memory usage.

```python
_SHARD_SHUFFLE_SIZE = 1000  # Adjust this value
_SAMPLE_SHUFFLE_SIZE = 2000  # Adjust this value
```

Additionally, ensure that you are not holding onto unnecessary data by using `deepcopy` and `gc.collect()` effectively.

------------------------------------------------------------------------------

Issue #249

Q: Should I use WebDataset or TorchData for my data handling needs in PyTorch?

A: Both WebDataset and TorchData are compatible and can be used for data
handling in PyTorch. WebDataset is useful for backward compatibility and can
work without Torch. It also has added features like `extract_keys`,
`rename_keys`, and `xdecode` to make the transition to TorchData easier. If you
are starting a new project, TorchData is recommended due to its integration with
PyTorch. However, if you have existing code using WebDataset, there is no
urgency to switch, and you can continue using it with minimal changes.

```python
# Example using WebDataset
import webdataset as wds

dataset = wds.WebDataset("data.tar").decode("rgb").to_tuple("jpg", "cls")

# Example using TorchData
import torchdata.datapipes as dp

dataset = dp.iter.WebDataset("data.tar").decode("rgb").to_tuple("jpg", "cls")
```

------------------------------------------------------------------------------

Issue #247

Q: Does WebDataset support loading images from nested tar files?

A: Yes, WebDataset can support loading images from nested tar files by defining
a custom decoder using Python's `tarfile` library. You can create a function to
expand the nested tar files and add the images to the sample as if they were
part of it all along. This can be achieved using the `map` function in
WebDataset. Here is an example:

```python
import tarfile
import io

def expand_tar_files(sample):
    stream = tarfile.open(fileobj=io.BytesIO(sample["tar"]))
    for tarinfo in stream:
        name = tarinfo.name
        data = stream.extractfile(tarinfo).read()
        sample[name] = data

ds = WebDataset(...).map(expand_tar_files).decode(...)
```

This approach allows you to handle nested tar files and load the images seamlessly into your dataset.

------------------------------------------------------------------------------

Issue #245

Q: How do I enable caching for lists of shards when using `wds.ResampledShards` in a WebDataset pipeline?

A: To enable caching for lists of shards when using `wds.ResampledShards`, you
need to replace `tarfile_to_samples` with `cached_tarfile_to_samples` in your
pipeline. `ResampledShards` does not inherently support caching, but you can
achieve caching by modifying the pipeline to use the caching version of the
tarfile extraction function. Here is an example of how to modify your pipeline:

```python
dataset = wds.DataPipeline(
    wds.ResampledShards(urls),
    wds.cached_tarfile_to_samples(cache_dir='./_mycache'),
    wds.shuffle(shuffle_vals[0]),
    wds.decode(wds.torch_audio),
    wds.map(partial(callback, sample_size=sample_size, sample_rate=sample_rate, verbose=verbose, **kwargs)),
    wds.shuffle(shuffle_vals[1]),
    wds.to_tuple(audio_file_ext),
    wds.batched(batch_size),
).with_epoch(epoch_len)
```

This modification ensures that the extracted samples are cached, improving the efficiency of reloading shards.

------------------------------------------------------------------------------

Issue #244

Q: What is the recommended way of combining multiple data sources with a specified frequency for sampling from each?

A: To combine multiple data sources with specified sampling frequencies, you can
use the `RandomMix` function. This allows you to mix datasets at the sample
level with non-integer weights. For example, if you want to sample from dataset
`A` 1.45 times more frequently than from dataset `B`, you can specify the
weights accordingly. Here is an example:

```python
ds1 = WebDataset("A/{00..99}.tar")
ds2 = WebDataset("B/{00..99}.tar")
mix = RandomMix([ds1, ds2], [1.45, 1.0])
```

This setup ensures that samples are drawn from `ds1` 1.45 times more frequently than from `ds2`.

------------------------------------------------------------------------------

Issue #239

Q: Is it possible to filter a WebDataset to select only a subset of categories without creating a new dataset?

A: Yes, it is possible to filter a WebDataset to select only a subset of
categories using a simple map function. This approach is efficient as long as
you don't discard more than 50-80% of the samples. For example, you can use the
following code to filter samples based on their class:

```python
def select(sample):
    if sample["cls"] in [0, 3, 9]:
        return sample
    else:
        return None

dataset = wds.WebDataset(...).decode().map(select)
```

However, if you need smaller subsets of your data, it is recommended to create a
new WebDataset to achieve efficient I/O. This is due to the inherent cost of
random disk accesses when selecting a small subset without creating a new
dataset.

------------------------------------------------------------------------------

Issue #238

Q: Why does shard caching with `s3cmd` in the URL create only a single file in the `cache_dir`?

A: When using shard caching with URLs like `pipe:s3cmd get
s3://bucket/path-{00008..00086}.tar -`, the caching mechanism incorrectly
identifies the filename as `s3cmd`. This happens because the `pipe_cleaner`
function in `webdataset/cache.py` tries to guess the actual URL from the "pipe:"
specification by looking for words that start with "s3". Since "s3cmd" fits this
pattern, it mistakenly uses "s3cmd" as the filename, leading to only one shard
being cached. To fix this, you can override the mapping by specifying the
`url_to_name` option to `cached_url_opener`.

```python
# Example of overriding the mapping
dataset = WebDataset(url, url_to_name=lambda url: hashlib.md5(url.encode()).hexdigest())
```

This ensures that each URL is uniquely identified, preventing filename conflicts in the cache.

------------------------------------------------------------------------------

Issue #237

Q: How do I handle filenames with multiple periods in WebDataset to avoid issues with key interpretation?

A: In WebDataset, periods in filenames are used to support multiple extensions,
which can lead to unexpected key interpretations. For example, a filename like
`./235342 Track 2.0 (Clean Version).mp3` might be interpreted with a key of `0
(clean version).mp3`. This is by design to facilitate downstream pipeline
processing. To avoid issues, it's recommended to handle this during dataset
creation. You can also use "glob" patterns like `*.mp3` to match extensions.
Here's an example of using glob patterns:

```python
import webdataset as wds

dataset = wds.WebDataset("data.tar").decode("torchrgb").to_tuple("*.mp3")
```

This approach ensures that your filenames are correctly interpreted and processed.

------------------------------------------------------------------------------

Issue #236

Q: How does WebDataset convert my local JPG images and text into .npy files?

A: WebDataset does not inherently convert your local JPG images and text into
.npy files. Instead, it writes data in the format specified by the file
extension in the sample keys. For example, if you write a sample with
`{"__key__": "xyz", "image.jpg": some_tensor}`, it will save the tensor as a
JPEG file named `xyz.image.jpg`. Conversely, if you write the same tensor with
`{"__key__": "xyz", "image.npy": some_tensor}`, it will save the tensor in NPY
format as `xyz.image.npy`. The conversion happens based on the file extension
provided in the sample keys.

```python
writer = ShardWriter(...)
image = rand((256,256,3))
sample["__key__"] = "dataset/sample00003"
sample["data.npy"] = image
sample["jpg"] = image
sample["something.png"] = image
writer.write(sample)
```

Upon reading, both ".jpg" and ".npy" files can be turned into tensors if you
call `.decode` with the appropriate arguments.

------------------------------------------------------------------------------

Issue #233

Q: How can I correctly split WebDataset shards across multiple nodes and workers in a PyTorch XLA setup?

A: To correctly split WebDataset shards across multiple nodes and workers in a
PyTorch XLA setup, you need to use the `split_by_node` and `split_by_worker`
functions. This ensures that each node and worker processes a unique subset of
the data. Here is an example of how to set this up:

```python
import os
import webdataset as wds
import torch
import torch.distributed as dist

if __name__ == "__main__":
    try:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group("nccl")
    except KeyError:
        rank = 0
        world_size = 1
        local_rank = 0
        dist.init_process_group(
            backend="nccl",
            init_method="tcp://127.0.0.1:12584",
            rank=rank,
            world_size=world_size,
        )

    dataset = wds.DataPipeline(
        wds.SimpleShardList("source-{000000..000999}.tar"),
        wds.split_by_node,
        wds.split_by_worker,
    )

    for idx, i in enumerate(iter(dataset)):
        if idx == 2:
            break
        print(f"rank: {rank}, world_size: {world_size}, {i}")
```

To ensure deterministic shuffling across nodes, use `detshuffle`:

```python
dataset = wds.DataPipeline(
    wds.SimpleShardList("source-{000000..000016}.tar"),
    wds.detshuffle(),
    wds.split_by_node,
    wds.split_by_worker,
)
```

This setup ensures that each node and worker processes a unique subset of the data, avoiding redundant processing.

------------------------------------------------------------------------------

Issue #227

Q: How can I leverage Apache Beam to build a WebDataset in Python for a large training set with complex pre-processing?

A: To build a WebDataset using Apache Beam, you can process shards in parallel
and each shard sequentially. Apache Beam allows you to handle large-scale data
processing efficiently. Below is a sample code snippet to write data to a
WebDataset tar file via Beam:

```python
import apache_beam as beam
from webdataset import ShardWriter

def process_element(element):
    # Perform your complex pre-processing here
    processed_sample = ...  # do something with element
    return processed_sample

def write_to_shard(element, shard_name):
    with ShardWriter(shard_name) as writer:
        writer.write(element)

def run_pipeline(input_pattern, output_pattern):
    with beam.Pipeline() as p:
        (p
         | 'ReadInput' >> beam.io.ReadFromText(input_pattern)
         | 'ProcessElement' >> beam.Map(process_element)
         | 'WriteToShard' >> beam.Map(write_to_shard, output_pattern))

input_pattern = 'gs://source_bucket/shard-*'
output_pattern = 'gs://dest_bucket/shard-{shard_id}.tar'
run_pipeline(input_pattern, output_pattern)
```

This example reads input data, processes each element, and writes the processed
data to a WebDataset shard. Adjust the `process_element` function to fit your
specific pre-processing needs.

------------------------------------------------------------------------------

Issue #225

Q: How can I ensure that WebLoader-generated batches are different when using
multi-node training with WebDataset for DDP?

A: When using WebDataset for DDP (Distributed Data Parallel) training, you need
to ensure that each node processes different data to avoid synchronization
issues. Here are some strategies:

1. **Use `resampled=True` and `with_epoch`**: This ensures that each worker gets
an infinite stream of samples, preventing DDP from hanging due to uneven data
distribution.
   ```python
   dataset = WebDataset(..., resampled=True).with_epoch(epoch_length)
   ```

2. **Equal Shard Distribution**: Ensure shards are evenly distributed across
nodes and use `.repeat(2).with_epoch(n)` to avoid running out of data.
   ```python
   dataset = WebDataset(...).repeat(2).with_epoch(epoch_length)
   ```

3. **Avoid Shard Splitting**: Have the same set of shuffled shards on all nodes,
effectively multiplying your epoch size by the number of workers.

4. **Nth Sample Selection**: Select every nth sample on each of the n nodes to ensure even distribution.

For validation, ensure the dataset size is divisible by the number of nodes to avoid hanging:
```python
dataset = dataset.batch(torch.distributed.get_world_size(), drop_last=True)
dataset = dataset.unbatch()
dataset = dataset.sharding_filter()
```

These methods help maintain synchronization and prevent DDP from hanging due to uneven data distribution.

------------------------------------------------------------------------------

Issue #220

Q: Why is my `.flac` sub-file with a key starting with `_` being decoded as
UTF-8 instead of as a FLAC file in WebDataset?

A: In WebDataset, keys that start with an underscore (`_`) are treated as
metadata and are automatically decoded as UTF-8. This is based on the convention
that metadata keys start with an underscore and are in UTF-8 format. However,
the system now checks for keys that start with double underscores (`__`), so
your case should work if you update your key to start with a double underscore.
This ensures that the `.flac` extension is recognized and the file is decoded
correctly.

```python
# Example key update
"_gbia0001334b.flac"  # Original key
"__gbia0001334b.flac"  # Updated key
```

This change reserves the single underscore for metadata while allowing other
files to be decoded based on their extensions.

------------------------------------------------------------------------------

Issue #219

Q: How do I resolve the `AttributeError: module 'webdataset' has no attribute 'ShardList'` error when using WebDataset?

A: The `ShardList` class has been renamed to `SimpleShardList` in WebDataset v2.
Additionally, the `splitter` argument is now called `nodesplitter`. To fix the
error, update your code to use `SimpleShardList` and `nodesplitter`. Here is an
example:

```python
urls = list(braceexpand.braceexpand("dataset-{000000..000999}.tar"))
dataset = wds.SimpleShardList(urls, nodesplitter=wds.split_by_node, shuffle=False)
dataset = wds.Processor(dataset, wds.url_opener)
dataset = wds.Processor(dataset, wds.tar_file_expander)
dataset = wds.Processor(dataset, wds.group_by_keys)
```

This should resolve the `AttributeError` and ensure your code is compatible with WebDataset v2.

------------------------------------------------------------------------------

Issue #216

Q: How can I use ShardWriter to write directly to a gcloud URL without storing shards locally?

A: The recommended usage of ShardWriter is to write to local disk and then copy
to the cloud. However, if your dataset is too large to store locally, you can
modify the ShardWriter to write directly to a gcloud URL. By changing the line
in `webdataset/writer.py` from:

```python
self.tarstream = TarWriter(open(self.fname, "wb"), **self.kw)
```

to:

```python
self.tarstream = TarWriter(self.fname, **self.kw)
```

you can enable direct writing to the cloud. This modification allows the
`TarWriter` to handle the gcloud URL directly, bypassing the need to store the
shards locally.

------------------------------------------------------------------------------

Issue #212

Q: How does WebDataset handle sharding and caching when downloading data from S3?

A: When using WebDataset with S3 and multiple shards, the shards are accessed
individually and handled in a streaming fashion by default, meaning nothing is
cached locally. If caching is enabled, each shard is first downloaded completely
before being used for further reading. This can cause a delay in training start
time, as the first shard must be fully downloaded before training begins. To
customize the cache file names, you can override the `url_to_name` argument.
Here is an example:

```python
dataset = wds.WebDataset('pipe:s3 http://url/dataset-{001..099}.tar', url_to_name=lambda url: url.split('/')[-1])
```

This behavior ensures that the data is processed correctly, but it may introduce delays when caching is enabled.

------------------------------------------------------------------------------

Issue #211

Q: How can I use ShardWriter to write data to remote URLs?

A: ShardWriter is designed to write data to local paths for simplicity and
better error handling. However, you can upload the data to a remote URL by using
a post-processing hook. This hook allows you to define a function that uploads
the shard to a remote location after it has been written locally. For example,
you can use the `gsutil` command to upload the shard to a Google Cloud Storage
bucket and then delete the local file:

```python
def upload_shard(fname):
    os.system(f"gsutil cp {fname} gs://mybucket")
    os.unlink(fname)

with ShardWriter(..., post=upload_shard) as writer:
    ...
```

This approach ensures that ShardWriter handles local file operations while you manage the remote upload process.

------------------------------------------------------------------------------

Issue #210

Q: How can I handle collation of dictionaries when using `default_collation_fn` in WebDataset?

A: The `default_collation_fn` in WebDataset expects samples to be a list or
tuple, not dictionaries. If you need to collate dictionaries, you can use a
custom collate function. Starting with PyTorch 1.11, you can use
`torch.utils.data.default_collate` which can handle dictionaries. You can pass
this function to the `batched` method. Here is an example of how to use a custom
collate function:

```python
from torch.utils.data import default_collate

def custom_collate_fn(samples):
    return default_collate(samples)

# Use the custom collate function
dataset = WebDataset(...).batched(20, collation_fn=custom_collate_fn)
```

This approach allows you to handle dictionaries without manually converting them to tuples.

------------------------------------------------------------------------------

Issue #209

Q: How can I ensure that each batch contains only one description per image
while using the entire dataset for training with WebDataset?

A: To ensure that each batch contains only one description per image while still
using the entire dataset for training with WebDataset, you can use a custom
transformation function. This function can be applied to the dataset to filter
and collate the samples as needed. Here is an example of how you can achieve
this:

```python
def my_transformation(src):
    seen_images = set()
    for sample in src:
        image_id = sample['image_id']
        if image_id not in seen_images:
            seen_images.add(image_id)
            yield sample

dataset = dataset.compose(my_transformation)
```

In this example, `my_transformation` ensures that each image is only included
once per batch by keeping track of seen images. You can further customize this
function to fit your specific requirements.

------------------------------------------------------------------------------

Issue #201

Q: How can I subsample a large dataset in the web dataset format without significantly slowing down iteration speed?

A: To subsample a large dataset like LAION 400M efficiently, you should perform
the `select(...)` operation before any decoding or data augmentation steps, as
these are typically the slowest parts of the pipeline. If you only need a small
percentage of the samples, it's best to generate the subset ahead of time as a
new dataset. You can use a small WebDataset/TarWriter pipeline to create the
subset, possibly with parallelization tools like `ray`, or use commands like
`tarp proc ... | tarp split ...`. For dynamic selection, consider splitting your
dataset into shards based on the categories you want to filter on. This approach
avoids the performance hit from random file accesses.

```python
def filter(sample):
    return sample['metadata'] in metadata_list

dataset = wds.WebDataset("path/to/dataset")
dataset = dataset.select(filter).decode("pil").to_tuple("jpg", "json")
```

By organizing your pipeline efficiently, you can maintain high iteration speeds while subsampling your dataset.

------------------------------------------------------------------------------

Issue #200

Q: How can I skip reading certain files in a tar archive when using the WebDataset library?

A: You cannot avoid reading files in a tar archive when using the WebDataset
library, especially in large-scale applications where files are streamed from a
network. However, you can avoid decoding samples that you don't need by using
the `map` or `select` functions before the `decode` statement. This approach is
recommended to improve efficiency. If you need fast random access to tar files
stored on disk, consider using the `wids` library, which provides this
capability but requires the files to be fully downloaded.

```python
# Example of using map to filter out unwanted samples before decoding
dataset = wds.WebDataset("data.tar").map(filter_function).decode("pil")
```

```python
# Example of using select to filter out unwanted samples before decoding
dataset = wds.WebDataset("data.tar").select(filter_function).decode("pil")
```

------------------------------------------------------------------------------

Issue #199

Q: Why is `wds.Processor` not included in the `v2` or `main` branch of
WebDataset, and how can I add preprocessing steps to the data?

A: The `wds.Processor` class, along with `wds.Shorthands` and `wds.Composable`,
has been removed from the `v2` and `main` branches of WebDataset because the
architecture for pipelines has been updated to align more closely with
`torchdata`. To add preprocessing steps to your data, you can use the `map`
function, which allows you to apply a function to each sample in the dataset.
The function you provide to `map` will receive the complete sample as an
argument. Additionally, you can write custom pipeline stages as callables. Here
is an example:

```python
def process(source):
    for sample in source:
        # Your preprocessing code goes here
        yield sample

ds = WebDataset(...).compose(process)
```

This approach ensures you have access to all the information in each sample for preprocessing.

------------------------------------------------------------------------------

Issue #196

Q: How can I speed up the `rsample` function when working with WebDataset tar files?

A: To speed up the `rsample` function in WebDataset, it's more efficient to
apply `rsample` on shards rather than individual samples. This approach avoids
the overhead of sequential reading of the entire tar file, which can be slow. By
sampling shards, you can reduce I/O operations and improve performance.
Additionally, using storage servers like AIStore can help perform sampling on
the server side, leveraging random access capabilities.

```python
# Example of using rsample on shards
import webdataset as wds

dataset = wds.WebDataset("shards-{000000..000099}.tar")
dataset = dataset.rsample(0.1)  # Apply rsample on shards
```

This method ensures efficient data processing without the need for random access on tar files.

------------------------------------------------------------------------------

Issue #194

Q: How can I use `WebDataset` with `DistributedDataParallel` (DDP) if `ddp_equalize` is not available?

A: The `ddp_equalize` method is no longer available in `WebDataset`. Instead,
you can use the `.repeat(2).with_epoch(n)` method to ensure each worker
processes the same number of batches. Here, `n` should be the total number of
samples divided by the batch size. This approach helps balance the dataset
across workers, though some batches may be missing or repeated. Alternatively,
consider using `torchdata` for better sharding and distributed training support.

Example:
```python
urls = "./shards/imagenet-train-{000000..001281}.tar"
dataset_size, batch_size = 1282000, 64
dataset = wds.WebDataset(urls).decode("pil").shuffle(5000).batched(batch_size, partial=False)
loader = wds.WebLoader(dataset, num_workers=4)
loader = loader.repeat(2).with_epoch(dataset_size // batch_size)
```

For more robust solutions, consider using PyTorch's synchronization methods or the upcoming `WebIndexedDataset`.

------------------------------------------------------------------------------

Issue #187

Q: Why is it recommended to avoid batching in the `DataLoader` and instead batch
in the dataset and then rebatch after the loader in PyTorch?

A: It is generally most efficient to avoid batching in the `DataLoader` because
data transfer between different processes (when using `num_workers` greater than
zero) is more efficient with larger amounts of data. Batching in the dataset and
then rebatching after the loader can optimize this process. For example, using
the WebDataset Dataloader, you can batch/rebatch as follows:

```python
loader = wds.WebLoader(dataset, num_workers=4, batch_size=8)
loader = loader.unbatched().shuffle(1000).batched(12)
```

This approach ensures efficient data transfer and processing.

------------------------------------------------------------------------------

Issue #185

Q: How can I include the original filename in the metadata when iterating through a WebDataset/WebLoader?

A: To include the original filename in the metadata when iterating through a
WebDataset/WebLoader, you can define a function to add the filename to the
metadata dictionary and then apply this function using `.map()`. Here's how you
can do it:

1. Define a function `getfname` to add the filename to the metadata:
    ```python
    def getfname(sample):
        sample["metadata"]["filename"] = sample["__key__"]
        return sample
    ```

2. Add `.map(getfname)` to your pipeline after the `rename` step:
    ```python
    pipeline = [
        wds.SimpleShardList(input_shards),
        wds.split_by_worker,
        wds.tarfile_to_samples(handler=log_and_continue),
        wds.select(filter_no_caption_or_no_image),
        wds.decode("pilrgb", handler=log_and_continue),
        wds.rename(image="jpg;png;jpeg;webp", text="txt", metadata="json"),
        wds.map(getfname),  # Add this line
        wds.map_dict(
            image=preprocess_img,
            text=lambda text: tokenizer(text),
            metadata=lambda x: x,
        ),
        wds.to_tuple("image", "text", "metadata"),
        wds.batched(args.batch_size, partial=not is_train),
    ]
    ```

This will ensure that the `metadata` dictionary in each sample contains the original filename under the key `filename`.

------------------------------------------------------------------------------

Issue #179

Q: How can I enable writing TIFF images using `TarWriter` in the `webdataset` library?

A: To enable writing TIFF images using `TarWriter` in the `webdataset` library,
you need to extend the default image encoder to handle TIFF format. This can be
done by modifying the `make_handlers` and `imageencoder` functions.
Specifically, you should add a handler for TIFF in `make_handlers` and ensure
that the quality is not reduced in `imageencoder`. Here are the necessary
modifications:

```python
# Add this in webdataset.writer.make_handlers
add_handlers(handlers, "tiff", lambda data: imageencoder(data, "tiff"))

# Modify this in webdataset.writer.imageencoder
if format in ["JPEG", "tiff"]:
    opts = dict(quality=100)
```

These changes will allow you to save TIFF images, which is useful for storing
binary masks in float32 format that cannot be saved using PNG, JPG, or PPM.

------------------------------------------------------------------------------

Issue #177

Q: Can we skip some steps when resuming training to avoid loading unused data into memory?

A: The recommended way to handle resuming training without reloading unused data
is to use the `resampled=True` option in WebDataset or the `wds.resampled`
pipeline stage. This approach ensures consistent training statistics upon
restarting, eliminating the need to skip samples. Achieving "each sample exactly
once per epoch" is complex, especially when restarting mid-epoch, and depends on
your specific environment (single/multiple nodes, worker usage, etc.).
WebDataset provides primitives like `slice`, `detshuffle`, and `rng=`, but it
can't automate this due to lack of environment context.

```python
# Example usage of resampled=True
import webdataset as wds

dataset = wds.WebDataset("data.tar").resampled(True)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32)
```

For more advanced handling, consider using frameworks like torchdata or Ray
Data, which have native support for WebDataset and their own mechanisms for
handling restarts.

------------------------------------------------------------------------------

Issue #172

Q: Why is the epoch count in `detshuffle` not incrementing across epochs in my WebDataset pipeline?

A: This issue arises because the default DataLoader setup in PyTorch creates new
worker processes each epoch when `persistent_workers=False`. This causes the
`detshuffle` instance to reset its epoch count, starting from -1 each time. To
maintain the epoch state across epochs, set `persistent_workers=True` in your
DataLoader. This ensures that the worker processes persist and the epoch state
is maintained. Alternatively, you can use resampling

------------------------------------------------------------------------------

Issue #171

Q: Why do I get an `ImportError: cannot import name 'PytorchShardList' from 'webdataset'` and how can I fix it?

A: The `PytorchShardList` class is not available in the version of the
`webdataset` package you are using. It was present in version 0.1 but has since
been replaced. To resolve this issue, you should use `SimpleShardList` instead.
Update your import statement as follows:

```python
from webdataset import SimpleShardList
```

This should resolve the import error and allow you to use the functionality
provided by `SimpleShardList`. Always refer to the latest documentation for the
most up-to-date information on the package.

------------------------------------------------------------------------------

Issue #170

Q: How can I correctly use glob patterns with WebDataset when my data is stored in Google Cloud Storage (GCS)?

A: When using WebDataset with data stored in GCS, glob patterns like
`training_*` are not automatically expanded. This can lead to unintended
behavior, such as treating the pattern as a single shard. To correctly use glob
patterns, you should manually list the files using a command like `gsutil ls`
and then pass the resulting list to WebDataset. Here’s an example:

```python
import os
import webdataset as wds

shard_list = list(os.popen("gsutil ls gs://BUCKET/PATH/training_*.tar").readlines())
train_data = wds.WebDataset(shard_list, shardshuffle=True, repeat=True)
```

This ensures that all matching files are included in the dataset, providing the
expected randomness and variety during training.

------------------------------------------------------------------------------

Issue #163

Q: How do I update my WebDataset code to use the new `wds.DataPipeline`
interface instead of the outdated `.compose(...)` method?

A: The `wds.DataPipeline` interface is the recommended way to work with
WebDataset as it is easier to use and extend. The `.compose(...)` method is
outdated and no longer includes the `source_` method. To update your code, you
can replace the `.compose(...)` method with a custom class that inherits from
`wds.DataPipeline` and `wds.compat.FluidInterface`. Here is an example of how to
re-implement `SampleEqually` using the new interface:

```python
import webdataset as wds

class SampleEqually(wds.DataPipeline, wds.compat.FluidInterface):
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

You can then use this class to sample equally from multiple datasets. This
approach ensures compatibility with the latest WebDataset API.

------------------------------------------------------------------------------

Issue #154

Q: How can I split a WebDataset into training and evaluation sets based on
target labels during runtime in PyTorch Lightning?

A: To split a WebDataset into training and evaluation sets based on target
labels during runtime, you can use the `select` method to filter samples
according to their labels. This approach allows you to dynamically create
different datasets for training and evaluation. For instance, you can filter out
samples with labels 0 and 1 for training and use samples with label 2 for
evaluation. Here is an example:

```python
training_ds = WebDataset(...).decode(...).select(lambda sample: sample["kind"] in [0, 1])
training_dl = WebDataloader(training_ds, ...)

val_ds = WebDataset(...).decode(...).select(lambda sample: sample["kind"] == 2)
val_dl = WebDataloader(val_ds, ...)
```

This method ensures that your training and evaluation datasets are correctly
split based on the target labels during runtime.

