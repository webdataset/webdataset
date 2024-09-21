
# WebDataset FAQ

This is a Frequently Asked Questions file for WebDataset.  It is
automatically generated from selected WebDataset issues using AI.

Since the entries are generated automatically, not all of them may
be correct.  When in doubt, check the original issue.

------------------------------------------------------------------------------

Issue #367

Q: How can I sample sequences of frames from large video datasets using WebDataset?

A: To sample sequences of frames from large video datasets with WebDataset, you
can precompute sampled sequences of frames and treat each collection as a batch.
Alternatively, you can split your videos into shorter clips with overlapping
frames, generate multiple samples from each clip, and shuffle the resulting
sequences. Here's a code snippet demonstrating how to generate and shuffle five-
frame sequences from 50-frame clips:

```python
from webdataset import WebDataset
import random

ds = WebDataset("video-clips-{000000..000999}.tar").decode()

def generate_clips(src):
    for sample in src:
        # assume that each video clip sample contains sample.000.jpg to sample.049.jpg images
        clip = [sample["%03d.jpg" % i] for i in range(50)]
        starts = random.sample(range(46), 10)  # Choose 10 starting points
        key = sample["__key__"]
        for i in starts:
            yield {
               "__key__": f"{key}-{i}",
               "sequence": clip[i:i+5],
            }

ds = ds.compose(generate_clips).shuffle(1000)
```

This approach allows you to work with large datasets by handling smaller,
manageable sequences, which can be efficiently preprocessed and shuffled to
create a diverse training set.

------------------------------------------------------------------------------

Issue #364

Q: How can I ensure that each validation sample is seen exactly once per epoch
in a multi-node setup using WebDataset with FSDP?

A: When using WebDataset in a multi-node setup with Fully Sharded Data Parallel
(FSDP), you can ensure that each validation sample is seen exactly once per
epoch by assigning each shard to a specific GPU. Since you have an equal number
of shards and GPUs, you can map each shard to a GPU. For the shard that is about
half the size, you can either accept that the corresponding GPU will do less
work, or you can split another shard to balance the load. To ensure that each
sample is loaded exactly once, you can use the `wds.ResampledShards` function
without resampling, and avoid using `ddp_equalize` since it is designed for
training rather than validation. Here's an example of how you might set up your
validation dataset:

```py
val_dataset = wds.DataPipeline(
    wds.ResampledShards(
        os.path.join('path', 'to',  'val_samples_{0000...xxxx}.tar')
    ),
    wds.tarfile_to_samples(),
    wds.decode(),
    wds.to_tuple("input.npy", "target.npy"),
    wds.batched(1)
).with_length(num_val_samples)
```

To ensure that the validation loop stops after all samples have been loaded, you
can use the length of the dataset to control the number of iterations in your
validation loop. This way, you can manually iterate over the dataset and stop
when you've reached the total number of samples.

------------------------------------------------------------------------------

Issue #331

Q: How can I handle gzipped tar files with WebDataset/WIDS?

A: When working with gzipped tar files in WebDataset or WIDS, it's important to
understand that random access to compressed files is not straightforward due to
the nature of compression. However, Python's `tarfile` library can handle gzip-
compressed streams using `tarfile.open("filename.tar.gz", "r:gz")`. For WIDS,
the best practice is to use uncompressed `.tar` files for the dataset, which
allows for efficient random access. If storage is a concern, you can compress
individual files within the tar archive (e.g., `.json.gz` instead of `.json`).
This approach provides a balance between storage efficiency and compatibility
with WIDS. Here's an example of how to compress individual files:

```python
import tarfile
import gzip

# Compress individual files and add them to a tar archive
with tarfile.open('archive.tar', 'w') as tar:
    with open('file.json', 'rb') as f_in:
        with gzip.open('file.json.gz', 'wb') as f_out:
            f_out.writelines(f_in)
    tar.add('file.json.gz', arcname='file.json.gz')
```

Remember that for WebDataset, you can use `.tar.gz` files directly, as it
supports on-the-fly decompression. If you encounter datasets that are not in
order, you can repack them using GNU tar with sorting to ensure that
corresponding files are adjacent, which is a requirement for WebDataset.

------------------------------------------------------------------------------

Issue #329

Q: How can I create a JSON metafile for random access in a WebDataset?

A: To create a JSON metafile for a WebDataset, you can use the `widsindex`
command that comes with the webdataset package. This command generates an index
file for a given list of WebDataset shards. The index file is in JSON format and
allows for efficient random access to the dataset. Here's a simple example of
how to use `widsindex`:

```bash
widsindex mydataset-0000.tar mydataset-0001.tar > mydataset-index.json
```

This command will create a JSON file named `mydataset-index.json` that contains
the index for the shards `mydataset-0000.tar` and `mydataset-0001.tar`.

------------------------------------------------------------------------------

Issue #319

Q: How can I handle complex hierarchical data structures in WebDataset?

A: When dealing with complex hierarchical data structures in WebDataset, it's
often more practical to use a flat file naming scheme and express the hierarchy
within a JSON metadata file. This approach simplifies the file naming while
allowing for detailed structuring of the data. You can sequentially number the
files and reference them in the JSON, which contains the structure of your
dataset, including frame order, timestamps, and other relevant information.

For example, instead of trying to express the hierarchy in the file names, you can name your files like this:

```
sample_0.000.jpg
sample_0.001.jpg
sample_0.002.jpg
sample_0.json
```

And then use a JSON file to define the structure:

```json
{
    "frames": ["000.jpg", "001.jpg", "002.jpg"],
    "timestamps": [10001, 10002, 10003],
    "duration": 3
}
```

This method keeps the file naming simple and leverages the JSON file to maintain
the hierarchical relationships within the dataset.

------------------------------------------------------------------------------

Issue #316

Q: Why am I getting a ValueError when trying to batch variable-length numpy arrays using webdataset?

A: The error you're encountering is due to the attempt to collate numpy arrays
with different shapes into a single batch. Since the `num_frames` dimension
varies, you cannot directly convert a list of such arrays into a single numpy
array without padding or truncating them to a uniform size. To resolve this, you
can specify a custom collation function that handles variable-length sequences
appropriately. This function can either pad the sequences to the same length or
store them in a data structure that accommodates variable lengths, such as a
list or a padded tensor. Here's an example of how to specify a custom collation
function:

```python
def custom_collate_fn(batch):
    # Handle variable-length sequences here, e.g., by padding
    # Return the batch in the desired format
    return batch

pipeline.extend([
    # ... other pipeline steps ...
    wds.batched(args.batch_size, collation_fn=custom_collate_fn, partial=not is_train)
])
```

By providing a custom collation function, you can ensure that the data is
prepared in a way that is compatible with your model's input requirements.

------------------------------------------------------------------------------

Issue #307

Q: Can I skip loading large files in a tar file when using WebDataset?

A: When working with `WebDataset`, it is not possible to skip the reading of
files within a tar archive that you do not need. The library operates on a
streaming basis, which means that all bytes are read sequentially. However, you
can filter out unwanted data after it has been read into memory. If performance
is a concern, consider creating a new dataset containing only the necessary
files. For indexed access to WebDataset files, you can use the "wids" interface,
which reads only the data you use from disk when working with local files.

Here's a short example of filtering out unwanted data after reading:

```python
dataset = wds.WebDataset(["path/to/dataset.tar"])
keys_to_keep = ["__key__", "__url__", "txt"]

def filter_keys(sample):
    return {k: sample[k] for k in keys_to_keep if k in sample}

filtered_dataset = dataset.map(filter_keys)
```

------------------------------------------------------------------------------

Issue #303

Q: Why does the number of steps per epoch change when increasing `num_workers` in DDP training with Webdataset?

A: When using multiple workers in a distributed data parallel (DDP) training
setup with Webdataset, the number of steps per epoch may change if the epoch
size is not properly configured to account for the parallelism introduced by the
workers. The `with_epoch` method should be applied to the `WebLoader` instead of
the `WebDataset` to ensure that the dataset is correctly divided among the
workers. Additionally, to maintain proper shuffling across workers, you may need
to add cross-worker shuffling. Here's an example of how to configure the loader:

```python
data = wds.WebDataset(self.url, resampled=True).shuffle(1000).map(preprocess_train)
loader = wds.WebLoader(data, pin_memory=True, shuffle=False, batch_size=20, num_workers=2).with_epoch(...)
```

For cross-worker shuffling, you can modify the loader like this:

```python
loader = loader.unbatched().shuffle(2000).batched(20).with_epoch(200)
```

------------------------------------------------------------------------------

Issue #291

Q: How can I skip a corrupt image sample when using NVIDIA DALI for data loading?

A: When working with NVIDIA DALI for data loading, you can handle corrupt or
missing data by using the `handler` parameter. This parameter allows you to
specify a behavior when a decoding error occurs. For example, you can use
`warn_and_continue` to issue a warning and skip the problematic sample, allowing
the data pipeline to continue processing the remaining samples. This is
particularly useful when dealing with large datasets where some samples may be
corrupt or unreadable.

Here's a short code example demonstrating how to use the `handler` parameter:

```python
from nvidia.dali.plugin import pytorch
import webdataset as wds

def warn_and_continue(e):
    print("Warning: skipping a corrupt sample.", e)

ds = (
    wds.WebDataset(url, handler=warn_and_continue, shardshuffle=True, verbose=verbose)
    .map(_mapper, handler=warn_and_continue)
    .to_tuple("jpg", "cls")
    .map_tuple(transform, identity, handler=warn_and_continue)
    .batched(batch_size)
)
```

By passing `warn_and_continue` to the `.map`, `.map_tuple`, or `.decode`
methods, you instruct DALI to handle exceptions gracefully and continue with the
next sample.

------------------------------------------------------------------------------

Issue #289

Q: Can WebDataset support interleaved datasets such as MMC4, where one example
may include a list of texts with several images?

A: Yes, WebDataset can support interleaved datasets like MMC4. You can organize
your dataset by creating a `.json` file that contains the hierarchical structure
and references to the image files. This `.json` file acts as a manifest for each
sample, detailing the associated text and images. The image files themselves are
stored alongside the `.json` file. Here's a simple example of how you might
structure a `.json` file for an interleaved dataset:

```json
{
  "text": ["This is the first text", "This is the second text"],
  "images": ["image1.jpg", "image2.jpg", "image3.jpg"]
}
```

And in your dataset, you would have the `.json` file and the referenced images in the same sample directory or archive.

------------------------------------------------------------------------------

Issue #283

Q: How can I authenticate to read objects from a private bucket with WebDataset?

A: To authenticate and read objects from a private bucket using WebDataset, you
need to provide the necessary credentials to the underlying command line
programs that WebDataset uses for data access. If you are using a storage
provider like NetApp, which is not directly supported by WebDataset's built-in
protocols, you can use the `pipe:` protocol to specify a custom command that
includes the necessary authentication steps. For example, you can create a shell
script that uses your storage provider's CLI tools to authenticate with your
`access key id` and `secret access key`, and then pass this script to
WebDataset:

```bash
# auth_script.sh
# This script authenticates and downloads a shard from a private bucket
# Replace <ACCESS_KEY>, <SECRET_KEY>, <BUCKET_NAME>, and <SHARD_NAME> with your actual values
netappcli --access-key <ACCESS_KEY> --secret-key <SECRET_KEY> download <BUCKET_NAME>/<SHARD_NAME>
```

Then, use this script with WebDataset:

```python
import webdataset as wds

# Use the 'pipe:' protocol with your authentication script
dataset = wds.WebDataset("pipe:./auth_script.sh")
```

Ensure that your script has the necessary permissions to be executed and that it
correctly handles the authentication and data retrieval process.

------------------------------------------------------------------------------

Issue #278

Q: Why is `with_epoch(N)` needed for multinode training with WebDataset?

A: When using WebDataset for training models in PyTorch, the `with_epoch(N)`
function is used to define the end of an epoch when working with an infinite
stream of samples. This is particularly important in distributed training
scenarios to ensure that all nodes process the same number of batches per epoch,
which helps in synchronizing the training process across nodes. Without
`with_epoch(N)`, the training loop would not have a clear indication of when an
epoch ends, potentially leading to inconsistent training states among different
nodes. WebDataset operates with the `IterableDataset` interface, which does not
support the `set_epoch` method used by `DistributedSampler` in PyTorch's
`DataLoader`. Therefore, `with_epoch(N)` serves as a mechanism to delineate
epochs in the absence of `set_epoch`.

```python
# Example of using with_epoch in a training loop
for epoch in range(num_epochs):
    for sample in webdataset_reader.with_epoch(epoch_length):
        train(sample)
```

------------------------------------------------------------------------------

Issue #264

Q: How can I include the file name (only the stem, not the extension) in the
`metadata` dictionary when using WebDataset?

A: When working with WebDataset, each sample in the dataset contains a special
key `__key__` that holds the file name without the extension. To include the
file name in the `metadata` dictionary, you can create a custom mapping function
that extracts the `__key__` and adds it to the `metadata`. Here's a short code
example on how to modify the pipeline to include the file name in the
`metadata`:

```python
def add_filename_to_metadata(sample):
    sample["metadata"]["filename"] = sample["__key__"]
    return sample

pipeline = [
    # ... (other pipeline steps)
    wds.map(add_filename_to_metadata),
    # ... (remaining pipeline steps)
]
```

This function should be added to the pipeline after the `wds.decode` step and
before the `wds.to_tuple` step. This way, the `metadata` dictionary will contain
the file name for each sample processed by the pipeline.

------------------------------------------------------------------------------

Issue #261

Q: Why is my WebDataset tar file unexpectedly large when saving individual tensors?

A: The large file size is due to the fact that each tensor is pointing to a
large underlying byte array buffer, which is being saved in its entirety. This
results in saving much more data than just the tensor's contents. To fix this,
you should clone the tensor before saving it to ensure that only the relevant
data is written to the file. Additionally, each file in a tar archive has a
512-byte header, which can add significant overhead when saving many small
files. To reduce file size, consider compressing the tar file or batching
tensors before saving.

Here's a code snippet showing how to clone the tensor before saving:

```python
with wds.TarWriter(f"/tmp/dest.tar") as sink:
    for i, d in tqdm(enumerate(tensordict), total=N):
        obj = {"__key__": f"{i}"}
        for k, v in d.items():
            buffer = io.BytesIO()
            torch.save(v.clone(), buffer)  # Clone the tensor here
            obj[f"{k}.pth"] = buffer.getvalue()
        sink.write(obj)
```

To compress the tar file, simply save it with a `.tar.gz` extension and use a compression library:

```python
with wds.TarWriter(f"/tmp/dest.tar.gz", compressor="gz") as sink:
    # ... rest of the code ...
```

------------------------------------------------------------------------------

Issue #260

Q: What is the purpose of the `.with_epoch()` method in WebDataset and could it be named more descriptively?

A: The `.with_epoch()` method in WebDataset is used to explicitly set the number
of samples that constitute an epoch during distributed training. This is
important for ensuring that each worker in a distributed system processes a full
epoch's worth of data. The name `.with_epoch()` might not be immediately clear,
but it is intended to indicate that the dataset is being configured with a
specific epoch length. A more descriptive name like `.set_epoch_size()` could
potentially convey the purpose more clearly. However, changing the method name
would be a breaking change for existing codebases. Improving the documentation
with examples can help clarify the usage:

```python
# Original method name
dataset = dataset.with_epoch(10000)

# Hypothetical more descriptive method name
dataset = dataset.set_epoch_size(10000)
```

In the meantime, users should refer to the improved documentation for guidance
on how to use the `.with_epoch()` method effectively.

------------------------------------------------------------------------------

Issue #257

Q: How can I efficiently load only the necessary auxiliary images for a sample
in my training configuration to save on I/O and decoding time?

A: When working with datasets that include a main image and multiple auxiliary
images, you can optimize the data loading process by selectively reading only
the required files. This can be achieved by using the `select_files` option in
WebDataset or similar tools, which allows you to specify which files to extract
from the dataset. By pre-selecting the files during the dataset preparation
phase, you ensure that your tar files contain exactly the files needed for
training, minimizing unnecessary I/O operations and decoding time for unused
images. Here's a short example of how you might use `select_files`:

```python
import webdataset as wds

# Define your selection criteria based on the training configuration
def select_files(sample):
    return [sample['main.jpg']] + [sample[f'aux{i}.jpg'] for i in range(number_of_aux_images)]

# Create a dataset and apply the selection
dataset = wds.WebDataset("dataset.tar").select(select_files)
```

This approach is more efficient than reading all files and discarding the
unneeded ones, as it avoids the overhead of reading and decoding data that will
not be used in the training process.

------------------------------------------------------------------------------

Issue #256

Q: Why does my training program using WebDataset consume so much memory and crash?

A: The memory consumption issue you're experiencing with WebDataset during
training is likely due to the shuffle buffer size. WebDataset uses in-memory
buffering to shuffle data, and if the buffer size is too large, it can consume a
significant amount of memory, especially when dealing with large datasets or
when running on systems with limited memory. The parameters
`_SHARD_SHUFFLE_SIZE` and `_SAMPLE_SHUFFLE_SIZE` control the number of shards
and samples kept in memory for shuffling. Reducing these values can help
mitigate memory usage issues. For example, you can try setting:

```python
_SHARD_SHUFFLE_SIZE = 1000  # Reduced from 2000
_SAMPLE_SHUFFLE_SIZE = 2500  # Reduced from 5000
```

Adjust these values based on your system's memory capacity and the size of your
dataset. Keep in mind that reducing the shuffle buffer size may affect the
randomness of your data shuffling and potentially the training results. It's a
trade-off between memory usage and shuffle effectiveness.

------------------------------------------------------------------------------

Issue #249

Q: Should I use WebDataset or TorchData for my data loading in PyTorch?

A: The choice between WebDataset and TorchData depends on your specific needs
and the context of your project. WebDataset is still a good choice if you
require backwards compatibility or if you need to work without PyTorch. It is
also being integrated with other frameworks like Ray, which may be beneficial
for certain use cases. However, it's important to note that as of July 2023,
active development on TorchData has been paused to re-evaluate its technical
design. This means that while TorchData is still usable, it may not receive
updates or new features in the near future. If you are starting a new project or
are able to adapt to changes, you might want to consider this factor. Here's a
simple example of how you might use WebDataset:

```python
import webdataset as wds

# Create a dataset
dataset = wds.WebDataset("path/to/data-{000000..000999}.tar")

# Iterate over the dataset
for sample in dataset:
    image, label = sample["image"], sample["label"]
    # process image and label
```

And here's how you might use TorchData:

```python
from torchdata.datapipes.iter import FileOpener, TarArchiveReader

# Create a data pipeline
datapipes = FileOpener("path/to/data.tar") \
    .parse(TarArchiveReader())

# Iterate over the data pipeline
for file_name, file_stream in datapipes:
    # process file_stream
```

Given the pause in TorchData development, you should consider the stability and
future support of the library when making your decision.

------------------------------------------------------------------------------

Issue #247

Q: How can I load images from nested tar files using webdataset?

A: To load images from nested tar files with webdataset, you can create a custom
decoder that handles `.tar` files using Python's `tarfile` module. This decoder
can be applied to your dataset with the `.map()` method, which allows you to
modify each sample in the dataset. The custom decoder will read the nested tar
file from the sample, extract its contents, and add them to the sample
dictionary. Here's a short example of how you can implement this:

```python
import io
import tarfile
from webdataset import WebDataset

def expand_tar_files(sample):
    stream = tarfile.open(fileobj=io.BytesIO(sample["tar"]))
    for tarinfo in stream:
        if tarinfo.isfile():
            name = tarinfo.name
            data = stream.extractfile(tarinfo).read()
            sample[name] = data
    return sample

ds = WebDataset("dataset.tar").map(expand_tar_files).decode("...")
```

In this example, `expand_tar_files` is a function that takes a sample from the
dataset, opens the nested tar file contained within it, and adds each file from
the nested tar to the sample. The `WebDataset` object is then created with the
path to the dataset tar file, and the `expand_tar_files` function is applied to
each sample in the dataset.

------------------------------------------------------------------------------

Issue #246

Q: What is the purpose of `.to_tuple()` in WebDataset and how does it handle missing files?

A: The `.to_tuple()` method in WebDataset is used to extract specific fields
from a dataset where each sample is a dictionary with keys corresponding to file
extensions. This method simplifies the process of preparing data for training by
converting dictionaries into tuples, which are more convenient to work with in
many machine learning frameworks. When you specify multiple file extensions
separated by semicolons, `.to_tuple()` will return the first file that matches
any of the given extensions. If a file with a specified extension is not present
in a sample, `.to_tuple()` will raise an error. To handle optional files, you
can use a custom function with `.map()` that uses the `get` method to return
`None` if a key is missing, thus avoiding errors and allowing for flexible data
structures.

Here's an example of using `.to_tuple()` with mandatory and optional files:

```python
# Mandatory jpg and txt, optional npy
def make_tuple(sample):
    return sample["jpg"], sample.get("npy"), sample["txt"]

ds = WebDataset(...) ... .map(make_tuple)
```

And here's how you might use `.to_tuple()` directly for mandatory files:

```python
ds = WebDataset(...) ... .to_tuple("jpg", "txt")
```

------------------------------------------------------------------------------

Issue #244

Q: How can I combine multiple data sources with a specified frequency for sampling from each?

A: To combine multiple data sources with non-integer sampling frequencies, you
can use the `RandomMix` function from the WebDataset library. This function
allows you to specify the relative sampling weights as floating-point numbers,
which can represent the desired sampling frequency from each dataset. Here's an
example of how to use `RandomMix` to combine two datasets with a specified
sampling frequency:

```python
from webdataset import WebDataset, RandomMix

ds1 = WebDataset('path_to_shards_A/{00..99}.tar')
ds2 = WebDataset('path_to_shards_B/{00..99}.tar')
mix = RandomMix([ds1, ds2], [1.45, 1.0])  # Sampling from ds1 1.45 times more frequently than ds2
```

This will create a mixed dataset where samples from `ds1` are drawn
approximately 1.45 times more often than samples from `ds2`.

------------------------------------------------------------------------------

Issue #239

Q: Can I filter a WebDataset to select only a subset of categories?

A: Yes, you can filter a WebDataset to select only a subset of categories by
using a map function. This is efficient as long as the subset is not too small;
otherwise, it can lead to inefficient I/O due to random disk accesses. For very
small subsets, it's recommended to create a new WebDataset. Here's a simple
example of how to filter categories:

```python
def select(sample):
    if sample["cls"] in [0, 3, 9]:  # Replace with desired categories
        return sample
    else:
        return None

dataset = wds.WebDataset(...).decode().map(select)
```

This approach works well when the number of classes is much larger than the
number of shards, and you're not discarding a significant portion of the data.
If you find yourself discarding a large percentage of the data, consider
creating a new WebDataset for efficiency.

------------------------------------------------------------------------------

Issue #237

Q: How does WebDataset handle filenames with multiple periods when extracting keys?

A: WebDataset uses periods to separate the base filename from the extension,
which can lead to unexpected keys when multiple periods are present in the base
filename. This is by design to support filenames with multiple extensions, such
as `.seg.jpg`. It's important to follow this convention when creating datasets
to avoid issues in downstream processing. If you have filenames with multiple
periods, consider renaming them before creating the dataset. For matching files,
you can use glob patterns like `*.mp3` to ensure you're working with the correct
file type.

```python
# Example of using a glob pattern to match files with the .mp3 extension
dataset = wds.Dataset("dataset.tar").select(lambda x: fnmatch.fnmatch(x, "*.mp3"))
```

------------------------------------------------------------------------------

Issue #236

Q: How does webdataset handle the conversion of tensors to different file formats like .jpg and .npy?

A: In webdataset, the conversion of tensors to specific file formats is
determined by the file extension you specify in the key when writing the data
using `ShardWriter`. There is no automatic conversion; the tensor is simply
saved in the format corresponding to the extension you provide. When reading the
data, you can decode the files into tensors using the appropriate arguments.
Here's a short example of how to write a tensor as different file formats:

```python
from webdataset import ShardWriter

writer = ShardWriter(...)

sample = {}
sample["__key__"] = "dataset/sample00003"
sample["image.jpg"] = some_tensor  # Will be saved as a JPEG file
sample["image.npy"] = some_tensor  # Will be saved as a NPY file

writer.write(sample)
```

When you write a sample with `{"__key__": "xyz", "image.jpg": some_tensor}`, a
JPEG file named `xyz.image.jpg` is created. Conversely, if you write
`{"__key__": "xyz", "image.npy": some_tensor}`, an NPY file named
`xyz.image.npy` is created.

------------------------------------------------------------------------------

Issue #233

Q: How do I ensure that WebDataset correctly splits shards across multiple nodes and workers?

A: When using WebDataset for distributed training across multiple nodes and
workers, it's important to use the `split_by_node` and `split_by_worker`
functions to ensure that each node and worker processes a unique subset of the
data. The `detshuffle` function can be used for deterministic shuffling of
shards before splitting. Here's a minimal example of how to set up the dataset
pipeline for multi-node training:

```python
import webdataset as wds

dataset = wds.DataPipeline(
    wds.SimpleShardList("source-{000000..000999}.tar"),
    wds.detshuffle(),
    wds.split_by_node,
    wds.split_by_worker,
)

for idx, item in enumerate(iter(dataset)):
    if idx < 2:  # Just for demonstration
        print(f"item: {item}")
```

Make sure you are using a recent version of WebDataset that supports these
features. If you encounter any issues, check the version and consider updating
to the latest release.

------------------------------------------------------------------------------

Issue #227

Q: How can I use Apache Beam to write data to a WebDataset tar file for large-scale machine learning datasets?

A: Apache Beam is a powerful tool for parallel data processing, which can be
used to build large datasets for machine learning. When dealing with datasets
larger than 10TB and requiring complex preprocessing, you can use Apache Beam to
process and write the data into a WebDataset tar file format. Below is a
simplified example of how you might set up your Beam pipeline to write to a
WebDataset. This example assumes you have a function `preprocess_sample` that
takes a sample and performs the necessary preprocessing:

```python
import apache_beam as beam
from webdataset import ShardWriter

def write_to_webdataset(sample):
    # Assuming 'preprocess_sample' is a function that preprocesses your data
    processed_sample = preprocess_sample(sample)
    # Write the processed sample to a shard using ShardWriter
    # This is a simplified example; you'll need to manage shards and temp files
    with ShardWriter("output_shard.tar", maxcount=1000) as sink:
        sink.write(processed_sample)

# Set up your Apache Beam pipeline
with beam.Pipeline() as pipeline:
    (
        pipeline
        | 'Read Data' >> beam.io.ReadFromSomething(...)  # Replace with your data source
        | 'Process and Write' >> beam.Map(write_to_webdataset)
    )
```

Remember to manage the sharding and temporary files appropriately, as the
`ShardWriter` will need to write to different shards based on your dataset's
partitioning. The `maxcount` parameter controls how many items are in each
shard. You will also need to handle the copying of the temporary shard files to
your destination bucket as needed.

------------------------------------------------------------------------------

Issue #225

Q: How can I ensure that Distributed Data Parallel (DDP) training with
WebDataset doesn't hang due to uneven data distribution across nodes?

A: When using WebDataset for DDP training, it's important to ensure that all
nodes receive the same number of samples to prevent hanging during
synchronization. One effective method is to create a number of shards that is
divisible by the total number of workers and ensure each shard contains the same
number of samples. Assign each worker the same number of shards to achieve exact
epochs with no resampling, duplication, or missing samples. If the dataset
cannot be evenly divided, you can use `resampled=True` to generate an infinite
stream of samples, and set an epoch length using `with_epoch`. This approach
allows for synchronization across workers even if the dataset size is not
divisible by the number of workers. Here's an example of setting an epoch
length:

```python
from webdataset import WebDataset

dataset = WebDataset(urls, resampled=True).with_epoch(epoch_length)
```

For validation, where you want to avoid arbitrary epoch lengths, you can drop
samples from the end of the validation set to make its size divisible by the
world size. This can be done using TorchData as follows:

```python
from torch.utils.data import DataLoader
import torch.distributed

dataset = dataset.batch(torch.distributed.get_world_size(), drop_last=True)
dataset = dataset.unbatch()
dataset = dataset.sharding_filter()
```

Remember to use the `sharding_filter` to ensure that each process only sees its own subset of the data.

------------------------------------------------------------------------------

Issue #219

Q: What should I use instead of `ShardList` in webdataset v2, and how do I specify a splitter?

A: In webdataset v2, the `ShardList` class has been renamed to
`SimpleShardList`. If you encounter an `AttributeError` stating that the module
`webdataset` has no attribute `ShardList`, you should replace it with
`SimpleShardList`. Additionally, the `splitter` argument has been changed to
`nodesplitter`. Here's how you can update your code to reflect these changes:

```python
urls = list(braceexpand.braceexpand("dataset-{000000..000999}.tar"))
dataset = wds.SimpleShardList(urls, splitter=wds.split_by_worker, nodesplitter=wds.split_by_node, shuffle=False)
dataset = wds.Processor(dataset, wds.url_opener)
dataset = wds.Processor(dataset, wds.tar_file_expander)
dataset = wds.Processor(dataset, wds.group_by_keys)
```

If you are using `WebDataset` and encounter a `TypeError` regarding an
unexpected keyword argument `splitter`, ensure that you are using the correct
argument name `nodesplitter` instead.

------------------------------------------------------------------------------

Issue #216

Q: Can I use `ShardWriter` to write directly to a cloud storage URL like Google Cloud Storage?

A: The `ShardWriter` from the `webdataset` library is primarily designed to
write shards to a local disk, and then these shards can be copied to cloud
storage. Writing directly to cloud storage is not the default behavior because
it can be less efficient and more error-prone due to network issues. However, if
you have a large dataset that cannot be stored locally, you can modify the
`ShardWriter` code to write directly to a cloud URL by changing the line where
the `TarWriter` is instantiated. Here's a short example of the modification:

```python
# Original line in ShardWriter
self.tarstream = TarWriter(open(self.fname, "wb"), **self.kw)

# Modified line to write directly to a cloud URL
self.tarstream = TarWriter(self.fname, **self.kw)
```

Please note that this is a workaround and may not be officially supported. It's
recommended to test thoroughly to ensure data integrity and handle any potential
exceptions related to network issues.

------------------------------------------------------------------------------

Issue #212

Q: Does WebDataset download all shards at once, and how does caching affect the download behavior?

A: WebDataset accesses shards individually and handles data in a streaming
fashion by default, meaning that shards are not cached locally unless caching is
explicitly enabled. When caching is enabled, each shard is downloaded completely
before being used, which can block training until the download is finished. This
behavior contrasts with the streaming mode, where training can start as soon as
the first batch is ready. The caching mechanism does not currently download
shards in parallel with training, which can lead to delays when starting the
training process. To change the local cache name when using `pipe:s3`, you can
override the `url_to_name` argument to map shard names to cache file names as
desired.

Here's an example of how to override the `url_to_name` function:

```python
import webdataset as wds

def custom_url_to_name(url):
    # Custom logic to convert URL to a cache filename
    return url.replace("http://url/dataset-", "").replace(".tar", ".cache")

dataset = wds.WebDataset("pipe:s3 http://url/dataset-{001..099}.tar", url_to_name=custom_url_to_name)
```

------------------------------------------------------------------------------

Issue #211

Q: How can I write to a remote location using ShardWriter?

A: ShardWriter is designed to write to local disk for simplicity and
reliability, but it provides a hook for uploading data to a remote location. You
can define a function that handles the upload process and then pass this
function to the `post` parameter of ShardWriter. Here's a short example of how
to use this feature:

```Python
def upload_shard(fname):
    os.system(f"gsutil cp {fname} gs://mybucket")
    os.unlink(fname)

with ShardWriter(..., post=upload_shard) as writer:
    # Your code to add data to the writer
    ...
```

This approach allows you to have control over the upload process and handle any
errors that may occur during the transfer to the remote storage.

------------------------------------------------------------------------------

Issue #210

Q: How does the `default_collation_fn` work in WebDataset when it seems to
expect a list or tuple, but the documentation suggests it should handle a
collection of samples as dictionaries?

A: The confusion arises from the mismatch between the documentation and the
actual implementation of `default_collation_fn`. The function is designed to
take a batch of samples and collate them into a single batch for processing.
However, the current implementation of `default_collation_fn` in WebDataset does
not handle dictionaries directly. Instead, it expects each sample in the batch
to be a list or tuple. If you have a batch of dictionaries, you would need to
convert them into a list or tuple format before using `default_collation_fn`.
Alternatively, you can use `torch.utils.data.default_collate` from PyTorch 1.11
or later, which can handle dictionaries, or you can provide a custom collate
function that handles dictionaries. Here's an example of a custom collate
function that could handle a list of dictionaries:

```python
def custom_collate_fn(batch):
    # Assuming each element in batch is a dictionary
    collated_batch = {}
    for key in batch[0].keys():
        collated_batch[key] = [d[key] for d in batch]
    return collated_batch
```

You can then pass this `custom_collate_fn` to your data loader.

------------------------------------------------------------------------------

Issue #209

Q: How can I ensure each batch contains only one description per image when using webdatasets?

A: To ensure that each batch contains only one description per image in
webdatasets, you can create a custom transformation function that acts as a
filter or collate function. This function can be composed with your dataset to
enforce the batching rule. You can use buffers or other conditional logic within
your transformation to manage the batching process. Here's a simple example of
how you might start implementing such a transformation:

```Python
def unique_image_collate(src):
    buffer = {}
    for sample in src:
        image_id = sample['image_id']
        if image_id not in buffer:
            buffer[image_id] = sample
            if len(buffer) == batch_size:
                yield list(buffer.values())
                buffer.clear()
        # Additional logic to handle leftovers, etc.
    if buffer:
        yield list(buffer.values())

dataset = dataset.compose(unique_image_collate)
```

This function collects samples in a buffer until it has a batch's worth of
unique images, then yields that batch and clears the buffer for the next batch.
You'll need to add additional logic to handle cases such as the end of an epoch
where the buffer may not be full.

------------------------------------------------------------------------------

Issue #201

Q: How can I efficiently subsample a large dataset without slowing down iteration speed?

A: When dealing with large datasets, such as LAION 400M, and needing to
subsample based on metadata, there are several strategies to maintain high I/O
performance. If the subset is small and static, it's best to create a new
dataset ahead of time. This can be done using a WebDataset/TarWriter pipeline or
with `tarp proc ... | tarp split ...` commands, potentially parallelizing the
process with tools like `ray`. If dynamic selection is necessary, consider
splitting the dataset into shards by the categories of interest. This approach
avoids random file accesses, which can significantly slow down data pipelines.
Here's a simple example of creating a subset using `tarp`:

```bash
tarp proc mydataset.tar -c 'if sample["metadata"] in metadata_list: yield sample'
tarp split -o subset-%06d.tar --size=1e9
```

Remember to perform filtering before any heavy operations like decoding or augmentation to avoid unnecessary processing.

------------------------------------------------------------------------------

Issue #196

Q: How can I speed up subsampling from a tar file when using WebDataset?

A: When working with WebDataset, it's important to remember that it is optimized
for streaming data and does not support efficient random access within tar
files. To speed up subsampling, you should avoid using very small probabilities
with `rsample` as it requires reading the entire stream. Instead, consider using
more shards and applying `rsample` to the shards rather than individual samples.
This approach avoids the overhead of sequential reading. Additionally, some
storage servers like AIStore can perform server-side sampling, which can be more
efficient as they can use random access.

```python
# Example of using rsample with shards
dataset = WebDataset("dataset-{0000..9999}.tar").rsample(0.1)
```

------------------------------------------------------------------------------

Issue #194

Q: How should I balance dataset elements across DDP nodes when using WebDataset?

A: When using WebDataset with Distributed Data Parallel (DDP) in PyTorch, you
may encounter situations where the dataset is not evenly distributed across the
workers. To address this, you can use the `.repeat()` method in combination with
`.with_epoch()` to ensure that each worker processes the same number of batches.
The `.repeat(2)` method is used to repeat the dataset twice, which should be
sufficient for most cases. If the dataset is highly unbalanced, you may need to
adjust this number. The `.with_epoch(n)` method is used to limit the number of
samples processed in an epoch to `n`, where `n` is typically set to the total
number of samples divided by the batch size. This combination ensures that each
epoch has a consistent size across workers, while also handling any imbalance in
the number of shards or samples per worker.

Here's an example of how to use these methods:

```python
batch_size = 64
epoch_size = 1281237  # Total number of samples in the dataset
loader = wds.WebLoader(dataset, num_workers=4)
loader = loader.repeat(2).with_epoch(epoch_size // batch_size)
```

This approach allows for a balanced distribution of data across DDP nodes, with
the caveat that some batches may be missing or repeated. It's a trade-off
between perfect balance and resource usage.

------------------------------------------------------------------------------

Issue #185

Q: How can I include the original file name in the metadata dictionary when iterating through a WebDataset?

A: When working with WebDataset, you can include the original file name in the
metadata dictionary by defining a function that extracts the `__key__` from the
sample and adds it to the metadata. You then apply this function using the
`.map()` method in your pipeline. Here's a short example of how to define and
use such a function:

```python
def add_filename_to_metadata(sample):
    sample["metadata"]["filename"] = sample["__key__"]
    return sample

# Add this to your pipeline after renaming the keys
pipeline.append(wds.map(add_filename_to_metadata))
```

This function should be added to the pipeline after the renaming step to ensure
that the `metadata` key is already present in the sample dictionary.

------------------------------------------------------------------------------

Issue #177

Q: How can I resume training from a specific step without iterating over unused data when using WebDataset?

A: When using WebDataset for training with large datasets, it's common to want
to resume training from a specific step without loading all the previous data
into memory. WebDataset provides a feature for this scenario through shard
resampling. By setting `resampled=True` or using the `wds.resampled` pipeline
stage, you can ensure that you get the same training statistics when restarting
your job without the need to skip samples manually. This approach is recommended
over trying to implement "each sample exactly once per epoch," which can be
complex and environment-dependent.

Here's a short example of how you might use the `resampled` option:

```python
from webdataset import WebDataset

dataset = WebDataset(urls).resampled(rng=my_random_state)
```

And here's how you might use the `wds.resampled` pipeline stage:

```python
import webdataset as wds

dataset = wds.WebDataset(urls).pipe(wds.resampled)
```

------------------------------------------------------------------------------

Issue #172

Q: Why does the `detshuffle` epoch count not increment across epochs when using WebDataset?

A: The issue with `detshuffle` not incrementing the epoch count across epochs is
likely due to the interaction between the DataLoader's worker process management
and the internal state of the `detshuffle`. When `persistent_workers=False`, the
DataLoader creates new worker processes each epoch, which do not retain the
state of the `detshuffle` instance. This results in the `detshuffle` epoch count
resetting each time. To maintain the state across epochs, you can set
`persistent_workers=True` in the DataLoader. Alternatively, you can manage the
epoch count externally and pass it to `detshuffle` if needed. Here's a short
example of how to set `persistent_workers`:

```python
from torch.utils.data import DataLoader

# Assuming 'dataset' is your WebDataset instance
loader = DataLoader(dataset, persistent_workers=True)
```

If you need to manage the epoch count externally, you could use an environment
variable or another mechanism to pass the epoch count to `detshuffle`. However,
this approach is less clean and should be used with caution, as it may introduce
complexity and potential bugs into your code.

------------------------------------------------------------------------------

Issue #171

Q: I'm getting an ImportError when trying to import `PytorchShardList` from `webdataset`. What should I do?

A: The `PytorchShardList` class has been removed in recent versions of the
`webdataset` package. If you are using version 0.1 of `webdataset`,
`PytorchShardList` was available, but in later versions, it has likely been
replaced with `SimpleShardList`. To resolve the ImportError, you should update
your import statement to use the new class name. Here's how you can import
`SimpleShardList`:

```python
from webdataset import SimpleShardList
```

If `SimpleShardList` does not meet your requirements, you may need to check the
documentation for the version of `webdataset` you are using to find the
appropriate replacement or consider downgrading to the version that contains
`PytorchShardList`.

------------------------------------------------------------------------------

Issue #170

Q: How do I use glob patterns with WebDataset to read data from Google Cloud Storage (GCS)?

A: WebDataset does not natively support glob patterns due to the lack of a
consistent API for globbing across different object stores. To use glob patterns
with files stored in GCS, you need to manually resolve the glob pattern using
`gsutil` and then pass the list of shards to WebDataset. Here's an example of
how to do this in Python:

```Python
import os
import webdataset as wds

# Use gsutil to resolve the glob pattern and get the list of shard URLs
shard_list = [shard.strip() for shard in os.popen("gsutil ls gs://BUCKET/PATH/training_*.tar").readlines()]

# Create the WebDataset with the resolved list of shard URLs
train_data = wds.WebDataset(shard_list, shardshuffle=True, repeat=True)
```

This approach ensures that you get the expected behavior when reading data from
shards that match a glob pattern in GCS. Remember to install `gsutil` and
authenticate with GCS before running the code.

