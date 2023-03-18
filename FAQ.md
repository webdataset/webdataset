Q: Can WebDataset be used with torch-xla in a multiprocessing context?

A: Yes, WebDataset can be used with torch-xla in a multiprocessing context. A
gist has been written for this purpose using the torch-xla distributed
`MpDeviceLoader` with shard splitting across accelerators and workers. The code
also includes checks to make sure that all the minibatches are unique. The gist
can be found at
https://gist.github.com/harpone/3b6003c22295a50cbd3d2cfc566dc115. It has been
suggested to include this in the WebDataset documentation.

---

Q: Why does WebDataset decode the file path separator in Windows incorrectly?

A: The issue is caused by brace-expand, which is used to expand input file
strings with glob style braces. Some file systems, including Windows, use a
backslash to separate directories in file paths. However, brace-expand also
considers the backslash as an escape character, which causes WebDataset to
decode it incorrectly. A possible solution is to represent paths using forward
slashes, which works across all platforms. Alternatively, you can translate the
backslash to a different character that isn't recognized as an escape character
by brace-expand. Or you can just expand the list of files using any Windows glob
function and then pass the resulting list to WebDataset.

---

Q: How can webdatasets be used with PyTorch Lightning?

A: Currently, webdataset datasets using the default PyTorch DataLoader or
WebLoader does not work with PyTorch Lightning. The user is required to write
their own wds.WebLoader with a custom length function as illustrated in
discussions on the issue tracker. The length function can raise TypeError or
NotImplementedError. It is possible to set the length attribute of the
webdataset DataLoader to a custom length function that raises an error of your
choosing or set it to False. Three alternatives have been suggested by users for
webdataset to work better with PyTorch Lightning. They include, exposing a way
to pass arguments to `wds.Processor`, making `wds.Processor` raise `TypeError`
when `length=False` or making `wds.Processor.__len__` raise `TypeError` instead
of ValueError. Documentation has also been suggested to make it easier for users
to integrate webdatasets and PyTorch Lightning.

---

Q: My training program stops randomly when using WebDataset, the GPU still has
high utility but the training does not go on and the error message is raised.
What could be the reason for this?

A: This issue mostly happens when doing multinode training. You need to make
sure that the number of workers corresponds to the number of shards in a
reasonable way. Secondly, when it comes to your validation set, the number of
nodes doing the validation depends on the size of the validation set. If the
validation set size is small, you can choose to run it on the node with the rank
0 or opt to distribute it across all nodes. In the case of a small validation
set, you can load it into memory and keep it there to speed things up. If you
choose to distribute the validation set and it's small, you can use
`WebDataset(...).islice(rank, 999999, world_size)` as your input. This way, the
same shard will be opened on all nodes with each node getting a different subset
of samples.

---

Q: Why is my dataset not completely shuffled even when buffer size and initial
are set to the size of the dataset?

A: The issue might be due to creating a toy dataset with 100 shards containing a
total of 10,000 integer values, with 16 processes reading those shards. This
leads to only 6 shards per worker in each epoch, causing integers that are near
each other to be included in the same batch, regardless of shuffling. One
alternative solution is to sample with replacement, by using an infinite data
stream, rebatching with `dataloader` and then shuffling. Another solution is to
not split the shards by identity and instead select by slice. This can be done
with the following code:

```
offset = worker_id + num_workers * node_id
splitsize = num_workes * world_size
dataset = WebDataset(..., nodesplitter=identity, splitter=identity).slice(offset, 999999, splitsize).shuffle(...).decode(...)
```

The dataset can then be loaded using `WebLoader`. Also, setting `epoch_shuffle`
to True can enable epoch-specific shuffling with the environment variable
`WDS_EPOCH`. Shuffling using per-sample methods such as random sampling can also
be done through filters or operations on the datasets. Furthermore, if complex
augmentation and I/O is required, nvlabs/tensorcom can be used.

---

Q: What does this issue fix in WebDataset?

A: This fix adds an option to shuffle the shards before splitting them between
nodes and workers. With epoch-based training loops, the idea is to set the
random seed to a different number every epoch, ensuring shards are distributed
differently at each iteration. Without this, each worker always sees the same
subset of shards at every epoch, which may limit the possibilities of which
datapoints are being contrasted. The fix introduces a `set_epoch` method, which
informs the dataloader of the current epoch, and uses that as a random seed for
shuffling shards before they are being split between workers and nodes.

---

Q: How should one handle storing small embeddings when using WebDataset?

A: Storing small embeddings using WebDataset can lead to significant space
overhead and slower performance when reading the dataset. However, the overhead
may not be significant enough to require a change in approach. To speed up
processing, the following can be tried:

- Use .npy or .ten storage format if you're storing tensors; they can probably
be decoded faster.
- Store data in 8-bit integer or float16 formats.
- Store batched data.
- Cache data in memory or in a database.

If using batched data or caching in a database, careful considerations should be
made regarding number of workers and shared database writes. Additionally, for
small records, it is recommended to reduce DataLoader IPC overhead. To handle
this, one can try chaining commands such as `unbatched()`, `shuffle()`, and
`batched()` during data loading. There is no recommended minimum file size when
using WebDataset, but it can be effective for storing non-image data such as a
mix of strings, ints, and floats using JSON.

---

Q: How can we fix the issue where invalid header arises when streaming images
from an s3 bucket in PyTorch using the WebDataset project?

A: The invalid header issue occurs due to missed included info in `stdout` for
streaming in `s3cmd` when downloading images. One solution to this issue is to
use the `quite` option while downloading data from `s3cmd`. The use case is to
filter the useless stdout or to note the case for users. There isn't really
anything WebDataset can do to fix it: once `s3cmd` mixes outputs, WebDataset
can't unmix them. A bug report can also be raised for programs like `s3cmd`
which print diagnostics to stdout when using stdout for data.

---

Q: How can I solve the "gopen handler not defined" error when trying to iterate
through a locally-stored dataset in WebDataset on Windows with file paths?

A: One solution proposed on the WebDataset issue tracker is to use the `file:`
scheme in the URL for local addresses, for example
`file:d:/some/path/train-{000000..000999}.tar`. Another solution is to use
Windows symlinks that include drive letters in the target. It is also suggested
to refer to the dataset path as "./data" regardless of machine and to create
symlinks for the actual data path to the "./data" path on each machine.

---

Q: Is there a concrete example of using tensorcom and webdataset?

A: Yes, it is possible to convert a tensorcom Connection object to a webloader
object. You can turn any object with an `__iter__` method into a `Processor`,
and there are some examples provided in the issue tracker. For instance, the
code snippet below shows how to create a rebatching pipeline using tensorcom and
webdataset.

```
connection = Connection(...) # or TensorcomDataset(...)
src = Processor(connection, utils.identity)
rebatched = src.unbatched().shuffle(1000).batched(300)
```

Here, `Connection` is the tensorcom connection object, `utils.identity` is the
function that is applied to each item in the connection, `src` is a tensorcom
backend processor, and `rebatched` is a webdataset processor that unbatchs,
shuffles, and rebatches the data in 300 batches.

---

Q: Is it possible to apply online filtering on a dataset with WebDataset?

A: Yes, you can apply online filtering to a dataset with WebDataset. Processing
is just a chain of iterators operating on a stream of data, so you can select,
reorder, and transform any way you like. You can use shorthand for selecting
samples such as in the following example:

```python
def selection(sample):
    return "training" in sample["__key__"]

ds = WebDataset(...).select(selection).decode(...).etc...
```

Note that this will still read all samples but then discard the samples it
doesn't want/need based on the selection function. When you train on very small
subsets of the original dataset (less than 10%) and you need maximal I/O
performance, you may want to prefilter the data based on some criteria and
generate a new dataset representing just that subset.

---

Q: Does webdataset support horovod?

A: According to the developer, webdataset provides standard PyTorch
IterableDataset instances and should be able to work with Horovod. There is not
an example provided for this case but the developer has added that you have
complete control over splitting so you can customize the dataset to work for
your use case. For instance, you can include use cases such as
`my_node_splitter` and `wds.split_by_worker` to customize splitting of the data.

---

Q: What happened to the `length` argument in WebDataset?

A: The `length` argument is no longer supported in the current version of
WebDataset (v2). In order to specify an explicit size, you should use
`.with_length(n)` instead. For v1, use `webdataset.FakeLength` for setting the
length. If you want to force a specific epoch length, use
`.repeat().slice(num_samples)`. The `__len__` method had to be removed because
PyTorch considers having a `__len__` method on an `IterableDataset` to be wrong.
The documentation has since been updated.

---

Q: What's wrong with the docstring of PytorchShardList?

A: The docstring of the constructor of PytorchShardList class in the
`shardlists.py` file of WebDataset is missing some parameters such as
`epoch_shuffle` and includes some deprecated parameters. As a result, users
might have trouble using those parameters or might be confused about whether to
use them or not. To ensure the proper usage of the PytorchShardList constructor,
users should rely on the actual code implementation rather than the docstring.
Here is an example of how to use epoch_shuffle:

```python
import webdataset as wds

shardlist = wds.PytorchShardList(
    "http://storage.googleapis.com/lpr-imagenet/imagenet-train-{000000..000499}.tar",
    epoch_shuffle=True
)
```

---

Q: What happened to the WebDataset documentation website?

A: The WebDataset documentation website is currently down and not up to date.
There are plans to improve the documentation by adding more notebooks in the
`docs` subdirectory. However, in the meantime, you may view an old version of
the documentation at https://rom1504.github.io/webdataset/.

---

Q: Does WebDataset internally deal with the `num_workers>1` when transforming
data, since it inherits from the PyTorch `IterableDataset`, which requires a
`worker_init_fn` to avoid duplicate data when using multiple workers?

A: WebDataset automatically deals with `num_workers > 1`. By default,
`wds.shuffle` seeds the random number generator (RNG) by the process ID and time
at the start of each epoch. If `split_by_worker` is used first, shards will
deterministically be assigned to workers; placing wds.shuffle after that
operation will shuffle shards with an unpredictable seed on each epoch. This is
what the WebDataset wrapper does by default. Detshuffle can be used to achieve
deterministic shuffling. The WebDataset interface is a small convenience for the
`DataPipeline` interface and only splits by workers, not by nodes. As such, it's
recommended to use the pipeline interface for more control. However, the
advantage of using WebDataset is that there aren't any duplicates, making it a
suitable solution for small-to-medium-sized datasets.

---

Q: What is the issue with the compose implementation in WebDataset, and how can
it be resolved?

A: The `compose()` function in WebDataset and its associated `source_` method
are outdated and no longer recommended. Instead, the `wds.DataPipeline`
interface should be used, which is easier to use and extend. Conversion from
`compose()` to `DataPipeline` is simple, and can be accomplished by referring to
the `compat.py` module to see how the implementation has changed.

---

Q: What happens when using glob pattern as shard URL in WebDataset?

A: When using a glob pattern as a shard URL in WebDataset, it will expand the
pattern as single shard, which means that every worker would read the same
samples in the same order from the same shards resulting in almost no
randomness. It's important to use the brace notation intended for WebDataset
which allows to specify the range of files. However, if it's necessary to use a
glob pattern, an additional step is needed to retrieve the list of files that
match the pattern before piping them to WebDataset by using an auxiliary Python
code that retrieves the files in a list and feeds that to WebDataset, like this:

```Python
shard_list = list(os.popen("gsutil ls gs://BUCKET/PATH/training_*.tar").readlines())
train_data = wds.WebDataset(shard_list, shardshuffle=True, repeat=True)
```

---

Q: Why am I getting an "ImportError" when trying to import "PytorchShardList"
from "webdataset"?

A: The error is caused by the fact that "PytorchShardList" has been removed from
the "webdataset" package since version 0.1. You should try using
"SimpleShardList" instead, like this: `from webdataset import SimpleShardList`.
Here's an example of how to use it:

```python
from torch.utils.data import Dataset, DataLoader
from webdataset import SimpleShardList, IterableDataset

# Define a function that parses a sample.
def parser(sample):
    # ...
    return parsed_sample

# Create a SimpleShardList and an IterableDataset.
shardlist = SimpleShardList("path/to/shards")
dataset = IterableDataset(shardlist).map(parser)

# Create a DataLoader from the Dataset.
dataloader = DataLoader(dataset, batch_size=32, num_workers=4)
```

---

Q: Why is the epoch count in DetShuffle not incrementing across epochs, causing
issues with shuffling shards across nodes and workers in WebDataset pipelines?

A: According to the WebDataset issue tracker, this issue is likely due to how
the DataLoader handles worker processes and internal state, and there are many
subtle interactions that can arise depending on the configuration used. The
recommended solution is to use resampling if possible, as that does not require
any configuration and will always provide reasonable results. It has also been
suggested that supporting the use of `WDS_EPOCH` in the `run()` method of
DetShuffle could be continued as a backup option for when
`persistent_workers=False` is used, though this is considered an "ugly"
solution.

---

Q: How can I use WebDataset together with DDP for training and what happened to
the ddp_equalize method?

A: Instead of using `.ddp_equalize` in WebDataset, you can use the
`.repeat(2).set_length(n)` method, where `n` depends on your preferences
regarding frequency of repeating samples and unequal sample distribution across
workers. Alternatively, you can do partial distributed SGD updates. Note that
these issues are being addressed in torchdata, which is merging all the
WebDataset functionality into its pipeline. While torchdata doesn't return an
actual WebDataset and doesn't have the `decode` method, it does have a
`webdataset()` method and provides flexible options for distributed training. An
example of building a pipeline for supervised vision with torchdata can be found
in their GitHub repository.

---

Q: What happened to `wds.Processor` in the `v2` or `main` branch?

A: Starting from the `v2` or `main` branch, the `wds.Processor` class is not
included anymore because the architecture for pipelines has changed to be more
in line with `torchdata`. To add preprocessing pipelines to the data, create a
pipeline stage as a callable and compose the dataset with it. When you write
`map(f)`, the function `f` gets the complete sample as an argument. A sample
callable can be done as follows:

```python
def process(source):
    for sample in source:
        ... code goes here ..

ds = WebDataset(...).compose(process)
```

---

Q: How can I subsample a large dataset in the WebDataset format without slowing
down the iteration speed too much?

A: To subsample a large dataset represented in the WebDataset format, there are
some tradeoffs that need to be considered to achieve optimal performance. First,
it is recommended to perform the `select(...)` operation before the `decode` and
any data augmentation since image decompression/augmentation is usually the
slowest part of the pipeline. Additionally, if only a small percentage of the
samples needs to be retained, generating a subset of the dataset ahead of time
is the best approach. This can be done through a small WebDataset/TarWriter
pipeline to generate the subset using `ray` for parallelization or by using
`tarp proc ... | tarp split ...`. If dynamic selection is required, it is best
to split the dataset into shards based on the categories to split on. While
these choices may seem cumbersome, they are necessary to achieve high
performance I/O as file formats that enable selecting subsets dynamically end up
using random file accesses, which slows down input pipelines.

---

Q: Can ShardWriter write to remote URLs?

A: No, ShardWriter cannot write to remote URLs directly. TarWriter, which is
used internally by ShardWriter, supports writing to remote URLs, but the error
handling gets complicated with remote writing. ShardWriter only writes to local
disks, but it provides a hook to upload the data. For example, you can use the
`post` argument of the ShardWriter constructor to pass a function that uploads
the sharded file to a remote URL using a command-line utility like `gsutil`.
Here's an example of how to use `post`:

```Python
import os
from webdataset import ShardWriter

def upload_shard(fname):
    os.system(f"gsutil cp {fname} gs://mybucket")  # replace with your preferred command
    os.unlink(fname)

with ShardWriter("mydata-%d.tar", maxsize=1000, post=upload_shard) as writer:
    # write data to the ShardWriter
```

This creates a ShardWriter that shards the data into files with names like
`mydata-0.tar`, `mydata-1.tar`, etc., and calls the `upload_shard` function to
upload each file to a remote bucket after it's written. The `upload_shard`
function sends each file to the `mybucket` bucket in Google Cloud Storage using
the `gsutil cp` command-line utility, then removes the file from the local disk.
Replace the `gs://mybucket` URL with the destination URL that you want to use.

---

Q: How do I use the `ShardList` object in WebDataset?

A: In the newer version of WebDataset (v2), the `ShardList` object has been
renamed to `SimpleShardList`. Additionally, the keyword argument `splitter` has
been renamed to `nodesplitter`. Here is an updated code example:

```python
import webdataset as wds

# Use braceexpand to create a list of urls
urls = list(wds.gopen("dataset-{000000..000999}.tar").readlines())
dataset = wds.SimpleShardList(urls, nodesplitter=wds.split_by_node, splitter=wds.split_by_worker, shuffle=False)
dataset = wds.Processor(dataset, wds.url_opener)
dataset = wds.Processor(dataset, wds.tar_file_expander)
dataset = wds.Processor(dataset, wds.group_by_keys)
```

---

