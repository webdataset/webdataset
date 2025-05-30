# Issues for webdataset/webdataset

Generated on 2025-05-07 06:05:59

## Open Issues

### Issue #460: Documentation for Webdataset + MultiNode and HF  SFTTrainer

* **Created by:** Ale9806
* **Created on:** 2025-04-16 08:24:25

#### Summary

The issue involves integrating SFTTrainer with WebDataset for multi-node usage,
which fails despite working on a       single node. The problem arises because
SFTTrainer requires a dataset, but available examples convert datasets into
dataloaders, leading to compatibility issues. Attempts to resolve this using the
repository's documentation have been  unsuccessful, and an example for multi-
node configuration is sought.

#### Comments

No comments on this issue.

---
### Issue #457: Sampling from different data set

* **Created by:** ttdd11
* **Created on:** 2025-03-25 13:02:43

#### Summary

The issue involves efficiently managing and sampling data from multiple sources
for training on multi-node TPU         machines without pre-combining and
shuffling the data, which is time-consuming. The current method involves
repeating  smaller datasets to balance their presence, but this requires
recreating all data archives if any dataset changes. The user seeks a solution
to sample from multiple sources dynamically, ensuring each node receives a
representative        distribution of data without pre-processing.

#### Comment Summary (1 comments)

You can utilize RandomMix to create web datasets from various data sources and
mix them according to specified         probabilities for model training. This
involves defining datasets like ds1, ds2, etc., and using wds.RandomMix to
combine them with a list of probabilities. This approach allows for flexible
data sampling from multiple sources.

#### Comment Details

* arijit-hub commented on 2025-05-02 22:46:13

---
### Issue #456: does webdataset have doc ?

* **Created by:** habaohaba
* **Created on:** 2025-03-16 15:27:45

#### Summary

The issue concerns the availability of documentation or an online resource for
viewing API information for the         webdataset library. Users are seeking a
centralized location to access comprehensive guides or references to better
understand and utilize the library's functionalities. The lack of easily
accessible documentation may hinder effective use and integration of the library
in projects. #enhancement #documentation

#### Comment Summary (2 comments)

The webdataset.md file on GitHub provides documentation for the WebDataset
library, which is designed for efficient    handling of large datasets in
machine learning workflows. The document outlines the basic usage, features, and
benefits of using WebDataset, such as its ability to stream data directly from
cloud storage and its compatibility     with PyTorch. However, the documentation
appears to be somewhat limited, suggesting that it may still be under
development or could benefit from further elaboration to enhance user
understanding.

#### Comment Details

* ManuelSokolov commented on 2025-03-17 13:23:32
* habaohaba commented on 2025-03-17 14:00:13

---
### Issue #453: Unexpected Repetition and Missing Files in WebDataset with `resampled=True` and Shuffling Enabled

* **Created by:** ManuelSokolov
* **Created on:** 2025-03-06 10:18:22

#### Summary

The issue involves using WebDataset to load .pt files into a machine learning
model, where enabling resampled=True and setting buffer=100 causes file
repetition and missing directories within epochs, and unexpected file changes
between   epochs. The problem is temporarily resolved by disabling resampling
and shuffling, but this removes the desired        shuffling capability. The
user seeks a solution to enable shuffling without causing these inconsistencies.
The problem likely stems from the interaction between resampling and shuffling
mechanisms, which may not handle the dataset's      structure as expected.

#### Comment Summary (3 comments)

The code for ResampledShards implements a sampling scheme with replacement,
allowing shards to be picked multiple      times per epoch and not necessarily
the same between epochs. The __iter__() method randomly selects shards,
resulting  in approximately one-third of shards not being picked, one-third
picked once, and one-third picked multiple times per  epoch. To avoid repeating
or losing shards, one might consider using shardshuffle and detshuffle options,
where        detshuffle=True applies a new random order each epoch, potentially
achieving the desired behavior without modifying    __iter__().

#### Comment Details

* gui11aume commented on 2025-03-14 17:59:02
* ManuelSokolov commented on 2025-03-17 12:50:15
* gui11aume commented on 2025-03-17 12:59:39

---
### Issue #452: Documentation pointer

* **Created by:** javirk
* **Created on:** 2025-03-03 14:37:24

#### Summary

The user is seeking comprehensive documentation for webdataset and wids, as they
have only found a FAQ and code        comments, which are insufficient for
beginners. They express difficulty in understanding the basics due to the lack
of detailed resources. The user appreciates the development effort but requires
more accessible documentation to          effectively utilize the tools.
#enhancement #documentation

#### Comments

No comments on this issue.

---
### Issue #450: How to implement batch sampler on webdataset?

* **Created by:** ManuelSokolov
* **Created on:** 2025-02-12 10:32:01

#### Summary

The user is attempting to implement a custom batch sampler function for a
webdataset in a bioML project to prevent experimental batch effects, ensuring
each batch contains data from a single experimental condition. While they
successfully implemented a custom sampler using PyTorch's DataLoader, they are
unable to find documentation or a method to achieve the same with webdataset,
which is crucial for their project. They seek guidance on creating a custom
sampler for webdataset to maintain data integrity across experimental
conditions.

#### Comment Summary (1 comments)

PyTorch distinguishes between indexed and iterable datasets, both of which are
compatible with DataLoader but are handled differently, with only indexed
datasets utilizing samplers. In the context of WebDataset, wids is used for
indexed datasets and webdataset for iterable datasets. For scalable data
handling, a custom batch function is recommended, particularly for DDP
training with iterable datasets, where resampling is advised. While the wids
library aligns more closely with traditional usage and is easier for DDP,
creating efficient custom samplers for large datasets can be challenging.

#### Comment Details

* tmbdev commented on 2025-02-12 15:52:32

---
### Issue #448: How to prevent shuffling due to num_workers != 0 ? Can WebLoader objects be aggregated?

* **Created by:** ManuelSokolov
* **Created on:** 2025-02-11 18:55:19

#### Summary

The issue involves using webdataset to load protein data for model training,
where each batch should contain data from only one protein. The current approach
causes random mixing of proteins across batches due to the num_workers setting.
A proposed solution involves creating separate loaders for each protein, but
merging these loaders into a single       WebLoader is problematic because Chain
is not iterable. The user seeks a method to merge these loaders while ensuring
each loader processes specific shards and applies the necessary collation
functions.

#### Comment Summary (2 comments)

The dataset structure can be challenging, especially with WebDataset, which
doesn't randomize shard order. Using       multiple DataLoader workers
introduces nondeterminism, and opening many WebDataset readers can lead to
excessive       resource use. To optimize, consider a custom batcher within the
IterableDataset framework to cache duplicate samples,  allowing for efficient
sequential reading and scaling with large datasets. Alternatively, use wids for
index           shuffling, though it's more complex. Direct access with wids or
mmtar is possible but may cause random access issues.  Shuffling only at the
start might impact performance compared to per-minibatch shuffling.

#### Comment Details

* tmbdev commented on 2025-02-12 15:41:45
* richardrl commented on 2025-02-16 09:23:09

---
### Issue #445: lmdb_cached method & multiple dataloader workers

* **Created by:** pkrack
* **Created on:** 2025-02-03 15:58:32

#### Summary

The issue concerns the lmdb_cached method in the WebDataset library, which may
not function correctly with multiple    DataLoader workers. The concern is that
the method iterates over the entire dataset once cached, potentially causing
each worker to return the entire dataset instead of a partitioned subset.
Additionally, if the source argument lacks   dictionaries with the __key__,
workers might overwrite each other's data, leading to incorrect caching
behavior. The   user suggests a workaround by modifying the cache path to
include the worker ID to ensure proper caching.

#### Comment Summary (1 comments)

LMDBCached has rarely been used in the past, but there is an acknowledgment that
it should be fixed.

#### Comment Details

* tmbdev commented on 2025-02-08 02:17:53

---
### Issue #441: How Column Store work with multiple node training?

* **Created by:** krNeko9t
* **Created on:** 2025-01-09 13:05:23

#### Summary

The issue involves using the Column Store from the WebDataset library in a
multi-node setup. The example specifies     that operations like shuffling and
decoding should occur after the add_column call, but it is unclear how to handle
the node splitter operation and resampling. The user attempts to create a
dataset with
nodesplitter=wds.shardlists.split_by_node and empty_check=True, but it does not
function as expected.

#### Comment Summary (7 comments)

The issue discussed revolves around the add_column function in a multiworker and
multinode environment, where the      split_by_worker and split_by_node settings
can lead to "empty shard" errors due to improper handling of shard
distribution. The solution involves setting both nodesplitter and workersplitter
to None for the WebDataset            initialized in add_column, ensuring that
the additional column source is consumed by the same worker on the same node.
Additionally, setting resampled=True is necessary if the main dataset is
resampled. This approach prevents             shard/sample mismatches and
abnormal memory consumption, especially in multi-GPU scenarios.

#### Comment Details

* tmbdev commented on 2025-02-08 02:39:59
* krNeko9t commented on 2025-02-08 02:48:48
* tmbdev commented on 2025-02-10 23:37:55
* Vinye commented on 2025-02-12 12:23:23
* krNeko9t commented on 2025-02-12 12:55:03
* tmbdev commented on 2025-02-12 15:20:29
* Vinye commented on 2025-02-12 16:11:05

---
### Issue #433: Performance vs. TFRecord

* **Created by:** guillaumeguy
* **Created on:** 2024-12-19 18:04:10

#### Summary

The issue involves a significant performance degradation, specifically a 2-3X
slowdown, when using WebDataset (WDS)    compared to TFRecord for streaming
tabular data from S3. The suspected cause is unnecessary array copies between
workers, which may be impacting efficiency. Profiling traces have been provided
for further analysis, and the          possibility of creating a dummy dataset
for profiling is offered.

#### Comment Summary (2 comments)

The comments suggest that the time difference in data processing is mainly due
to I/O, with TFRecord benefiting from   gzip compression, unlike WebDataset. To
improve WebDataset's I/O speed, gzip compression can be applied, and the
system will handle decompression automatically. Although WebDataset and PyTorch
loaders have more overhead for small   records, this can be mitigated by
batching data in .tar files and unbatching during loading, though this overhead
is   likely not a significant issue in the current context.

#### Comment Details

* tmbdev commented on 2025-02-08 02:06:14
* guillaumeguy commented on 2025-02-25 19:54:33

---
### Issue #432: Possibly reading from tarfiles before file is streamed from S3?

* **Created by:** jiamenguk
* **Created on:** 2024-12-12 14:17:51

#### Summary

The issue involves streaming .tar and .tar.gz files from S3-compatible storage,
where a slow connection may cause the  filename to be read from the tarball, but
the subsequent bytes fail to load, resulting in a tarfile.ReadError due to   an
"unexpected end of data." This error occurs when using a command to stream files
via AWS S3, and attempts are being made to mitigate the issue by piping the
output through cat. The problem may be related to previously reported issues
#21 and #35.

#### Comment Summary (4 comments)

To handle incomplete file streaming from S3 or similar object stores, one
approach is to implement an exception        handler to skip problematic files,
while another is to use a local cache directory to download files before
accessing  them. Some users have experienced similar issues with GCS,
encountering "unexpected end of data" errors. Solutions     include adjusting
buffer size in the webdataset/cache.py file and writing scripts to retry file
retrieval until        successful, with caching as an additional strategy to
improve reliability.

#### Comment Details

* jiamenguk commented on 2024-12-12 20:39:57
* dakl commented on 2024-12-16 08:22:16
* jiamenguk commented on 2024-12-16 10:17:55
* tmbdev commented on 2025-02-08 01:51:22

---
### Issue #430: How to Download CLIP embeddings for LAION 400

* **Created by:** Divyanshupy
* **Created on:** 2024-11-26 21:34:32

#### Summary

The user is experiencing difficulty locating and downloading the precomputed
CLIP embeddings, which are part of the    LAION-400 dataset and amount to 1TB in
size. They are seeking guidance on how to access these embeddings. This issue
primarily concerns data accessibility and user guidance. #enhancement #error

#### Comment Summary (1 comments)

The phrase "Where does it mention that?" is typically used to request
clarification or a specific reference within a   document or conversation. It
implies that the speaker is seeking evidence or a citation for a claim or
statement made. This question is often used in discussions to ensure accuracy
and accountability in the information being presented.

#### Comment Details

* tmbdev commented on 2024-12-11 02:36:49

---
### Issue #429: v0.2.107 not available on PyPI

* **Created by:** psobot
* **Created on:** 2024-11-26 15:00:40

#### Summary

The user is requesting the release of version 0.2.107 of the WebDataset package
on PyPI, as the current version        available (v0.2.100) lacks important
fixes, including those from a specific pull request. This update is crucial for
users who need the latest improvements and bug fixes. The request is directed to
a project maintainer, indicating the  need for timely action. #enhancement
#error

#### Comment Summary (1 comments)

The speaker is uncertain about a previous issue but is committed to resolving it
quickly and will attempt to address   it as soon as possible.

#### Comment Details

* tmbdev commented on 2024-12-11 02:35:29

---
### Issue #428:  Bug: Inconsistency between IndexTarSamples and MMIndexedTar filename filtering

* **Created by:** avkondepudi
* **Created on:** 2024-11-22 21:34:02

#### Summary

The issue arises from inconsistent filename filtering between the
IndexTarSamples and MMIndexedTar classes, leading to index misalignment and
potential data corruption. IndexTarSamples filters filenames containing periods,
while          MMIndexedTar does not, causing mismatches when accessing files.
Additionally, ShardWriter silently truncates keys      longer than 128
characters, saving them in the index without their data. The suggested fix
includes aligning filtering logic, adding key length validation, and
implementing warnings or errors for overly long keys.

#### Comment Summary (1 comments)

The comments express gratitude and indicate an intention to review the provided
information or material.

#### Comment Details

* tmbdev commented on 2024-12-11 02:34:53

---
### Issue #422: load tensor type webdataset into multi GPUs

* **Created by:** lyb369
* **Created on:** 2024-10-29 15:07:45

#### Summary

The issue involves loading a WebDataset containing embeddings generated from
prompts via a text encoder for multi-GPU  training. The user is seeking
assistance to efficiently manage this process, likely to optimize performance
and ensure compatibility across multiple GPUs. Any guidance or solutions from
those experienced in handling such datasets in a    distributed training
environment would be greatly appreciated. #performance #enhancement

#### Comments

No comments on this issue.

---
### Issue #419: Webloader not yielding all the samples 

* **Created by:** crscarpinteiro
* **Created on:** 2024-10-24 16:49:59

#### Summary

The issue with the dataloader not providing all samples may stem from an
incorrect definition of the epoch length, as  the with_epoch and with_length
methods are set based on dataset_size, num_gpus, and batch_size, which might not
align  with the actual dataset size. Additionally, the use of resample and
shuffle parameters could affect the sample         distribution, but since
resampling is disabled during validation, the problem likely lies in the epoch
configuration.  Double-check the calculations for dataset_size // (num_gpus *
batch_size) to ensure they match the intended number of  samples per epoch.

#### Comments

No comments on this issue.

---
### Issue #417: collated nested dicts are not supported in `wds.filters._unbatched`

* **Created by:** v-iashin
* **Created on:** 2024-10-22 15:06:30

#### Summary

The issue in the webdataset library's filters.py arises when handling the meta
field in a sample dictionary, where the code fails if the meta dictionary
contains more than one key or if the number of elements in meta does not match
the   batch size. This can lead to errors during data processing, particularly
when dealing with complex metadata structures in batched datasets. The problem
highlights a need for more robust handling of metadata to ensure compatibility
with   varying data structures and batch sizes.

#### Comment Summary (1 comments)

Unbatched is the opposite of batched, and the current output is incorrect for a
batched process. The expected output   should resemble a structured dictionary
with tensors and metadata, as shown in the example. Improving the
documentation might help clarify the expected behavior and output.

#### Comment Details

* tmbdev commented on 2024-12-11 02:27:48

---
### Issue #415: Default DirectoryShardList glob pattern is incorrect

* **Created by:** thecodingwizard
* **Created on:** 2024-10-18 23:11:21

#### Summary

The issue arises from the use of a default glob pattern *.{tar,tgz,tar.tgz} in
the DirectoryShardList class, which is  not supported by Python's globbing as it
does not handle curly braces. This results in the inability to correctly
iterate over tar files in a directory using the fluid interface, as demonstrated
by the StopIteration error when       attempting to access files. The problem
necessitates using the pipeline pattern to work around the default pattern
limitation.

#### Comment Summary (1 comments)

The comments express gratitude for the report and a commitment to addressing the
issue promptly. The speaker           acknowledges the problem and indicates an
intention to resolve it swiftly. Overall, the tone is appreciative and
proactive.

#### Comment Details

* tmbdev commented on 2024-12-11 02:23:01

---
### Issue #409: contract of expand_urls changed

* **Created by:** rwightman
* **Created on:** 2024-10-09 16:23:57

#### Summary

The expand_urls function in the webdataset library no longer accepts a list as
input, despite the documentation        suggesting otherwise, leading to
interface breaks that affect downstream usage. This change has forced users to
pin to older versions, which is not ideal for maintaining up-to-date software.
The issue highlights the need for consistent   documentation and backward
compatibility in public interfaces. #error #serious #enhancement

#### Comment Summary (1 comments)

Development and the prevention of reversions or incompatibilities rely heavily
on test cases, which are often          incomplete. To address this, submitting
a comprehensive test case is recommended. This approach ensures better
coverage and stability in the development process.

#### Comment Details

* tmbdev commented on 2024-12-10 23:48:30

---
### Issue #402: cache_dir behavior changed from 0.2.88

* **Created by:** rmekdma
* **Created on:** 2024-09-25 03:16:26

#### Summary

In version 0.2.88 of the WebDataset library, a change was introduced where the
destination no longer includes the base name, differing from previous behavior.
This alteration presents two options: either revert to the original behavior
prior to version 0.2.88 or document this modification as a breaking change in
the changelog. Addressing this issue is  crucial to maintain consistency and
inform users of the change.

#### Comment Summary (2 comments)

The issue seems to be an unintentional change where URL conversion to cached
names is now managed by a function that   strips directory names by default, but
can be overridden. However, the code inconsistently rejects slashes in cache
names, which is likely incorrect. The author acknowledges the problem and plans
to investigate and resolve it.

#### Comment Details

* tmbdev commented on 2024-09-26 13:39:27
* tmbdev commented on 2024-09-26 15:11:33

---
### Issue #380: webdataset.github.io cache / streaming info incorrect

* **Created by:** corinblaikie
* **Created on:** 2024-07-25 22:28:27

#### Summary

The WebDataset documentation claims that caching occurs in parallel with dataset
iteration, allowing training to start immediately without waiting for shard
downloads. However, a recent issue highlights that this behavior has changed;
now, each shard is fully downloaded before being used, which affects the
immediacy of training start times. This       change was made to simplify error
handling, but it contradicts the original documentation.

#### Comment Summary (1 comments)

The comments indicate an acknowledgment and a commitment to make the necessary
updates.

#### Comment Details

* tmbdev commented on 2024-08-17 00:11:05

---
### Issue #377: add LocalWids class

* **Created by:** tmbdev
* **Created on:** 2024-07-19 04:52:21

#### Summary

The issue involves adding a LocalWids class designed to facilitate quick,
unindexed local access to a collection of    shards, aiming to minimize overhead
during random access operations. This enhancement is intended to improve
performance by providing a more efficient way to handle data shards without the
need for indexing, thus optimizing     data retrieval processes. The
implementation focuses on reducing latency and resource consumption, making it a
valuable addition for systems requiring rapid data access.

#### Comments

No comments on this issue.

---
### Issue #368: Readme is missing images

* **Created by:** liambsmith
* **Created on:** 2024-06-20 19:35:14

#### Summary

The issue involves missing images in the README file due to the absence of the
"readme_files" directory, which is      typically used to store these images.
This could lead to incomplete documentation and confusion for users trying to
understand the project. Ensuring the directory is created and populated with the
necessary images will resolve the     issue. #error #enhancement

#### Comments

No comments on this issue.

---
### Issue #366: column store API

* **Created by:** tmbdev
* **Created on:** 2024-06-17 02:55:40

#### Summary

The issue proposes adding a small API to enable column store functionality,
allowing samples to be composed from       multiple shards in parallel. This
would involve implementing a .join_columns(shard_name_function) method to
encapsulate the existing code from the provided example. This enhancement aims
to streamline the process of handling   data across shards, improving efficiency
and usability.

#### Comments

No comments on this issue.

---
### Issue #365: Memory leak at the end of an epoch of training with OpenCLIP on a WebDataset

* **Created by:** fartashf
* **Created on:** 2024-06-16 04:02:37

#### Summary

The issue involves a potential memory leak when using OpenCLIP with WebDataset,
where memory usage increases at the    start of each new epoch if
persistent_workers=True. Disabling persistent workers resolves the leak but is
not          practical according to the WebDataset FAQ. The problem may be
linked to how OpenCLIP handles the WebDataset pipeline,  possibly due to
improper cleanup of partially processed shards, and is related to discussions
and alternative          implementations in the WebDataset community.

#### Comment Summary (6 comments)

The memory leak issue arises from the repeated creation of new iterators each
epoch, which call url_opener without     closing the stream opened by
gopen.gopen, leading to increased memory usage, especially when using pipe:aws
cp -. This problem doesn't occur with pipe:cat, suggesting a potential issue
with how aws cp - handles unclosed pipes. A          temporary workaround for
OpenCLIP is to reduce the number of epochs, thus minimizing the frequency of
dataloader       resets, but this limits evaluations and checkpoints. Further
investigation and a more robust solution are needed to    address this across
different codebases.

#### Comment Details

* tmbdev commented on 2024-06-17 02:52:20
* fartashf commented on 2024-06-17 15:27:09
* fartashf commented on 2024-07-10 02:07:50
* fartashf commented on 2024-07-14 06:50:19
* mbuckler commented on 2024-09-16 22:01:23
* fartashf commented on 2024-09-16 22:16:52

---
### Issue #364: Validation Set Distributed Sampling when using WebDataset with FSDP

* **Created by:** adamcatto
* **Created on:** 2024-06-10 17:59:10

#### Summary

The issue involves distributing validation samples across multiple GPUs in a
multi-node setup using WebDataset and     FSDP, ensuring each sample is
processed exactly once per epoch. The user has an equal number of shards as
GPUs, but    one shard is smaller, complicating equal distribution. They are
considering using ddp_equalize with WebLoader to       manage sample
distribution but are unsure if it guarantees that all samples are loaded exactly
once per epoch. The     challenge is to balance the workload across GPUs while
adhering to the requirement of processing each sample once.

#### Comments

No comments on this issue.

---
### Issue #363: Incorrect Documentation for default_collation_fn

* **Created by:** buddit1
* **Created on:** 2024-06-10 02:21:56

#### Summary

The issue highlights a discrepancy between the documentation and the actual
implementation of the default_collation_fn function in the webdataset library.
The docstring incorrectly states that the function accepts a collection of
samples as dictionaries and returns a dictionary, whereas the function actually
expects samples to be lists or tuples and      returns a list. The user offers
to submit a pull request to update the documentation to reflect the correct
behavior   of the function.

#### Comment Summary (1 comments)

The comments indicate a request for a pull request (PR) to be submitted. This
suggests that there is a desire to       incorporate changes or improvements
into the codebase. The brevity of the comment implies that the necessary context
or details have already been discussed or are understood by the involved
parties.

#### Comment Details

* tmbdev commented on 2024-06-17 02:48:38

---
### Issue #362: batched augmentations with kornia

* **Created by:** edgarriba
* **Created on:** 2024-06-04 08:36:52

#### Summary

A founder and maintainer of the Kornia library is exploring the possibility of
implementing function hooks to          preprocess or augment images and labels
in batches directly on device tensors. This enhancement aims to improve the
efficiency and performance of image processing tasks by leveraging device-level
operations. The exploration could lead to significant improvements in how Kornia
handles image data, potentially offering more streamlined and faster
processing capabilities.

#### Comment Summary (1 comments)

The speaker expresses willingness to investigate the matter and inquires about
the specific types of hooks required.

#### Comment Details

* tmbdev commented on 2024-06-17 02:48:23

---
### Issue #361: ShardListDataset does not work with multiprocessing_context=spawn

* **Created by:** bgedik
* **Created on:** 2024-06-02 06:35:25

#### Summary

The issue arises from the use of lambda functions in wids.cache_localname and
wids.default_localname, which leads to   an AttributeError when attempting to
pickle these local objects. This error can be resolved by replacing the lambda
functions with callable classes, CacheLocalname and DefaultLocalname, which
properly handle the creation of local      names for shards and ensure
compatibility with pickling. The solution involves defining these classes with
an __init__ method to set up directories and a __call__ method to process URLs.

#### Comments

No comments on this issue.

---
### Issue #360: PyPI packages not available for 2.90 or 2.88

* **Created by:** bgedik
* **Created on:** 2024-05-28 23:11:32

#### Summary

The user is inquiring about the availability of newer versions of the webdataset
package on PyPI, as the latest        version listed is 0.2.86. They are
interested in knowing if there are plans to publish more recent updates to the
package on the platform. This suggests a need for maintaining up-to-date package
versions for users relying on PyPI    for their dependencies. #enhancement
#update

#### Comment Summary (1 comments)

The user plans to release a new update in the coming days, as they have been
busy installing six new machines in their home server and upgrading to version
24.04.

#### Comment Details

* tmbdev commented on 2024-06-17 02:53:14

---
### Issue #357: Memory leak during training with standard DataLoader coupled with WebDataset dataloader

* **Created by:** vishaal27
* **Created on:** 2024-05-20 15:15:32

#### Summary

The issue arises when training CLIP models using a combination of a standard
dataloader and a webdataset loader,       leading to excessive swap space usage
and job termination. While CPU memory usage remains stable, the swap space
depletes rapidly when both loaders are used together, unlike when only the
standard dataloader is employed. This       suggests a potential inefficiency or
memory leak in the webdataset loader or its interaction with the standard
dataloader, causing the system to rely heavily on swap space, ultimately
resulting in the job being killed.

#### Comment Summary (5 comments)

The memory leak issue in PyTorch was linked to the itertools.cycle function, but
in this case, the problem seems to be related to the WebLoader and possibly the
repeat=True argument in the dataset, which doesn't use itertools.cycle. The
repeat argument sets self.repetitions and iterates through the dataset multiple
times, which might be causing memory   issues. WebDataset itself doesn't retain
data, except for buffering in the shuffle function. A user inquired about
shuffling keys before loading data into memory to avoid filling a buffer with
in-memory undecoded byte numpy arrays,   suggesting a need for more efficient
data handling.

#### Comment Details

* vishaal27 commented on 2024-05-20 17:36:40
* volcverse commented on 2024-05-31 08:44:16
* vishaal27 commented on 2024-05-31 10:01:12
* tmbdev commented on 2024-06-17 03:09:29
* noamsgl commented on 2025-04-10 16:29:21

---
### Issue #349: Update pypi with 0.2.88?

* **Created by:** chrisdxie
* **Created on:** 2024-04-15 15:06:36

#### Summary

The issue concerns the delay in updating the PyPI repository to reflect version
0.2.88, as it currently only shows     version 0.2.86. This discrepancy has been
previously reported in issue #342, indicating a potential oversight or delay in
the release process. Prompt resolution is necessary to ensure users have access
to the latest version. #enhancement #error

#### Comments

No comments on this issue.

---
### Issue #325: Buckets of elements of different sizes

* **Created by:** ethansmith2000
* **Created on:** 2024-01-18 01:46:54

#### Summary

The issue involves using webdataset to load datasets organized by image size
into separate folders, each containing    tar files, and creating iterators for
each dataset. The approach causes memory issues that scale unpredictably with
the number of nodes and workers, with memory usage increasing by approximately
7GB per step, leading to premature      termination despite having a 1TB RAM
allowance. The user is exploring whether the switching between datasets
contributes to the problem and is considering alternative solutions, such as
using map-style datasets with a sampler   in PyTorch, to manage memory more
effectively.

#### Comment Summary (4 comments)

The issue of training jobs being killed was traced to a memory leak caused by
the itertools.cycle function, not the    WebDataset itself. The problem arose
when using both an image data-loader and a WebDataset loader together, leading
to excessive swap space usage. The solution involved removing the cycle
function, which resolved the memory leak.         Additionally, it was suggested
to avoid using shuffle in each component dataset and instead create a new
IterableDataset with a shuffle method to manage data loading efficiently. This
approach ensures that data is properly  shuffled and batched without causing
memory issues.

#### Comment Details

* vishaal27 commented on 2024-05-20 15:00:24
* vishaal27 commented on 2024-05-20 17:37:21
* tmbdev commented on 2024-06-17 03:14:36
* rardz commented on 2024-08-14 02:40:30

---
### Issue #323: AttributeError: module 'webdataset' has no attribute 'Dataset'

* **Created by:** YYDreamzure
* **Created on:** 2024-01-08 09:50:40

#### Summary

The issue is that the wds module no longer has the Dataset attribute, which may
cause confusion or errors for users    following the documentation that still
references wds.Dataset. This discrepancy suggests that the documentation is
outdated or the module has been updated without corresponding documentation
changes. Users relying on wds.Dataset need to find an alternative or updated
method to achieve the same functionality. #error #documentation

#### Comment Summary (1 comments)

The issue was resolved by downgrading the webdataset package to version 0.1.103
using the command pip install          webdataset==0.1.103.

#### Comment Details

* harikudil commented on 2024-01-12 01:46:52

---
### Issue #314: Please provide a mechanism for detecting when one tarball has been fully consumed

* **Created by:** pabl0
* **Created on:** 2023-11-30 15:44:41

#### Summary

The issue involves the inability to track when a tar file has been fully
consumed by tariterators.tar_file_expander,   especially for remote files
represented as io.BytesIO objects, which lack filename or URL information. While
local     files can be managed by monkey patching the tar_file_iterator
function, remote files present a challenge. Addressing   this issue might be
facilitated by resolving #311, potentially allowing integration with the close
method.

#### Comment Summary (2 comments)

The __index_in_shard__ metadata can be added to samples to indicate when the
last shard is fully consumed, similar to  the functionality in the new "wids"
library, which also supports random access. A quick implementation of this idea
has proven effective, and it has been used to resolve issues with single-sample
tars that previously encountered       duplicate key problems.

#### Comment Details

* tmbdev commented on 2024-01-04 19:29:41
* jpc commented on 2024-10-23 20:29:06

---
### Issue #310: Problem about webdataset load the same tar file by epoch

* **Created by:** si22bao
* **Created on:** 2023-11-16 13:13:53

#### Summary

The issue involves a webdataset where ResampledShards is designed to ensure each
GPU and worker loads different tar    files by repeating them. However, the
dataset reloads the same tar file from the beginning at the start of a new
epoch, likely due to the worker in the dataloader being reset with each new
iteration. To resolve this, consider       implementing a mechanism to persist
the state of the worker across epochs or modify the dataloader to maintain shard
progress between epochs.

#### Comment Summary (1 comments)

The pipeline may restart from the beginning of a shard at the next epoch,
especially if there's only one shard per     worker, leading to partial reads.
To mitigate this, using "repeatedly" can help, but for distributed training with
small datasets, the new "wids" interface is recommended, with examples available
in the ./examples subdirectory.       Additionally, there's a need to determine
if changing "with_epoch" could resolve this unexpected behavior.

#### Comment Details

* tmbdev commented on 2024-01-04 17:47:53

---
### Issue #308: Curl in popen doesn't close if there are no tar shard on an AWS.

* **Created by:** dmasny99
* **Created on:** 2023-10-31 13:57:53

#### Summary

In a custom implementation of the OpenCLIP project using webdataset, a problem
arises when curl within gopen fails to  retrieve tar files due to network
errors, causing the pipe to remain open improperly. This results in a timeout
for    PyTorch's watchdog collective operations. The issue can be resolved by
adjusting flags in the curl command, suggesting it might be a common problem for
others using similar setups.

#### Comment Summary (1 comments)

Error handling for missing shards is minimal, typically resulting in job
termination. To handle network errors while   continuing the loading process,
it's recommended to use a shell script that repeatedly attempts to download the
shard  using "curl." This approach ensures resilience against transient network
issues.

#### Comment Details

* tmbdev commented on 2024-01-04 18:05:30

---
### Issue #306: Loss of performance

* **Created by:** karamavusibrahim
* **Created on:** 2023-10-25 19:54:18

#### Summary

The user is experiencing a performance drop and increased GPU memory usage when
switching from a custom dataset to     using webdataset for training with two
A100 GPUs. The training process is slower, and memory usage is higher with
webdataset, which might be expected due to different data handling and loading
mechanisms. Additionally, the user      encounters an AttributeError related to
the NoneType object when attempting to use webdataset in a batched format,
likely due to compatibility issues with the accelerate library and WebLoader.

#### Comments

No comments on this issue.

---
### Issue #303: how to use num_workers in ddp training?

* **Created by:** zhangvia
* **Created on:** 2023-10-08 10:01:42

#### Summary

When using the WebDataset in distributed data parallel (DDP) training, setting
num_workers to 0 results in the         expected number of steps per epoch,
calculated as 8000 / 2 / 20 = 200. However, increasing num_workers to 16 causes
the total steps to incorrectly inflate to 3200, suggesting a miscalculation or
misconfiguration in how data is being   distributed or processed across workers.
This discrepancy may be due to how the data loader handles batching and
worker processes, potentially leading to data being processed multiple times.
Investigating the data loading and       worker synchronization logic might help
resolve this issue.

#### Comment Summary (1 comments)

To optimize your data loading process, consider moving the with_epoch method
from WebDataset to WebLoader.             Additionally, enhance data randomness
by implementing cross-worker shuffling using
unbatched().shuffle(2000).batched(20).with_epoch(200) on the WebLoader. This
approach should improve the efficiency    and randomness of your data pipeline.

#### Comment Details

* tmbdev commented on 2023-10-10 17:21:58

---
### Issue #301: too many files when using multiprocessing with pytorch dataloader and `decode`

* **Created by:** ekorman
* **Created on:** 2023-09-30 19:14:05

#### Summary

The issue arises when using the .decode() method in conjunction with multiple
workers in a PyTorch DataLoader, leading to an OSError due to too many open
files. This behavior is observed as the number of open files increases with each
iteration when decode is set to True, but remains stable when decode is False or
num_workers is set to 0. The problem  seems to be related to file handling
within the decoding process, potentially indicating a resource management issue
in the webdataset library when used with multiple workers.

#### Comment Summary (2 comments)

Users are experiencing issues with webdataset==0.2.86 and torch==2.3.1, and are
seeking solutions. They are inquiring  if @tmbdev is aware of the problem and if
it is being addressed.

#### Comment Details

* mhyatt000 commented on 2024-07-22 02:31:57
* Aceticia commented on 2025-03-21 16:14:11

---
### Issue #295: How can I use cache by "wds.DataPipeline(*pipeline)"

* **Created by:** Yangr116
* **Created on:** 2023-09-19 07:03:24

#### Summary

The issue involves a request for an example of implementing caching within the
wds.DataPipeline(*pipeline) function,   specifically when the first element of
the pipeline is wds.SimpleShardList(input_shards). This suggests a need for
optimizing data retrieval processes by storing frequently accessed data
temporarily to improve performance. The user   is likely seeking guidance on
enhancing the efficiency of data processing in a WebDataset pipeline.

#### Comments

No comments on this issue.

---
### Issue #288: Problem with parsing filenames

* **Created by:** gau-nernst
* **Created on:** 2023-08-14 06:53:32

#### Summary

The issue arises from the base_plus_ext() function in the WebDataset library,
which incorrectly splits keys containing periods (.) when extracting file
extensions, leading to unexpected key formats during data loading. The user
expects   keys like 'S_02_voice/S_02/multiple_sentences/S_02_4011_VE4.wav' but
receives
'S_02_voice/S_02/multiple_sentences/S_02_4011_VE4' instead. Possible solutions
include repacking data to avoid periods in keys or modifying base_plus_ext() to
use os.path.splitext() for more accurate splitting. Documentation updates
could also help warn users about this behavior.

#### Comment Summary (1 comments)

The user agrees with @gau-nernst and shares their experience of naming crawled
images using a timestamp format like    f"{datetime.now().isoformat()}.jpg".
They encountered a ValueError due to the TAR archives not being in the expected
WebDataset format, as the files didn't share the same prefix or types. They
suggest that warning users to avoid using  dots in file names is good advice,
given the library's regex requirements.

#### Comment Details

* phunc20 commented on 2025-01-15 10:09:10

---
### Issue #287: Compatibility with JSONL

* **Created by:** zanussbaum
* **Created on:** 2023-08-09 03:13:48

#### Summary

The issue involves the use of JSONL files with the WebDataset pipeline, which is
not yet documented, raising questions about compatibility and implementation.
The user inquires whether a modification of the tarfile_to_samples function is
necessary to handle JSONL files, seeking a small example for guidance. This
suggests a need for documentation and      potential code adaptation to support
JSONL files effectively.

#### Comments

No comments on this issue.

---
### Issue #286: When I have a duplicate entry I get a secondary error

* **Created by:** enjalot
* **Created on:** 2023-08-04 17:28:50

#### Summary

The issue arises from a ValueError indicating a duplicate file name in a tar
file, specifically for the file
261c5b95-bf20-4515-a28e-462c93821190.jpg. This error is compounded by a
subsequent NameError due to an undefined       variable 'source' during
exception handling. The problem likely stems from improper error handling and
file management within the tariterators.py script of the WebDataset project.

#### Comments

No comments on this issue.

---
### Issue #282: "Issues with 'with_epoch' Configuration Relative to 'batch_size' in Distributed Training Using PyTorch Lightning and WebDataset"

* **Created by:** JiaqiLiao77
* **Created on:** 2023-07-17 12:52:06

#### Summary

When using PyTorch Lightning with WebDataset for distributed DDP training, an
issue arises where the with_epoch        setting behaves differently based on
the use of the dataloader. Without specifying a batch_size, with_epoch is
applied to each sample, but when batch_size is set in the dataloader, with_epoch
corresponds to the number of batches instead. This discrepancy affects how
epochs are defined and managed during training, potentially impacting the
training        process and results.

#### Comment Summary (1 comments)

The comment clarifies that the "with_epoch" function slices the stream based on
items rather than samples,             necessitating the division of the desired
epoch size by the batch size. It raises a question about potential issues
with this approach, implying a need to consider any problems that might arise
from this method of slicing.

#### Comment Details

* tmbdev commented on 2023-07-27 08:07:23

---
### Issue #279: issue in tariterators.py

* **Created by:** omar-ahmd
* **Created on:** 2023-07-12 11:41:53

#### Summary

In line 244, the code mistakenly uses the variable source instead of filesample,
leading to an error due to source     being undefined. This issue likely causes
a runtime error, disrupting the program's execution. Correcting the variable
name should resolve the error and improve the program's functionality. #error
#serious

#### Comment Summary (1 comments)

The issue has been identified and resolved, thanks to your observation.

#### Comment Details

* tmbdev commented on 2023-07-27 08:15:32

---
### Issue #272: The intended copy behaviour of `compose` is not achieved.

* **Created by:** ferrine
* **Created on:** 2023-06-01 09:06:50

#### Summary

The issue arises in the compose method of the webdataset library, where a
shallow copy of the pipeline is made,        leading to unintended modifications
of the original DataPipeline object when the append method is used. This results
in unexpected behavior, such as modifying p0 when creating new pipelines p1 and
p2 from p0. The expected behavior is   that the compose method should not affect
the original pipeline p0.

#### Comment Summary (2 comments)

The user is willing to open a pull request (PR) to address the issue quickly.
The recipient appreciates the report and apologizes for the delayed response,
expressing willingness to include the PR or investigate the issue themselves if
a PR is not provided.

#### Comment Details

* ferrine commented on 2023-06-01 09:17:20
* tmbdev commented on 2023-07-27 08:29:41

---
### Issue #271: resume dataloader

* **Created by:** mactavish91
* **Created on:** 2023-05-22 16:32:48

#### Summary

The issue involves resuming distributed training with WebDataset after an
interruption, specifically when handling a   large dataset of up to 2 billion
samples. The goal is to continue training from the point of interruption without
retraining previously processed data, ensuring efficient use of resources and
time. Implementing a checkpointing       mechanism that saves the state of the
training process, including the current epoch and sample index, can help achieve
this. #performance #enhancement

#### Comment Summary (6 comments)

The comments discuss handling distributed training with resampling, which
simplifies the process by eliminating the    need to resume training, as
restarting yields the same sample distribution. However, tracking epochs and
samples       requires managing the distributed state of PyTorch, which
WebDataset currently cannot do. Suggestions include making   the pipeline state
more explicit and providing an API for managing shard states, while
acknowledging that resampling   maintains a uniform sample distribution without
a defined epoch end. Alternatives like WIDS and recent implementations in
datasets for dataloader resuming are mentioned, with a potential extension in
webdataset through NVIDIA's           Megatron-Energon.

#### Comment Details

* tmbdev commented on 2023-07-27 08:38:11
* ethanhe42 commented on 2024-09-12 15:55:21
* tmbdev commented on 2024-09-24 19:22:19
* tmbdev commented on 2024-09-24 19:23:06
* lhoestq commented on 2024-10-01 10:18:49
* ethanhe42 commented on 2025-01-09 18:26:12

---
### Issue #269: Arg `repeat` of class `WebDataset` is not used

* **Created by:** ychfan
* **Created on:** 2023-05-02 23:42:21

#### Summary

The issue involves the repeat parameter being ignored in the __init__ function
of the webdataset library, specifically in the compat.py file. This oversight
may lead to unexpected behavior when users attempt to utilize the repeat
functionality, potentially affecting the intended data processing workflow.
Addressing this issue would ensure that    the repeat parameter is properly
recognized and utilized, enhancing the library's functionality. #error
#enhancement

#### Comment Summary (3 comments)

The comments discuss issues with the WebDataset API, specifically regarding the
repeat argument, which seems outdated  due to recent code changes. Users note
that without setting with_epoch, workers will endlessly repeat their shard
shares. It is suggested that using the .repeat() method explicitly might resolve
the issue, and the repeat argument is likely a remnant that should be removed.

#### Comment Details

* Burningdust21 commented on 2023-09-18 13:34:40
* qibin2020 commented on 2024-01-28 10:25:59
* tmbdev commented on 2024-02-28 03:53:12

---
### Issue #267: Sharded Dataset Has Long Delay Before First Batch (and caching=False)

* **Created by:** cupdike
* **Created on:** 2023-04-30 18:58:03

#### Summary

The issue involves inefficient processing of a sharded dataset of large images
using WebDataset, where the initial     metadata and tar download phases are
followed by a prolonged idle period before batch processing begins. This results
in an uneven cadence of batch completions, with the first batch taking
significantly longer than subsequent ones,      leading to suboptimal GPU
utilization. The user seeks a solution to initiate batch processing sooner and
maintain a    consistent processing rhythm to enhance performance.

#### Comment Summary (8 comments)

The discussion revolves around performance issues with WebDataset, particularly
long delays in GPU utilization and     file loading times, possibly due to
caching or configuration issues. Users report long waits before GPU activity
begins and variability in file load times, despite not enabling caching.
Suggestions include checking for large batch  sizes, shuffle buffers, slow disk
access, and DataLoader configurations, with a focus on I/O bottlenecks due to
hardware limitations. Alternatives like Deep Lake and FFCV are mentioned, but no
definitive solution is provided,      highlighting the need for optimized
configurations and hardware improvements.

#### Comment Details

* tmbdev commented on 2023-05-01 17:05:39
* cwerner commented on 2023-05-27 09:23:08
* cupdike commented on 2023-06-05 17:50:53
* cwerner commented on 2024-02-16 22:28:11
* cupdike commented on 2024-02-16 22:37:06
* cwerner commented on 2024-02-16 23:08:41
* cupdike commented on 2024-02-16 23:18:29
* tmbdev commented on 2024-02-28 13:05:15

---
### Issue #262: Are there plans to support WebP?

* **Created by:** CS123n
* **Created on:** 2023-04-13 07:31:55

#### Summary

The issue involves the manual transfer of WebP images into a TarWriter object,
as opposed to JPEG and PNG formats      which can be stored directly. This
discrepancy suggests a potential limitation or lack of support for WebP format
in   the current implementation, requiring additional steps to handle WebP
images. Addressing this could enhance the        functionality and streamline
the process for users dealing with various image formats.

#### Comment Summary (1 comments)

You can either add a handler to webdataset.writer.default_handlers or override
the encoder in TarWriter for            customization. ImageIO appears to
support this functionality natively, suggesting it might be added by default in
the  future. Improved documentation and a more user-friendly API are needed.

#### Comment Details

* tmbdev commented on 2023-04-28 20:30:53

---
### Issue #260: [REQUEST] rename with_epoch()  method of WebDataset

* **Created by:** ozanciga
* **Created on:** 2023-04-03 16:24:25

#### Summary

The issue highlights confusion over the naming of a method crucial for
distributed training, specifically the          .with_epoch method, which is not
clearly explained in the manual. The user suggests that a more descriptive name,
such as .set_one_epoch_samples(), could improve clarity and usability. This
change could enhance understanding and reduce   the need for trial and error
when using the function.

#### Comment Summary (2 comments)

The comments discuss the challenges of changing established names in code,
highlighting that while renaming methods    like with_epoch() to something
clearer like .set_one_epoch_samples() can improve understanding, such changes
can be    incompatible. There's a suggestion to enhance documentation with more
details and examples to clarify usage without    altering existing names.
Improving documentation is seen as a viable solution to address confusion
without breaking    compatibility.

#### Comment Details

* tmbdev commented on 2023-04-28 21:20:08
* HuangWanqiu commented on 2024-02-03 12:23:23

---
### Issue #256: Why does it consume so much memory?

* **Created by:** LinB203
* **Created on:** 2023-03-10 01:50:26

#### Summary

The issue involves a memory leak in a program that processes a video encoded as
16 images, an image in .pth format, a  title, and 8 captions using a custom
decode function. Despite attempts to manage memory using deepcopy, del, and
gc.collect(), the program crashes after extended training sessions, indicating
ineffective memory release. The user    seeks advice on potential code issues or
insights into how WebDataset handles memory management.

#### Comment Summary (6 comments)

The comments discuss a memory issue encountered during training with a custom
Decode function in a data pipeline from  OpenCLIP, using WebDataset with 200
.tar files, each containing 5,000 data sets. The problem persists even when
reducing the batch size from 80 to 40 across 8 A100 GPUs, each with 10 workers.
The issue is suspected to stem from    the shuffle buffer size, with suggestions
to reduce _SHARD_SHUFFLE_SIZE and _SAMPLE_SHUFFLE_SIZE to decrease memory
usage. A user inquires if reducing these buffer sizes resolved the problem,
referencing a similar issue on GitHub.

#### Comment Details

* LinB203 commented on 2023-03-10 01:51:17
* LinB203 commented on 2023-03-10 01:57:28
* LinB203 commented on 2023-03-10 02:01:00
* LinB203 commented on 2023-03-10 07:24:36
* tmbdev commented on 2023-03-18 19:39:54
* vishaal27 commented on 2024-05-20 15:11:28

---
### Issue #254: Using s3fs as handler instead of "pipe:" doesn't open the files. 

* **Created by:** ahmadsachalrb
* **Created on:** 2023-03-09 07:24:47

#### Summary

The user is experiencing high RAM consumption leading to crashes when using the
"pipe:" method in webdataset for       loading data from S3, and attempts to use
s3fs as a workaround. However, they encounter an "Attribute Error: has no
'startswith'" when trying to use s3fs as a handler in webdataset, and are
uncertain if the pipe URL utilizes s3fs.     They seek a solution to efficiently
load data without excessive RAM usage or errors.

#### Comment Summary (1 comments)

To address the issue temporarily, users can utilize the Pipeline interface as a
workaround for opening URLs. This      method serves as an interim solution
until a permanent fix is implemented.

#### Comment Details

* tmbdev commented on 2023-03-18 19:43:00

---
### Issue #251: In DDP training, .with_length() meets "_pickle.PicklingError: Can't pickle <class 'webdataset.pipeline.WebDataset_Length'>: attribute lookup WebDataset_Length on webdataset.pipeline failed "

* **Created by:** JingyunLiang
* **Created on:** 2023-03-01 13:49:09

#### Summary

In Distributed Data Parallel (DDP) training, an error occurs when using
.with_length() due to a _pickle.PicklingError, which indicates that the
WebDataset_Length class cannot be pickled because its attribute lookup fails.
This issue      likely arises from the multiprocessing module's inability to
serialize the WebDataset_Length class, which is necessary for parallel
processing. Addressing this error may involve ensuring that the class is defined
in a way that supports   pickling or adjusting the multiprocessing strategy.

#### Comment Summary (3 comments)

The bug in DDP was fixed by hacking the class to override the __len__ method
before creating an instance, as           .with_length() affects the instance
method, not the class. This approach is necessary because pickle can only
serialize top-module level functions, requiring a direct class modification. The
solution involves setting             wds.WebDataset.__len__ and
wds.WebLoader.__len__ to a custom function, but issues persist if the code is
added after   dataset creation or when using
multiprocessing.set_start_method('spawn') with with_length.

#### Comment Details

* JingyunLiang commented on 2023-03-01 14:10:25
* zhangvia commented on 2023-10-08 08:07:07
* npuichigo commented on 2024-11-25 08:16:57

---
### Issue #250: Using DDP with WebDataset in pytorch lightning

* **Created by:** adhakal224
* **Created on:** 2023-02-20 20:25:19

#### Summary

The user is experiencing issues with using WebDataset for multi-GPU training,
receiving a ValueError indicating the    need for an explicit nodesplitter for
multi-node training. They have tried two approaches to resolve this: using
.with_epochs and ddp_equalize, but their training remains slow. They seek
clarification on the differences between     these methods, the role of
dataset_size, and best practices for using WebDataset with PyTorch Lightning in
a           multi-GPU, multi-node setup. They also request guidance on
optimizing their current dataloader configuration for       better performance.

#### Comment Summary (25 comments)

Users are discussing the use of Distributed Data Parallel (DDP) with WebDataset
in PyTorch Lightning, referencing the  OpenCLIP codebase. Some users experience
issues with training getting stuck and confusion about the resample +
with_epoch approach. Others report successful training but note the lack of
documentation on optimal usage. The        conversation includes code snippets
and links to examples, with discussions on setting with_epoch and num_workers
for  efficient data loading. Users also mention the deprecated ddp_equalize and
suggest using wids for indexed datasets.    The need for better documentation
and examples is a recurring theme.

#### Comment Details

* superhero-7 commented on 2023-02-26 14:54:23
* adhakal224 commented on 2023-02-26 18:35:47
* superhero-7 commented on 2023-02-27 01:34:36
* adhakal224 commented on 2023-03-01 05:44:05
* superhero-7 commented on 2023-03-01 07:33:34
* tmbdev commented on 2023-03-03 20:34:21
* cliffzhao commented on 2023-03-09 01:24:08
* ForJadeForest commented on 2023-03-18 15:48:03
* cliffzhao commented on 2023-03-20 01:06:07
* tmbdev commented on 2023-03-20 08:05:33
* ForJadeForest commented on 2023-03-24 15:46:03
* urinieto commented on 2023-03-28 23:08:05
* RoyJames commented on 2023-06-06 13:00:37
* alexqdh commented on 2023-06-12 11:43:33
* jrcavani commented on 2023-09-28 02:02:50
* laolongboy commented on 2023-10-09 08:59:32
* HuangChiEn commented on 2024-03-19 10:02:35
* tmbdev commented on 2024-03-20 23:58:11
* tmbdev commented on 2024-03-21 00:03:34
* tmbdev commented on 2024-03-21 00:14:48
* HuangChiEn commented on 2024-03-21 03:04:22
* harrytranx commented on 2024-06-12 02:13:05
* HuangChiEn commented on 2024-06-12 08:20:34
* amyxlu commented on 2024-08-03 16:59:01
* HuangChiEn commented on 2024-08-08 00:25:19

---
### Issue #242: MultiShardSample uses same seed on every worker

* **Created by:** iceboundflame
* **Created on:** 2023-01-23 04:20:35

#### Summary

The MultiShardSample feature in the webdataset library is designed to combine
multiple datasets with different         sampling weights, but its current
implementation uses the same random number generator (RNG) for every worker
without  seeding it according to the worker ID. This results in each PyTorch
DataLoader worker process receiving the same set   of shards, which is not the
intended behavior. The README.md suggests this capability might still be a
work-in-progress (WIP) feature, and the issue is observed in version 0.2.31.

#### Comment Summary (1 comments)

The issue of PyTorch DataLoader workers receiving the same set of shards is not
a bug but a feature that can be        managed by splitting the shard list among
workers. By using a pipeline with wds.MultiShardSample and extending it with
wds.split_by_node and wds.split_by_worker, you can effectively distribute shards
across workers and nodes. The         provided split_by_node and split_by_worker
functions utilize islice to ensure each worker processes a unique subset of
shards, leveraging PyTorch's worker information to achieve efficient data
distribution.

#### Comment Details

* SCZwangxiao commented on 2023-03-10 05:26:35

---
### Issue #240: AttributeError: 'DataPipeline' object has no attribute 'with_epoch'

* **Created by:** varadgunjal
* **Created on:** 2023-01-12 03:02:58

#### Summary

The issue arises when creating a wds.DataPipeline object with a SimpleShardList
and additional processing functions,   resulting in an unspecified error. This
could be due to an incorrect installation or compatibility issue between
webdataset and the specified version of torch. Ensure all dependencies are
correctly installed and compatible, and     consider providing the exact error
message for more precise troubleshooting.

#### Comments

No comments on this issue.

---
### Issue #237: Periods in base filename interpreted as extensions

* **Created by:** iceboundflame
* **Created on:** 2022-12-22 20:07:06

#### Summary

The issue arises from filenames containing periods in the base part, such as
./235342 Track 2.0 (Clean Version).mp3,   which result in unexpected keys like 0
(clean version).mp3. This design choice, intended to support multiple
extensions, leads to sporadic downstream errors, such as ValueError, when the
expected file extensions are not found.  It is recommended to update the
documentation to clarify this behavior and prevent confusion.

#### Comment Summary (1 comments)

The use of "." for multiple extensions in WebDataset is intentional and common,
as it simplifies downstream pipeline   creation, despite being a restriction on
tar files. While it's possible to map file names in the input pipeline to
address this, it's not recommended; handling it during dataset creation is
preferable. Additionally, "glob" patterns   like "*.mp3" can be used to match
extensions, and documentation improvements are planned to address these
conventions.

#### Comment Details

* tmbdev commented on 2023-01-31 21:49:04

---
### Issue #234: Checkpoint support

* **Created by:** yncxcw
* **Created on:** 2022-12-13 21:03:31

#### Summary

The issue concerns the lack of support for checkpointing in WebDataset, which is
needed to resume data loading from a  specific index after a crash during
training. The user seeks a feature where the index of the current sample being
loaded is saved during checkpointing, allowing WebDataset to continue from that
point upon resuming. This enhancement  would improve the robustness and
efficiency of training workflows using WebDataset.

#### Comment Summary (1 comments)

The use case appears limited due to the challenge of ensuring restartability
with multiple worker processes. The       preferred method is resampled
training, which allows for restarting without statistical anomalies and
accommodates any number of workers. This approach ensures consistency and
flexibility in training processes.

#### Comment Details

* tmbdev commented on 2022-12-20 22:00:04

---
### Issue #233: Updated syntax for multinode usage

* **Created by:** app/
* **Created on:** 2022-12-10 23:19:56

#### Summary

The issue involves setting up a multi-core PyTorch training environment on
Google Colab using WebDataset with data     stored in GCP Buckets. The user is
experiencing unexpected behavior where every core processes every file, despite
the documentation suggesting that a default nodesplitter and split_by_worker
should distribute shards across cores. The    user seeks advice on implementing
a basic version with shard and sample shuffling, and is considering using the
main   release of WebDataset rather than the v2 branch, while also contemplating
providing a minimum working example for      clarity.

#### Comment Summary (4 comments)

The comments discuss achieving the correct data splitting behavior using the
WebDataset library in a distributed       setting. Initially, the user
successfully implemented node and worker splitting with a visual test, using
torch.distributed for process management. They noted that using detshuffle()
ensures deterministic shuffling across    nodes, while shuffle() results in non-
deterministic shuffling due to different random seeds. The user also highlighted
that WebDataset does not perform node splitting by default and should ideally
raise an error in a multinode setup,     which it currently does not. They plan
to enhance error detection and documentation.

#### Comment Details

* jrcavani commented on 2023-09-27 21:04:23
* jrcavani commented on 2023-09-27 21:18:31
* jrcavani commented on 2023-09-28 01:35:37
* tmbdev commented on 2023-10-10 17:26:46

---
### Issue #228: when converting dataset to wds, data is getting larger.

* **Created by:** AlvL1225
* **Created on:** 2022-11-23 08:10:48

#### Summary

The issue involves a significant increase in dataset size when converting from
LMDB format to WebDataset (wds) format  using ShardWriter. The original dataset,
consisting of 880k image-text pairs, was 202GB in LMDB format but expanded to
474GB in wds format, even when attempting to compress with tar.gz. This suggests
inefficiencies in the conversion or   compression process, potentially due to
the storage format or compression settings used.

#### Comment Summary (5 comments)

The user is experiencing a significant decrease in decoding speed when using the
decode("rgb") function with           WebDataset, dropping from 1600 it/s to 30
it/s, and is questioning if this is normal. They also note that setting
num_workers=8 with batch_size=32 does not utilize multiple threads as expected,
possibly due to insufficient shards    for the number of workers. Additionally,
there is a concern about the size of tar files being larger than expected,
potentially due to differences in image formats or tar file handling, and a
suggestion to use JPEG for better          performance.

#### Comment Details

* AlvL1225 commented on 2022-11-23 09:07:25
* AlvL1225 commented on 2022-11-25 06:43:12
* tmbdev commented on 2022-12-09 18:23:43
* tmbdev commented on 2022-12-09 18:25:35
* gau-nernst commented on 2024-10-28 01:46:01

---
### Issue #219: AttributeError: module 'webdataset' has no attribute 'ShardList'

* **Created by:** drscotthawley
* **Created on:** 2022-10-25 03:44:54

#### Summary

The issue arises from attempting to use ShardList from the webdataset module,
which results in an AttributeError       indicating that ShardList is not an
attribute of the module. This suggests a possible documentation error or a
version mismatch where the ShardList class may have been renamed, removed, or
not yet implemented in the installed version of  the library. Users should
verify the library version and consult the latest documentation or source code
for updates   or alternative methods.

#### Comment Summary (2 comments)

The user attempted to use wds.WebDataset with a splitter to resolve a ValueError
related to multi-node training but    encountered a TypeError due to an
unexpected keyword argument. The issue arises because in version 2, ShardList
has    been renamed to SimpleShardList, and splitter is now referred to as
nodesplitter. The user acknowledges the need to    update their worksheet to
reflect these changes.

#### Comment Details

* drscotthawley commented on 2022-10-25 03:48:10
* tmbdev commented on 2022-12-09 18:48:55

---
### Issue #217: Decompression vs Sequential Read

* **Created by:** Stack-Attack
* **Created on:** 2022-10-18 05:21:21

#### Summary

The issue involves optimizing data read performance in an environment
constrained by HDD read speeds, particularly     when dealing with sharded data.
The current approach achieves optimal performance with a single worker due to
maximal  sequential read speeds, but there's interest in leveraging CPU
resources by compressing data with GZIP. Two potential  solutions are proposed:
arranging parallel processes to read from the same shard at offset intervals or
maintaining    minimal read processes with a parallel decompression algorithm
like PIGZ/PUGZ. The user seeks validation of these      assumptions and
potential alternative solutions.

#### Comment Summary (2 comments)

The comments discuss strategies for efficiently reading multiple shards from a
rotational drive, highlighting the slow performance of simultaneous reads.
Suggestions include copying a single shard to a RAM file system or SSD, using a
web server to manage sequential access, and employing file locking with a script
to ensure sequential reads. Additionally, a Python multiprocessing approach is
recommended for parallel processing of data. The web server is noted to be
faster than disk I/O because it manages requests sequentially, reducing the
overhead of simultaneous disk access. The         approach is not perfect but
offers a balance between simplicity and performance.

#### Comment Details

* tmbdev commented on 2022-12-20 23:07:02
* Stack-Attack commented on 2023-01-16 04:55:07

---
### Issue #216: Shard writer with a gcloud url 

* **Created by:** Fawzikhattar
* **Created on:** 2022-10-13 14:49:34

#### Summary

The issue involves the shard writer in the WebDataset library not functioning
correctly with a Google Cloud URL. The   problem appears to be resolved by
setting the stream to self.fname at a specific line in the code. Additionally,
there is a question about the necessity of opening the file before passing it to
the TarWriter constructor. #error           #enhancement

#### Comment Summary (3 comments)

The recommended usage of ShardWriter involves writing data to a local disk
before copying it to the cloud, but some    users face challenges with this
approach due to large dataset sizes that exceed local storage capacity. An
enhancement to allow direct writing to the cloud is suggested, and a user has
found a workaround by modifying a specific line in   the code to enable this
functionality. This modification allows them to bypass local storage limitations
and write     directly to the cloud effectively.

#### Comment Details

* tmbdev commented on 2022-12-20 23:08:48
* adamklec commented on 2023-03-26 18:23:35
* justinpinkney commented on 2023-06-15 11:21:18

---
### Issue #212: Webdataset sharding download

* **Created by:** itepifanio
* **Created on:** 2022-09-19 15:07:26

#### Summary

The issue concerns the use of webdataset with S3 and multiple shards,
specifically regarding the behavior of automatic sharding and sample caching.
The user is uncertain if all 100 tar files are downloaded at once or iteratively
when     accessing items, and whether the entire shard is downloaded when using
sample caching. Clarification in the            documentation is needed to
understand the data download process and optimize performance.

#### Comment Summary (5 comments)

The comments discuss the behavior of a data processing system with caching and
streaming options. By default, data is  streamed without local caching, but
enabling caching downloads each shard completely before use, which can delay
training. Users can customize cache file names using the url_to_name argument.
Concerns are raised about the delay in  training start when caching is enabled,
as it waits for the first shard to download completely, unlike streaming,
which starts training immediately. Additionally, there's confusion about why
subsequent shards aren't downloaded       concurrently during training, despite
having multiple workers.

#### Comment Details

* tmbdev commented on 2022-12-20 23:11:17
* itepifanio commented on 2022-12-21 00:37:16
* nauyan commented on 2023-02-25 05:55:16
* tmbdev commented on 2023-03-18 21:44:12
* cirquit commented on 2023-04-03 21:45:06

---
### Issue #211: ShardWriter works only with local paths

* **Created by:** artemkurylev
* **Created on:** 2022-09-13 16:22:15

#### Summary

The issue highlights that ShardWriter is currently unable to write to remote
URLs, despite utilizing TarWriter         internally, which suggests a potential
for this capability. The user suggests that enabling this feature would be
beneficial, although there might be logical reasons for its current limitation.
Implementing this feature could        enhance the functionality of ShardWriter
by allowing remote URL writing.

#### Comment Summary (2 comments)

Error handling for remote writing can be complex, so ShardWriter simplifies the
process by writing to the local disk   and providing a hook for data upload. The
provided Python code demonstrates using a post hook to upload files to
Google Cloud Storage and then delete them locally. Additionally, there's a need
to improve the documentation and       function comments, which seems to address
a workaround for issue #216.

#### Comment Details

* tmbdev commented on 2022-09-15 07:07:58
* justinpinkney commented on 2023-06-15 11:25:44

---
### Issue #210: Collate function confusion

* **Created by:** KayARS
* **Created on:** 2022-09-09 10:20:18

#### Summary

The issue revolves around a misunderstanding of the default_collation_fn
function's expected input format. The         function's comment suggests it
takes a collection of dictionaries to create a batch, but the code asserts that
the     input should be a list or tuple, leading to confusion. The user is
loading multiple numpy arrays stored in a           dictionary and is unsure if
they need to convert these to tuples, which would be cumbersome. Clarification
on the      expected input format and whether conversion is necessary is needed.

#### Comment Summary (3 comments)

The current collation function doesn't support dictionaries, but an enhancement
is planned. Starting with PyTorch      1.11, torch.utils.data.default_collate
can be used to address this issue by allowing custom collate functions for
batched data. Additionally, the unbatch function in the WebDataset library also
lacks support for dicts, and a pull    request to enable this feature is being
considered.

#### Comment Details

* tmbdev commented on 2022-09-15 07:09:04
* jenuk commented on 2022-10-21 14:24:35
* tmbdev commented on 2024-05-15 00:25:54

---
### Issue #201: How to best subsample a dataset?

* **Created by:** ThaddaeusWiedemer
* **Created on:** 2022-08-03 15:24:42

#### Summary

The issue involves subsampling a large dataset (LAION 400M) in the web dataset
format by filtering samples based on a  metadata list, which significantly
reduces iteration speed from 1.26 it/s to 4.52 it/s. The user seeks an efficient
method to construct a subsampled dataset that maintains iteration speed and
allows saving the dataset iterator for     future use without duplicating data
due to limited disk space. Suggestions for optimizing the filtering process or
alternative approaches to achieve these goals are requested.

#### Comment Summary (1 comments)

To optimize performance, consider doing select(...) before decode and data
augmentation, as image decompression and    augmentation are typically the
slowest parts of the pipeline. If retaining a small sample percentage, pre-
generate the subset using a WebDataset/TarWriter pipeline with parallelization
tools like ray, or use tarp proc ... | tarp split    .... For dynamic selection,
split datasets into shards based on categories to avoid the slowdowns associated
with      random file accesses in formats like zip or lmdb, which are necessary
trade-offs for achieving high-performance I/O.

#### Comment Details

* tmbdev commented on 2022-08-12 18:12:44

---
### Issue #199: Why is `wds.Processor` not included in the `v2` or `main` branch.

* **Created by:** abhayraw1
* **Created on:** 2022-07-25 10:51:27

#### Summary

The wds.Processor class, previously documented for adding a preprocessing
pipeline to data, is missing from the main   branch, raising questions about its
intentional removal and the lack of guidance on alternative methods for
preprocessing. Users need a workaround to access all sample information for
preprocessing. Clarification on this       change and updated documentation
would be beneficial.

#### Comment Summary (2 comments)

The documentation needs updating due to changes in the pipeline architecture to
align with torchdata, which led to the exclusion of certain features. When using
map(f), the function f receives the complete sample, allowing access to all
information. Additionally, pipeline stages can be written as callables, enabling
custom processing of datasets using   functions like process in the example
provided.

#### Comment Details

* dandelin commented on 2022-08-12 05:00:23
* tmbdev commented on 2022-08-12 18:21:51

---
### Issue #192: Require support for the realse notes

* **Created by:** slyviacassell
* **Created on:** 2022-06-27 08:37:10

#### Summary

The absence of release notes for each version makes it challenging for users and
developers to track changes and       understand the differences between
versions, potentially leading to confusion and inefficiencies in usage and
development. Providing detailed release notes would enhance transparency and
improve user experience by clearly        communicating updates, bug fixes, and
new features. This issue highlights the need for better documentation practices
to support effective version management. #enhancement #usability

#### Comment Summary (1 comments)

The plan is to create a release notes file, ensuring that the update primarily
focuses on bug fixes without            introducing any incompatible changes.

#### Comment Details

* tmbdev commented on 2022-07-18 17:51:06

---
### Issue #190: multiple seeding / shuffle problems

* **Created by:** rwightman
* **Created on:** 2022-06-11 00:20:05

#### Summary

The issue involves challenges in achieving deterministic shuffling during
distributed training due to the lack of a    reliable epoch counter shared
across the training loop and pipeline, which can lead to out-of-sync epochs
among        workers. Additionally, the use of random.Random() with a tuple for
seeding introduces inconsistencies across different Python runtimes, as the hash
used for the seed can vary between interpreter invocations and machines. These
problems   highlight significant concerns in maintaining consistency and
reliability in distributed training environments.

#### Comment Summary (2 comments)

The comments discuss two main points: (1) achieving exact shuffling across
distributed nodes requires careful epoch    setup or external synchronization,
with Redis being a past solution, though future improvements from torchdata are
anticipated; (2) training results in ResampledShards() due to different seeding
of distributed workers, confirmed by   testing, and while the PYTHONHASHSEED
environment variable can be used, code changes are planned to eliminate this
necessity.

#### Comment Details

* rwightman commented on 2022-06-11 16:12:49
* tmbdev commented on 2022-07-18 17:47:22

---
### Issue #183: Request to publish PyPI release with latest version

* **Created by:** reuben
* **Created on:** 2022-05-30 14:14:54

#### Summary

The desired commit from the WebDataset repository is not available in any
current PyPI release, as the PyPI versions   are lagging behind the latest
updates in the repository. This discrepancy prevents users from easily accessing
the     latest features and fixes through standard package management. Consider
manually installing the package from the       repository or requesting an
updated release on PyPI. #enhancement #error

#### Comments

No comments on this issue.

---
### Issue #181: Reading a wds with .npz files leads to pickle " Object arrays cannot be loaded when allow_pickle=False"

* **Created by:** charlescurt
* **Created on:** 2022-05-19 00:52:06

#### Summary

The issue involves implementing a dataloader for a tar file stored in S3, which
contains .npz arrays and .cls files.   While the .cls files load correctly, the
.npz arrays cannot be transferred to the specified device, causing a problem  in
the data processing pipeline. Assistance is requested to resolve the issue of
transferring .npz arrays to the       device.

#### Comment Summary (1 comments)

The suggestion is to use the s5cmd cat s3://test.tar command instead of s5cmd
get for accessing the contents of the    test.tar file stored in an S3 bucket.
This approach might be more efficient or suitable for the user's needs. The
context implies a preference for directly streaming or viewing the file contents
rather than downloading it.

#### Comment Details

* brtasavpatel commented on 2022-06-30 20:58:25

---
### Issue #174: Webdataset fails on importing torch class

* **Created by:** f90
* **Created on:** 2022-04-26 16:00:02

#### Summary

The issue arises because the webdataset package version 0.2.5 does not list
torch as a dependency, leading to a        ModuleNotFoundError when attempting
to import TarWriter in a non-Torch project. This error occurs because webdataset
internally imports IterableDataset from torch, which is not installed by
default. To resolve this, torch should be     added as a dependency in
webdataset's package configuration, or the import statement should be modified
to avoid       requiring torch when it's not necessary.

#### Comment Summary (2 comments)

The previous version of webdataset (<= 0.2) did not have a dependency on torch,
suggesting the possibility of removing this dependency. The task has been
completed, but there is a need to add integration tests for non-PyTorch and
TensorFlow environments.

#### Comment Details

* rom1504 commented on 2022-04-26 22:31:20
* tmbdev commented on 2022-04-29 16:04:53

---
### Issue #173: should tariterators group_by_keys have a handler

* **Created by:** rwightman
* **Created on:** 2022-04-23 15:00:06

#### Summary

The group_by_keys function currently throws an exception when encountering
duplicate keys, which is problematic for    datasets with non-unique prefixes
across tar files. The user suggests adding an option to handle exceptions within
the loop, allowing the process to continue by ignoring conflicting entries. This
enhancement would improve robustness when dealing with unordered or externally
controlled datasets.

#### Comments

No comments on this issue.

---
### Issue #172: detshuffle epoch count the same across epochs

* **Created by:** rwightman
* **Created on:** 2022-04-22 21:32:02

#### Summary

The issue involves a pipeline similar to the OpenImages notebook, with
additional shuffle and worker/node splitters,   where accuracy has worsened
after transitioning from older WebDataset 0.1 code. The problem appears to be
improper     shuffling, as shards are not shuffled across nodes and workers as
expected, with the epoch count in detshuffle not     incrementing correctly. The
root cause is unclear, but it seems the pipeline might be copied each epoch,
preventing    the detshuffle instance from initializing properly after the first
epoch, leading to potential misuse of the API.

#### Comment Summary (3 comments)

The comments discuss issues with the default DataLoader setup in PyTorch,
particularly when persistent_workers=False,  which causes new workers to be
created each epoch, potentially affecting state persistence. Setting
persistent_workers=True helps maintain state across epochs. The interactions
between worker processes and internal     state are complex, but torchdata is
expected to simplify this. The recommended approach is to use resampling for its
simplicity and reliability. Additionally, there's a suggestion to support the
use of WDS_EPOCH in detshuffle to ensure compatibility, with documentation and
code comments to address potential issues when persistent_workers=False.

#### Comment Details

* rwightman commented on 2022-04-22 21:54:44
* tmbdev commented on 2022-04-25 07:24:49
* rwightman commented on 2022-04-25 15:32:47

---
### Issue #170: Unintended behaviour when using glob pattern as shard URL

* **Created by:** f90
* **Created on:** 2022-04-09 14:18:38

#### Summary

The issue involves unexpected training behavior when using WebDataset (wds) with
Google Cloud Storage (GCS) paths      containing wildcard asterisks. The user
observed that the model's loss decreased linearly, indicating limited data
variability, because WebDataset treated the wildcard path as a single shard
rather than expanding it to multiple       files. This misuse of WebDataset, due
to misunderstanding of its path handling, suggests a need for improved error
handling or built-in globbing to prevent similar issues.

#### Comment Summary (1 comments)

The lack of globbing support is due to the absence of a consistent API across
different object stores, limiting        support to reading the contents of
individual objects. To implement globbing, a workaround involves using a command
like gsutil ls to list matching files and then processing them with a tool like
wds.WebDataset. Additional             documentation and error checks are needed
to improve this approach.

#### Comment Details

* tmbdev commented on 2022-04-19 17:27:44

---
### Issue #164: bug: wds append the splitter twice

* **Created by:** maxwellzh
* **Created on:** 2022-03-11 07:16:48

#### Summary

The issue arises when the input urls is a string pattern like
data-{000..123).tar, causing the webdataset library to   append both
nodesplitter and workersplitter twice. This results in the yielded data being
less than expected,          potentially due to incorrect handling of the URL
pattern expansion or splitting logic. The problem may affect data     processing
efficiency and accuracy, requiring a review of the splitting mechanism to ensure
correct data distribution.

#### Comment Summary (1 comments)

The issue was identified and fixed, with appreciation expressed for catching it.
Additionally, there is a pending task to add a test case.

#### Comment Details

* tmbdev commented on 2022-03-17 00:19:23

---
### Issue #163: Compose implementation outdated

* **Created by:** LexBosch
* **Created on:** 2022-03-08 14:02:45

#### Summary

The issue arises when attempting to use the compose method on a WebDataset
object, resulting in an error related to    the compose function in the
webdataset library. This suggests a possible misunderstanding of the
documentation or an   outdated implementation, as the error indicates a problem
with the compose method's usage or its compatibility with    the current library
version. Reviewing the library's documentation for updates or changes in the API
might resolve the issue.

#### Comment Summary (5 comments)

The recommended interface for working with datasets is the wds.DataPipeline
interface, which is noted for its ease of  use and extensibility. The
.compose(...) interface has undergone changes, notably the removal of source_,
but          conversion is straightforward, as seen in compat.py. Users are
seeking examples and documentation updates for using    SampleEqually with the
new API, and a sample implementation has been provided, highlighting the need
for official      documentation to reflect these changes.

#### Comment Details

* tmbdev commented on 2022-03-17 00:18:10
* cfanyyx commented on 2023-02-26 15:46:53
* cyrildiagne commented on 2023-03-22 18:25:38
* dome272 commented on 2023-04-02 02:49:08
* samar-khanna commented on 2023-05-05 05:55:28

---
### Issue #159: [Documentation request] Batching clarification

* **Created by:** austinmw
* **Created on:** 2022-02-25 17:04:21

#### Summary

The issue revolves around the confusion in the "complete pipeline" example in
the README, specifically regarding the   sequence of batching operations:
dataset.batched(16), wds.WebLoader(..., batch_size=8), .unbatched(), and
.batched(12). The user is unclear about the rationale behind batching,
unbatching, and re-batching, suspecting it      might be related to improved
shuffling, and seeks guidance on choosing the initial batch size. Additionally,
there's   uncertainty about initializing the dataloader with ResampledShards for
multinode batching and calculating epochs to    ensure each image is trained on
once per epoch.

#### Comment Summary (1 comments)

The comments express confusion about the "complete pipeline" example in the new
readme, specifically regarding the     sequence of batching and unbatching
operations and their impact on shuffling. There are questions about the
rationale  behind the initial batch size choice and whether the same dataloader
initialization should be used for multinode       batching with ResampledShards.
Additionally, clarification is sought on how to set with_epoch() to ensure each
image   is trained on roughly once per epoch, given a specific dataset and
hardware configuration. The commenters also request an explanation of why
loader.num_workers * .with_epoch(N) is associated with "nsamples" when it seems
to relate more   to iterations or training steps per epoch.

#### Comment Details

* dxli94 commented on 2022-04-08 10:18:55

---
### Issue #158: resampled=True and with_length did not work for "exact epoch" when using shuffle

* **Created by:** kamo262
* **Created on:** 2022-02-25 16:43:35

#### Summary

The issue involves unexpected behavior when using the resampled option with the
WebDataset library for "exact epoch"   processing. The user expected the length
of keys to be 4096, matching the number of samples in 000.tar, but observed a
smaller count (e.g., 3723) when using .shuffle(4096). Removing the .shuffle
operation resulted in the expected length, suggesting a misunderstanding or
potential issue with how shuffling interacts with resampling in the dataset
processing.

#### Comment Summary (3 comments)

The comments discuss a potential bug related to the use of resampled/with_epoch
in a data processing pipeline, which   should result in epochs of exact length
but seems to be causing issues. A user provided a 000.tar file, generated with a
script, to help diagnose the problem, noting that there are overlapping keys
within an epoch, despite the total      number of keys being 4096. Another user
reports a similar issue where the number of batches exceeds expectations,
questioning if their calculation of epoch_size is incorrect, and provides a code
snippet for context.

#### Comment Details

* tmbdev commented on 2022-02-25 17:29:43
* kamo262 commented on 2022-02-26 01:00:16
* rishikanthc commented on 2022-05-23 23:01:09

---
### Issue #157: The group_by_keys function in tariterators.py

* **Created by:** EEthinker
* **Created on:** 2022-02-23 23:39:02

#### Summary

The issue arises in the group_by_keys function within tariterators.py, where a
ValueError is raised if a suffix is     found in current_sample, indicating a
duplicate file name in a tar file. The error is inconsistent, occurring only
when the suffix is actively added as a key to the dictionary, which naturally
results in its presence in               current_sample. The logic behind
treating this as an error is unclear, and guidance is needed to understand and
resolve this behavior.

#### Comment Summary (10 comments)

The issue discussed involves errors in WebDataset when tar files contain
duplicate file names, which are not allowed   as WebDataset requires unique file
names across the entire dataset. This error can occur when tar files are
shuffled   and duplicates end up next to each other. Solutions include using
explicit pipeline construction, renaming files, or   ensuring each tar file
contains multiple samples. A proposed change (#327) aims to address this issue,
and a potential fix involves merging files into larger tar files or adjusting
the resampling settings. An explicit checker for         duplicate file names is
suggested for future improvements.

#### Comment Details

* tmbdev commented on 2022-02-25 17:26:39
* EEthinker commented on 2022-02-25 17:28:43
* tmbdev commented on 2022-02-28 03:48:21
* parkitny commented on 2022-12-24 13:13:22
* tmbdev commented on 2023-03-18 21:50:35
* FuchenUSTC commented on 2023-04-23 06:01:24
* tmbdev commented on 2023-04-30 02:59:07
* piEsposito commented on 2024-01-29 19:57:11
* jpc commented on 2024-06-09 19:44:14
* weixi-feng commented on 2024-09-16 22:55:56

---
### Issue #156: Ability to use globbing

* **Created by:** nils-werner
* **Created on:** 2022-02-22 15:26:06

#### Summary

The issue involves the need for dynamic dataset access using globbing patterns
like dataset-*.tar instead of           specifying each shard explicitly with
dataset-{000000..000010}.tar. This would simplify the process by allowing users
to access all available shards without prior knowledge of their exact names,
enhancing flexibility and efficiency in   data handling. Implementing globbing
would streamline dataset management and improve performance by reducing manual
input requirements.

#### Comment Summary (1 comments)

The comments suggest that instead of passing a string, you can pass a list to
the WebDataset by using shards =         list(glob(pattern)). Additionally,
there's a note to add support for iterables in addition to lists in the future.

#### Comment Details

* tmbdev commented on 2022-02-25 17:23:11

---
### Issue #155: Documentation website is 404

* **Created by:** lozhn
* **Created on:** 2022-02-18 15:24:13

#### Summary

The documentation for WebDataset, previously accessible online, is currently
returning a 404 error, indicating that    the page is not found. This issue
could impact users' ability to access important information and guidance on
using    the WebDataset tool. It is essential to address this to ensure users
can continue to utilize the documentation         effectively. #error #serious

#### Comment Summary (5 comments)

The documentation for the project is outdated, and there are plans to improve it
by adding more notebooks in the docs  subdirectory for version 0.2.x. Meanwhile,
users can access the old documentation at the provided link or by checking  out
to version 1 and opening index.html in a browser. There is some uncertainty
about the project's current activity   and the location of the latest
documentation.

#### Comment Details

* tmbdev commented on 2022-02-19 00:01:50
* nils-werner commented on 2022-02-24 15:45:41
* lozhn commented on 2022-02-25 07:12:32
* rom1504 commented on 2022-02-27 11:50:41
* tianzhipeng-git commented on 2025-01-07 13:53:39

---
### Issue #136: Wrong params in the docstring of PytorchShardList

* **Created by:** mrsalehi
* **Created on:** 2022-01-15 04:20:33

#### Summary

The PytorchShardList class in webdataset/webdataset/shardlists.py has an
incomplete and outdated docstring for its     constructor, missing parameters
like epoch_shuffle and including deprecated ones. This discrepancy can lead to
confusion for developers relying on the documentation for implementation
details. Updating the docstring to accurately reflect the current parameters is
necessary for clarity and usability.

#### Comments

No comments on this issue.

---
### Issue #133: Add verbose flag for ShardWriter constructor

* **Created by:** Midren
* **Created on:** 2021-12-21 07:49:16

#### Summary

The ShardWriter class uses the self.verbose variable to determine whether to log
the creation of new tar files, but    currently lacks a mechanism to set this
variable through its constructor. This limitation restricts the flexibility of
the class, as users cannot control logging behavior during instantiation.
Enhancing the constructor to accept a        verbosity parameter would improve
usability and configurability. #enhancement

#### Comments

No comments on this issue.

---
### Issue #129: [FEATURE] Add NamedTuple or Dataclass conversion

* **Created by:** cwerner
* **Created on:** 2021-12-08 15:09:02

#### Summary

The issue suggests adding conversion functions to transform data into NamedTuple
or Dataclass types, similar to        existing to_tuple and to_dict functions.
This enhancement would provide more structured data handling options,
potentially improving code readability and maintainability. Implementing these
functions could streamline data         manipulation processes.

#### Comment Summary (1 comments)

The comments collectively express agreement and appreciation for a suggestion
that was made, indicating it is          well-received and considered
beneficial.

#### Comment Details

* tmbdev commented on 2021-12-09 17:50:31

---
### Issue #128: How to retrieve pairwise samples?

* **Created by:** nils-werner
* **Created on:** 2021-12-07 12:52:44

#### Summary

The issue arises when attempting to retrieve pairwise overlapping samples using
webdataset and                         more_itertools.pairwise, resulting in an
error because the samples are tuples of two dictionaries instead of a single
dictionary. The proposed solution involves repacking the tuples into a single
dictionary and merging arrays, but the   process feels overly complex. A cleaner
approach might involve creating a custom transformation function that directly
handles the tuple structure, simplifying the pipeline and reducing the need for
multiple transformation steps.

#### Comment Summary (2 comments)

The comments discuss the process of decoding data using dictionaries, suggesting
that decoding should occur before     converting data to tuples and pairing
them. This approach is recommended to avoid doubling the CPU cost and because
pairing before decoding offers no memory benefits. The conversation also
highlights the importance of shuffling data   before pairing to achieve varied
results, although one user prefers not to shuffle first due to the need for
context   between contiguous audio samples. The use of functions with .map or
.then is advised for simplifying complex           operations.

#### Comment Details

* tmbdev commented on 2021-12-09 17:27:33
* nils-werner commented on 2021-12-09 21:34:00

---
### Issue #119: Add changelog and/ or release notes

* **Created by:** cwerner
* **Created on:** 2021-10-29 10:44:21

#### Summary

A user of the webdataset library appreciates its utility in simplifying dataset
handling for a desktop deep learning   project but suggests adding a Release
Notes or ChangeLog file to easily track version changes and new features. The
user has pinned the project to version 0.1.62 due to an API change in later
versions that breaks their existing code,  which they plan to adapt soon. This
enhancement would aid in deciding when to upgrade versions without disrupting
current workflows.

#### Comment Summary (1 comments)

Improving release management, change logs, and version management is essential,
and efforts will be made in that       direction. For now, pinning to a specific
version is advisable, and installing from PyPI is recommended as packaged
releases are made when the software is stable.

#### Comment Details

* tmbdev commented on 2021-11-02 08:16:13

---
### Issue #117: cache_size is being ignored

* **Created by:** asavpatel
* **Created on:** 2021-10-20 19:51:50

#### Summary

The issue involves a malfunction in the caching mechanism for shards downloaded
from S3, where specifying cache_size   fails to limit the cache growth in the
designated cache_dir. Despite accepting cache_size as an input parameter, the
code does not utilize it effectively, leading to uncontrolled cache expansion.
This could result in storage            constraints being exceeded during
training.

#### Comments

No comments on this issue.

---
### Issue #112: gsutil cat intermittently fails

* **Created by:** egafni
* **Created on:** 2021-09-24 06:29:38

#### Summary

The error message indicates a failure when using gsutil cat with webdataset,
despite having num_workers=4, which       should prevent excessive requests. The
error suggests an issue with the subprocess execution or resource limits,
possibly related to the gsutil command's parallel settings. Improving the error
message for clarity and checking       resource limits or subprocess handling
might help resolve the issue.

#### Comment Summary (35 comments)

The discussion revolves around issues with using gsutil for streaming data from
Google Cloud Storage (GCS) during      training, particularly with PyTorch's
WebDataset. Users experience frequent connection errors and process leaks when
using gsutil cat or gsutil cp for large datasets, leading to training
interruptions. Some have tried alternatives like using the Google Cloud Storage
API directly, which reduces errors but impacts performance due to blocking
network      reads. Suggestions include using caching, pre-copying data, or
switching to HTTP access via curl. The conversation     also touches on the
challenges of maintaining parallelism and reliability when accessing GCS from
outside Google's     infrastructure.

#### Comment Details

* tmbdev commented on 2021-10-15 08:55:28
* asavpatel commented on 2021-11-14 21:47:05
* tmbdev commented on 2021-11-15 17:11:06
* asavpatel commented on 2021-11-16 18:55:01
* rwightman commented on 2022-03-12 21:11:28
* rom1504 commented on 2022-03-14 01:05:10
* rwightman commented on 2022-03-14 05:34:35
* rwightman commented on 2022-03-16 18:01:54
* rom1504 commented on 2022-03-16 18:54:10
* rom1504 commented on 2022-03-16 18:55:14
* rom1504 commented on 2022-03-16 18:58:50
* rom1504 commented on 2022-03-16 19:03:06
* rwightman commented on 2022-03-16 19:05:14
* rwightman commented on 2022-03-16 19:14:06
* rom1504 commented on 2022-03-16 19:41:02
* rwightman commented on 2022-03-16 22:29:27
* rwightman commented on 2022-03-17 00:54:38
* rwightman commented on 2022-03-17 20:45:41
* rwightman commented on 2022-03-18 17:44:04
* rwightman commented on 2022-03-23 16:38:58
* rom1504 commented on 2022-03-23 19:29:49
* tmbdev commented on 2022-03-25 23:02:09
* rwightman commented on 2022-03-26 00:32:07
* tmbdev commented on 2022-03-28 15:21:27
* calclavia commented on 2022-04-16 05:09:50
* tmbdev commented on 2022-04-25 07:19:22
* calclavia commented on 2022-04-27 05:53:16
* tmbdev commented on 2022-04-29 16:45:04
* calclavia commented on 2022-05-04 06:22:19
* tmbdev commented on 2022-05-05 05:35:17
* kylesargent commented on 2023-07-15 17:42:13
* tmbdev commented on 2023-07-27 08:47:26
* adrienpl commented on 2024-08-01 07:56:13
* tmbdev commented on 2024-09-24 19:50:33
* rwightman commented on 2024-09-24 20:16:54

---
### Issue #101: Cross-validation with Webdataset?

* **Created by:** BryanWBear
* **Created on:** 2021-08-17 01:48:15

#### Summary

The issue concerns the lack of examples or guidance on implementing K-fold
cross-validation using a webdataset, which  is a method for splitting data into
K subsets to train and validate machine learning models. This technique is
crucial for assessing model performance and ensuring robust evaluation, but the
absence of clear examples makes it challenging for practitioners to apply it
effectively with webdatasets. Providing detailed documentation or examples would
enhance usability and facilitate better model validation practices.

#### Comment Summary (4 comments)

The comments discuss a method for cross-validation using integer hashing of
sample keys to partition data into         training and validation sets. The
approach involves computing a hash from the sample key and using it to determine
which samples to drop or keep during training and evaluation. Additionally,
there's a mention of a hardcoded           dataset_size in a WebDataset example,
with a suggestion to update it, as Lightning requires a length for progress
bars. The user confirms the method answers their question and expresses
gratitude for the help.

#### Comment Details

* tmbdev commented on 2021-08-19 06:35:51
* BryanWBear commented on 2021-08-19 21:58:36
* austinmw commented on 2022-01-19 19:53:01
* tmbdev commented on 2024-05-18 18:46:09

---
### Issue #95: Webdataset support for Tensorflow

* **Created by:** retazo0018
* **Created on:** 2021-07-26 11:39:52

#### Summary

The user is experiencing two issues with using Webdataset for TensorFlow. First,
on Windows OS, they encounter a       "zmq.error.ZMQError: Protocol not
supported" error, which does not occur on Linux, possibly due to the use of the
IPC  protocol for socket connections. Second, on Linux in a multi-GPU
environment, the program freezes or fails to          terminate when ZMQ workers
are set to more than one, indicating potential issues with parallel processing
or resource  management.

#### Comment Summary (4 comments)

The comments discuss issues with UNIX IPC not being available on Windows and
suggest modifying the protocol in the     source code, possibly replacing IPC
with TCP to address program freezes. Debugging tools like "pyrasite" or
"pdb-clone" are recommended to identify where the program hangs, which may be
related to IPC limitations.              Additionally, there is excitement about
the development of native Tensorflow support for WebDataset, with a request
for a timeline, and a suggestion that adding support for Jax and Tensorflow
could enhance WebDataset adoption.

#### Comment Details

* tmbdev commented on 2021-07-27 16:49:59
* retazo0018 commented on 2021-07-29 04:47:51
* grofte commented on 2021-12-07 15:09:21
* rom1504 commented on 2022-03-14 01:09:40

---
### Issue #90: Processes not cleaned up?

* **Created by:** thecooltechguy
* **Created on:** 2021-07-14 17:56:58

#### Summary

The issue involves a failure in a multi-GPU PyTorch training job using
Webdataset to stream data from S3, where an     error occurs at the end of the
second epoch due to a failure to create new OS threads. This is suspected to be
caused  by Webdataset not properly terminating s5cmd processes after they finish
piping data, leading to an accumulation of    open processes that exceed the
system's limits. The problem does not occur when using local file inputs,
suggesting a  potential bug in Webdataset's process management for S3 data
inputs.

#### Comment Summary (16 comments)

The comments discuss issues with WebDataset and process management during
distributed training. WebDataset doesn't     inherently kill processes; it
creates a pipe for commands, and processes may linger if epochs are interrupted.
A       Golang error suggests resource shortages, possibly memory, despite ample
resources on EKS nodes. The user suspects     Lightning's limit_train_batches
might cause process accumulation. They explore WebDataset's methods for
equalizing     batch distribution and note that caching might be causing
excessive process creation. A workaround involves modifying  WebDataset's
caching logic to prevent unnecessary process launches. Persistent workers in
PyTorch DataLoader might     also contribute to the issue.

#### Comment Details

* tmbdev commented on 2021-07-24 17:41:58
* thecooltechguy commented on 2021-07-25 06:15:47
* thecooltechguy commented on 2021-07-25 06:28:07
* tmbdev commented on 2021-07-27 16:39:20
* tmbdev commented on 2021-07-27 16:43:38
* thecooltechguy commented on 2021-07-27 19:41:16
* thecooltechguy commented on 2021-07-27 20:32:33
* thecooltechguy commented on 2021-07-28 00:16:50
* thecooltechguy commented on 2021-07-28 11:04:41
* tmbdev commented on 2021-08-02 16:47:12
* EEthinker commented on 2022-02-24 03:07:13
* EEthinker commented on 2022-02-24 03:08:08
* yao-eastside commented on 2022-07-09 23:54:32
* yao-eastside commented on 2022-07-10 00:17:10
* yao-eastside commented on 2022-07-10 00:43:58
* bryant1410 commented on 2023-08-04 15:19:53

---
### Issue #89: Is there a concrete example of using tensorcom and webdataset?

* **Created by:** naba89
* **Created on:** 2021-07-14 14:29:45

#### Summary

The issue involves creating a distributed preprocessing pipeline using
Tensorcom, where data is sent in larger batch   sizes to optimize network
bandwidth. The challenge arises in converting the Tensorcom Connection object to
a webloader object for unbatching, shuffling, and re-batching with a different
batch size, as well as setting up the Tensorcom URL to be recognized by the
webdataset interface. The user seeks examples or guidance on implementing a
Tensorcom backend  webdataset client capable of rebatching.

#### Comment Summary (1 comments)

The code snippet demonstrates how to transform any object with an __iter__
method into a Processor by using a          Connection or TensorcomDataset
object. It shows the creation of a Processor instance, src, which is then
manipulated   through a series of method calls: unbatched(), shuffle(1000), and
batched(300). This sequence effectively processes    the data by unbatching,
shuffling with a buffer size of 1000, and rebatching into groups of 300.

#### Comment Details

* tmbdev commented on 2021-07-24 17:15:05

---
### Issue #88: local files gopen

* **Created by:** SeppeLampe
* **Created on:** 2021-07-14 11:49:25

#### Summary

The issue involves a ValueError when iterating through a locally-stored dataset
due to urlparse not returning a        .scheme of 'file' for local addresses,
but rather the disk name, which is unrecognized by the gopen_schemes
dictionary. This results in a gopen_error instead of using gopen_file to open
the URL. A temporary solution has been   shared on StackOverflow, but a more
stable fix is sought.

#### Comment Summary (1 comments)

The comments discuss using file path patterns and symlinks for managing data
locations across different systems. One   suggestion is using a pattern like
file:d:/some/path/train-{000000..000999}.tar for file naming. Another approach
is   using symlinks, even on Windows, to maintain consistent data references
like "./data" across different machines,       enhancing convenience and
flexibility.

#### Comment Details

* tmbdev commented on 2021-07-24 17:07:56

---
### Issue #84: Incompletely read temporary cache files are not discarded

* **Created by:** thecooltechguy
* **Created on:** 2021-06-17 17:43:34

#### Summary

The issue involves the Webdataset library creating multiple temporary files for
shard files when using a large dataset with multiple workers and GPUs. This
occurs due to intermittent network issues, such as "broken pipe" errors, when
reading shard files from S3, leading to incomplete reads and repeated caching
attempts. The problem seems to stem from the absence of a cleanup mechanism for
incomplete temporary shard files in the code, specifically lacking an
os.remove(self.tempname) call when complete=False.

#### Comment Summary (2 comments)

The comments suggest a potential typo in line 69 of the code snippet, where
self.tempsuffix is used without being      defined in the class. The user plans
to replace it with os.remove(self.tempname) as a fix and intends to add a test
case to verify the solution.

#### Comment Details

* thecooltechguy commented on 2021-06-17 17:50:39
* tmbdev commented on 2021-06-18 03:31:24

---
### Issue #83: Included useless stdout in gopen when use s3

* **Created by:** 6gsn
* **Created on:** 2021-06-13 11:44:29

#### Summary

The issue involves streaming images from S3 buckets using PyTorch, where the
images appear broken due to a ReadError   caused by extraneous information
included in the stdout during streaming. This was traced back to the s3cmd
command,   which outputs status information that interferes with the data
stream. The problem was resolved by using the -q        (quiet) option in s3cmd
to suppress the extraneous output. It is recommended to filter unnecessary
stdout or inform    users about this potential issue.

#### Comment Summary (1 comments)

WebDataset cannot resolve issues caused by s3cmd mixing outputs, as it cannot
separate them once combined. It's        recommended that tools like s3cmd avoid
printing diagnostics to stdout when it's used for data output. Filing a bug
report with s3cmd might be a good course of action.

#### Comment Details

* tmbdev commented on 2021-06-18 03:34:33

---
### Issue #79: Guidance on using Webdataset for small embeddings

* **Created by:** tcwalther
* **Created on:** 2021-06-05 11:11:33

#### Summary

The issue involves using WebDataset to store small embeddings and output
classes, which results in significant space   overhead and poor performance
during data reading. The user prefers WebDataset's format but finds it
inefficient for   small data, as each embedding is serialized individually,
leading to these inefficiencies. A potential solution could  involve batching
multiple samples into a single file to reduce overhead and improve read
performance, or exploring     alternative storage formats that better handle
small data sizes.

#### Comment Summary (2 comments)

Your current records have a 25% space and performance overhead, but you can read
them at 30,000 per second, which      might be sufficient without further
optimization. To improve efficiency, consider using .npy or .ten formats,
storing  data in 8-bit integer or float16, batching data, and utilizing caching.
For batched data, use WebDataset and WebLoader to manage data efficiently, and
consider caching in memory or on disk. Reducing DataLoader IPC overhead can also
help, especially for small records. JSON is suitable for mixed data types, and
WebDataset is versatile for both PyTorch and  Keras.

#### Comment Details

* tmbdev commented on 2021-06-06 15:58:41
* grofte commented on 2021-11-09 13:17:35

---
### Issue #70: Need guideline in subclassing, etc

* **Created by:** rongcuid
* **Created on:** 2021-05-16 02:36:20

#### Summary

The current documentation makes it challenging to navigate the code base for
constructing custom datasets,             particularly regarding the handling of
worker info when creating an IterableDataset that feeds a WDS iterator to
DataLoader. Additionally, there is uncertainty about controlling sampling weight
in shards for data augmentation and   managing biased datasets, which may
require experimenting with a custom IterableDataset class to prevent duplicate
data. Improved guidance on these aspects would enhance usability and efficiency.

#### Comment Summary (2 comments)

The experiment indicates that using WebDataset (WDS) iterators directly
eliminates the need to manually handle worker  information, enhancing
performance compared to other methods. However, the documentation for WDS is
lacking and could  benefit from improvements, as tracing the codebase was
necessary to understand its functionality. The WebDataset       function
simplifies creating a dataset pipeline similar to PyTorch's, and while it offers
many options, the essential  setup involves using ShardList with methods like
tar_file_expander and group_by_keys. Custom shard sampling can be
implemented by subclassing IterableDataset and Composable. Future updates may
include JSON-based dataset descriptions  for easier shard-level resampling.

#### Comment Details

* rongcuid commented on 2021-05-16 03:05:10
* tmbdev commented on 2021-05-16 17:15:28

---
### Issue #68: Program stop at some iteration

* **Created by:** XuyangBai
* **Created on:** 2021-05-13 01:08:51

#### Summary

The user is experiencing issues with the webdataset library where their training
program halts at random iterations    despite high GPU utilization, without
progressing further. Additionally, when using a small validation set with only
one shard, the program raises a warning world_size 4 > num_shards 0 and fails to
start validation when training with 4 GPUs. The user is currently bypassing
validation to continue training but still encounters the stopping issue. They
are seeking solutions for these problems while using webdataset version 0.1.62.

#### Comment Summary (6 comments)

In multinode training, ensure the number of workers aligns with the number of
shards, especially for validation sets,  where num_workers should be 0 or 1 if
there's only one shard. For small validation sets, running them on node 0 is
efficient, but if distributed validation is needed, use
WebDataset(...).islice(rank, 999999, world_size) to distribute samples across
nodes. If num_workers=0 causes issues, it might be due to fewer shards than
nodes, which can be         resolved by using smaller shards or slicing by node.
Documentation and additional checks for these scenarios are       suggested
improvements.

#### Comment Details

* tmbdev commented on 2021-05-16 17:00:45
* XuyangBai commented on 2021-05-17 01:56:25
* tmbdev commented on 2021-05-18 17:47:27
* tmbdev commented on 2021-05-18 17:49:40
* XuyangBai commented on 2021-05-24 12:27:32
* tmbdev commented on 2021-05-24 15:30:22

---
### Issue #66: Pytorch Lightning integration

* **Created by:** hal-314
* **Created on:** 2021-05-12 12:38:54

#### Summary

The issue involves the incompatibility of the webdataset with PyTorch Lightning
when using the default PyTorch         DataLoader or WebLoader, particularly
with infinite dataloaders. The current implementation requires a custom length
function to handle iterable datasets, as PyTorch Lightning expects a TypeError
or NotImplementedError when a           dataloader declares a __len__ method but
is an iterable. Proposed solutions include modifying wds.WebLoader to allow
passing arguments to wds.Processor or changing wds.Processor to raise a
TypeError when length=False, though these      changes may not be backward
compatible.

#### Comment Summary (4 comments)

The integration with tmbdev/webdataset-lightning is now functional, even in a
multinode setup. There is a suggestion   to make "DataStream" an explicit class
in the webdataset library to handle details, and the importance of
documentation is emphasized, as understanding the integration process took
considerable effort. Community members are  willing to contribute to
documentation and development, with ongoing work on a central webdataset/AIStore
for a        university lab and experience with PyTorch Lightning.

#### Comment Details

* tmbdev commented on 2021-05-16 16:42:07
* hal-314 commented on 2021-05-17 07:44:43
* tmbdev commented on 2021-05-18 17:51:17
* codestar12 commented on 2021-12-23 15:15:32

---
### Issue #47: Resized Dataset for multiprocessing data loading

* **Created by:** tottenjordan
* **Created on:** 2021-03-15 16:35:00

#### Summary

The issue arises when using the ResizedDataset class in a distributed PyTorch
XLA proof of concept, leading to errors  after approximately 40 training steps.
The error message indicates a mismatch between the reported length of the
IterableDataset and the number of samples fetched, suggesting improper
configuration of the dataset replica across     workers. To resolve this, ensure
that the IterableDataset is correctly configured for multiprocessing, possibly
by     reviewing the PyTorch documentation on IterableDataset and adjusting the
dataset splitting logic accordingly.

#### Comment Summary (9 comments)

The comments discuss the complexities of using PyTorch's DataLoader with large
datasets, particularly in distributed   settings. The use of worker_init_fn is
highlighted for configuring dataset copies differently across workers, but it's
noted that this might not be necessary when using wds.Dataset. The discussion
emphasizes that exact epochs are less    critical for large datasets, and
WebDataset's default behavior of iterating over all shards on all nodes can
simplify  training. For smaller datasets, exact epochs might still be desired,
and the comments suggest avoiding ResizedDataset  and not setting the length on
WebDataset unless done correctly. The conversation also touches on handling
length       mismatches and suggests using .unbatched().shuffle().batched() for
better data shuffling. The user reports issues with BrokenPipe errors and length
mismatches but finds that training proceeds well with adjusted configurations.

#### Comment Details

* tottenjordan commented on 2021-03-15 17:17:37
* tmbdev commented on 2021-03-16 07:21:49
* tottenjordan commented on 2021-03-16 23:09:33
* tottenjordan commented on 2021-03-17 03:02:48
* tmbdev commented on 2021-03-17 08:10:19
* tottenjordan commented on 2021-03-17 16:36:55
* tmbdev commented on 2021-03-17 18:25:36
* tottenjordan commented on 2021-03-18 14:52:52
* tmbdev commented on 2021-03-19 04:51:40

---
### Issue #39: Question about .shuffle function reproducibility

* **Created by:** vincentlux
* **Created on:** 2021-01-26 06:20:44

#### Summary

The issue concerns the need for setting a seed in the shuffle function to ensure
reproducibility of results. This is   important for scenarios where consistent
outcomes are required across different runs of the same code. Implementing a
seed feature would enhance the function's utility by providing deterministic
behavior. #enhancement

#### Comment Summary (2 comments)

In WebDataset, there are two types of shuffles: shard shuffles and sample
shuffles, both of which you can control. By  default, it uses random.shuffle for
shards and random.randint for samples, allowing you to set random.seed for
reproducibility. For more control, override the rng= argument in
iterators.shuffle for samples and the random argument in ShardList/WebDataset
for shards. A test case needs to be added.

#### Comment Details

* tmbdev commented on 2021-02-08 05:13:36
* vincentlux commented on 2021-02-14 06:43:30

---
### Issue #36: Using webdataset with torch-xla in multiprocessing context

* **Created by:** harpone
* **Created on:** 2021-01-15 12:17:52

#### Summary

The gist implements a solution using torch-xla's MpDeviceLoader to distribute
workloads across accelerators and        workers, ensuring that all minibatches
are unique through shard splitting. This approach can enhance the efficiency
and reliability of distributed training by preventing data duplication across
devices. It could serve as a valuable    addition to the official documentation
to guide users in optimizing distributed machine learning tasks.

#### Comment Summary (1 comments)

The comments suggest adding information to the documentation.

#### Comment Details

* tmbdev commented on 2021-02-08 06:36:32

---
### Issue #25: What is the recommended way of using webdataset with pytorch-lightning and ddp?

* **Created by:** isty2e
* **Created on:** 2020-10-13 17:09:45

#### Summary

The user is attempting to build a LightningDataModule using webdataset but is
facing challenges due to inexperience    with IterableDataset. Initially, they
followed an example that led to duplicate batching issues. After researching,
they discovered suggestions to either let the dataset handle batching or use
wds.MultiDataset, but replacing           DataLoader with MultiDataset did not
resolve the issue. They are seeking advice on the recommended method for
distributed data parallel (DDP) processing.

#### Comment Summary (12 comments)

The discussion revolves around the use of the seed_everything function in
PyTorch Lightning with ddp parallel          training, highlighting potential
issues with identical shuffling across processes. While some users prioritize
reproducibility, others focus on efficient data distribution, especially when
using WebDataset. WebDataset's hooks     allow for custom shard selection and
shuffling, addressing distributed training constraints. There's ongoing
collaboration to integrate WebDataset with PyTorch Lightning, despite challenges
with TPUs and torch-xla. PyTorch is   redesigning its I/O pipeline, potentially
incorporating WebDataset's functionality, while users continue to adapt and
share solutions for current frameworks.

#### Comment Details

* harpone commented on 2020-10-14 12:22:42
* isty2e commented on 2020-10-14 14:11:55
* harpone commented on 2020-10-15 07:25:38
* tmbdev commented on 2020-10-20 06:34:30
* tchaton commented on 2020-10-29 20:03:15
* tchaton commented on 2021-01-09 21:40:24
* harpone commented on 2021-01-11 07:50:03
* tmbdev commented on 2021-02-08 06:49:23
* hav4ik commented on 2021-03-18 07:45:56
* harpone commented on 2021-03-18 07:50:26
* hav4ik commented on 2021-03-24 15:25:33
* tmbdev commented on 2021-03-30 09:03:26

---
### Issue #13: How should I write the targets such that it is automatically recognized as a torch.LongTensor instead of Int?

* **Created by:** giovannipcarvalho
* **Created on:** 2020-09-17 16:19:27

#### Summary

The issue concerns the lack of explicit casting to long for targets in the
ImageNet example using nn.CrossEntropyLoss, which typically requires target
tensors to be of type long. The user is unsure how this behavior is achieved
without   access to the dataset and has attempted to use an identity function to
convert inputs to np.int64, but this approach   disrupts automatic batching by
producing a list of scalar tensors. The user seeks clarification on how the
example     handles target types without explicit casting.

#### Comment Summary (1 comments)

When using WebDataset with DataLoader and a positive batch_size, the collate_fn
in DataLoader is responsible for       converting List[int] to tensors. This
conversion is also handled by the collate_fn used with the .batch method in
WebDataset. You can override these defaults by specifying explicit collate_fns
in either DataLoader or WebDataset, and there's a need to check and improve the
consistency of the two default collate_fns.

#### Comment Details

* tmbdev commented on 2020-09-18 09:29:26

---
### Issue #12: Is MultiDataset a complete replacement for torch.utils.data.DataLoader? 

* **Created by:** wongjoel
* **Created on:** 2020-09-14 14:31:34

#### Summary

When using webdataset with PyTorch Lightning, training stalls at epoch 0 if
dataloaders are instances of MultiDataset. Switching to
torch.utils.data.DataLoader resolves the issue, suggesting a compatibility
problem. It is unclear if      MultiDataset is intended to replace DataLoader,
and further investigation is needed to ensure compatibility with       PyTorch
Lightning.

#### Comment Summary (3 comments)

The DataLoader class is complex and has issues, particularly with
IterableDatasets, while MultiDataset is an           experimental alternative
that addresses some of these limitations, such as sample splitting among workers
and dataset  length determination. Although MultiDataset is not a direct
replacement for DataLoader, it offers advantages in        containerized
environments and simpler shard assignment. Additionally, Tensorcom is being
developed as another         alternative, running data loaders as separate
processes and supporting RDMA and GPUdirect for large-scale,             high-
performance training.

#### Comment Details

* tmbdev commented on 2020-09-14 19:55:26
* wongjoel commented on 2020-09-15 06:18:34
* tmbdev commented on 2020-09-15 21:56:59

---
