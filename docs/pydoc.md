
# Module `webdataset.multi`

```
Help on module webdataset.multi in webdataset:

NAME
    webdataset.multi

CLASSES
    torch.utils.data.dataset.IterableDataset(torch.utils.data.dataset.Dataset)
        MultiDataset(torch.utils.data.dataset.IterableDataset, webdataset.dataset.Pipeline)
        MultiDatasetIterator
    webdataset.dataset.Pipeline(builtins.object)
        MultiDataset(torch.utils.data.dataset.IterableDataset, webdataset.dataset.Pipeline)
    
    class MultiDataset(torch.utils.data.dataset.IterableDataset, webdataset.dataset.Pipeline)
     |  MultiDataset(dataset, workers=4, output_size=10000, nominal=None, pin_memory=True)
     |  
     |  An iterable Dataset.
     |  
     |  All datasets that represent an iterable of data samples should subclass it.
     |  Such form of datasets is particularly useful when data come from a stream.
     |  
     |  All subclasses should overwrite :meth:`__iter__`, which would return an
     |  iterator of samples in this dataset.
     |  
     |  When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
     |  item in the dataset will be yielded from the :class:`~torch.utils.data.DataLoader`
     |  iterator. When :attr:`num_workers > 0`, each worker process will have a
     |  different copy of the dataset object, so it is often desired to configure
     |  each copy independently to avoid having duplicate data returned from the
     |  workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
     |  process, returns information about the worker. It can be used in either the
     |  dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
     |  :attr:`worker_init_fn` option to modify each copy's behavior.
     |  
     |  Example 1: splitting workload across all workers in :meth:`__iter__`::
     |  
     |      >>> class MyIterableDataset(torch.utils.data.IterableDataset):
     |      ...     def __init__(self, start, end):
     |      ...         super(MyIterableDataset).__init__()
     |      ...         assert end > start, "this example code only works with end >= start"
     |      ...         self.start = start
     |      ...         self.end = end
     |      ...
     |      ...     def __iter__(self):
     |      ...         worker_info = torch.utils.data.get_worker_info()
     |      ...         if worker_info is None:  # single-process data loading, return the full iterator
     |      ...             iter_start = self.start
     |      ...             iter_end = self.end
     |      ...         else:  # in a worker process
     |      ...             # split workload
     |      ...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
     |      ...             worker_id = worker_info.id
     |      ...             iter_start = self.start + worker_id * per_worker
     |      ...             iter_end = min(iter_start + per_worker, self.end)
     |      ...         return iter(range(iter_start, iter_end))
     |      ...
     |      >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
     |      >>> ds = MyIterableDataset(start=3, end=7)
     |  
     |      >>> # Single-process loading
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
     |      [3, 4, 5, 6]
     |  
     |      >>> # Mult-process loading with two worker processes
     |      >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
     |      [3, 5, 4, 6]
     |  
     |      >>> # With even more workers
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=20)))
     |      [3, 4, 5, 6]
     |  
     |  Example 2: splitting workload across all workers using :attr:`worker_init_fn`::
     |  
     |      >>> class MyIterableDataset(torch.utils.data.IterableDataset):
     |      ...     def __init__(self, start, end):
     |      ...         super(MyIterableDataset).__init__()
     |      ...         assert end > start, "this example code only works with end >= start"
     |      ...         self.start = start
     |      ...         self.end = end
     |      ...
     |      ...     def __iter__(self):
     |      ...         return iter(range(self.start, self.end))
     |      ...
     |      >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
     |      >>> ds = MyIterableDataset(start=3, end=7)
     |  
     |      >>> # Single-process loading
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
     |      [3, 4, 5, 6]
     |      >>>
     |      >>> # Directly doing multi-process loading yields duplicate data
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
     |      [3, 3, 4, 4, 5, 5, 6, 6]
     |  
     |      >>> # Define a `worker_init_fn` that configures each dataset copy differently
     |      >>> def worker_init_fn(worker_id):
     |      ...     worker_info = torch.utils.data.get_worker_info()
     |      ...     dataset = worker_info.dataset  # the dataset copy in this worker process
     |      ...     overall_start = dataset.start
     |      ...     overall_end = dataset.end
     |      ...     # configure the dataset to only process the split workload
     |      ...     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
     |      ...     worker_id = worker_info.id
     |      ...     dataset.start = overall_start + worker_id * per_worker
     |      ...     dataset.end = min(dataset.start + per_worker, overall_end)
     |      ...
     |  
     |      >>> # Mult-process loading with the custom `worker_init_fn`
     |      >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
     |      [3, 5, 4, 6]
     |  
     |      >>> # With even more workers
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))
     |      [3, 4, 5, 6]
     |  
     |  Method resolution order:
     |      MultiDataset
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      webdataset.dataset.Pipeline
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, dataset, workers=4, output_size=10000, nominal=None, pin_memory=True)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |  
     |  __len__(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from torch.utils.data.dataset.IterableDataset:
     |  
     |  __add__(self, other)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from torch.utils.data.dataset.Dataset:
     |  
     |  __getitem__(self, index)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from torch.utils.data.dataset.Dataset:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from webdataset.dataset.Pipeline:
     |  
     |  batched(self, batchsize, partial=True)
     |  
     |  decode(self, decoder='rgb', handler=<function reraise_exception at 0x7fd34997a700>)
     |      Decode the data with the given decoder.
     |  
     |  map(self, f, handler=<function reraise_exception at 0x7fd34997a700>)
     |      Apply function `f` to each sample.
     |  
     |  map_dict(self, handler=<function reraise_exception at 0x7fd34997a700>, **kw)
     |      Transform each sample by applying functions to corresponding fields.
     |  
     |  map_tuple(self, *args, handler=<function reraise_exception at 0x7fd34997a700>)
     |      Apply a list of functions to the tuple.
     |  
     |  pipe(self, stage)
     |      Add a pipline stage (a function taking an iterator and returning another iterator).
     |  
     |  rename(self, handler=<function reraise_exception at 0x7fd34997a700>, **kw)
     |      Rename fields in the sample, dropping all unmatched fields.
     |  
     |  reseed_rng(self)
     |  
     |  select(self, predicate, **kw)
     |      Select samples based on a predicate.
     |  
     |  shuffle(self, size, rng=None, **kw)
     |      Shuffle the data.
     |  
     |  to_tuple(self, *args, handler=<function reraise_exception at 0x7fd34997a700>)
     |      Extract fields from the sample in order and yield tuples.
     |  
     |  unbatched(self)
    
    class MultiDatasetIterator(torch.utils.data.dataset.IterableDataset)
     |  MultiDatasetIterator(dataset=None, workers=4, output_size=100, pin_memory=True)
     |  
     |  An iterable Dataset.
     |  
     |  All datasets that represent an iterable of data samples should subclass it.
     |  Such form of datasets is particularly useful when data come from a stream.
     |  
     |  All subclasses should overwrite :meth:`__iter__`, which would return an
     |  iterator of samples in this dataset.
     |  
     |  When a subclass is used with :class:`~torch.utils.data.DataLoader`, each
     |  item in the dataset will be yielded from the :class:`~torch.utils.data.DataLoader`
     |  iterator. When :attr:`num_workers > 0`, each worker process will have a
     |  different copy of the dataset object, so it is often desired to configure
     |  each copy independently to avoid having duplicate data returned from the
     |  workers. :func:`~torch.utils.data.get_worker_info`, when called in a worker
     |  process, returns information about the worker. It can be used in either the
     |  dataset's :meth:`__iter__` method or the :class:`~torch.utils.data.DataLoader` 's
     |  :attr:`worker_init_fn` option to modify each copy's behavior.
     |  
     |  Example 1: splitting workload across all workers in :meth:`__iter__`::
     |  
     |      >>> class MyIterableDataset(torch.utils.data.IterableDataset):
     |      ...     def __init__(self, start, end):
     |      ...         super(MyIterableDataset).__init__()
     |      ...         assert end > start, "this example code only works with end >= start"
     |      ...         self.start = start
     |      ...         self.end = end
     |      ...
     |      ...     def __iter__(self):
     |      ...         worker_info = torch.utils.data.get_worker_info()
     |      ...         if worker_info is None:  # single-process data loading, return the full iterator
     |      ...             iter_start = self.start
     |      ...             iter_end = self.end
     |      ...         else:  # in a worker process
     |      ...             # split workload
     |      ...             per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
     |      ...             worker_id = worker_info.id
     |      ...             iter_start = self.start + worker_id * per_worker
     |      ...             iter_end = min(iter_start + per_worker, self.end)
     |      ...         return iter(range(iter_start, iter_end))
     |      ...
     |      >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
     |      >>> ds = MyIterableDataset(start=3, end=7)
     |  
     |      >>> # Single-process loading
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
     |      [3, 4, 5, 6]
     |  
     |      >>> # Mult-process loading with two worker processes
     |      >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
     |      [3, 5, 4, 6]
     |  
     |      >>> # With even more workers
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=20)))
     |      [3, 4, 5, 6]
     |  
     |  Example 2: splitting workload across all workers using :attr:`worker_init_fn`::
     |  
     |      >>> class MyIterableDataset(torch.utils.data.IterableDataset):
     |      ...     def __init__(self, start, end):
     |      ...         super(MyIterableDataset).__init__()
     |      ...         assert end > start, "this example code only works with end >= start"
     |      ...         self.start = start
     |      ...         self.end = end
     |      ...
     |      ...     def __iter__(self):
     |      ...         return iter(range(self.start, self.end))
     |      ...
     |      >>> # should give same set of data as range(3, 7), i.e., [3, 4, 5, 6].
     |      >>> ds = MyIterableDataset(start=3, end=7)
     |  
     |      >>> # Single-process loading
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=0)))
     |      [3, 4, 5, 6]
     |      >>>
     |      >>> # Directly doing multi-process loading yields duplicate data
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2)))
     |      [3, 3, 4, 4, 5, 5, 6, 6]
     |  
     |      >>> # Define a `worker_init_fn` that configures each dataset copy differently
     |      >>> def worker_init_fn(worker_id):
     |      ...     worker_info = torch.utils.data.get_worker_info()
     |      ...     dataset = worker_info.dataset  # the dataset copy in this worker process
     |      ...     overall_start = dataset.start
     |      ...     overall_end = dataset.end
     |      ...     # configure the dataset to only process the split workload
     |      ...     per_worker = int(math.ceil((overall_end - overall_start) / float(worker_info.num_workers)))
     |      ...     worker_id = worker_info.id
     |      ...     dataset.start = overall_start + worker_id * per_worker
     |      ...     dataset.end = min(dataset.start + per_worker, overall_end)
     |      ...
     |  
     |      >>> # Mult-process loading with the custom `worker_init_fn`
     |      >>> # Worker 0 fetched [3, 4].  Worker 1 fetched [5, 6].
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=2, worker_init_fn=worker_init_fn)))
     |      [3, 5, 4, 6]
     |  
     |      >>> # With even more workers
     |      >>> print(list(torch.utils.data.DataLoader(ds, num_workers=20, worker_init_fn=worker_init_fn)))
     |      [3, 4, 5, 6]
     |  
     |  Method resolution order:
     |      MultiDatasetIterator
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, dataset=None, workers=4, output_size=100, pin_memory=True)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |  
     |  __next__(self)
     |  
     |  terminate(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from torch.utils.data.dataset.IterableDataset:
     |  
     |  __add__(self, other)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from torch.utils.data.dataset.Dataset:
     |  
     |  __getitem__(self, index)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from torch.utils.data.dataset.Dataset:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FUNCTIONS
    D(*args)
    
    copy_and_delete_tensors(sample, pin_memory=True)
    
    maybe_copy(a, pin_memory)
    
    omp_warning()

DATA
    timeout = 0.1
    verbose = 0

FILE
    /home/tmb/proj/webdataset/webdataset/multi.py



```

# Module `webdataset.__init__`

```
Help on module webdataset.__init__ in webdataset:

NAME
    webdataset.__init__

DESCRIPTION
    # Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
    # This file is part of the WebDataset library.
    # See the LICENSE file for licensing terms (BSD-style).
    #
    # flake8: noqa

DATA
    __all__ = ['tenbin', 'dataset', 'writer']

FILE
    /home/tmb/proj/webdataset/webdataset/__init__.py



```

# Module `webdataset.autodecode`

```
Help on module webdataset.autodecode in webdataset:

NAME
    webdataset.autodecode

DESCRIPTION
    Train PyTorch models directly from POSIX tar archive, locally
    or over HTTP connections.

FUNCTIONS
    imagehandler(data, imagespec)
        Decode image data using the given `imagespec`.
        
        The `imagespec` specifies whether the image is decoded
        to numpy/torch/pi, decoded to uint8/float, and decoded
        to l/rgb/rgba:
        
        - l8: numpy uint8 l
        - rgb8: numpy uint8 rgb
        - rgba8: numpy uint8 rgba
        - l: numpy float l
        - rgb: numpy float rgb
        - rgba: numpy float rgba
        - torchl8: torch uint8 l
        - torchrgb8: torch uint8 rgb
        - torchrgba8: torch uint8 rgba
        - torchl: torch float l
        - torchrgb: torch float rgb
        - torch: torch float rgb
        - torchrgba: torch float rgba
        - pill: pil None l
        - pil: pil None rgb
        - pilrgb: pil None rgb
        - pilrgba: pil None rgba

DATA
    __all__ = ['WebDataset', 'tariterator', 'default_handlers', 'imagehand...
    default_handlers = {'l': {'avi': <webdataset.autodecode.TorchVideoLoad...

FILE
    /home/tmb/proj/webdataset/webdataset/autodecode.py



```

# Module `webdataset.filters`

```
Help on module webdataset.filters in webdataset:

NAME
    webdataset.filters

DESCRIPTION
    Train PyTorch models directly from POSIX tar archive, locally
    or over HTTP connections.

DATA
    __all__ = ['WebDataset', 'tariterator', 'default_handlers', 'imagehand...

FILE
    /home/tmb/proj/webdataset/webdataset/filters.py



```

# Module `webdataset.dataset`

```
Help on module webdataset.dataset in webdataset:

NAME
    webdataset.dataset

DESCRIPTION
    Train PyTorch models directly from POSIX tar archive, locally
    or over HTTP connections.

CLASSES
    torch.utils.data.dataset.IterableDataset(torch.utils.data.dataset.Dataset)
        Dataset(torch.utils.data.dataset.IterableDataset, SampleIterator)
    SampleIterator(Pipeline)
        Dataset(torch.utils.data.dataset.IterableDataset, SampleIterator)
    
    class Dataset(torch.utils.data.dataset.IterableDataset, SampleIterator)
     |  Dataset(urls, *, length=None, open_fn=<function reader at 0x7f9fd898a550>, handler=<function reraise_exception at 0x7f9facbfc700>, tarhandler=None, prepare_for_worker=True, initial_pipeline=None, shard_selection=<function worker_urls at 0x7f9fd898b820>)
     |  
     |  Iterate over sharded datasets.
     |  
     |  This class combines several function: it is a container for a list of
     |  shards, it is a container for a processing pipelines, and it handles
     |  some bookkeeping related to DataLoader.
     |  
     |  Method resolution order:
     |      Dataset
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      SampleIterator
     |      Pipeline
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, urls, *, length=None, open_fn=<function reader at 0x7f9fd898a550>, handler=<function reraise_exception at 0x7f9facbfc700>, tarhandler=None, prepare_for_worker=True, initial_pipeline=None, shard_selection=<function worker_urls at 0x7f9fd898b820>)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |  
     |  __len__(self)
     |      Return the nominal length of the dataset.
     |  
     |  shard_fn(self)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from torch.utils.data.dataset.IterableDataset:
     |  
     |  __add__(self, other)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from torch.utils.data.dataset.Dataset:
     |  
     |  __getitem__(self, index)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from torch.utils.data.dataset.Dataset:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from SampleIterator:
     |  
     |  raw_samples(self, urls)
     |  
     |  samples(self, urls)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Pipeline:
     |  
     |  batched(self, batchsize, partial=True)
     |  
     |  decode(self, decoder='rgb', handler=<function reraise_exception at 0x7f9facbfc700>)
     |      Decode the data with the given decoder.
     |  
     |  map(self, f, handler=<function reraise_exception at 0x7f9facbfc700>)
     |      Apply function `f` to each sample.
     |  
     |  map_dict(self, handler=<function reraise_exception at 0x7f9facbfc700>, **kw)
     |      Transform each sample by applying functions to corresponding fields.
     |  
     |  map_tuple(self, *args, handler=<function reraise_exception at 0x7f9facbfc700>)
     |      Apply a list of functions to the tuple.
     |  
     |  pipe(self, stage)
     |      Add a pipline stage (a function taking an iterator and returning another iterator).
     |  
     |  rename(self, handler=<function reraise_exception at 0x7f9facbfc700>, **kw)
     |      Rename fields in the sample, dropping all unmatched fields.
     |  
     |  reseed_rng(self)
     |  
     |  select(self, predicate, **kw)
     |      Select samples based on a predicate.
     |  
     |  shuffle(self, size, rng=None, **kw)
     |      Shuffle the data.
     |  
     |  to_tuple(self, *args, handler=<function reraise_exception at 0x7f9facbfc700>)
     |      Extract fields from the sample in order and yield tuples.
     |  
     |  unbatched(self)

FUNCTIONS
    ignore_and_continue(exn)
        Called in an exception handler to ignore any exception and continue.
    
    ignore_and_stop(exn)
        Called in an exception handler to ignore any exception and stop further processing.
    
    reraise_exception(exn)
        Called in an exception handler to re-raise the exception.
    
    warn_and_continue(exn)
        Called in an exception handler to ignore any exception, isssue a warning, and continue.
    
    warn_and_stop(exn)
        Called in an exception handler to ignore any exception and stop further processing.

DATA
    __all__ = ['Dataset', 'tariterator', 'default_handlers', 'imagehandler...

FILE
    /home/tmb/proj/webdataset/webdataset/dataset.py



```

# Module `webdataset.gopen`

```
Help on module webdataset.gopen in webdataset:

NAME
    webdataset.gopen - Open URLs by calling subcommands.

FUNCTIONS
    gopen(url, mode='rb', bufsize=8192, **kw)

DATA
    __all__ = ['gopen', 'gopen_schemes']
    gopen_schemes = {'__default__': <function gopen_objectio>, 'ftps': <fu...

FILE
    /home/tmb/proj/webdataset/webdataset/gopen.py



```

# Module `webdataset.checks`

```
Help on module webdataset.checks in webdataset:

NAME
    webdataset.checks

DESCRIPTION
    # Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
    # This file is part of the objectio library.
    # See the LICENSE file for licensing terms (BSD-style).
    #

FUNCTIONS
    check(value, msg='')
        Check value for membership; raise ValueError if fails.
    
    checkcallable(value, msg='')
        Check value for membership; raise ValueError if fails.
    
    checkmember(value, values, msg='')
        Check value for membership; raise ValueError if fails.
    
    checknotnone(value, msg='')
        Check value for membership; raise ValueError if fails.
    
    checkrange(value, lo, hi, msg='')
        Check value for membership; raise ValueError if fails.
    
    checktype(value, types, msg='')
        Type check value; raise ValueError if fails.

FILE
    /home/tmb/proj/webdataset/webdataset/checks.py



```

# Module `webdataset.tenbin`

```
Help on module webdataset.tenbin in webdataset:

NAME
    webdataset.tenbin - Binary tensor encodings for PyTorch and NumPy.

DESCRIPTION
    This defines efficient binary encodings for tensors. The format is 8 byte
    aligned and can be used directly for computations when transmitted, say,
    via RDMA. The format is supported by WebDataset with the `.ten` filename
    extension. It is also used by Tensorcom, Tensorcom RDMA, and can be used
    for fast tensor storage with LMDB and in disk files (which can be memory
    mapped)
    
    Data is encoded as a series of chunks:
    
    - magic number (int64)
    - length in bytes (int64)
    - bytes (multiple of 64 bytes long)
    
    Arrays are a header chunk followed by a data chunk.
    Header chunks have the following structure:
    
    - dtype (int64)
    - 8 byte array name
    - ndim (int64)
    - dim[0]
    - dim[1]
    - ...

FUNCTIONS
    load(fname, infos=False, nocheck=False)
        Read a list of arrays from a file, with magics, length, and padding.
    
    read(stream, n=999999, infos=False)
        Read a list of arrays from a stream, with magics, length, and padding.
    
    save(fname, *args, infos=None, nocheck=False)
        Save a list of arrays to a file, with magics, length, and padding.
    
    sctp_recv(socket, infos=False, maxsize=100000000)
        Receive arrays as an SCTP datagram.
        
        This is just a convenience function and illustration.
        For more complex networking needs, you may want
        to call sctp_recv and decode_buffer directly.
    
    sctp_send(socket, dest, l, infos=None)
        Send arrays as an SCTP datagram.
        
        This is just a convenience function and illustration.
        For more complex networking needs, you may want
        to call encode_buffer and sctp_send directly.
    
    write(stream, l, infos=None)
        Write a list of arrays to a stream, with magics, length, and padding.
    
    zrecv_multipart(socket, infos=False)
        Receive arrays as a multipart ZMQ message.
    
    zrecv_single(socket, infos=False)
        Receive arrays as a single part ZMQ message.
    
    zsend_multipart(socket, l, infos=None)
        Send arrays as a multipart ZMQ message.
    
    zsend_single(socket, l, infos=None)
        Send arrays as a single part ZMQ message.

DATA
    __all__ = ['read', 'write', 'save', 'load', 'zsend_single', 'zrecv_sin...

FILE
    /home/tmb/proj/webdataset/webdataset/tenbin.py



```

# Module `webdataset.writer`

```
Help on module webdataset.writer in webdataset:

NAME
    webdataset.writer

DESCRIPTION
    # Copyright (c) 2017-2019 NVIDIA CORPORATION. All rights reserved.
    # This file is part of the WebDataset library.
    # See the LICENSE file for licensing terms (BSD-style).
    #

CLASSES
    builtins.object
        ShardWriter
        TarWriter
    
    class ShardWriter(builtins.object)
     |  ShardWriter(pattern, maxcount=100000, maxsize=3000000000.0, post=None, **kw)
     |  
     |  Like TarWriter but splits into multiple shards.
     |  
     |  :param pattern: output file pattern
     |  :param maxcount: maximum number of records per shard (Default value = 100000)
     |  :param maxsize: maximum size of each shard (Default value = 3e9)
     |  :param kw: other options passed to TarWriter
     |  
     |  Methods defined here:
     |  
     |  __enter__(self)
     |  
     |  __exit__(self, *args, **kw)
     |  
     |  __init__(self, pattern, maxcount=100000, maxsize=3000000000.0, post=None, **kw)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  close(self)
     |  
     |  finish(self)
     |  
     |  next_stream(self)
     |  
     |  write(self, obj)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class TarWriter(builtins.object)
     |  TarWriter(fileobj, user='bigdata', group='bigdata', mode=292, compress=None, encoder=True, keep_meta=False)
     |  
     |  A class for writing dictionaries to tar files.
     |  
     |  :param fileobj: fileobj: file name for tar file (.tgz/.tar) or open file descriptor
     |  :param encoder: sample encoding (Default value = True)
     |  :param compress:  (Default value = None)
     |  
     |  `True` will use an encoder that behaves similar to the automatic
     |  decoder for `Dataset`. `False` disables encoding and expects byte strings
     |  (except for metadata, which must be strings). The `encoder` argument can
     |  also be a `callable`, or a dictionary mapping extensions to encoders.
     |  
     |  The following code will add two file to the tar archive: `a/b.png` and
     |  `a/b.output.png`.
     |  
     |  ```Python
     |      tarwriter = TarWriter(stream)
     |      image = imread("b.jpg")
     |      image2 = imread("b.out.jpg")
     |      sample = {"__key__": "a/b", "png": image, "output.png": image2}
     |      tarwriter.write(sample)
     |  ```
     |  
     |  Methods defined here:
     |  
     |  __enter__(self)
     |  
     |  __exit__(self, exc_type, exc_val, exc_tb)
     |  
     |  __init__(self, fileobj, user='bigdata', group='bigdata', mode=292, compress=None, encoder=True, keep_meta=False)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  close(self)
     |      Close the tar file.
     |  
     |  dwrite(self, key, **kw)
     |      Convenience function for `write`.
     |      
     |      Takes key as the first argument and key-value pairs for the rest.
     |      Replaces "_" with ".".
     |  
     |  write(self, obj)
     |      Write a dictionary to the tar file.
     |      
     |      :param obj: dictionary of objects to be stored
     |      :returns: size of the entry
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

DATA
    __all__ = ['TarWriter', 'ShardWriter']

FILE
    /home/tmb/proj/webdataset/webdataset/writer.py



```
