
# Module `webdataset.iterators`

```
Help on module webdataset.iterators in webdataset:

NAME
    webdataset.iterators

DESCRIPTION
    A collection of iterators implementing useful functionality for
    transforming datasets in processing pipelines.
    
    These functions are plain iterator functions. You can find curried versions
    in webdataset.filters, and you can find IterableDataset wrappers in
    webdataset.processing.

DATA
    __all__ = ['WebDataset', 'tariterator', 'default_handlers', 'imagehand...

FILE
    /home/tmb/proj/webdataset/webdataset/iterators.py



```

# Module `webdataset.fluid`

```
Help on module webdataset.fluid in webdataset:

NAME
    webdataset.fluid - A fluid interface for constructing datasets.

CLASSES
    torch.utils.data.dataset.IterableDataset(torch.utils.data.dataset.Dataset)
        Dataset
    
    class Dataset(torch.utils.data.dataset.IterableDataset)
     |  Dataset(*args, **kwds)
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
     |      Dataset
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      typing.Generic
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __getattr__(self, name)
     |  
     |  __init__(self, urls, *, length=True, splitter=<function split_by_worker at 0x7f3d8dd039d0>, handler=<function reraise_exception at 0x7f3d8de85040>, shuffle=False, cache_dir='', cache_size=1000000000000000, cache_name=<function shard_uuid at 0x7f3d8dd03310>, cache_verbose=1)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |  
     |  __len__(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __parameters__ = ()
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from torch.utils.data.dataset.IterableDataset:
     |  
     |  __add__(self, other: torch.utils.data.dataset.Dataset[+T_co])
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from torch.utils.data.dataset.IterableDataset:
     |  
     |  __orig_bases__ = (torch.utils.data.dataset.Dataset[+T_co],)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from torch.utils.data.dataset.Dataset:
     |  
     |  __getitem__(self, index) -> +T_co
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
     |  Class methods inherited from typing.Generic:
     |  
     |  __class_getitem__(params) from builtins.type
     |  
     |  __init_subclass__(*args, **kwargs) from builtins.type
     |      This method is called when a class is subclassed.
     |      
     |      The default implementation does nothing. It may be
     |      overridden to extend subclasses.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from typing.Generic:
     |  
     |  __new__(cls, *args, **kwds)
     |      Create and return a new object.  See help(type) for accurate signature.

DATA
    __all__ = ['FluidPipes', 'Dataset']

FILE
    /home/tmb/proj/webdataset/webdataset/fluid.py



```

# Module `webdataset.multi`

```
Help on module webdataset.multi in webdataset:

NAME
    webdataset.multi

CLASSES
    builtins.object
        DistLoader
        DistSender
        Finished
        MultiLoader
    
    class DistLoader(builtins.object)
     |  DistLoader(sockname)
     |  
     |  Methods defined here:
     |  
     |  __init__(self, sockname)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class DistSender(builtins.object)
     |  DistSender(sockname)
     |  
     |  Methods defined here:
     |  
     |  __init__(self, sockname)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  send(self, sample)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Finished(builtins.object)
     |  Finished(**kw)
     |  
     |  Methods defined here:
     |  
     |  __init__(self, **kw)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class MultiLoader(builtins.object)
     |  MultiLoader(dataset, workers=4, verbose=False, nokill=False, prefix='/tmp/_multi-')
     |  
     |  Methods defined here:
     |  
     |  __init__(self, dataset, workers=4, verbose=False, nokill=False, prefix='/tmp/_multi-')
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |  
     |  kill(self)
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FUNCTIONS
    reader(dataset, sockname, index)

DATA
    all_pids = set()
    the_protocol = 5

FILE
    /home/tmb/proj/webdataset/webdataset/multi.py



```

# Module `webdataset.dbcache`

```
Help on module webdataset.dbcache in webdataset:

NAME
    webdataset.dbcache

CLASSES
    torch.utils.data.dataset.IterableDataset(torch.utils.data.dataset.Dataset)
        DBCache
    
    class DBCache(torch.utils.data.dataset.IterableDataset)
     |  DBCache(*args, **kwds)
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
     |      DBCache
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      typing.Generic
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __call__(self, source)
     |      Call self as a function.
     |  
     |  __init__(self, dbname, size, source=None, shuffle=False, verbose=True)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |  
     |  __len__(self)
     |  
     |  dbiter(self)
     |  
     |  getmeta(self, key)
     |  
     |  key_exists(self, key)
     |  
     |  setmeta(self, key, value)
     |  
     |  source_(self, source)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __parameters__ = ()
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from torch.utils.data.dataset.IterableDataset:
     |  
     |  __add__(self, other: torch.utils.data.dataset.Dataset[+T_co])
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes inherited from torch.utils.data.dataset.IterableDataset:
     |  
     |  __orig_bases__ = (torch.utils.data.dataset.Dataset[+T_co],)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from torch.utils.data.dataset.Dataset:
     |  
     |  __getitem__(self, index) -> +T_co
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
     |  Class methods inherited from typing.Generic:
     |  
     |  __class_getitem__(params) from builtins.type
     |  
     |  __init_subclass__(*args, **kwargs) from builtins.type
     |      This method is called when a class is subclassed.
     |      
     |      The default implementation does nothing. It may be
     |      overridden to extend subclasses.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from typing.Generic:
     |  
     |  __new__(cls, *args, **kwds)
     |      Create and return a new object.  See help(type) for accurate signature.

FUNCTIONS
    get_uuid(data)

FILE
    /home/tmb/proj/webdataset/webdataset/dbcache.py



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
    imagehandler(imagespec)

DATA
    __all__ = ['WebDataset', 'tariterator', 'default_handlers', 'imagehand...

FILE
    /home/tmb/proj/webdataset/webdataset/autodecode.py



```

# Module `webdataset.utils`

```
Help on module webdataset.utils in webdataset:

NAME
    webdataset.utils

FUNCTIONS
    ignore_and_continue(exn)
        Called in an exception handler to ignore any exception and continue.
    
    ignore_and_stop(exn)
        Called in an exception handler to ignore any exception and stop further processing.
    
    reraise_exception(exn)
        Called in an exception handler to re-raise the exception.
    
    warn_and_stop(exn)
        Called in an exception handler to ignore any exception and stop further processing.

DATA
    __all__ = ['reraise_exception', 'ignore_and_continue', 'ignore_and_sto...

FILE
    /home/tmb/proj/webdataset/webdataset/utils.py



```

# Module `webdataset.mock`

```
Help on module webdataset.mock in webdataset:

NAME
    webdataset.mock

CLASSES
    builtins.object
        DataLoader
        IterableDataset
    
    class DataLoader(builtins.object)
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class IterableDataset(builtins.object)
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FILE
    /home/tmb/proj/webdataset/webdataset/mock.py



```

# Module `webdataset.filters`

```
problem in webdataset.filters - AttributeError: module 'webdataset.iterators' has no attribute 'map_stream'


```

# Module `webdataset.dataset`

```
Help on module webdataset.dataset in webdataset:

NAME
    webdataset.dataset

DESCRIPTION
    Train PyTorch models directly from POSIX tar archive, locally
    or over HTTP connections.

FUNCTIONS
    reraise_exception(exn)
        Called in an exception handler to re-raise the exception.

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
    gopen_schemes = {'__default__': <function gopen_error>, 'ftps': <funct...

FILE
    /home/tmb/proj/webdataset/webdataset/gopen.py



```

# Module `webdataset.tariterators`

```
Help on module webdataset.tariterators in webdataset:

NAME
    webdataset.tariterators

DESCRIPTION
    Train PyTorch models directly from POSIX tar archive, locally
    or over HTTP connections.

FUNCTIONS
    reraise_exception(exn)
        Called in an exception handler to re-raise the exception.

DATA
    __all__ = ['Dataset', 'tariterator', 'default_handlers', 'imagehandler...

FILE
    /home/tmb/proj/webdataset/webdataset/tariterators.py



```

# Module `webdataset.workerenv`

```
Help on module webdataset.workerenv in webdataset:

NAME
    webdataset.workerenv

DESCRIPTION
    Functions related to splitting datasets by node and worker.
    This follows the PyTorch model of magic global functions and environment
    settings and supplies the default node and worker splitting functions.
    This is provided mainly because PyTorch users expect something like this
    to exist. The cleaner and safer way of dealing with node and worker splitting
    is via explicit functions.

CLASSES
    builtins.object
        WorkerEnvironment
            TorchWorkerEnvironment
    
    class TorchWorkerEnvironment(WorkerEnvironment)
     |  TorchWorkerEnvironment(group=None)
     |  
     |  Method resolution order:
     |      TorchWorkerEnvironment
     |      WorkerEnvironment
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, group=None)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from WorkerEnvironment:
     |  
     |  __str__(self)
     |      Return str(self).
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from WorkerEnvironment:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class WorkerEnvironment(builtins.object)
     |  WorkerEnvironment(rank=0, world_size=1, worker=0, nworkers=1)
     |  
     |  Methods defined here:
     |  
     |  __init__(self, rank=0, world_size=1, worker=0, nworkers=1)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __str__(self)
     |      Return str(self).
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FUNCTIONS
    get_worker_environment()
    
    split_by_node(urls, env=None)
        Selects a subset of urls based on node info.
        
        Used as a shard selection function in Dataset.
    
    split_by_worker(urls, env=None)
        Selects a subset of urls based on worker info.
        
        Used as a shard selection function in Dataset.
    
    worker_id()

DATA
    too_few_shards_warning = 1
    worker_environment = None

FILE
    /home/tmb/proj/webdataset/webdataset/workerenv.py



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

# Module `webdataset.bench`

```
Help on module webdataset.bench in webdataset:

NAME
    webdataset.bench

CLASSES
    builtins.object
        TotalSize
    
    class TotalSize(builtins.object)
     |  Methods defined here:
     |  
     |  __call__(self, sample)
     |      Call self as a function.
     |  
     |  __init__(self)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)

FUNCTIONS
    main(args)

FILE
    /home/tmb/proj/webdataset/webdataset/bench.py



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
     |  ShardWriter(pattern, maxcount=100000, maxsize=3000000000.0, post=None, start_shard=0, **kw)
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
     |  __init__(self, pattern, maxcount=100000, maxsize=3000000000.0, post=None, start_shard=0, **kw)
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

# Module `webdataset.shardcache`

```
Help on module webdataset.shardcache in webdataset:

NAME
    webdataset.shardcache

CLASSES
    io.RawIOBase(_io._RawIOBase, io.IOBase)
        CacheStream
    
    class CacheStream(io.RawIOBase)
     |  CacheStream(fname, stream, verbose=False)
     |  
     |  Base class for raw binary I/O.
     |  
     |  Method resolution order:
     |      CacheStream
     |      io.RawIOBase
     |      _io._RawIOBase
     |      io.IOBase
     |      _io._IOBase
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, fname, stream, verbose=False)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  close(self, complete=False)
     |      Flush and close the IO object.
     |      
     |      This method has no effect if the file is already closed.
     |  
     |  read(self, n)
     |  
     |  readinto(self, b)
     |  
     |  ----------------------------------------------------------------------
     |  Data and other attributes defined here:
     |  
     |  __abstractmethods__ = frozenset()
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from _io._RawIOBase:
     |  
     |  readall(self, /)
     |      Read until EOF, using multiple read() call.
     |  
     |  write(...)
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from _io._IOBase:
     |  
     |  __del__(...)
     |  
     |  __enter__(...)
     |  
     |  __exit__(...)
     |  
     |  __iter__(self, /)
     |      Implement iter(self).
     |  
     |  __next__(self, /)
     |      Implement next(self).
     |  
     |  fileno(self, /)
     |      Returns underlying file descriptor if one exists.
     |      
     |      OSError is raised if the IO object does not use a file descriptor.
     |  
     |  flush(self, /)
     |      Flush write buffers, if applicable.
     |      
     |      This is not implemented for read-only and non-blocking streams.
     |  
     |  isatty(self, /)
     |      Return whether this is an 'interactive' stream.
     |      
     |      Return False if it can't be determined.
     |  
     |  readable(self, /)
     |      Return whether object was opened for reading.
     |      
     |      If False, read() will raise OSError.
     |  
     |  readline(self, size=-1, /)
     |      Read and return a line from the stream.
     |      
     |      If size is specified, at most size bytes will be read.
     |      
     |      The line terminator is always b'\n' for binary files; for text
     |      files, the newlines argument to open can be used to select the line
     |      terminator(s) recognized.
     |  
     |  readlines(self, hint=-1, /)
     |      Return a list of lines from the stream.
     |      
     |      hint can be specified to control the number of lines read: no more
     |      lines will be read if the total size (in bytes/characters) of all
     |      lines so far exceeds hint.
     |  
     |  seek(...)
     |      Change stream position.
     |      
     |      Change the stream position to the given byte offset. The offset is
     |      interpreted relative to the position indicated by whence.  Values
     |      for whence are:
     |      
     |      * 0 -- start of stream (the default); offset should be zero or positive
     |      * 1 -- current stream position; offset may be negative
     |      * 2 -- end of stream; offset is usually negative
     |      
     |      Return the new absolute position.
     |  
     |  seekable(self, /)
     |      Return whether object supports random access.
     |      
     |      If False, seek(), tell() and truncate() will raise OSError.
     |      This method may need to do a test seek().
     |  
     |  tell(self, /)
     |      Return current stream position.
     |  
     |  truncate(...)
     |      Truncate file to size bytes.
     |      
     |      File pointer is left unchanged.  Size defaults to the current IO
     |      position as reported by tell().  Returns the new size.
     |  
     |  writable(self, /)
     |      Return whether object was opened for writing.
     |      
     |      If False, write() will raise OSError.
     |  
     |  writelines(self, lines, /)
     |      Write a list of lines to stream.
     |      
     |      Line separators are not added, so it is usual for each of the
     |      lines provided to have a line separator at the end.
     |  
     |  ----------------------------------------------------------------------
     |  Static methods inherited from _io._IOBase:
     |  
     |  __new__(*args, **kwargs) from builtins.type
     |      Create and return a new object.  See help(type) for accurate signature.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors inherited from _io._IOBase:
     |  
     |  __dict__
     |  
     |  closed

FUNCTIONS
    cache_shards(urls, cache_dir='./data', cache_size=1000000000000000.0, cache_name=<function guess_shard at 0x7f8138f43280>, verbose=False)
    
    guess_shard(path)
    
    shard_uuid(path)

FILE
    /home/tmb/proj/webdataset/webdataset/shardcache.py



```
