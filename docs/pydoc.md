
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
        Dataset
        webdataset.webdataset.WebDataset
    
    class Dataset(torch.utils.data.dataset.IterableDataset)
     |  Dataset(urls, *, keys=<function base_plus_ext at 0x7fd35cd940e0>, suffixes=None, length=None, epochs=1, opener=<function reader at 0x7fd35d0ee290>, handler=<function reraise_exception at 0x7fd36531c200>, shuffle=False, prepare_for_worker=True, initial_pipeline=None)
     |  
     |  Iterate over sharded datasets.
     |  
     |  :param urls: shard spec or list of shards
     |  :param prepare_for_worker: callable called in each worker before anything else is done
     |  
     |  Method resolution order:
     |      Dataset
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, urls, *, keys=<function base_plus_ext at 0x7fd35cd940e0>, suffixes=None, length=None, epochs=1, opener=<function reader at 0x7fd35d0ee290>, handler=<function reraise_exception at 0x7fd36531c200>, shuffle=False, prepare_for_worker=True, initial_pipeline=None)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |  
     |  __len__(self)
     |  
     |  decode(self, decoder='rgb', handler=<function reraise_exception at 0x7fd36531c200>)
     |      Decode the data with the given decoder.
     |  
     |  map(self, f, handler=<function reraise_exception at 0x7fd36531c200>)
     |      Apply function `f` to each sample.
     |  
     |  map_dict(self, handler=<function reraise_exception at 0x7fd36531c200>, **kw)
     |      Transform each sample by applying functions to corresponding fields.
     |  
     |  map_tuple(self, *args, handler=<function reraise_exception at 0x7fd36531c200>)
     |      Apply a list of functions to the tuple.
     |  
     |  pipe(self, stage)
     |      Add a pipline stage (a function taking an iterator and returning another iterator).
     |  
     |  raw_iter(self)
     |      Iterate over samples.
     |  
     |  rename(self, handler=<function reraise_exception at 0x7fd36531c200>, **kw)
     |      Rename fields in the sample, dropping all unmatched fields.
     |  
     |  shard_selection(self)
     |      Contains the logic for self.subset shard selection.
     |  
     |  shuffle(self, size)
     |      Shuffle the data.
     |  
     |  to_tuple(self, *args, handler=<function reraise_exception at 0x7fd36531c200>)
     |      Extract fields from the sample in order and yield tuples.
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
    
    class WebDataset(torch.utils.data.dataset.IterableDataset)
     |  WebDataset(urls, *, extensions=None, decoder='rgb', transforms=None, pipeline=None, epochs=1, keys=<function base_plus_ext at 0x7fd35cd939e0>, opener=<function reader at 0x7fd35d0ee290>, verbose=False, shuffle=0, associate=None, prepare_for_worker=True, length=None, handler=<function reraise_exception at 0x7fd35d0f0830>)
     |  
     |  Iterate over sharded datasets.
     |  
     |  :param urls: shard spec or list of shards
     |  :param extensions: extensions to extract (Default value = None, can be either list of lists or "a;b c")
     |  :param decode: decoder to apply to files in tarfiles (Default value = True, based on extension)
     |  :param transforms: list of functions to apply to unbatched samples (Default value = None)
     |  :param pipeline: function that maps the iterator, e.g. for batching
     |  :param opener: either a function that returns a stream or a string that is invoked via Popen
     |  :param verbose: verbose output
     |  :param shuffle: if >0, then shuffle shards, and shuffle samples with a buffer of the given size
     |  :param associate: a callable or dictionary that returns additional information to associate with each sample
     |  :param prepare_for_worker: callable called in each worker before anything else is done
     |  :param extra_meta: associates subset info with each sample record
     |  
     |  The decoder can be True (default decoder), False (no decoder), a callable (called
     |  decode the sample, or a dictionary mapping filename extensions to callables for
     |  the decoding.
     |  
     |  Method resolution order:
     |      WebDataset
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, urls, *, extensions=None, decoder='rgb', transforms=None, pipeline=None, epochs=1, keys=<function base_plus_ext at 0x7fd35cd939e0>, opener=<function reader at 0x7fd35d0ee290>, verbose=False, shuffle=0, associate=None, prepare_for_worker=True, length=None, handler=<function reraise_exception at 0x7fd35d0f0830>)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |      Iterate over samples.
     |  
     |  __len__(self)
     |  
     |  shard_selection(self)
     |      Contains the logic for self.subset shard selection.
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
    __all__ = ['Dataset', 'WebDataset', 'tariterator', 'default_handlers',...

FILE
    /home/tmb/proj/webdataset/webdataset/dataset.py



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

# Module `webdataset.io`

```
Help on module webdataset.io in webdataset:

NAME
    webdataset.io - Open URLs by calling subcommands.

FUNCTIONS
    gopen(url, mode='rb', bufsize=8192)

DATA
    __all__ = ['gopen', 'scheme_to_command']

FILE
    /home/tmb/proj/webdataset/webdataset/io.py



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

# Module `webdataset.webdataset`

```
Help on module webdataset.webdataset in webdataset:

NAME
    webdataset.webdataset

DESCRIPTION
    Train PyTorch models directly from POSIX tar archive, locally
    or over HTTP connections.

CLASSES
    torch.utils.data.dataset.IterableDataset(torch.utils.data.dataset.Dataset)
        WebDataset
    
    class WebDataset(torch.utils.data.dataset.IterableDataset)
     |  WebDataset(urls, *, extensions=None, decoder='rgb', transforms=None, pipeline=None, epochs=1, keys=<function base_plus_ext at 0x7fc0405949e0>, opener=<function reader at 0x7fc04f4b1290>, verbose=False, shuffle=0, associate=None, prepare_for_worker=True, length=None, handler=<function reraise_exception at 0x7fc04f4b9830>)
     |  
     |  Iterate over sharded datasets.
     |  
     |  :param urls: shard spec or list of shards
     |  :param extensions: extensions to extract (Default value = None, can be either list of lists or "a;b c")
     |  :param decode: decoder to apply to files in tarfiles (Default value = True, based on extension)
     |  :param transforms: list of functions to apply to unbatched samples (Default value = None)
     |  :param pipeline: function that maps the iterator, e.g. for batching
     |  :param opener: either a function that returns a stream or a string that is invoked via Popen
     |  :param verbose: verbose output
     |  :param shuffle: if >0, then shuffle shards, and shuffle samples with a buffer of the given size
     |  :param associate: a callable or dictionary that returns additional information to associate with each sample
     |  :param prepare_for_worker: callable called in each worker before anything else is done
     |  :param extra_meta: associates subset info with each sample record
     |  
     |  The decoder can be True (default decoder), False (no decoder), a callable (called
     |  decode the sample, or a dictionary mapping filename extensions to callables for
     |  the decoding.
     |  
     |  Method resolution order:
     |      WebDataset
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, urls, *, extensions=None, decoder='rgb', transforms=None, pipeline=None, epochs=1, keys=<function base_plus_ext at 0x7fc0405949e0>, opener=<function reader at 0x7fc04f4b1290>, verbose=False, shuffle=0, associate=None, prepare_for_worker=True, length=None, handler=<function reraise_exception at 0x7fc04f4b9830>)
     |      Initialize self.  See help(type(self)) for accurate signature.
     |  
     |  __iter__(self)
     |      Iterate over samples.
     |  
     |  __len__(self)
     |  
     |  shard_selection(self)
     |      Contains the logic for self.subset shard selection.
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
    tariterator(fileobj, keys=<function base_plus_ext at 0x7fc0405949e0>, decoder=True, suffixes=None, tar_errors=<function reraise_exception at 0x7fc04f4b9830>, decode_errors=<function reraise_exception at 0x7fc04f4b9830>)
        Iterate through training samples stored in a sharded tar file.
        
        :param fileobj: a Python file-like object
        :param check_sorted:  check whether the input is actually properly sorted (Default value = False)
        :param keys:  key extraction function (Default value = base_plus_ext)
        :param decoder: value decoding function (Default value = True)
        
        The key extraction function takes a string representing a pathname and
        returns a pair (__key__, suffix).
        
        The decoder takes the entire sample as a dict and returns the
        decoded sample as a dict.

DATA
    __all__ = ['Dataset', 'WebDataset', 'tariterator', 'default_handlers',...

FILE
    /home/tmb/proj/webdataset/webdataset/webdataset.py



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
    default_handlers = {'l': {'class': <function maybe_int>, 'cls': <funct...

FILE
    /home/tmb/proj/webdataset/webdataset/autodecode.py



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
