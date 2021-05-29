
# Module `webdataset.dbcache`

```
Help on module webdataset.dbcache in webdataset:

NAME
    webdataset.dbcache - Cache training samples in an SQLite3 database.

CLASSES
    torch.utils.data.dataset.IterableDataset(torch.utils.data.dataset.Dataset)
        DBCache
    
    class DBCache(torch.utils.data.dataset.IterableDataset)
     |  DBCache(*args, **kwds)
     |  
     |  An IterableDataset that caches its inputs.
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
     |  __init__(self, dbname, size, source=None, shuffle=False, verbose=True)
     |      Create a DBCache for the given file name and of the given size.
     |      
     |      :param dbname: file name
     |      :param size: number of samples to be cached
     |      :param source: data source
     |      :param shuffle: shuffle data on return
     |      :param verbose: print progress messages
     |  
     |  __iter__(self)
     |      Iterate over the samples in the dataset.
     |      
     |      If no cache is defined, just iterates over the source dataset.
     |      
     |      If a cache is set and it is full, iterates over the samples in the cache.
     |      
     |      If a cache is set and not full, adds samples to the cache from the source
     |      and yields them.
     |  
     |  __len__(self)
     |      Return the number of samples in the cache.
     |  
     |  dbiter(self)
     |      Iterate over the samples in the cache.
     |  
     |  getmeta(self, key)
     |      Get the metadata for the given key.
     |      
     |      :param key: key to be retrieved
     |  
     |  key_exists(self, key)
     |      Check whether a key exists in the database.
     |      
     |      :param key: key
     |  
     |  setmeta(self, key, value)
     |      Set the metadata for the given key.
     |      
     |      :param key: key
     |      :param value: value to be set (a string)
     |  
     |  source_(self, source)
     |      Set the dataset source for this cache.
     |      
     |      :param source: an IterableDataset
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
        Compute a UUID for data.
        
        :param data: byte array

FILE
    /home/tmb/proj/webdataset/webdataset/dbcache.py



```

# Module `webdataset.filters`

```
Help on module webdataset.filters in webdataset:

NAME
    webdataset.filters - A small curry wrapper for the functions in the `iterators` package.

CLASSES
    builtins.object
        Curried
        Curried2
    
    class Curried(builtins.object)
     |  Curried(f)
     |  
     |  Helper class for currying pipeline stages.
     |  
     |  We use this roundabout construct because it can be pickled.
     |  
     |  Methods defined here:
     |  
     |  __call__(self, *args, **kw)
     |      Curry with the given arguments.
     |  
     |  __init__(self, f)
     |      Store the function for future currying.
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Curried2(builtins.object)
     |  Curried2(f, *args, **kw)
     |  
     |  Helper class for currying pipeline stages.
     |  
     |  We use this roundabout construct becauce it can be pickled.
     |  
     |  Methods defined here:
     |  
     |  __call__(self, data)
     |      Call the curried function with the given argument.
     |  
     |  __init__(self, f, *args, **kw)
     |      Create a curried function.
     |  
     |  __repr__(self)
     |      Compute a string representation.
     |  
     |  __str__(self)
     |      Compute a string representation.
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
    associate = <webdataset.filters.Curried object>
    batched = <webdataset.filters.Curried object>
    decode = <webdataset.filters.Curried object>
    info = <webdataset.filters.Curried object>
    map = <webdataset.filters.Curried object>
    map_dict = <webdataset.filters.Curried object>
    map_tuple = <webdataset.filters.Curried object>
    rename = <webdataset.filters.Curried object>
    select = <webdataset.filters.Curried object>
    shuffle = <webdataset.filters.Curried object>
    to_tuple = <webdataset.filters.Curried object>
    unbatched = <webdataset.filters.Curried object>

FILE
    /home/tmb/proj/webdataset/webdataset/filters.py



```

# Module `webdataset.autodecode`

```
Help on module webdataset.autodecode in webdataset:

NAME
    webdataset.autodecode - Automatically decode webdataset samples.

CLASSES
    builtins.object
        Continue
        Decoder
        ImageHandler
    
    class Continue(builtins.object)
     |  Continue(key, data)
     |  
     |  Special class for continuing decoding.
     |  
     |  This is mostly used for decompression, as in:
     |  
     |      def decompressor(key, data):
     |          if key.endswith(".gz"):
     |              return Continue(key[:-3], decompress(data))
     |          return None
     |  
     |  Methods defined here:
     |  
     |  __init__(self, key, data)
     |      __init__.
     |      
     |      :param key:
     |      :param data:
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class Decoder(builtins.object)
     |  Decoder(handlers, pre=None, post=None, only=None)
     |  
     |  Decode samples using a list of handlers.
     |  
     |  For each key/data item, this iterates through the list of
     |  handlers until some handler returns something other than None.
     |  
     |  Methods defined here:
     |  
     |  __call__(self, sample)
     |      Decode an entire sample.
     |      
     |      :param sample: the sample
     |  
     |  __init__(self, handlers, pre=None, post=None, only=None)
     |      Create a Decoder.
     |      
     |      :param handlers: main list of handlers
     |      :param pre: handlers called before the main list (.gz handler by default)
     |      :param post: handlers called after the main list (default handlers by default)
     |      :param only: a list of extensions; when give, only ignores files with those extensions
     |  
     |  decode(self, sample)
     |      Decode an entire sample.
     |      
     |      :param sample: the sample, a dictionary of key value pairs
     |  
     |  decode1(self, key, data)
     |      Decode a single field of a sample.
     |      
     |      :param key: file name extension
     |      :param data: binary data
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class ImageHandler(builtins.object)
     |  ImageHandler(imagespec, extensions=['jpg', 'jpeg', 'png', 'ppm', 'pgm', 'pbm', 'pnm'])
     |  
     |  Decode image data using the given `imagespec`.
     |  
     |  The `imagespec` specifies whether the image is decoded
     |  to numpy/torch/pi, decoded to uint8/float, and decoded
     |  to l/rgb/rgba:
     |  
     |  - l8: numpy uint8 l
     |  - rgb8: numpy uint8 rgb
     |  - rgba8: numpy uint8 rgba
     |  - l: numpy float l
     |  - rgb: numpy float rgb
     |  - rgba: numpy float rgba
     |  - torchl8: torch uint8 l
     |  - torchrgb8: torch uint8 rgb
     |  - torchrgba8: torch uint8 rgba
     |  - torchl: torch float l
     |  - torchrgb: torch float rgb
     |  - torch: torch float rgb
     |  - torchrgba: torch float rgba
     |  - pill: pil None l
     |  - pil: pil None rgb
     |  - pilrgb: pil None rgb
     |  - pilrgba: pil None rgba
     |  
     |  Methods defined here:
     |  
     |  __call__(self, key, data)
     |      Perform image decoding.
     |      
     |      :param key: file name extension
     |      :param data: binary data
     |  
     |  __init__(self, imagespec, extensions=['jpg', 'jpeg', 'png', 'ppm', 'pgm', 'pbm', 'pnm'])
     |      Create an image handler.
     |      
     |      :param imagespec: short string indicating the type of decoding
     |      :param extensions: list of extensions the image handler is invoked for
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
    basichandlers(key, data)
        Handle basic file decoding.
        
        This function is usually part of the post= decoders.
        This handles the following forms of decoding:
        
        - txt -> unicode string
        - cls cls2 class count index inx id -> int
        - json jsn -> JSON decoding
        - pyd pickle -> pickle decoding
        - pth -> torch.loads
        - ten tenbin -> fast tensor loading
        - mp messagepack msg -> messagepack decoding
        - npy -> Python NPY decoding
        
        :param key: file name extension
        :param data: binary data to be decoded
    
    call_extension_handler(key, data, f, extensions)
        Call the function f with the given data if the key matches the extensions.
        
        :param key: actual key found in the sample
        :param data: binary data
        :param f: decoder function
        :param extensions: list of matching extensions
    
    gzfilter(key, data)
        Decode .gz files.
        
        This decodes compressed files and the continues decoding.
        
        :param key: file name extension
        :param data: binary data
    
    handle_extension(extensions, f)
        Return a decoder function for the list of extensions.
        
        Extensions can be a space separated list of extensions.
        Extensions can contain dots, in which case the corresponding number
        of extension components must be present in the key given to f.
        Comparisons are case insensitive.
        
        Examples:
        handle_extension("jpg jpeg", my_decode_jpg)  # invoked for any file.jpg
        handle_extension("seg.jpg", special_case_jpg)  # invoked only for file.seg.jpg
    
    imagehandler(imagespec, extensions=['jpg', 'jpeg', 'png', 'ppm', 'pgm', 'pbm', 'pnm'])
        Create an image handler.
        
        This is just a lower case alias for ImageHander.
        
        :param imagespec: textual image spec
        :param extensions: list of extensions the handler should be applied for
    
    torch_audio(key, data)
        Decode audio using the torchaudio library.
        
        :param key: file name extension
        :param data: data to be decoded
    
    torch_loads(data)
        Load data using torch.loads, importing torch only if needed.
        
        :param data: data to be decoded
    
    torch_video(key, data)
        Decode video using the torchvideo library.
        
        :param key: file name extension
        :param data: data to be decoded

DATA
    default_post_handlers = [<function basichandlers>]
    default_pre_handlers = [<function gzfilter>]
    image_extensions = ['jpg', 'jpeg', 'png', 'ppm', 'pgm', 'pbm', 'pnm']
    imagespecs = {'l': ('numpy', 'float', 'l'), 'l8': ('numpy', 'uint8', '...

FILE
    /home/tmb/proj/webdataset/webdataset/autodecode.py



```

# Module `webdataset.dataset`

```
Help on module webdataset.dataset in webdataset:

NAME
    webdataset.dataset - Train PyTorch models directly from POSIX tar archive.

DESCRIPTION
    Code works locally or over HTTP connections.

CLASSES
    builtins.object
        BatchedLength
        Composable
            DatasetTest(torch.utils.data.dataset.IterableDataset, Composable, Shorthands)
            Processor(torch.utils.data.dataset.IterableDataset, Composable, Shorthands)
            Repeatedly(torch.utils.data.dataset.IterableDataset, Composable, Shorthands)
            ResampledShards(torch.utils.data.dataset.IterableDataset, Composable)
            ShardList(torch.utils.data.dataset.IterableDataset, Composable)
        Shorthands
    torch.utils.data.dataset.IterableDataset(torch.utils.data.dataset.Dataset)
        ChoppedDataset
        DatasetTest(torch.utils.data.dataset.IterableDataset, Composable, Shorthands)
        MockDataset
        Processor(torch.utils.data.dataset.IterableDataset, Composable, Shorthands)
        Repeatedly(torch.utils.data.dataset.IterableDataset, Composable, Shorthands)
        ResampledShards(torch.utils.data.dataset.IterableDataset, Composable)
        ShardList(torch.utils.data.dataset.IterableDataset, Composable)
    
    class BatchedLength(builtins.object)
     |  BatchedLength(batchsize, partial: bool)
     |  
     |  Compute the batched length of a dataset.
     |  
     |  We make this a class rather than a closure so that it can be pickled.
     |  
     |  Methods defined here:
     |  
     |  __call__(self, length)
     |      Compute the number of batches for the given length.
     |      
     |      :param length: number of samples
     |  
     |  __init__(self, batchsize, partial: bool)
     |      Initialize.
     |      
     |      :param batchsize: batch size
     |      :param partial: allow partial batches
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class ChoppedDataset(torch.utils.data.dataset.IterableDataset)
     |  ChoppedDataset(*args, **kwds)
     |  
     |  Change the actual and nominal length of an IterableDataset.
     |  
     |  This will continuously iterate through the original dataset, but
     |  impose new epoch boundaries at the given length/nominal.
     |  This exists mainly as a workaround for the odd logic in DataLoader.
     |  It is also useful for choosing smaller nominal epoch sizes with
     |  very large datasets.
     |  
     |  Method resolution order:
     |      ChoppedDataset
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      typing.Generic
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __getstate__(self)
     |      Return the pickled state of the dataset.
     |      
     |      This resets the dataset iterator, since that can't be pickled.
     |  
     |  __init__(self, dataset, length=None, nominal=None)
     |      Create a ChoppedDataset.
     |      
     |      :param dataset: IterableDataset
     |      :param length: declared length of the dataset
     |      :param nominal: nominal length of dataset (if different from declared)
     |  
     |  __iter__(self)
     |      Return an iterator over the dataset.
     |      
     |      This iterator returns as many samples as given by the `length` parameter.
     |  
     |  __len__(self)
     |      Return the length of the dataset.
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
    
    class Composable(builtins.object)
     |  A mixin implementing composability of data pipelines.
     |  
     |  Methods defined here:
     |  
     |  __init__(self)
     |      Initialize the composable mixin.
     |  
     |  compose(self, constructor, *args, **kw)
     |      Compose this processor with another IterableDataset.
     |      
     |      The constructor should be of the form `__init__(self, source_dataset, ...)`
     |  
     |  source_(self, source)
     |      Set the source for this dataset.
     |      
     |      :param source: source dataset, should be an IterableDataset instance
     |  
     |  then(self, f, *args, length=True, **kw)
     |      Compose this processor with a new processor defined by a function.
     |      
     |      The function is of the form:
     |      
     |          def my_process(source, ...):
     |              for sample in source:
     |                  ...
     |                  result = ...
     |                  yield result
     |  
     |  ----------------------------------------------------------------------
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class DatasetTest(torch.utils.data.dataset.IterableDataset, Composable, Shorthands)
     |  DatasetTest(*args, **kwds)
     |  
     |  Perform final checks on an IterableDataset and permit easy mock tests.
     |  
     |  This is the implementation of the `Shorthands.test` method; you usually
     |  do not need to construct it explicitly.
     |  
     |  Method resolution order:
     |      DatasetTest
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      typing.Generic
     |      Composable
     |      Shorthands
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, length=None, checker=None, mock_sample=None, mock_length=10000, mock=False)
     |      Create a DatasetTest.
     |      
     |      :param length: length of the dataset
     |      :param checker: any kind of final checking function you want to run over samples
     |      :param mock_sample: mock sample
     |      :param mock_length: size of mocked dataset
     |      :param mock: turning mocking on/off
     |  
     |  __iter__(self)
     |      Return an iterator either over the mock object or the underlying dataset.
     |  
     |  __len__(self)
     |      Return the length of the test object.
     |      
     |      This is either the length of the mock object when in mock mode,
     |      otherwise the length of the underlying dataset/data loader.
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
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Composable:
     |  
     |  compose(self, constructor, *args, **kw)
     |      Compose this processor with another IterableDataset.
     |      
     |      The constructor should be of the form `__init__(self, source_dataset, ...)`
     |  
     |  source_(self, source)
     |      Set the source for this dataset.
     |      
     |      :param source: source dataset, should be an IterableDataset instance
     |  
     |  then(self, f, *args, length=True, **kw)
     |      Compose this processor with a new processor defined by a function.
     |      
     |      The function is of the form:
     |      
     |          def my_process(source, ...):
     |              for sample in source:
     |                  ...
     |                  result = ...
     |                  yield result
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Shorthands:
     |  
     |  associate(self, associator)
     |      Slice the stream of training samples.
     |      
     |      Associates information from the associator with the current sample.
     |      The associator should either be a function or a hash table. It is
     |      invoked with the sample key as an argument and must return a dictionary
     |      of information that is merged with the sample.
     |      
     |      :param associator: callable or dictionary-like object
     |  
     |  batched(self, batchsize, collation_fn=<function default_collation_fn at 0x7f8d75246dc0>, partial=True)
     |      Compute batches for the given dataset.
     |      
     |      :param batchsize: desired batchsize
     |      :param collation_fn: collation function to turn list of objects into batches
     |      :param partial: return partial batches
     |  
     |  dbcache(self, fname, size)
     |      Cache training samples in an SQLite database.
     |      
     |      This is useful for testing and for running validation tests.
     |      
     |      :param fname: filename for the sqlite database
     |      :param size: number of samples to be cached
     |  
     |  ddp_equalize(self, length)
     |      Equalize number of training samples in DistributedDataParallel training.
     |      
     |      Torch's DistributedDataParallel requires the same number of samples in
     |      all participating compute nodes.
     |      
     |      Use with `loader = loader.ddp_equalize(number_of_batches)`
     |      
     |      
     |      You need to specify the number of batches you want to equalize to.
     |      This is usually the number of samples in the dataset divided by the batch size.
     |      
     |      :param length: number of batches in the dataset
     |  
     |  decode(self, *args, pre=None, post=None, only=None, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Decode samples.
     |      
     |      This is a special form of mapping over samples given as dicts.
     |      A list of all decoders is formed from `pre + args + post`.
     |      For each dict entry, the decoders on that list are invoked in sequence
     |      until one of them decodes the sample. That decoded value is then stored
     |      in the dictionary and the next dictionary entry is decoded.
     |      
     |      The `pre` and `post` decoder lists are set to common defaults (including `.gz` decoding).
     |      You can specify decoders for your application in the `args` argument.
     |      String arguments like "pil" are a shorthand for image decoder functions like
     |      `webdataset.imagehandler("pil")`. All other decoders must be specified as
     |      functions.
     |      
     |      :param args: list of decoder functions; a string like "pil" is a shorthand for common image decoders
     |      :param pre: a list of decoder functions that is always carried out before args
     |      :param post: a list of decoder functions that is always carried out after args
     |      :param only: limit decoding to the list of these fields
     |      :param handler: exception handler
     |  
     |  map(self, f, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Map a function over a stream of samples.
     |      
     |      This may be a tuple stream or a stream of dicts.
     |      
     |      :param f: The function to be mapped.
     |      :param handler: The exception handling strategy.
     |  
     |  map_dict(self, handler=<function reraise_exception at 0x7f8d7524a820>, **kw)
     |      Map the fields of a dictionary.
     |      
     |      :param handler: exeption handler
     |      :param kw: list of key=function mappers
     |  
     |  map_tuple(self, *args, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Map a tuple.
     |      
     |      :param args: List of functions corresponding to the fields of the tuple.
     |      :param handler: exception handler
     |  
     |  rename(self, handler=<function reraise_exception at 0x7f8d7524a820>, **kw)
     |      Rename fields in a dictionary based sample.
     |      
     |      This works on dictionary input streams. A keyword argument like
     |      `new="old"` renames extension/key "old" to "new".
     |      
     |      :param handler: exception handler
     |      :param kw: list of renames
     |  
     |  repeat(self, nepochs=None, nbatches=None, nsamples=None, batchsize=<function guess_batchsize at 0x7f8d752463a0>)
     |      Repeat samples from the source dataset iterator.
     |      
     |      With no arguments, repeat infinitely.
     |      
     |      :param nepochs: maximum number of epochs
     |      :param nbatches: maximum number of batches
     |      :param nsamples: maximum number of samples
     |      :param batchsize: integer giving batchsize, or function to compute it
     |  
     |  rsample(self, p=0.5)
     |      Randomly subsample a stream of samples.
     |      
     |      :param args: probability of including a sample in the output stream.
     |  
     |  select(self, predicate, **kw)
     |      Select samples matching some predicate.
     |      
     |      :param predicate: predicate used to select samples
     |  
     |  shuffle(self, size, **kw)
     |      Shuffle the dataset using an internal shuffle buffer.
     |      
     |      This will buffer up `initial` samples. Then it will add new samples to
     |      the internal buffer and return random samples from the buffer, simultaneously
     |      filling up the buffer to the given size.
     |      
     |      Using initial < size will result in less initial randomness but faster
     |      startups.
     |      
     |      :param size: size of the shuffle buffer
     |      :param initial: buffer this many samples before yield training samples
     |      :param handler: The exception handling strategy.
     |      :param kw: other keywords for iterators.shuffle
     |  
     |  slice(self, *args)
     |      Slice the stream of training samples.
     |      
     |      This takes the usual islice arguments of ([start], stop, [step])
     |      
     |      :param args: arguments to itertools.islice
     |  
     |  test(self, length=None, checker=None, mock_sample=None, mock_length=None, mock=False)
     |      A quick and simple way of switching to a mock dataset at the end of a pipeline.
     |      
     |      Use with `loader = loader.test(mock_sample=..., mock_length=...)
     |      You can turn on mocking with `loader.mock = True`
     |      
     |      :param length: length of the dataset
     |      :param checker: any kind of final checking function you want to run over samples
     |      :param mock_sample: mock sample
     |      :param mock_length: size of mocked dataset
     |      :param mock: turning mocking on/off
     |  
     |  to_tuple(self, *args, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Convert a dictionary-based sample to a tuple.
     |      
     |      Field names to be extracted can be specified as a Python list
     |      or as a string. "__key__ jpg;png cls" will extract a triple, with the
     |      first entry being the key, the second being a JPEG or PNG image, and
     |      the third being the contents of the cls file.
     |      
     |      :param args: field names
     |      :param handler: exception handler
     |  
     |  unbatched(self, length=None)
     |      Take a stream of batches and turn it back into a stream of samples.
     |      
     |      :param length: user-supplied length for the unbatched dataset.
    
    class MockDataset(torch.utils.data.dataset.IterableDataset)
     |  MockDataset(*args, **kwds)
     |  
     |  MockDataset.
     |  
     |  A mock dataset for performance testing and unit testing.
     |  
     |  Method resolution order:
     |      MockDataset
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      typing.Generic
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, sample, length)
     |      Create a mock dataset instance.
     |      
     |      :param sample: the sample to be returned repeatedly
     |      :param length: the length of the mock dataset
     |  
     |  __iter__(self)
     |      Return an iterator over this mock dataset.
     |  
     |  __len__(self)
     |      Return the length of this mock dataset.
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
    
    class Processor(torch.utils.data.dataset.IterableDataset, Composable, Shorthands)
     |  Processor(*args, **kwds)
     |  
     |  A class that turns a function into an IterableDataset.
     |  
     |  Method resolution order:
     |      Processor
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      typing.Generic
     |      Composable
     |      Shorthands
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, source, f, *args, _kwa={}, length=True, **kw)
     |      Create a processor.
     |      
     |      The function should take an iterator as an argument and yield
     |      processed samples. The function is invoked as `f(source, *args, **kw)`.
     |      
     |      The `length` can be specified as `True`, in which case the value
     |      is taken from the source dataset, as a callable, in which case
     |      the length is the result of applying the callable to the source
     |      dataset, or as an integer, in which case the length returned by
     |      `__len__` is that integer.
     |      
     |      :param source: source dataset, an IterableDataset
     |      :param f: function implementing the processor
     |      :param args: extra arguments to the processor after the source iterator
     |      :param _kwa: keyword arguments
     |      :param length: specified length for the output
     |      :param kw: extra keyword arguments
     |  
     |  __iter__(self)
     |      Return an iterator over the source dataset processed by the given function.
     |  
     |  __len__(self)
     |      Return the length of this dataset; see above how this is computed.
     |  
     |  source_(self, source)
     |      Set the source dataset.
     |      
     |      :param source: source dataset
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
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Composable:
     |  
     |  compose(self, constructor, *args, **kw)
     |      Compose this processor with another IterableDataset.
     |      
     |      The constructor should be of the form `__init__(self, source_dataset, ...)`
     |  
     |  then(self, f, *args, length=True, **kw)
     |      Compose this processor with a new processor defined by a function.
     |      
     |      The function is of the form:
     |      
     |          def my_process(source, ...):
     |              for sample in source:
     |                  ...
     |                  result = ...
     |                  yield result
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Shorthands:
     |  
     |  associate(self, associator)
     |      Slice the stream of training samples.
     |      
     |      Associates information from the associator with the current sample.
     |      The associator should either be a function or a hash table. It is
     |      invoked with the sample key as an argument and must return a dictionary
     |      of information that is merged with the sample.
     |      
     |      :param associator: callable or dictionary-like object
     |  
     |  batched(self, batchsize, collation_fn=<function default_collation_fn at 0x7f8d75246dc0>, partial=True)
     |      Compute batches for the given dataset.
     |      
     |      :param batchsize: desired batchsize
     |      :param collation_fn: collation function to turn list of objects into batches
     |      :param partial: return partial batches
     |  
     |  dbcache(self, fname, size)
     |      Cache training samples in an SQLite database.
     |      
     |      This is useful for testing and for running validation tests.
     |      
     |      :param fname: filename for the sqlite database
     |      :param size: number of samples to be cached
     |  
     |  ddp_equalize(self, length)
     |      Equalize number of training samples in DistributedDataParallel training.
     |      
     |      Torch's DistributedDataParallel requires the same number of samples in
     |      all participating compute nodes.
     |      
     |      Use with `loader = loader.ddp_equalize(number_of_batches)`
     |      
     |      
     |      You need to specify the number of batches you want to equalize to.
     |      This is usually the number of samples in the dataset divided by the batch size.
     |      
     |      :param length: number of batches in the dataset
     |  
     |  decode(self, *args, pre=None, post=None, only=None, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Decode samples.
     |      
     |      This is a special form of mapping over samples given as dicts.
     |      A list of all decoders is formed from `pre + args + post`.
     |      For each dict entry, the decoders on that list are invoked in sequence
     |      until one of them decodes the sample. That decoded value is then stored
     |      in the dictionary and the next dictionary entry is decoded.
     |      
     |      The `pre` and `post` decoder lists are set to common defaults (including `.gz` decoding).
     |      You can specify decoders for your application in the `args` argument.
     |      String arguments like "pil" are a shorthand for image decoder functions like
     |      `webdataset.imagehandler("pil")`. All other decoders must be specified as
     |      functions.
     |      
     |      :param args: list of decoder functions; a string like "pil" is a shorthand for common image decoders
     |      :param pre: a list of decoder functions that is always carried out before args
     |      :param post: a list of decoder functions that is always carried out after args
     |      :param only: limit decoding to the list of these fields
     |      :param handler: exception handler
     |  
     |  map(self, f, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Map a function over a stream of samples.
     |      
     |      This may be a tuple stream or a stream of dicts.
     |      
     |      :param f: The function to be mapped.
     |      :param handler: The exception handling strategy.
     |  
     |  map_dict(self, handler=<function reraise_exception at 0x7f8d7524a820>, **kw)
     |      Map the fields of a dictionary.
     |      
     |      :param handler: exeption handler
     |      :param kw: list of key=function mappers
     |  
     |  map_tuple(self, *args, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Map a tuple.
     |      
     |      :param args: List of functions corresponding to the fields of the tuple.
     |      :param handler: exception handler
     |  
     |  rename(self, handler=<function reraise_exception at 0x7f8d7524a820>, **kw)
     |      Rename fields in a dictionary based sample.
     |      
     |      This works on dictionary input streams. A keyword argument like
     |      `new="old"` renames extension/key "old" to "new".
     |      
     |      :param handler: exception handler
     |      :param kw: list of renames
     |  
     |  repeat(self, nepochs=None, nbatches=None, nsamples=None, batchsize=<function guess_batchsize at 0x7f8d752463a0>)
     |      Repeat samples from the source dataset iterator.
     |      
     |      With no arguments, repeat infinitely.
     |      
     |      :param nepochs: maximum number of epochs
     |      :param nbatches: maximum number of batches
     |      :param nsamples: maximum number of samples
     |      :param batchsize: integer giving batchsize, or function to compute it
     |  
     |  rsample(self, p=0.5)
     |      Randomly subsample a stream of samples.
     |      
     |      :param args: probability of including a sample in the output stream.
     |  
     |  select(self, predicate, **kw)
     |      Select samples matching some predicate.
     |      
     |      :param predicate: predicate used to select samples
     |  
     |  shuffle(self, size, **kw)
     |      Shuffle the dataset using an internal shuffle buffer.
     |      
     |      This will buffer up `initial` samples. Then it will add new samples to
     |      the internal buffer and return random samples from the buffer, simultaneously
     |      filling up the buffer to the given size.
     |      
     |      Using initial < size will result in less initial randomness but faster
     |      startups.
     |      
     |      :param size: size of the shuffle buffer
     |      :param initial: buffer this many samples before yield training samples
     |      :param handler: The exception handling strategy.
     |      :param kw: other keywords for iterators.shuffle
     |  
     |  slice(self, *args)
     |      Slice the stream of training samples.
     |      
     |      This takes the usual islice arguments of ([start], stop, [step])
     |      
     |      :param args: arguments to itertools.islice
     |  
     |  test(self, length=None, checker=None, mock_sample=None, mock_length=None, mock=False)
     |      A quick and simple way of switching to a mock dataset at the end of a pipeline.
     |      
     |      Use with `loader = loader.test(mock_sample=..., mock_length=...)
     |      You can turn on mocking with `loader.mock = True`
     |      
     |      :param length: length of the dataset
     |      :param checker: any kind of final checking function you want to run over samples
     |      :param mock_sample: mock sample
     |      :param mock_length: size of mocked dataset
     |      :param mock: turning mocking on/off
     |  
     |  to_tuple(self, *args, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Convert a dictionary-based sample to a tuple.
     |      
     |      Field names to be extracted can be specified as a Python list
     |      or as a string. "__key__ jpg;png cls" will extract a triple, with the
     |      first entry being the key, the second being a JPEG or PNG image, and
     |      the third being the contents of the cls file.
     |      
     |      :param args: field names
     |      :param handler: exception handler
     |  
     |  unbatched(self, length=None)
     |      Take a stream of batches and turn it back into a stream of samples.
     |      
     |      :param length: user-supplied length for the unbatched dataset.
    
    class Repeatedly(torch.utils.data.dataset.IterableDataset, Composable, Shorthands)
     |  Repeatedly(*args, **kwds)
     |  
     |  Repeatedly yield samples from a dataset.
     |  
     |  Method resolution order:
     |      Repeatedly
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      typing.Generic
     |      Composable
     |      Shorthands
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, nepochs=None, nbatches=None, nsamples=None, batchsize=None, length=None)
     |      Create an instance of Repeatedly.
     |      
     |      :param nepochs: repeat for a maximum of nepochs
     |      :param nbatches: repeat for a maximum of nbatches
     |      :param nsamples: repeat for a maximum of nsamples (requires batchsize)
     |      :param batchsize: integer or function of sample returning batch size
     |  
     |  __iter__(self)
     |      Return an iterator that iterates repeatedly over a source.
     |  
     |  __len__(self)
     |      Return the length of the source.
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
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Composable:
     |  
     |  compose(self, constructor, *args, **kw)
     |      Compose this processor with another IterableDataset.
     |      
     |      The constructor should be of the form `__init__(self, source_dataset, ...)`
     |  
     |  source_(self, source)
     |      Set the source for this dataset.
     |      
     |      :param source: source dataset, should be an IterableDataset instance
     |  
     |  then(self, f, *args, length=True, **kw)
     |      Compose this processor with a new processor defined by a function.
     |      
     |      The function is of the form:
     |      
     |          def my_process(source, ...):
     |              for sample in source:
     |                  ...
     |                  result = ...
     |                  yield result
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Shorthands:
     |  
     |  associate(self, associator)
     |      Slice the stream of training samples.
     |      
     |      Associates information from the associator with the current sample.
     |      The associator should either be a function or a hash table. It is
     |      invoked with the sample key as an argument and must return a dictionary
     |      of information that is merged with the sample.
     |      
     |      :param associator: callable or dictionary-like object
     |  
     |  batched(self, batchsize, collation_fn=<function default_collation_fn at 0x7f8d75246dc0>, partial=True)
     |      Compute batches for the given dataset.
     |      
     |      :param batchsize: desired batchsize
     |      :param collation_fn: collation function to turn list of objects into batches
     |      :param partial: return partial batches
     |  
     |  dbcache(self, fname, size)
     |      Cache training samples in an SQLite database.
     |      
     |      This is useful for testing and for running validation tests.
     |      
     |      :param fname: filename for the sqlite database
     |      :param size: number of samples to be cached
     |  
     |  ddp_equalize(self, length)
     |      Equalize number of training samples in DistributedDataParallel training.
     |      
     |      Torch's DistributedDataParallel requires the same number of samples in
     |      all participating compute nodes.
     |      
     |      Use with `loader = loader.ddp_equalize(number_of_batches)`
     |      
     |      
     |      You need to specify the number of batches you want to equalize to.
     |      This is usually the number of samples in the dataset divided by the batch size.
     |      
     |      :param length: number of batches in the dataset
     |  
     |  decode(self, *args, pre=None, post=None, only=None, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Decode samples.
     |      
     |      This is a special form of mapping over samples given as dicts.
     |      A list of all decoders is formed from `pre + args + post`.
     |      For each dict entry, the decoders on that list are invoked in sequence
     |      until one of them decodes the sample. That decoded value is then stored
     |      in the dictionary and the next dictionary entry is decoded.
     |      
     |      The `pre` and `post` decoder lists are set to common defaults (including `.gz` decoding).
     |      You can specify decoders for your application in the `args` argument.
     |      String arguments like "pil" are a shorthand for image decoder functions like
     |      `webdataset.imagehandler("pil")`. All other decoders must be specified as
     |      functions.
     |      
     |      :param args: list of decoder functions; a string like "pil" is a shorthand for common image decoders
     |      :param pre: a list of decoder functions that is always carried out before args
     |      :param post: a list of decoder functions that is always carried out after args
     |      :param only: limit decoding to the list of these fields
     |      :param handler: exception handler
     |  
     |  map(self, f, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Map a function over a stream of samples.
     |      
     |      This may be a tuple stream or a stream of dicts.
     |      
     |      :param f: The function to be mapped.
     |      :param handler: The exception handling strategy.
     |  
     |  map_dict(self, handler=<function reraise_exception at 0x7f8d7524a820>, **kw)
     |      Map the fields of a dictionary.
     |      
     |      :param handler: exeption handler
     |      :param kw: list of key=function mappers
     |  
     |  map_tuple(self, *args, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Map a tuple.
     |      
     |      :param args: List of functions corresponding to the fields of the tuple.
     |      :param handler: exception handler
     |  
     |  rename(self, handler=<function reraise_exception at 0x7f8d7524a820>, **kw)
     |      Rename fields in a dictionary based sample.
     |      
     |      This works on dictionary input streams. A keyword argument like
     |      `new="old"` renames extension/key "old" to "new".
     |      
     |      :param handler: exception handler
     |      :param kw: list of renames
     |  
     |  repeat(self, nepochs=None, nbatches=None, nsamples=None, batchsize=<function guess_batchsize at 0x7f8d752463a0>)
     |      Repeat samples from the source dataset iterator.
     |      
     |      With no arguments, repeat infinitely.
     |      
     |      :param nepochs: maximum number of epochs
     |      :param nbatches: maximum number of batches
     |      :param nsamples: maximum number of samples
     |      :param batchsize: integer giving batchsize, or function to compute it
     |  
     |  rsample(self, p=0.5)
     |      Randomly subsample a stream of samples.
     |      
     |      :param args: probability of including a sample in the output stream.
     |  
     |  select(self, predicate, **kw)
     |      Select samples matching some predicate.
     |      
     |      :param predicate: predicate used to select samples
     |  
     |  shuffle(self, size, **kw)
     |      Shuffle the dataset using an internal shuffle buffer.
     |      
     |      This will buffer up `initial` samples. Then it will add new samples to
     |      the internal buffer and return random samples from the buffer, simultaneously
     |      filling up the buffer to the given size.
     |      
     |      Using initial < size will result in less initial randomness but faster
     |      startups.
     |      
     |      :param size: size of the shuffle buffer
     |      :param initial: buffer this many samples before yield training samples
     |      :param handler: The exception handling strategy.
     |      :param kw: other keywords for iterators.shuffle
     |  
     |  slice(self, *args)
     |      Slice the stream of training samples.
     |      
     |      This takes the usual islice arguments of ([start], stop, [step])
     |      
     |      :param args: arguments to itertools.islice
     |  
     |  test(self, length=None, checker=None, mock_sample=None, mock_length=None, mock=False)
     |      A quick and simple way of switching to a mock dataset at the end of a pipeline.
     |      
     |      Use with `loader = loader.test(mock_sample=..., mock_length=...)
     |      You can turn on mocking with `loader.mock = True`
     |      
     |      :param length: length of the dataset
     |      :param checker: any kind of final checking function you want to run over samples
     |      :param mock_sample: mock sample
     |      :param mock_length: size of mocked dataset
     |      :param mock: turning mocking on/off
     |  
     |  to_tuple(self, *args, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Convert a dictionary-based sample to a tuple.
     |      
     |      Field names to be extracted can be specified as a Python list
     |      or as a string. "__key__ jpg;png cls" will extract a triple, with the
     |      first entry being the key, the second being a JPEG or PNG image, and
     |      the third being the contents of the cls file.
     |      
     |      :param args: field names
     |      :param handler: exception handler
     |  
     |  unbatched(self, length=None)
     |      Take a stream of batches and turn it back into a stream of samples.
     |      
     |      :param length: user-supplied length for the unbatched dataset.
    
    class ResampledShards(torch.utils.data.dataset.IterableDataset, Composable)
     |  ResampledShards(*args, **kwds)
     |  
     |  An iterable dataset yielding a list of urls.
     |  
     |  Method resolution order:
     |      ResampledShards
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      typing.Generic
     |      Composable
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, urls, nshards=9223372036854775807, length=None)
     |      Sample shards from the shard list with replacement.
     |      
     |      :param urls: a list of URLs as a Python list or brace notation string
     |  
     |  __iter__(self)
     |      Return an iterator over the shards.
     |  
     |  __len__(self)
     |      Return the user-specified length of this dataset.
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
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Composable:
     |  
     |  compose(self, constructor, *args, **kw)
     |      Compose this processor with another IterableDataset.
     |      
     |      The constructor should be of the form `__init__(self, source_dataset, ...)`
     |  
     |  source_(self, source)
     |      Set the source for this dataset.
     |      
     |      :param source: source dataset, should be an IterableDataset instance
     |  
     |  then(self, f, *args, length=True, **kw)
     |      Compose this processor with a new processor defined by a function.
     |      
     |      The function is of the form:
     |      
     |          def my_process(source, ...):
     |              for sample in source:
     |                  ...
     |                  result = ...
     |                  yield result
    
    ResizedDataset = class ChoppedDataset(torch.utils.data.dataset.IterableDataset)
     |  ResizedDataset(*args, **kwds)
     |  
     |  Change the actual and nominal length of an IterableDataset.
     |  
     |  This will continuously iterate through the original dataset, but
     |  impose new epoch boundaries at the given length/nominal.
     |  This exists mainly as a workaround for the odd logic in DataLoader.
     |  It is also useful for choosing smaller nominal epoch sizes with
     |  very large datasets.
     |  
     |  Method resolution order:
     |      ChoppedDataset
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      typing.Generic
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __getstate__(self)
     |      Return the pickled state of the dataset.
     |      
     |      This resets the dataset iterator, since that can't be pickled.
     |  
     |  __init__(self, dataset, length=None, nominal=None)
     |      Create a ChoppedDataset.
     |      
     |      :param dataset: IterableDataset
     |      :param length: declared length of the dataset
     |      :param nominal: nominal length of dataset (if different from declared)
     |  
     |  __iter__(self)
     |      Return an iterator over the dataset.
     |      
     |      This iterator returns as many samples as given by the `length` parameter.
     |  
     |  __len__(self)
     |      Return the length of the dataset.
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
    
    class ShardList(torch.utils.data.dataset.IterableDataset, Composable)
     |  ShardList(*args, **kwds)
     |  
     |  An iterable dataset yielding a list of urls.
     |  
     |  Method resolution order:
     |      ShardList
     |      torch.utils.data.dataset.IterableDataset
     |      torch.utils.data.dataset.Dataset
     |      typing.Generic
     |      Composable
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, urls, shuffle=False, nodesplitter=True, splitter=True, length=None)
     |      Create a ShardList.
     |      
     |      :param urls: a list of URLs as a Python list or brace notation string
     |      :param shuffle: whether to shuffle the URLs
     |      :param nodesplitter: function for splitting urls across nodes (None: don't split)
     |      :param splitter: function for splitting urls across workers (None: don't split)
     |      :param length: user-specified length; this is returned unchanged by the len() function
     |  
     |  __iter__(self)
     |      Return an iterator over the shards.
     |  
     |  __len__(self)
     |      Return the user-specified length of this dataset.
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
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from Composable:
     |  
     |  compose(self, constructor, *args, **kw)
     |      Compose this processor with another IterableDataset.
     |      
     |      The constructor should be of the form `__init__(self, source_dataset, ...)`
     |  
     |  source_(self, source)
     |      Set the source for this dataset.
     |      
     |      :param source: source dataset, should be an IterableDataset instance
     |  
     |  then(self, f, *args, length=True, **kw)
     |      Compose this processor with a new processor defined by a function.
     |      
     |      The function is of the form:
     |      
     |          def my_process(source, ...):
     |              for sample in source:
     |                  ...
     |                  result = ...
     |                  yield result
    
    class Shorthands(builtins.object)
     |  A convenient set of shorthands for common data transformations.
     |  
     |  Methods defined here:
     |  
     |  associate(self, associator)
     |      Slice the stream of training samples.
     |      
     |      Associates information from the associator with the current sample.
     |      The associator should either be a function or a hash table. It is
     |      invoked with the sample key as an argument and must return a dictionary
     |      of information that is merged with the sample.
     |      
     |      :param associator: callable or dictionary-like object
     |  
     |  batched(self, batchsize, collation_fn=<function default_collation_fn at 0x7f8d75246dc0>, partial=True)
     |      Compute batches for the given dataset.
     |      
     |      :param batchsize: desired batchsize
     |      :param collation_fn: collation function to turn list of objects into batches
     |      :param partial: return partial batches
     |  
     |  dbcache(self, fname, size)
     |      Cache training samples in an SQLite database.
     |      
     |      This is useful for testing and for running validation tests.
     |      
     |      :param fname: filename for the sqlite database
     |      :param size: number of samples to be cached
     |  
     |  ddp_equalize(self, length)
     |      Equalize number of training samples in DistributedDataParallel training.
     |      
     |      Torch's DistributedDataParallel requires the same number of samples in
     |      all participating compute nodes.
     |      
     |      Use with `loader = loader.ddp_equalize(number_of_batches)`
     |      
     |      
     |      You need to specify the number of batches you want to equalize to.
     |      This is usually the number of samples in the dataset divided by the batch size.
     |      
     |      :param length: number of batches in the dataset
     |  
     |  decode(self, *args, pre=None, post=None, only=None, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Decode samples.
     |      
     |      This is a special form of mapping over samples given as dicts.
     |      A list of all decoders is formed from `pre + args + post`.
     |      For each dict entry, the decoders on that list are invoked in sequence
     |      until one of them decodes the sample. That decoded value is then stored
     |      in the dictionary and the next dictionary entry is decoded.
     |      
     |      The `pre` and `post` decoder lists are set to common defaults (including `.gz` decoding).
     |      You can specify decoders for your application in the `args` argument.
     |      String arguments like "pil" are a shorthand for image decoder functions like
     |      `webdataset.imagehandler("pil")`. All other decoders must be specified as
     |      functions.
     |      
     |      :param args: list of decoder functions; a string like "pil" is a shorthand for common image decoders
     |      :param pre: a list of decoder functions that is always carried out before args
     |      :param post: a list of decoder functions that is always carried out after args
     |      :param only: limit decoding to the list of these fields
     |      :param handler: exception handler
     |  
     |  map(self, f, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Map a function over a stream of samples.
     |      
     |      This may be a tuple stream or a stream of dicts.
     |      
     |      :param f: The function to be mapped.
     |      :param handler: The exception handling strategy.
     |  
     |  map_dict(self, handler=<function reraise_exception at 0x7f8d7524a820>, **kw)
     |      Map the fields of a dictionary.
     |      
     |      :param handler: exeption handler
     |      :param kw: list of key=function mappers
     |  
     |  map_tuple(self, *args, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Map a tuple.
     |      
     |      :param args: List of functions corresponding to the fields of the tuple.
     |      :param handler: exception handler
     |  
     |  rename(self, handler=<function reraise_exception at 0x7f8d7524a820>, **kw)
     |      Rename fields in a dictionary based sample.
     |      
     |      This works on dictionary input streams. A keyword argument like
     |      `new="old"` renames extension/key "old" to "new".
     |      
     |      :param handler: exception handler
     |      :param kw: list of renames
     |  
     |  repeat(self, nepochs=None, nbatches=None, nsamples=None, batchsize=<function guess_batchsize at 0x7f8d752463a0>)
     |      Repeat samples from the source dataset iterator.
     |      
     |      With no arguments, repeat infinitely.
     |      
     |      :param nepochs: maximum number of epochs
     |      :param nbatches: maximum number of batches
     |      :param nsamples: maximum number of samples
     |      :param batchsize: integer giving batchsize, or function to compute it
     |  
     |  rsample(self, p=0.5)
     |      Randomly subsample a stream of samples.
     |      
     |      :param args: probability of including a sample in the output stream.
     |  
     |  select(self, predicate, **kw)
     |      Select samples matching some predicate.
     |      
     |      :param predicate: predicate used to select samples
     |  
     |  shuffle(self, size, **kw)
     |      Shuffle the dataset using an internal shuffle buffer.
     |      
     |      This will buffer up `initial` samples. Then it will add new samples to
     |      the internal buffer and return random samples from the buffer, simultaneously
     |      filling up the buffer to the given size.
     |      
     |      Using initial < size will result in less initial randomness but faster
     |      startups.
     |      
     |      :param size: size of the shuffle buffer
     |      :param initial: buffer this many samples before yield training samples
     |      :param handler: The exception handling strategy.
     |      :param kw: other keywords for iterators.shuffle
     |  
     |  slice(self, *args)
     |      Slice the stream of training samples.
     |      
     |      This takes the usual islice arguments of ([start], stop, [step])
     |      
     |      :param args: arguments to itertools.islice
     |  
     |  test(self, length=None, checker=None, mock_sample=None, mock_length=None, mock=False)
     |      A quick and simple way of switching to a mock dataset at the end of a pipeline.
     |      
     |      Use with `loader = loader.test(mock_sample=..., mock_length=...)
     |      You can turn on mocking with `loader.mock = True`
     |      
     |      :param length: length of the dataset
     |      :param checker: any kind of final checking function you want to run over samples
     |      :param mock_sample: mock sample
     |      :param mock_length: size of mocked dataset
     |      :param mock: turning mocking on/off
     |  
     |  to_tuple(self, *args, handler=<function reraise_exception at 0x7f8d7524a820>)
     |      Convert a dictionary-based sample to a tuple.
     |      
     |      Field names to be extracted can be specified as a Python list
     |      or as a string. "__key__ jpg;png cls" will extract a triple, with the
     |      first entry being the key, the second being a JPEG or PNG image, and
     |      the third being the contents of the cls file.
     |      
     |      :param args: field names
     |      :param handler: exception handler
     |  
     |  unbatched(self, length=None)
     |      Take a stream of batches and turn it back into a stream of samples.
     |      
     |      :param length: user-supplied length for the unbatched dataset.
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
    WebDataset(urls, shardshuffle=True, cache_dir='', cache_size=1000000000000000, cache_name=<function shard_uuid at 0x7f8d7524a1f0>, cache_verbose=1, multimode=None, splitter=True, nodesplitter=True, handler=<function reraise_exception at 0x7f8d7524a820>, length=None, warn_empty=True)
        Return a pipeline for WebDataset-style data files.
        
        This is a convenience function for constructing a partial pipeline
        that reads from a set of sharded tar files, extracts the individual
        files, and groups them together into samples (dictionaries).
        
        You can use all the methods from `Composable` (`then`, `compose`) and
        from `Shorthands` (`batched`, `unbatched`, `decode`, `shuffle`, etc.)
        on the result.
        
        The `multimode` argument determines how to handle shard splitting across
        different nodes and workers:
        
        - None: split shards based on node/worker
        - "nodeworker": split shards both by node and worker
        - "worker": split shards by worker only (all shards on each node)
        - "resampled": infinite stream of samples, with all shards on all nodes
        - "sliced": all shards on all nodes, but split by samples
        
        :param urls: the source URLs, specified either as a list or as a brace-expanded string
        :param multimode: how to handle multimode processing
        :param shardshuffle: boolean indicating whether the shards should be shuffled or not
        :param splitter: a function called for splitting shards among workers (True: PyTorch default, None: no splitting)
        :param nodesplitter: a function called for splitting shards among nodes (True: PyTOrch default, None: no splitting)
        :param handler: an error handler
        :param length: the length of this dataset, should be an integer
        :param cache_dir: when set, caches shards in this directory
        :param cache_size: when set, specifies a maximum size for the shard cache
        :param cache_name: when set, specifies how shards should be named in the cache
        :param cache_verbose: when set, prints information about caching
        :param warn_empty: warn when no samples are generated at all
    
    WebLoader(*args, **kw)
        Return a small wrapper around torch.utils.data.DataLoader.
        
        This wrapper works identically to the original `DataLoader`, but adds
        alls the convenience functions and filters for WebDataset.
        
        You can use all the methods from `Composable` (`then`, `compose`) and
        from `Shorthands` (`batched`, `unbatched`, `decode`, `shuffle`, etc.)
        on the result.
        
        :param args: forwarded to `DataLoader`
        :param kw: forwarded to `DataLoader`
    
    warn_no_samples(data)
        Warn if the iterator yields no samples.

DATA
    default_cache_dir = ''
    default_cache_size = 1000000000000000
    default_cache_verbose = 1

FILE
    /home/tmb/proj/webdataset/webdataset/dataset.py



```

# Module `webdataset.handlers`

```
Help on module webdataset.handlers in webdataset:

NAME
    webdataset.handlers - Pluggable exception handlers.

DESCRIPTION
    These are functions that take an exception as an argument and then return...
    
    - the exception (in order to re-raise it)
    - True (in order to continue and ignore the exception)
    - False (in order to ignore the exception and stop processing)
    
    They are used as handler= arguments in much of the library.

FUNCTIONS
    ignore_and_continue(exn)
        Call in an exception handler to ignore any exception and continue.
    
    ignore_and_stop(exn)
        Call in an exception handler to ignore any exception and stop further processing.
    
    reraise_exception(exn)
        Call in an exception handler to re-raise the exception.
    
    warn_and_continue(exn)
        Call in an exception handler to ignore any exception, isssue a warning, and continue.
    
    warn_and_stop(exn)
        Call in an exception handler to ignore any exception and stop further processing.

FILE
    /home/tmb/proj/webdataset/webdataset/handlers.py



```

# Module `webdataset.checks`

```
Help on module webdataset.checks in webdataset:

NAME
    webdataset.checks - A collection of simple runtime checks.

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

# Module `webdataset.__init__`

```
Help on module webdataset.__init__ in webdataset:

NAME
    webdataset.__init__ - Exported globals for webdataset library.

FILE
    /home/tmb/proj/webdataset/webdataset/__init__.py



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
    bytedata(a)
        Return a the raw data corresponding to a.
    
    bytelen(a)
        Determine the length of a in bytes.
    
    check_acceptable_input_type(data, allow64)
        Check that the data has an acceptable type for tensor encoding.
        
        :param data: array
        :param allow64: allow 64 bit types
    
    check_infos(data, infos, required_infos=None)
        Verify the info strings.
    
    decode_buffer(buf, infos=False)
        Decode a byte array into a list of arrays.
    
    decode_chunks(buf)
        Decode a byte array into a list of chunks.
    
    decode_header(h)
        Decode a byte array into an array header.
    
    decode_list(l, infos=False)
        Given a list of byte arrays, decode them into arrays.
    
    encode_buffer(l, infos=None)
        Encode a list of arrays into a single byte array.
    
    encode_chunks(l)
        Encode a list of chunks into a single byte array, with lengths and magics..
    
    encode_header(a, info='')
        Encode an array header as a byte array.
    
    encode_list(l, infos=None)
        Given a list of arrays, encode them into a list of byte arrays.
    
    load(fname, infos=False, nocheck=False)
        Read a list of arrays from a file, with magics, length, and padding.
    
    read(stream, n=9223372036854775807, infos=False)
        Read a list of arrays from a stream, with magics, length, and padding.
    
    read_chunk(stream)
        Read a byte chunk from a stream with magics, length, and padding.
    
    roundup(n, k=64)
        Round up to the next multiple of 64.
    
    save(fname, *args, infos=None, nocheck=False)
        Save a list of arrays to a file, with magics, length, and padding.
    
    str64(s)
        Convert a string to an int64.
    
    unstr64(i)
        Convert an int64 to a string.
    
    write(stream, l, infos=None)
        Write a list of arrays to a stream, with magics, length, and padding.
    
    write_chunk(stream, buf)
        Write a byte chunk to the stream with magics, length, and padding.

DATA
    long_to_short = {'float16': 'f2', 'float32': 'f4', 'float64': 'f8', 'i...
    magic = 9110334830257984638
    magic_bytes = b'~TenBin~'
    magic_str = '~TenBin~'
    short_to_long = {'f2': 'float16', 'f4': 'float32', 'f8': 'float64', 'i...

FILE
    /home/tmb/proj/webdataset/webdataset/tenbin.py



```

# Module `webdataset.bench`

```
Help on module webdataset.bench in webdataset:

NAME
    webdataset.bench - A simple command line program to benchmark I/O speeds.

CLASSES
    builtins.object
        TotalSize
    
    class TotalSize(builtins.object)
     |  Keep track of the total size of samples.
     |  
     |  Methods defined here:
     |  
     |  __call__(self, sample)
     |      Add sample to the counter.
     |      
     |      :param sample: undecoded sample to be added
     |  
     |  __init__(self)
     |      Create a TotalSize counter.
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
        Perform benchmarking.
        
        :param args: argparse result with command line arguments

FILE
    /home/tmb/proj/webdataset/webdataset/bench.py



```

# Module `webdataset.fluid`

```
Help on module webdataset.fluid in webdataset:

NAME
    webdataset.fluid - A deprecated interface to WebDataset.

CLASSES
    torch.utils.data.dataset.IterableDataset(torch.utils.data.dataset.Dataset)
        Dataset
    
    class Dataset(torch.utils.data.dataset.IterableDataset)
     |  Dataset(*args, **kwds)
     |  
     |  This class works almost identically to WebDataset but with internal state.
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
     |      Forward method calls to the underlying WebDataset and update the internal pipe.
     |  
     |  __init__(self, urls, *, length=True, splitter=<function split_by_worker at 0x7f6590352dc0>, handler=<function reraise_exception at 0x7f65903c8820>, shuffle=False, cache_dir='', cache_size=1000000000000000, cache_name=<function shard_uuid at 0x7f65903c81f0>, cache_verbose=1)
     |      Create a Dataset instance. See WebDataset for documentation.
     |  
     |  __iter__(self)
     |      Return an iterator over the underlying dataset.
     |  
     |  __len__(self)
     |      Return the length of the underlying dataset.
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
    default_cache_dir = ''
    default_cache_size = 1000000000000000
    default_cache_verbose = 1

FILE
    /home/tmb/proj/webdataset/webdataset/fluid.py



```

# Module `webdataset.iterators`

```
Help on module webdataset.iterators in webdataset:

NAME
    webdataset.iterators - A collection of iterators for data transformations.

DESCRIPTION
    These functions are plain iterator functions. You can find curried versions
    in webdataset.filters, and you can find IterableDataset wrappers in
    webdataset.processing.

FUNCTIONS
    associate(data, associator, **kw)
        Associate additional data with samples.
    
    batched(data, batchsize=20, collation_fn=<function default_collation_fn at 0x7f90abf6fdc0>, partial=True)
        Create batches of the given size.
        
        :param data: iterator
        :param batchsize: target batch size
        :param tensors: automatically batch lists of ndarrays into ndarrays
        :param partial: return partial batches
        :returns: iterator
    
    compose(*args)
        Compose a sequence of functions (left-to-right).
    
    compose2(f, g)
        Compose two functions, g(f(x)).
    
    decode(data, *args, handler=<function reraise_exception at 0x7f90abf6df70>, **kw)
        Decode data based on the decoding functions given as arguments.
    
    default_collation_fn(samples, combine_tensors=True, combine_scalars=True)
        Take a collection of samples (dictionaries) and create a batch.
        
        If `tensors` is True, `ndarray` objects are combined into
        tensor batches.
        
        :param dict samples: list of samples
        :param bool tensors: whether to turn lists of ndarrays into a single ndarray
        :returns: single sample consisting of a batch
        :rtype: dict
    
    getfirst(a, keys, default=None, missing_is_error=True)
        Get the first matching key from a dictionary.
        
        Keys can be specified as a list, or as a string of keys separated by ';'.
    
    identity(x)
        Return the argument.
    
    info(data, fmt=None, n=3, every=-1, width=50, stream=<_io.TextIOWrapper name='<stderr>' mode='w' encoding='utf-8'>, name='')
        Print information about the samples that are passing through.
        
        :param data: source iterator
        :param fmt: format statement (using sample dict as keyword)
        :param n: when to stop
        :param every: how often to print
        :param width: maximum width
        :param stream: output stream
        :param name: identifier printed before any output
    
    map(data, f, handler=<function reraise_exception at 0x7f90abf6df70>)
        Map samples.
    
    map_dict(data, handler=<function reraise_exception at 0x7f90abf6df70>, **kw)
        Map the entries in a dict sample with individual functions.
    
    map_tuple(data, *args, handler=<function reraise_exception at 0x7f90abf6df70>)
        Map the entries of a tuple with individual functions.
    
    parse_field_spec(fields)
        Parse a specification for a list of fields to be extracted.
        
        Keys are separated by spaces in the spec. Each key can itself
        be composed of key alternatives separated by ';'.
    
    pipeline(source, *args)
        Write an input pipeline; first argument is source, rest are filters.
    
    reduce(...)
        reduce(function, sequence[, initial]) -> value
        
        Apply a function of two arguments cumulatively to the items of a sequence,
        from left to right, so as to reduce the sequence to a single value.
        For example, reduce(lambda x, y: x+y, [1, 2, 3, 4, 5]) calculates
        ((((1+2)+3)+4)+5).  If initial is present, it is placed before the items
        of the sequence in the calculation, and serves as a default when the
        sequence is empty.
    
    rename(data, handler=<function reraise_exception at 0x7f90abf6df70>, **kw)
        Rename samples based on keyword arguments.
    
    reraise_exception(exn)
        Reraises the given exception; used as a handler.
        
        :param exn: exception
    
    rsample(data, p=0.5)
        Randomly subsample a stream of data.
    
    select(data, predicate)
        Select samples based on a predicate.
        
        :param data: source iterator
        :param predicate: predicate (function)
    
    shuffle(data, bufsize=1000, initial=100, rng=<module 'random' from '/usr/lib/python3.8/random.py'>, handler=None)
        Shuffle the data in the stream.
        
        This uses a buffer of size `bufsize`. Shuffling at
        startup is less random; this is traded off against
        yielding samples quickly.
        
        data: iterator
        bufsize: buffer size for shuffling
        returns: iterator
        rng: either random module or random.Random instance
    
    to_tuple(data, *args, handler=<function reraise_exception at 0x7f90abf6df70>)
        Convert dict samples to tuples.
    
    transform_with(sample, transformers)
        Transform a list of values using a list of functions.
        
        sample: list of values
        transformers: list of functions
        
        If there are fewer transformers than inputs, or if a transformer
        function is None, then the identity function is used for the
        corresponding sample fields.
    
    unbatched(data)
        Turn batched data back into unbatched data.

FILE
    /home/tmb/proj/webdataset/webdataset/iterators.py



```

# Module `webdataset.writer`

```
Help on module webdataset.writer in webdataset:

NAME
    webdataset.writer - Classes and functions for writing tar files and WebDataset files.

CLASSES
    builtins.object
        ShardWriter
        TarWriter
    
    class ShardWriter(builtins.object)
     |  ShardWriter(pattern, maxcount=100000, maxsize=3000000000.0, post=None, start_shard=0, **kw)
     |  
     |  Like TarWriter but splits into multiple shards.
     |  
     |  Methods defined here:
     |  
     |  __enter__(self)
     |      Enter context.
     |  
     |  __exit__(self, *args, **kw)
     |      Exit context.
     |  
     |  __init__(self, pattern, maxcount=100000, maxsize=3000000000.0, post=None, start_shard=0, **kw)
     |      Create a ShardWriter.
     |      
     |      :param pattern: output file pattern
     |      :param maxcount: maximum number of records per shard (Default value = 100000)
     |      :param maxsize: maximum size of each shard (Default value = 3e9)
     |      :param kw: other options passed to TarWriter
     |  
     |  close(self)
     |      Close the stream.
     |  
     |  finish(self)
     |      Finish all writing (use close instead).
     |  
     |  next_stream(self)
     |      Close the current stream and move to the next.
     |  
     |  write(self, obj)
     |      Write a sample.
     |      
     |      :param obj: sample to be written
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
     |      Enter context.
     |  
     |  __exit__(self, exc_type, exc_val, exc_tb)
     |      Exit context.
     |  
     |  __init__(self, fileobj, user='bigdata', group='bigdata', mode=292, compress=None, encoder=True, keep_meta=False)
     |      Create a tar writer.
     |      
     |      :param fileobj: stream to write data to
     |      :param user: user for tar files
     |      :param group: group for tar files
     |      :param mode: mode for tar files
     |      :param compress: desired compression
     |      :param encoder: encoder function
     |      :param keep_meta: keep metadata (entries starting with "_")
     |  
     |  close(self)
     |      Close the tar file.
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

FUNCTIONS
    bytestr(data)
        Convert data into a bytestring.
        
        Uses str and ASCII encoding for data that isn't already in string format.
        
        :param data: data
    
    encode_based_on_extension(sample, handlers)
        Encode an entire sample with a collection of handlers.
        
        :param sample: data sample (a dict)
        :param handlers: handlers for encoding
    
    encode_based_on_extension1(data, tname, handlers)
        Encode data based on its extension and a dict of handlers.
        
        :param data: data
        :param tname: file extension
        :param handlers: handlers
    
    imageencoder(image, format='PNG')
        Compress an image using PIL and return it as a string.
        
        Can handle float or uint8 images.
        
        :param image: ndarray representing an image
        :param format: compression format (PNG, JPEG, PPM)
    
    make_encoder(spec)
        Make an encoder function from a specification.
        
        :param spec: specification
    
    make_handlers()
        Create a list of handlers for encoding data.
    
    torch_dumps(data)
        Dump data into a bytestring using torch.dumps.
        
        This delays importing torch until needed.
        
        :param data: data to be dumped

DATA
    default_handlers = {'default': {'class': <function make_handlers.<loca...

FILE
    /home/tmb/proj/webdataset/webdataset/writer.py



```

# Module `webdataset.multi`

```
Help on module webdataset.multi in webdataset:

NAME
    webdataset.multi - An alternative to DataLoader using ZMQ.

DESCRIPTION
    This implements MultiLoader, an alternative to DataLoader when torch
    is not available. Subprocesses communicate with the loader through
    ZMQ, provided for high performance multithreaded queueing.

CLASSES
    builtins.object
        EOF
        MultiLoader
    
    class EOF(builtins.object)
     |  EOF(**kw)
     |  
     |  A class that indicates that a data stream is finished.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, **kw)
     |      Initialize the class with the kw as instance variables.
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
     |  Alternative to PyTorch DataLoader based on ZMQ.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, dataset, workers=4, verbose=False, nokill=False, prefix='/tmp/_multi-')
     |      Create a MultiLoader for a dataset.
     |      
     |      This creates ZMQ sockets, spawns `workers` subprocesses, and has them send data
     |      to the socket.
     |      
     |      :param dataset: source dataset
     |      :param workers: number of workers
     |      :param verbose: report progress verbosely
     |      :param nokill: don't kill old processes when restarting (allows multiple loaders)
     |      :param prefix: directory prefix for the ZMQ socket
     |  
     |  __iter__(self)
     |      Return an iterator over this dataloader.
     |  
     |  kill(self)
     |      kill.
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
        Read samples from the dataset and send them over the socket.
        
        :param dataset: source dataset
        :param sockname: name for the socket to send data to
        :param index: index for this reader, using to indicate EOF

DATA
    all_pids = set()
    the_protocol = 5

FILE
    /home/tmb/proj/webdataset/webdataset/multi.py



```

# Module `webdataset.shardcache`

```
Help on module webdataset.shardcache in webdataset:

NAME
    webdataset.shardcache - Implement caching for shards.

CLASSES
    io.RawIOBase(_io._RawIOBase, io.IOBase)
        CacheStream
    
    class CacheStream(io.RawIOBase)
     |  CacheStream(fname, stream, verbose=False)
     |  
     |  Cache raw IO stream.
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
     |      Create a shard cache.
     |      
     |      :param fname: file name for the cache file
     |      :param stream: stream to be cached
     |      :param verbose: verbose output on progress
     |  
     |  close(self, complete=False)
     |      Close both the cache file and the original stream.
     |      
     |      :param complete: indicate whether the stream was fully read (if not, the cache file is discarded)
     |  
     |  read(self, n)
     |      Read n bytes from the stream and write them to the cache file.
     |      
     |      :param n: number of bytes
     |  
     |  readinto(self, b)
     |      Read data into a buffer.
     |      
     |      :param b: buffer
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
    cache_shards(urls, cache_dir='./data', cache_size=1000000000000000.0, cache_name=<function guess_shard at 0x7f6210c12160>, verbose=False)
        Implement shard caching.
        
        When caching is off, just iterates through the list of shards.
        
        When caching is on (cache_dir is not None), opens each shard with caching
        an returns a dictionary consisting of a URL and a stream.
        
        :param urls: list of URLs
        :param cache_dir: directory used for caching
        :param cache_size: cache size
        :param cache_name: function computing cache names
        :param verbose: verbose caching info
    
    guess_shard(path)
        Guess the shard from a given path.
    
    shard_uuid(path)
        Compute a UUID for a shard path.

FILE
    /home/tmb/proj/webdataset/webdataset/shardcache.py



```

# Module `webdataset.gopen`

```
Help on module webdataset.gopen in webdataset:

NAME
    webdataset.gopen - Open URLs by calling subcommands.

CLASSES
    builtins.object
        Pipe
    
    class Pipe(builtins.object)
     |  Pipe(*args, mode=None, timeout=7200.0, ignore_errors=False, ignore_status=[], **kw)
     |  
     |  Wrapper class for subprocess.Pipe.
     |  
     |  This class looks like a stream from the outside, but it checks
     |  subprocess status and handles timeouts with exceptions.
     |  This way, clients of the class do not need to know that they are
     |  dealing with subprocesses.
     |  
     |  :param *args: passed to `subprocess.Pipe`
     |  :param **kw: passed to `subprocess.Pipe`
     |  :param timeout: timeout for closing/waiting
     |  :param ignore_errors: don't raise exceptions on subprocess errors
     |  :param ignore_status: list of status codes to ignore
     |  
     |  Methods defined here:
     |  
     |  __enter__(self)
     |      Context handler.
     |  
     |  __exit__(self, etype, value, traceback)
     |      Context handler.
     |  
     |  __init__(self, *args, mode=None, timeout=7200.0, ignore_errors=False, ignore_status=[], **kw)
     |      Create an IO Pipe.
     |  
     |  check_status(self)
     |      Poll the process and handle any errors.
     |  
     |  close(self)
     |      Wrap stream.close, wait for the subprocess, and handle errors.
     |  
     |  handle_status(self)
     |      Check the status variable and raise an exception if necessary.
     |  
     |  read(self, *args, **kw)
     |      Wrap stream.read and checks status.
     |  
     |  readLine(self, *args, **kw)
     |      Wrap stream.readLine and checks status.
     |  
     |  write(self, *args, **kw)
     |      Wrap stream.write and checks status.
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
    gopen(url, mode='rb', bufsize=8192, **kw)
        Open the URL.
        
        This uses the `gopen_schemes` dispatch table to dispatch based
        on scheme.
        
        Support for the following schemes is built-in: pipe, file,
        http, https, sftp, ftps, scp.
        
        When no scheme is given the url is treated as a file.
        
        You can use the OPEN_VERBOSE argument to get info about
        files being opened.
        
        :param url: the source URL
        :param mode: the mode ("rb", "r")
        :param bufsize: the buffer size
    
    gopen_curl(url, mode='rb', bufsize=8192)
        Open a URL with `curl`.
        
        :param url: url (usually, http:// etc.)
        :param mode: file mode
        :param bufsize: buffer size
    
    gopen_error(url, *args, **kw)
        Raise a value error.
        
        :param url: url
        :param args: other arguments
        :param kw: other keywords
    
    gopen_file(url, mode='rb', bufsize=8192)
        Open a file.
        
        This works for local files, files over HTTP, and pipe: files.
        
        :param url: URL to be opened
        :param mode: mode to open it with
        :param bufsize: requested buffer size
    
    gopen_pipe(url, mode='rb', bufsize=8192)
        Use gopen to open a pipe.
        
        :param url: a pipe: URL
        :param mode: desired mode
        :param bufsize: desired buffer size
    
    reader(url, **kw)
        Open url with gopen and mode "rb".
        
        :param url: source URL
        :param kw: other keywords forwarded to gopen
    
    set_options(obj, timeout=None, ignore_errors=None, ignore_status=None, handler=None)
        Set options for Pipes.
        
        This function can be called on any stream. It will set pipe options only
        when its argument is a pipe.
        
        :param obj: any kind of stream
        :param timeout: desired timeout
        :param ignore_errors: desired ignore_errors setting
        :param ignore_status: desired ignore_status setting
        :param handler: desired error handler

DATA
    PIPE = -1
    gopen_schemes = {'__default__': <function gopen_error>, 'ftps': <funct...
    info = {}

FILE
    /home/tmb/proj/webdataset/webdataset/gopen.py



```

# Module `webdataset.tariterators`

```
Help on module webdataset.tariterators in webdataset:

NAME
    webdataset.tariterators - Low level iteration functions for tar archives.

FUNCTIONS
    base_plus_ext(path)
        Split off all file extensions.
        
        Returns base, allext.
        
        :param path: path with extensions
        :param returns: path with all extensions removed
    
    group_by_keys(data, keys=<function base_plus_ext at 0x7fc3e98275e0>, lcase=True, suffixes=None, handler=None)
        Return function over iterator that groups key, value pairs into samples.
        
        :param keys: function that splits the key into key and extension (base_plus_ext)
        :param lcase: convert suffixes to lower case (Default value = True)
    
    shardlist(urls, *, shuffle=False)
        Given a list of URLs, yields that list, possibly shuffled.
    
    tar_file_expander(data, handler=<function reraise_exception at 0x7fc3e9827820>)
        Expand a stream of open tar files into a stream of tar file contents.
        
        This returns an iterator over (filename, file_contents).
    
    tar_file_iterator(fileobj, skip_meta='__[^/]*__($|/)', handler=<function reraise_exception at 0x7fc3e9827820>)
        Iterate over tar file, yielding filename, content pairs for the given tar stream.
        
        :param fileobj: byte stream suitable for tarfile
        :param skip_meta: regexp for keys that are skipped entirely (Default value = r"__[^/]*__($|/)")
    
    url_opener(data, handler=<function reraise_exception at 0x7fc3e9827820>, **kw)
        Given a stream of url names (packaged in `dict(url=url)`), yield opened streams.
    
    valid_sample(sample)
        Check whether a sample is valid.
        
        :param sample: sample to be checked

DATA
    meta_prefix = '__'
    meta_suffix = '__'
    trace = False

FILE
    /home/tmb/proj/webdataset/webdataset/tariterators.py



```

# Module `webdataset.mock`

```
Help on module webdataset.mock in webdataset:

NAME
    webdataset.mock - Mock implementations of torch interfaces when torch is not available.

CLASSES
    builtins.object
        DataLoader
        IterableDataset
    
    class DataLoader(builtins.object)
     |  Empty implementation of DataLoader when torch is not available.
     |  
     |  Data descriptors defined here:
     |  
     |  __dict__
     |      dictionary for instance variables (if defined)
     |  
     |  __weakref__
     |      list of weak references to the object (if defined)
    
    class IterableDataset(builtins.object)
     |  Empty implementation of IterableDataset when torch is not available.
     |  
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

# Module `webdataset.workerenv`

```
Help on module webdataset.workerenv in webdataset:

NAME
    webdataset.workerenv - Functions related to splitting datasets by node and worker.

DESCRIPTION
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
     |  TorchWorkerEnvironment.
     |  
     |  Method resolution order:
     |      TorchWorkerEnvironment
     |      WorkerEnvironment
     |      builtins.object
     |  
     |  Methods defined here:
     |  
     |  __init__(self, group=None)
     |      Initialize the worker environment for Torch.
     |      
     |      :param group: torch.distributed group for determining rank/size
     |  
     |  ----------------------------------------------------------------------
     |  Methods inherited from WorkerEnvironment:
     |  
     |  __str__(self)
     |      __str__.
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
     |  Encapsulates the runtime environment of the worker.
     |  
     |  Methods defined here:
     |  
     |  __init__(self, rank=0, world_size=1, worker=0, nworkers=1)
     |      Initialize the worker environment.
     |  
     |  __str__(self)
     |      __str__.
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
        Get the current worker environment.
    
    nodeslice(source, env=None)
        Slice the source based on the rank and worker number.
    
    split_by_node(urls, env=None)
        Select a subset of urls based on node info.
        
        Used as a shard selection function in Dataset.
    
    split_by_worker(urls, env=None)
        Select a subset of urls based on worker info.
        
        Used as a shard selection function in Dataset.
    
    worker_id()
        Return an identifier for the current worker.

DATA
    too_few_shards_warning = 1
    worker_environment = None

FILE
    /home/tmb/proj/webdataset/webdataset/workerenv.py



```

# Module `webdataset.utils`

```
Help on module webdataset.utils in webdataset:

NAME
    webdataset.utils - Miscellaneous utility functions.

FUNCTIONS
    guess_batchsize(batch)
        Guess the batch size by looking at the length of the first element in a tuple.
    
    identity(x)
        Return the argument as is.
    
    lookup_sym(sym, modules)
        Look up a symbol in a list of modules.
    
    repeatedly(source, nepochs=None, nbatches=None, nsamples=None, batchsize=<function guess_batchsize at 0x7fd418fc93a0>)
        Repeatedly yield samples from an iterator.
    
    repeatedly0(loader, nepochs=9223372036854775807, nbatches=9223372036854775807)
        Repeatedly returns batches from a DataLoader.
    
    safe_eval(s, expr='{}')
        Evaluate the given expression more safely.

FILE
    /home/tmb/proj/webdataset/webdataset/utils.py



```
