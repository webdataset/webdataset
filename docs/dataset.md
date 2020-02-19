# webdataset.dataset
Train PyTorch models directly from POSIX tar archive, locally
or over HTTP connections.

## imagehandler
```python
imagehandler(data, imagespec)
```
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


## tariterator
```python
tariterator(fileobj, keys=<function base_plus_ext at 0x7f8dc3a395f0>, decoder=True, suffixes=None, errors=True, container=None)
```

Iterate through training samples stored in a sharded tar file.

- fileobj: a Python file-like object
- check_sorted:  check whether the input is actually properly sorted (Default value = False)
- keys:  key extraction function (Default value = base_plus_ext)
- decoder: value decoding function (Default value = True)

The key extraction function takes a string representing a pathname and
returns a pair (__key__, suffix).

The decoder takes the entire sample as a dict and returns the
decoded sample as a dict.

## WebDataset
```python
WebDataset(self, urls, *, size=None, extensions=None, decoder='rgb', transforms=None, pipeline=None, epochs=1, keys=<function base_plus_ext at 0x7f8dc3a395f0>, opener=<function reader at 0x7f8dc3aa6440>, errors=True, verbose=False, shuffle=0, associate=None, prepare_for_worker=True, container=None, extra_meta=False)
```
Iterate over sharded datasets.

- urls: shard spec or list of shards
- extensions: extensions to extract (Default value = None, can be either list of lists or "a;b c")
- decode: decoder to apply to files in tarfiles (Default value = True, based on extension)
- transforms: list of functions to apply to unbatched samples (Default value = None)
- pipeline: function that maps the iterator, e.g. for batching
- opener: either a function that returns a stream or a string that is invoked via Popen
- epochs: how often to iterate through the shards before finishing the iterator
- verbose: verbose output
- shuffle: if >0, then shuffle shards, and shuffle samples with a buffer of the given size
- associate: a callable or dictionary that returns additional information to associate with each sample
- prepare_for_worker: callable called in each worker before anything else is done
- container: if given, treats the tar file as a record file of containers (protobufs, msgpack, etc.)
- extra_meta: associates subset info with each sample record

The decoder can be True (default decoder), False (no decoder), a callable (called
decode the sample, or a dictionary mapping filename extensions to callables for
the decoding.

### shard_selection
```python
WebDataset.shard_selection(self)
```
Contains the logic for self.subset shard selection.
