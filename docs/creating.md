```python
import webdataset as wds
import torchvision
import sys
```

# Creating a WebDataset

## Using `tar`

Since WebDatasets are just regular tar files, you can usually create them by just using the `tar` command. All you have to do is to arrange for any files that should be in the same sample to share the same basename. Many datasets already come that way. For those, you can simply create a WebDataset with

```
$ tar --sort=name -cf dataset.tar dataset/
```

If your dataset has some other directory layout, you may need a different file name in the archive from the name on disk. You can use the `--transform` argument to GNU tar to transform file names. You can also use the `-T` argument to read the files from a text file and embed other options in that text file.

## The `tarp create` Command

The [`tarp`](https://github.com/tmbdev/tarp) command is a little utility for manipulating `tar` archives. Its `create` subcommand makes it particularly simple to construct tar archives from files. The `tarp create` command takes a recipe for building
a tar archive that contains lines of the form:

```
archive-name-1 source-name-1
archive-name-2 source-name-2
...
```

The source name can either be a file, "text:something", or "pipe:something".

## Programmatically in Python

You can also create a WebDataset with library functions in this library:

- `webdataset.TarWriter` takes dictionaries containing key value pairs and writes them to disk
- `webdataset.ShardWriter` takes dictionaries containing key value pairs and writes them to disk as a series of shards

### Direct Conversion of Any Dataset

Here is a quick way of converting an existing dataset into a WebDataset; this will store all tensors as Python pickles:


```python
dataset = torchvision.datasets.MNIST(root="./temp", download=True)
sink = wds.TarWriter("mnist.tar")
for index, (input, output) in enumerate(dataset):
    if index%1000==0:
        print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
    sink.write({
        "__key__": "sample%06d" % index,
        "input.pyd": input,
        "output.pyd": output,
    })
sink.close()
```

     59000


```python
!ls -l mnist.tar
!tar tvf mnist.tar | head
```

    -rw-rw-r-- 1 tmb tmb 276490240 Oct 31 14:05 mnist.tar
    -r--r--r-- bigdata/bigdata 845 2020-10-31 14:05 sample000000.input.pyd
    -r--r--r-- bigdata/bigdata   5 2020-10-31 14:05 sample000000.output.pyd
    -r--r--r-- bigdata/bigdata 845 2020-10-31 14:05 sample000001.input.pyd
    -r--r--r-- bigdata/bigdata   5 2020-10-31 14:05 sample000001.output.pyd
    -r--r--r-- bigdata/bigdata 845 2020-10-31 14:05 sample000002.input.pyd
    -r--r--r-- bigdata/bigdata   5 2020-10-31 14:05 sample000002.output.pyd
    -r--r--r-- bigdata/bigdata 845 2020-10-31 14:05 sample000003.input.pyd
    -r--r--r-- bigdata/bigdata   5 2020-10-31 14:05 sample000003.output.pyd
    -r--r--r-- bigdata/bigdata 845 2020-10-31 14:05 sample000004.input.pyd
    -r--r--r-- bigdata/bigdata   5 2020-10-31 14:05 sample000004.output.pyd
    tar: write error


Storing data as Python pickles allows most common Python datatypes to be stored, it is lossless, and the format is fast to decode.
However, it is uncompressed and cannot be read by non-Python programs. It's often better to choose other storage formats, e.g.,
taking advantage of common image compression formats.

### Direct Conversion of Any Dataset with Compression

If you know that the input is an image and the output is an integer class, you can also write something like this:


```python
dataset = torchvision.datasets.MNIST(root="./temp", download=True)
sink = wds.TarWriter("mnist.tar")
for index, (input, output) in enumerate(dataset):
    if index%1000==0:
        print(f"{index:6d}", end="\r", flush=True, file=sys.stderr)
    sink.write({
        "__key__": "sample%06d" % index,
        "ppm": input,
        "cls": output,
    })
sink.close()
```

     59000


```python
!ls -l mnist.tar
!tar tvf mnist.tar | head
```

    -rw-rw-r-- 1 tmb tmb 276490240 Oct 31 14:05 mnist.tar
    -r--r--r-- bigdata/bigdata   1 2020-10-31 14:05 sample000000.cls
    -r--r--r-- bigdata/bigdata 797 2020-10-31 14:05 sample000000.ppm
    -r--r--r-- bigdata/bigdata   1 2020-10-31 14:05 sample000001.cls
    -r--r--r-- bigdata/bigdata 797 2020-10-31 14:05 sample000001.ppm
    -r--r--r-- bigdata/bigdata   1 2020-10-31 14:05 sample000002.cls
    -r--r--r-- bigdata/bigdata 797 2020-10-31 14:05 sample000002.ppm
    -r--r--r-- bigdata/bigdata   1 2020-10-31 14:05 sample000003.cls
    -r--r--r-- bigdata/bigdata 797 2020-10-31 14:05 sample000003.ppm
    -r--r--r-- bigdata/bigdata   1 2020-10-31 14:05 sample000004.cls
    -r--r--r-- bigdata/bigdata 797 2020-10-31 14:05 sample000004.ppm
    tar: write error


All we needed to do was to change the key from `.input.pyd` to `.ppm`; this will trigger using an image compressor (in this case, writing the image in PPM format). You can use different image types depending on what speed, compression, and quality tradeoffs you want to make. If you want to encode data yourself, you can simply convert it to a byte string yourself, store it under the desired key in the sample, and that binary string will get written out.

### Using `TarWriter`/`ShardWriter` with Binary Data (Lossless Writing)

The `assert` statements in that loop are not necessary, but they document and illustrate the expectations for this
particular dataset. Generally, the ".jpg" encoder can actually encode a wide variety of array types as images. The
".cls" encoder always requires an integer for encoding.

Here is how you can use `TarWriter` for writing a dataset without using an encoder:

```Python
sink = wds.TarWriter("dest.tar", encoder=False)
for basename in basenames:
    with open(f"{basename}.png", "rb") as stream):
        image = stream.read()
    cls = lookup_cls(basename)
    sample = {
        "__key__": basename,
        "input.png": image,
        "target.cls": cls
    }
    sink.write(sample)
sink.close()
```

Since no encoder is used, if you want to be able to read this data with the default decoder, `image` must contain a byte string corresponding to a PNG image (as indicated by the ".png" extension on its dictionary key), and `cls` must contain an integer encoded in ASCII (as indicated by the ".cls" extension on its dictionary key).
