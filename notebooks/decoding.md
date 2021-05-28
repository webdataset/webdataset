```python
%pylab inline

import torch
from torch.utils.data import IterableDataset
from torchvision import transforms
import webdataset as wds
from itertools import islice
```

    Populating the interactive namespace from numpy and matplotlib


# Data Decoding

Data decoding is a special kind of transformations of samples. You could simply write a decoding function like this:

```Python
def my_sample_decoder(sample):
    result = dict(__key__=sample["__key__"])
    for key, value in sample.items():
        if key == "png" or key.endswith(".png"):
            result[key] = mageio.imread(io.BytesIO(value))
        elif ...:
            ...
    return result

dataset = wds.Processor(dataset, wds.map, my_sample_decoder)
```

This gets tedious, though, and it also unnecessarily hardcodes the sample's keys into the processing pipeline. To help with this, there is a helper class that simplifies this kind of code. The primary use of `Decoder` is for decoding compressed image, video, and audio formats, as well as unzipping `.gz` files.

Here is an example of automatically decoding `.png` images with `imread` and using the default `torch_video` and `torch_audio` decoders for video and audio:

```Python
def my_png_decoder(key, value):
    if not key.endswith(".png"):
        return None
    assert isinstance(value, bytes)
    return imageio.imread(io.BytesIO(value))

dataset = wds.Decoder(my_png_decoder, wds.torch_video, wds.torch_audio)(dataset)
```

You can use whatever criteria you like for deciding how to decode values in samples. When used with standard `WebDataset` format files, the keys are the full extensions of the file names inside a `.tar` file. For consistency, it's recommended that you primarily rely on the extensions (e.g., `.png`, `.mp4`) to decide which decoders to use. There is a special helper function that simplifies this:

```Python
def my_decoder(value):
    return imageio.imread(io.BytesIO(value))
    
dataset = wds.Decoder(wds.handle_extension(".png", my_decoder))(dataset)
```

If you want to "decode everyting" automatically and even override some extensions, you can use something like:


```python
url = "http://storage.googleapis.com/nvdata-openimages/openimages-train-000000.tar"
url = f"pipe:curl -L -s {url} || true"

def png_decoder_16bpp(key, data):
    ...

dataset = wds.WebDataset(url).decode(
    wds.handle_extension("left.png", png_decoder_16bpp),
    wds.handle_extension("right.png", png_decoder_16bpp),
    wds.imagehandler("torchrgb"),
    wds.torch_audio,
    wds.torch_video
)
```

This code would...

- handle any file with a ".left.png" or ".right.png" extension using a special 16bpp PNG decoder function
- decode all other image extensions to three channel Torch tensors
- decode audio files using the `torchaudio` library
- decode video files using the `torchvideo` library

In order to decode images, audio, and video, it would dynamically load the `Pillow`, `torchaudio`, and `torchvideo` libraries.

# Automatic Decompression

The default decoder handles compressed files automatically. That is `.json.gz` is decompressed first using the `gzip` library and then treated as if it had been called `.json`.

In other words, you can store compressed files directly in a `WebDataset` and decompression is handled for you automatically.

If you want to add your own decompressors, look at the implementation of `webdataset.autodecode.gzfilter`.
