def pil_resize(sample, key=None, shape=None):
    import PIL.Image

    assert key is not None
    assert shape is not None
    image = sample[key]
    image = image.resize(shape, PIL.Image.BILINEAR)
    sample[key] = image
    return sample
