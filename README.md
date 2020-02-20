[![Test](https://github.com/tmbdev/webdataset/workflows/Test/badge.svg)](https://github.com/tmbdev/webdataset/actions?query=workflow%3ATest)
[![TestPip](https://github.com/tmbdev/webdataset/workflows/TestPip/badge.svg)](https://github.com/tmbdev/webdataset/actions?query=workflow%3ATestPip)
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/tmbdev/webdataset/?ref=repository-badge)

# WebDataset

WebDataset is a PyTorch Dataset (IterableDataset) implementation providing efficient access to datasets stored in POSIX tar archives.

Storing data in POSIX tar archives greatly speeds up I/O operations on rotational storage and on networked file systems because it permits all I/O operations to operate as large sequential reads and writes.

WebDataset fulfills a similar function to Tensorflow's TFRecord/tf.Example classes, but it is much easier to adopt because it does not actually require any kind of data conversion: data is stored in exactly the same format inside tar files as it is on disk, and all preprocessing and data augmentation code remains unchanged.

Documentation: [ReadTheDocs](http://webdataset.readthedocs.io)

# Using WebDataset

Here is an example of an Imagenet input pipeline used for training common visual object recognition models. Note that this code is identical to the standard `FileDataset` I/O except for the single call that constructs the `WebDataset`.

        import torch
        from torchvision import transforms
        import webdataset as wds

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])

        preproc = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]) 

        path = "http://server/imagenet_train-{0000..0147}.tgz"

        dataset = wds.WebDataset(path,
                                 decoder="pil",
                                 extensions="jpg;png cls",
                                 transforms=[preproc, lambda x: x-1])

        loader = torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=4)
        for xs, ys in loader:
            train_batch(xs, ys)

# Creating WebDataset

In order to permit record sequential access to data, WebDataset only requires that the files comprising a single training samples are stored adjacent to each other inside the tar archive. Such archives can be easily created using GNU tar:

        tar --sorted -cf dataset.tar dir

On BSD and OSX, you can use:

        find dir -type f -print | sort | tar -T - -cf dataset.tar

Very large datasets are best stored as shards, each comprising a number of samples. Shards can be shuffled, read, and processed in parallel. The companion `tarproc` library permits easy sharding, as well as parallel processing of web datsets and shards. The `tarproc` programs simply operate as filters on tar streams, so for sharding, you can use a command like this:

        tar --sorted -cf - dir | tarsplit -s 1e9 -o out


# TODO

 - support `image.*` and `image=jpg,png,jpeg` syntax for extensions
