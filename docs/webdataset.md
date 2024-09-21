# WebDataset API

## Fluid Interfaces

The `FluidInterface` class provides a way to create fluent interfaces for
chaining operations on datasets.
Most operations are contained in the `FluidInterface` mixin class.
`with_epoch` sets the epoch size (number of samples per epoch), effectively
an `itertools.islice` over the dataset.

::: webdataset.WebDataset

::: webdataset.WebLoader

::: webdataset.FluidInterface

::: webdataset.with_epoch

## Writing WebDatasets

::: webdataset.ShardWriter

::: webdataset.TarWriter

## Low Level I/O

::: webdataset.gopen.gopen

## Error Handling

::: webdataset.ignore_and_continue

::: webdataset.ignore_and_stop

::: webdataset.reraise_exception

::: webdataset.warn_and_continue

::: webdataset.warn_and_stop
