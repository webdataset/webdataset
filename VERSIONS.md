## Commit: 0.2.100 -> 0.2.100-24-g2d9423d

c67bab8 -> HEAD @ 2024-09-01 00:36:42 -0400

- Enhanced the `WebDataset` library with expanded docstrings and clearer argument descriptions for various classes and functions, improving code documentation and maintainability.
- Introduced a new feature in the `WebDataset` library that allows loading PyTorch weights in a more selective manner, with the ability to enable weights-only loading through an environment variable.
- Refactored the `WebDataset` library to include detailed docstrings for the `FileCache` class and the `cached_url_opener` function, providing clear explanations of their purposes, parameters, return values, and potential exceptions.
- Added a `pytest-cov` task for coverage reporting during testing and introduced JSON and LCOV output for coverage analysis.
- Updated the `webdataset` package with additional inline documentation for several functions and classes, enhancing readability and maintainability.
- Implemented various updates to the `WebDataset` library, including the removal of old docs, renaming of documentation sources, and cleanup of code and documentation.
- Made several updates to the `WebDataset` library to support a new feature for loading PyTorch weights selectively, including changes to test functions and the introduction of a new environment variable for weights-only loading.
- Updated the `WebDataset` library with expanded docstrings and clearer function signatures, improving the documentation and function signatures in the `multi.py` file.
- Enhanced the `WebDataset` library with expanded docstrings for mixing functions in the `webdataset/mix.py` file, providing clearer guidance on the usage and behavior of the mixing utilities.
- Expanded the docstrings for exception handling functions in the `webdataset/handlers.py` file, now including `Args` and `Returns` sections.
- Updated the `filters.py` file in the `WebDataset` library with expanded and detailed docstrings for the `FilterFunction` and `RestCurried` classes, as well as the `pipelinefilter`, `reraise_exception`, and `identity` functions.
- Refined the documentation of several classes in the `webdataset/extradatasets.py` file, including more detailed docstrings with clear descriptions, argument specifications, and return values.
- Improved the `webdataset/downloader.py` file with better docstrings and minor code cleanup, including the removal of a redundant import statement for `gopen`.
- Modified the `webdataset/__init__.py` file to remove unused code dependencies by eliminating the import statement for `cbors2_to_samples` and `cbors_to_samples` from the `.cborsiterators` module.
- Introduced support for PyTorch weights-only loading in the `webdataset` package, including a new environment variable `WDS_PYTORCH_WEIGHTS_ONLY` that enables this feature within the `torch_loads` function.
- Made several updates to the `WebDataset` library to improve warning messages and functionality, including changes to the `WebDataset` class to issue warnings when `shardshuffle` is set for resampled datasets or to `True`.
- Updated the `webdataset/cache.py` file with several changes, including a new docstring at the top of the file, detailed docstrings for the `FileCache` class and the `cached_url_opener` function, and minor code cleanup.
- Added `pytest-cov` to `requirements.dev.txt` for coverage reporting during testing and updated `autodecode.py` and `cache.py` in the `webdataset` package with additional inline documentation for several functions.

## Commit: 0.2.99 -> 0.2.100

defd638 -> c67bab8 @ 2024-09-01 00:34:22 -0400

This update introduces improvements to the release process automation for the `webdataset` package. The `tasks.py` script has been modified to include additional git commands that streamline the process of pushing changes to the repository, tagging the current version, and pushing tags to the remote repository. Additionally, the `webdataset/__init__.py` file has been updated to ensure consistency within the package.

## Commit: 0.2.97 -> 0.2.99

9e397e3 -> defd638 @ 2024-09-01 00:14:14 -0400

- The `tasks.py` script has been updated to include additional whitespace for readability and to modify the `summarize_version` function with a more structured subprocess call.
- In `setup.py`, the `version` variable has been updated.
- The `webdataset/__init__.py` file has been modified to reflect the new version.
- A small fix has been applied to `wids/wids.py` where the `IndexedTarSamples` class is now instantiated with a named argument `path`.
- The `tasks.py` script has been enhanced with new utility functions for reading, incrementing, and writing version numbers, as well as for creating a release, which includes tagging and pushing to the repository. These changes streamline the release process and improve code maintainability.

## Commit: 0.2.96 -> 0.2.97

7e33c40 -> 9e397e3 @ 2024-08-12 20:36:05 -0400

- Simplified the shard indexing process in `wids/wids_index.py` by replacing the `SimpleDownloader` with a direct call to `wids_dl.download_file` for downloading shards, and removed the associated release step for the downloader, streamlining the code.
- Updated the `__version__` variable in `webdataset/__init__.py` to reflect the new release.

## Commit: 0.2.95 -> 0.2.96

dedf42e -> 7e33c40 @ 2024-08-12 20:21:55 -0400

The recent update includes the following changes:

- The `webdataset` package has been updated with a new release.
- Modifications were made to the `setup.py` script to support the latest release.
- The `__version__` string in the `webdataset/__init__.py` file has been updated to reflect the new release of the package.

## Commit: 0.2.90 -> 0.2.95

be4f3d8 -> dedf42e @ 2024-05-13 14:57:35 -0400

The provided git diff indicates several updates and enhancements to the WebDataset library:

- The `Decoder` class now includes an assertion to ensure the `handlers` parameter is a list, improving input validation.
- The `FileCache` class has been modified to correctly yield file streams after handling exceptions, ensuring each cached file is returned only once.
- The `FluidInterface` class has been updated to correctly pass `batchsize` and `collation_fn` as arguments to the `filters.batched` function.
- A warning has been introduced in `compat.py` to notify users when `shardshuffle` is set to `None`, advising them to set it explicitly to `False` or a number.
- The `tar_file_expander` function in `tariterators.py` now yields an EOF marker at the end of each shard to prevent mixing samples from different shards.
- Various other minor enhancements and bug fixes have been implemented to improve the functionality and robustness of the library.

These changes aim to improve the usability and reliability of the WebDataset library for handling large-scale datasets in machine learning workflows.

## Commit: 0.2.88 -> 0.2.90

f11fd66 -> be4f3d8 @ 2024-03-13 14:39:27 -0700

- Introduced an `empty_check` option in `WebDataset` to detect cases where workers receive no shards, preventing infinite loops in pipelines when no samples are generated.
- Added a check in `ResampledShards` to raise a `ValueError` if no shards are found when `empty_check` is set to `True`.
- Implemented a fix in `ResampledShardlist` to correct the order of arguments.
- Added a `check_empty` function in `compat.py` to raise a `ValueError` if a dataset is empty, which can be turned off with `empty_check=False`.
- Modified `DataPipeline` to stop iterating if no samples are found, avoiding unnecessary looping over empty datasets.
- Skipped a test in `test_fluid.py` due to inaccessible remote data and potential deprecation of YAML specs.
- Adjusted tests in `test_loaders.py` to account for the new `empty_check` feature and added a test to ensure a `ValueError` is raised when a dataset is empty.
- Renamed `COMMITS.md` to `VERSIONS` and included multiple untracked files in the repository.

## Commit: 0.2.86 -> 0.2.88

457c4ce -> f11fd66 @ 2023-12-06 21:21:25 -0800

- Enhanced the `ShardListDataset` class to support a new `lru_size` parameter for specifying the size of the in-memory LRU cache.
- Introduced `ChunkedSampler` and `DistributedChunkedSampler` classes for efficient sampling in chunks, with optional shuffling within each chunk, and support for distributed training environments.
- Added `lengths_to_ranges`, `intersect_range`, `intersect_ranges`, and `iterate_ranges` utility functions to assist with range-based sampling operations.
- Implemented `DirectoryShardList` as an `IterableDataset` for iterating over shards in a directory, with various strategies for selecting and recycling shards.
- Created `download_file_no_log`, `download_file`, and `download_and_open` functions in `wids_dl.py` for downloading files with optional verbose logging and support for multiple download handlers.
- Added `keep_while_reading` as a cleanup callback for `MMIndexedTar` to manage file lifecycle based on reader activity.
- Developed `RandomShardDownloader` for randomly downloading shards to a directory, with strategies for updating or replacing shards.
- Added `ULockFile` class for non-blocking exclusive file locking using `fcntl`.
- Implemented `keep_most_recent_files` function to maintain the most recent files in a directory based on size and file count constraints.
- Created `ExclusiveLock` class for simple exclusive file locking and `create_cleanup_background_process` for background directory cleanup.
- Introduced `compute_sample_weights` utility function for calculating sample weights based on a list of `(n, w)` pairs.
- Added `deprecated` and `obsolete` decorators for marking functions as deprecated or obsolete, with support for environment variable-based suppression of obsolescence errors.
- Updated `ShardListDataset` to include `__dataset__` and `__index__` keys in samples, and to check for excessive cache misses.
- Refactored `ShardedSampler` into `ShardListSampler` for clarity and consistency with other sampler classes.
- Improved `IndexedTarSamples` class with support for both file paths and streams, MD5 checksum verification, and proper resource cleanup.
- Enhanced `LRUCache` with a `clear` method for releasing all cached objects and ensuring cleanup upon object deletion.
- Streamlined `ConcurrentDownloader` by removing unnecessary release methods and simplifying the download process.
- Optimized `MMIndexedTar` with a more efficient index-building process and added a `close` method for resource management.
- Refined `wids_specs.py` with better error handling and support for loading remote dataset descriptions.
- General code cleanup, including removal of unused variables, simplification of complex expressions, and improvements to comments and documentation.

## Commit: 0.2.85 -> 0.2.86

404b538 -> 457c4ce @ 2023-12-06 00:09:39 -0800

This update introduces a range of improvements and bug fixes across multiple files in the project. Key changes include:

- Refactoring of import statements for better readability and organization.
- Enhancement of the `ShardListDataset` and `ShardedSampler` classes in `wids.py` to improve dataset handling and sampling functionality.
- Improvements to error handling and exception messages to provide clearer debugging information.
- Code cleanup in various test files (`test_cache.py`, `test_decode.py`, `test_fluid.py`, etc.) to remove unnecessary imports and improve code structure.
- Introduction of new utility functions like `compute_file_md5sum` in `wids.py` for computing MD5 checksums of files.
- Optimization of iteration patterns in `shardlists.py` and `utils.py` to make the code more efficient.
- Adjustments to `gopen.py` to streamline the handling of read and write modes.
- Refactoring of `autodecode.py` to enhance the decoding process of webdataset samples.
- General code cleanup and refactoring across various modules (`pipeline.py`, `filters.py`, `multi.py`, etc.) to improve maintainability and performance.

These changes aim to enhance the functionality, efficiency, and readability of the codebase, contributing to a more robust and user-friendly library.

## Commit: 0.2.84 -> 0.2.85

065208d -> 404b538 @ 2023-12-05 16:28:26 -0800

- Bug fixes in `wids.py` for local filename defaulting and cache directory handling, including a new `cache_localname` function for when a cache directory is explicitly given.
- Improvements to `ShardListDataset` class to handle cache directory and localname function more effectively, with added verbosity for shard information.
- Adjustments to `wids_specs.py` to ensure `shardlist` is not `None` and to properly rebase URLs in a shardlist when a base is provided.
- Refactoring in `test_wids.py` to streamline assertions and remove unnecessary comments, with additional tests for `ShardListDataset` cache behavior and `TestSpecs` for shardlist extraction and dataset length verification.

## Commit: 0.2.83 -> 0.2.84

81bdb5c -> 065208d @ 2023-12-05 02:28:06 -0800

- Enhanced the `ShardListDataset` class to support loading datasets from JSON descriptors with URL resolution and optional dataset naming.
- Implemented a new caching mechanism using SQLite to store URL-to-path mappings, improving cache file name handling.
- Refactored the `load_dsdesc_and_resolve` function to handle dataset descriptions more robustly, including remote references and rebasing shard URLs.
- Updated the `widsindex` command-line tool to provide more detailed dataset information, including dataset names and URLs when available.
- Improved the URL handling in `wids_specs.py` by adding functions to merge base URLs with relative URLs and to rebase shard lists.
- Added a hashing function to generate dataset names based on the input string for better identification.
- Made various optimizations and code cleanups in `wids.py` and `wids_index.py` to streamline dataset processing and indexing.

## Commit: 0.2.79 -> 0.2.83

efb4a1e -> 81bdb5c @ 2023-11-22 08:43:16 -0800

- The `webdataset` package has been restructured, moving several test directories from `webdataset/tests` to `tests`.
- The `wids` module has been moved out of the `webdataset` directory and is now a top-level module.
- The `tasks.py` file has been updated to reflect changes in the Docker base image from Ubuntu 20.04 to Ubuntu 22.04 and includes additional tasks related to `pipx` and local testing with Docker.
- The `setup.py` script has been modified to include new entry points for the `widsindex` command line program and to update the package list to reflect the new location of the `wids` module.
- The `wids_index.py` and `wids_specs.py` files have been removed from the `webdataset/wids` directory and new versions have been added to the `wids` directory with significant updates, including new functions for handling shard index creation and updates.
- The `ShardListDataset` class in `wids.py` has been updated with new parameters and functionality, including the ability to specify a cache directory and to interpret transformations.
- The `wids_dl.py` file has been updated with new download handlers and functions to support file copying and URL merging.
- The `autodecode.py` file has been updated to ensure that the result of image decoding is a copy of the array when converting to a PyTorch tensor.
- A new command line interface has been added for creating, updating, and displaying information about shard index files, as well as for retrieving individual samples from a dataset.

## Commit: 0.2.77 -> 0.2.79

8e52bca -> efb4a1e @ 2023-11-14 14:13:38 -0800

- The `webdataset` package has made debugging output for `wids` optional, controlled by the `WIDS_VERBOSE` environment variable.
- A warning message is now displayed when ignoring files that are not in a subdirectory within the `group_by_key` function in `wids.py`.
- The `LRUShards` class now conditionally prints download information based on the `WIDS_VERBOSE` environment variable.
- The `ShardListDataset` class also conditionally prints shard information based on the `WIDS_VERBOSE` environment variable.

## Commit: 0.2.76 -> 0.2.77

09db99d -> 8e52bca @ 2023-11-14 14:12:56 -0800

This update introduces the missing `testdata/testgz.tar` file to the repository. Additionally, the `webdataset` package initialization file `__init__.py` has been updated to reflect this new addition. The changes ensure that the necessary test data is now included in the package for proper functionality.

## Commit: 0.2.75 -> 0.2.76

3959abb -> 09db99d @ 2023-11-09 12:31:00 -0800

- Added decompression support for `.gz` files in the `webdataset` library, allowing for the reading of gzipped content directly within the dataset.
- Implemented a new test case `TestGz` in `test_wids.py` to ensure that gzipped text files are correctly read and decompressed, verifying the content matches expected values.
- Modified the `default_decoder` function in `wids.py` to handle `.gz` file extensions by decompressing the content and updating the sample dictionary with the decompressed stream.

## Commit: 0.2.74 -> 0.2.75

d46f93a -> 3959abb @ 2023-11-08 10:26:20 -0800

- The `TarWriter` class in the `webdataset` package now features automatic compression for files with a `.gz` extension. This enhancement ensures that when such files are written using `TarWriter`, they are compressed without requiring additional steps.
- A new test case `test_writer_gz` has been added to `webdataset/tests/test_writer.py` to verify the functionality of the `.gz` compression within `TarWriter`. This test checks that a file with a `.gz` extension is correctly compressed and can be read back properly.
- The `encode_based_on_extension1` function within `webdataset/writer.py` has been updated to handle automatic compression for byte and string data if the target name ends with `.gz`. The function strips the `.gz` extension, compresses the data using `gzip`, and then proceeds with the usual encoding based on the file extension.

## Commit: 0.2.73 -> 0.2.74

039f70f -> d46f93a @ 2023-11-01 12:18:20 -0700

The `webdataset` package has introduced a context manager to its fluid interface, allowing for the use of `with` statements for cleaner resource management. Specifically, the `WebDataset` class within `webdataset/compat.py` now includes `__enter__` and `__exit__` methods, enabling the class instances to be used in `with` blocks, which ensures that resources are properly closed after use. Additionally, a `close` method has been added to the `DataPipeline` class in `webdataset/pipeline.py` to facilitate the closing of the pipeline. A test case `test_dataset_context` has been added to `webdataset/tests/test_fluid.py` to verify the functionality of the context manager in the dataset operations.

## Commit: 0.2.72 -> 0.2.73

e7507c9 -> 039f70f @ 2023-10-31 11:02:27 -0700

- Introduced new test cases for specifications and fixes in the `webdataset` library, specifically within the `test_wids.py` file, enhancing the robustness of the dataset handling.
- Refactored the `wids.py` module by extracting functions related to loading and parsing dataset descriptions into a new module named `wids_specs.py`. This change improves the modularity and readability of the codebase.
- Added functionality to handle dataset descriptions in JSON format, allowing for remote or local dataset descriptions to be loaded and processed.
- Implemented new methods in `wids_specs.py` to validate and extract shard lists from dataset descriptions, ensuring that the shard lists are well-formed and adhere to the expected structure.
- Made `ShardListDataset` class in `wids.py` compatible with both string paths and IO streams for loading shard lists, providing more flexibility in how dataset descriptions are provided.

## Commit: 0.2.71 -> 0.2.72

3c40a3e -> e7507c9 @ 2023-10-30 10:55:47 -0700

- The `webdataset` package has been updated with improvements to the `ShardListDataset` class, including the ability to load shard lists from JSON files and the addition of a `print` statement to display shard information.
- A new `json` import has been added to support JSON operations within the `wids.py` module.
- The `load_remote_shardlist` function now correctly handles string sources by opening and loading them as JSON.
- The `ShardListDataset` class now correctly initializes the `lengths` attribute from its own `shards` attribute instead of the external `shards` variable.
- A `return self` statement has been added to the `add_transform` method of `ShardListDataset` to allow for method chaining.
- The `wids_index.py` script has been modified to accept input from `stdin` if a single dash ("-") is provided as the file argument.
- The `main` function in `wids_index.py` now includes a conditional to add the dataset name to the result dictionary only if it is provided.
- The result dictionary in `wids_index.py` now includes a `wids_version` key with a value of 1, and the `files` key has been renamed to `shardlist`.

## Commit: 0.2.69 -> 0.2.71

a02f440 -> 3c40a3e @ 2023-10-30 10:15:54 -0700

- The `webdataset` package has enhanced the handling of default decoders by introducing the ability to specify the format for image data, with support for both `PIL` and `numpy` formats. This is achieved through the use of the `partial` function from the `functools` module to set the format parameter in the `default_decoder` function.
- The `ShardListDataset` class now accepts a `transformations` parameter that can be set to either `"PIL"` or `"numpy"` to determine the format of the decoded images. It also ensures that the transformations provided are callable, enhancing the robustness of the dataset handling.
- The `default_decoder` function has been updated to raise a `ValueError` if an unknown format is specified, improving error handling and user feedback.

## Commit: 0.2.68 -> 0.2.69

4bb1a6b -> a02f440 @ 2023-10-28 20:09:23 -0700

- The `webdataset` package has been updated with a new release.
- A debugging statement printing out file names and type flags has been removed from the `MMIndexedTar` class within the `webdataset/wids/wids_mmtar.py` file.

## Commit: 0.2.65 -> 0.2.68

5dc6332 -> 4bb1a6b @ 2023-10-25 13:04:31 -0700

This update introduces significant changes to the `webdataset` package, particularly around the handling of TAR files. The `wids_tar` and `wids_mmtar` modules have been refactored out, with `wids_mmtar` adding support for memory-mapped TAR file reading, which should improve performance when accessing TAR archives. The `TarFileReader` class has been moved to `wids_tar.py`, and a new `MMIndexedTar` class has been added to `wids_mmtar.py` for memory-mapped operations. Additionally, the `ShardListDataset` class now supports an optional `use_mmap` argument to leverage the new memory-mapped functionality. The `autodecode.py` module has been updated to ensure that numpy arrays are copied when converting to PyTorch tensors, addressing potential issues with memory sharing. The `tasks.py` script has been simplified by removing input prompts related to the working tree's cleanliness. The `ShardedSampler` class has been documented to clarify its purpose in sampling from a `ShardListDataset`. Lastly, the `imageio` library usage has been updated to the new `imageio.v3` API for image reading.

## Commit: 0.2.63 -> 0.2.65

c0e388d -> 5dc6332 @ 2023-10-20 13:25:46 -0700

- Enhanced the `webdataset` package with a new default decoder function `default_decoder` that handles common file extensions for web datasets, improving ease of use and integration with different data types.
- Implemented an indexing system for TAR files to speed up access, with the ability to load an existing index file or create a new one if it doesn't exist, indicated by the addition of `find_index_file` and modifications to `TarFileReader`.
- Added caching functionality to `ShardListDataset` with a Least Recently Used (LRU) cache mechanism to manage local storage of shards, and the ability to keep files after use.
- Improved the robustness of the `ConcurrentDownloader` by ensuring that temporary files are cleaned up properly after downloads.
- Extended the `ShardListDataset` class to allow for custom transformations to be applied to the dataset, enhancing flexibility for data preprocessing.

## Commit: 0.2.61 -> 0.2.63

977ee91 -> c0e388d @ 2023-10-18 11:54:21 -0700

The `webdataset` package has been updated with enhancements to its functionality. The `filters.py` module now includes an update to the `pipelinefilter` decorator, which turns the decorated function into one that is partially applied for all arguments except the first. Additionally, the `functools.update_wrapper` function is used to preserve the original function's metadata. The package structure has been expanded to include a new subpackage `webdataset.wids`.

## Commit: 0.2.60 -> 0.2.61

fe15d64 -> 977ee91 @ 2023-10-17 22:48:40 -0700

The `webdataset` package has been updated with the following changes:

- The `setup.py` script has been modified.
- Adjustments have been made to the `__init__.py` file within the `webdataset` directory.

These changes are part of routine maintenance and do not indicate any new features or bug fixes.

## Commit: 0.2.58 -> 0.2.60

b7be4da -> fe15d64 @ 2023-10-17 21:26:27 -0700

- The `webdataset` package has introduced a new file `webdataset/wids/wids.py` with a significant addition of code.
- A new class `ShardListDataset` has been implemented, providing an indexable dataset based on a list of shards, which can be local files or URLs, and includes caching functionality.
- The `ShardListDataset` class includes methods to return the total number of samples, cache statistics, check cache misses, retrieve a shard based on an index, and close the dataset.
- A `ShardedSampler` class has been added to sample from a `ShardListDataset

## Commit: 0.2.59 -> 0.2.58

b092eb6 -> b7be4da @ 2023-09-21 16:17:57 -0700

This git diff indicates a significant refactoring and enhancement of the `webdataset` library, particularly focusing on the `wids` module. Key changes include:

- The `wids` module has been restructured, with several files being moved into a new `wids` subdirectory.
- Functionality for reading JSON `wids` has been added, improving the handling of remote or local dataset descriptions.
- A new `load_remote_shardlist` function has been introduced to load dataset descriptions using web client APIs.
- The `ShardListDataset` class now supports loading from a remote shard list if a string is provided, enhancing flexibility in dataset management.
- The `LRUShards` class has been updated with a clearer eviction mechanism and a `clear` method to reset the cache.
- Several utility functions have been added or improved, such as `check_shards`, `set_all`, and `extract_shardlist`, to better manage and validate shard lists.
- The `ShardListDataset` class has been optimized to handle cumulative lengths and mapping indices to shards more efficiently.
- The `wids_index.py` script has been updated to create a shard index for a set of files, with expanded brace expressions in file names and additional command-line arguments for customization.

These updates aim to improve the efficiency, usability, and maintainability of the `webdataset` library, particularly for users working with large-scale datasets and web-based data sources.

## Commit: 0.2.57 -> 0.2.59

abc1a5d -> b092eb6 @ 2023-09-21 15:54:07 -0700

The `ShardWriter` class in the `webdataset/writer.py` file has been updated to allow the `verbose` level to be set as an initialization parameter. Previously, the verbosity level was hardcoded to `1`, but now it can be customized by passing the `verbose` argument during the creation of a `ShardWriter` instance. This change introduces more flexibility in controlling the output verbosity of the `ShardWriter` class.

## Commit: 0.2.53 -> 0.2.57

e54effd -> abc1a5d @ 2023-09-12 16:23:51 -0700

- The `ShardWriter` class has been modified to use `TarWriter` directly when opening tar files, which simplifies the code by removing an unnecessary file opening step.
- In `pipeline.py`, the `DataPipeline.compose` method has been updated to make a copy of the pipeline before appending new stages, ensuring that the original pipeline remains unmodified.
- The `pipe_cleaner` function in `cache.py` has been enhanced to handle HDFS URLs, expanding the range of supported storage backends.
- The `webdataset/__init__.py` and `setup.py` files have been updated to reflect the changes in the codebase.

## Commit: 0.2.51 -> 0.2.53

faa774e -> e54effd @ 2023-06-11 21:08:54 -0700

This update introduces significant enhancements to the `webdataset` library, focusing on improving exception handling, adding new features, and expanding test coverage:

- Improved exception handling in `tariterators.py` by ensuring that exceptions include the file object causing the issue, enhancing error traceability.
- Added a new feature for indexed web datasets, allowing efficient access to samples within tar files. This is facilitated by the new `IndexedTarSamples` class and related functions for computing MD5 checksums and the number of samples.
- Implemented a Least Recently Used (LRU) caching mechanism for managing shards in web datasets, which includes classes like `LRUCache` and `LRUShards`. This feature helps in optimizing the access to shards by caching them and reducing unnecessary downloads.
- Extended the test suite with multiple new test files (`test_wids.py`, `test_wids_dl.py`, `test_wids_lru.py`) to ensure the robustness of the new features, including tests for the LRU cache, shard list dataset, and download functionalities.
- Introduced new scripts (`wids.py`, `wids_bench.py`, `wids_dl.py`, `wids_index.py`, `wids_lru.py`) that provide additional functionality for downloading, indexing, and benchmarking web datasets, as well as managing the LRU cache for shards.
- The `ShardListDataset` class has been added to enable the creation of datasets from a list of shards, with the ability to cache and access them efficiently.
- A new `ShardedSampler` class is included to facilitate sampling from sharded datasets, allowing for batched and shuffled access to the data.

These changes aim to make the `webdataset` library more robust, efficient, and easier to use when working with large-scale datasets stored in tar files.

## Commit: 0.2.50 -> 0.2.51

e4c30ef -> faa774e @ 2023-06-11 09:47:53 -0700

- The `webdataset` package has been updated to include a new `rename_files` argument in the `cached_tarfile_samples` function within the `cache.py` module. This argument allows for renaming files when expanding tar files and grouping them into samples.
- The `rename_files` parameter has also been added to the `tar_file_expander` function call within `cached_tarfile_samples` to support the new functionality.

