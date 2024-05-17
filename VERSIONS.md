## Commit: 0.2.90 -> 0.2.90-18-g38dab69

be4f3d8 -> HEAD @ 2024-05-13 14:57:35 -0400

- **FAQ Generation**: Introduced an automatic FAQ generation script that summarizes issues into a `FAQ.md` file using AI. This includes tasks for creating and summarizing FAQ entries.
- **FluidInterface Class**: Fixed argument passing in the `listed` method to correctly pass `batchsize` and `collation_fn` to the `filters.batched` function.
- **Decoder Class**: Added an assertion to ensure the `handlers` parameter is a list during initialization.
- **FileCache Class**: Ensured the file stream is yielded correctly after handling exceptions, and each cached file is returned only once.
- **WebDataset**: Added a warning in `compat.py` to notify users when `shardshuffle` is set to `None`, advising them to set it explicitly to `False` or a number.
- **Tariterators**: Added an EOF signal at the end of each tarfile to terminate the accumulation of samples, preventing mixing of samples from different shards.
- **Shuffle Function**: Corrected the random seed initialization in the `_shuffle` function to use the provided seed.

## Commit: 0.2.88 -> 0.2.90

f11fd66 -> be4f3d8 @ 2024-03-13 14:39:27 -0700

- Introduced an `empty_check` option to `WebDataset` and `ResampledShards` to handle cases where no shards are available, raising a `ValueError` if no samples are found.
- Modified `webdataset/shardlists.py` to fix the order of arguments in `ResampledShardlist`.
- Added a new test `test_check_empty_throws_ValueError` to ensure the `empty_check` functionality works as expected.
- Updated `test_fluid.py` to skip a test due to inaccessible remote data.
- Enhanced `DataPipeline` to stop looping if no samples are generated.
- Added multiple untracked files and renamed `COMMITS.md` to `VERSIONS`.

## Commit: 0.2.86 -> 0.2.88

457c4ce -> f11fd66 @ 2023-12-06 21:21:25 -0800

- Introduced new `train-resnet50-wids.py` example script for training ResNet50 with WIDS.
- Enhanced `tasks.py` with additional tasks for testing, notebook processing, and Docker builds.
- Added comprehensive test coverage for various modules including `wids_dl`, `wids_mmtar`, `wids_cleanup`, and more.
- Implemented `LRUCleanup` class for efficient cache management.
- Improved `FileCache` and `StreamingOpen` classes for better file handling and caching.
- Added `RandomShardDownloader` class for downloading shards randomly from a source directory.
- Enhanced `WebDataset` class with better URL handling and cache management.
- Introduced `ChunkedSampler` and `DistributedChunkedSampler` for efficient data sampling in distributed environments.
- Added `keep_most_recent_files` function and `ExclusiveLock` class for file management and locking.
- Improved `download_file` and `download_and_open` functions for better file downloading and handling.
- Enhanced `MMIndexedTar` class with better memory-mapped file handling and cleanup callbacks.
- Updated `ShardListDataset` and `LRUShards` classes for better shard management and caching.
- Added new utility functions for handling file patterns and deprecation warnings.

## Commit: 0.2.85 -> 0.2.86

404b538 -> 457c4ce @ 2023-12-06 00:09:39 -0800

- Introduced a new `cleanup` task in `tasks.py` that runs `autoflake`, `isort`, and `black` for code formatting and cleanup.
- Added type annotations to various test functions in `tests/test_wids.py` to improve code clarity and type checking.
- Enhanced `ShardListDataset` and `ShardedSampler` classes in `wids/wids.py` with type annotations and additional docstrings for better code documentation and readability.
- Improved the `compute_file_md5sum` function in `wids/wids.py` to handle both filenames and file objects, with added examples in the docstring.
- Refactored several functions to use `yield from` for more concise and efficient code.
- Added new test cases for `ShardedSampler` in `tests/test_wids.py` to ensure proper functionality and uniqueness of indexes.
- Updated `resolve_dsdesc` and `load_dsdesc_and_resolve` functions in `wids/wids_specs.py` to handle `None` options more gracefully.
- Improved error handling and logging in various modules, including `webdataset/filters.py` and `webdataset/gopen.py`.
- Enhanced the `TarWriter` class in `webdataset/writer.py` with better handling of file objects and compression options.

## Commit: 0.2.84 -> 0.2.85

065208d -> 404b538 @ 2023-12-05 16:28:26 -0800

- **Bug Fixes and Improvements:**
  - Fixed bugs in local name defaulting and other areas.
  - Enhanced `ShardListDataset` to handle cache directories and local names more flexibly.
  - Improved `hash_localname` and `default_localname` functions to better manage shard URLs.
  - Added `cache_localname` function for better cache management.
  - Updated tests in `test_wids.py` to reflect changes in shard handling and caching.
  - Ensured `rebase_shardlist` and `resolve_dsdesc` functions handle shard lists correctly.
  - Improved error handling and assertions in `wids_specs.py`.

## Commit: 0.2.83 -> 0.2.84

81bdb5c -> 065208d @ 2023-12-05 02:28:06 -0800

- **Refactored**: Improved URL handling and caching mechanisms in `wids.py` by introducing `hash_localname` and enhancing `default_localname` to use URL-safe quoting.
- **Enhanced**: Added support for dataset name hashing and base64 encoding for cache file names.
- **Updated**: `ShardListDataset` class to handle dataset descriptors with nested datasets and improved shard URL resolution.
- **Improved**: `wids_index.py` to provide more detailed dataset information, including nested datasets.
- **Refined**: `wids_specs.py` to better handle remote dataset descriptions, rebase shard URLs, and resolve nested dataset references.
- **Fixed**: Various issues related to shard indexing and cache management.

## Commit: 0.2.79 -> 0.2.83

efb4a1e -> 81bdb5c @ 2023-11-22 08:43:16 -0800

- **Refactored Directory Structure:**
  - Moved `webdataset/tests` to `tests`.
  - Moved `webdataset/wids` to `wids`.

- **New Features:**
  - Added `widsindex` command line program.
  - Introduced `pipx` task in `tasks.py` for installing packages with `pipx`.
  - Added Docker support for local testing with `dockerlocal` task.

- **Enhancements:**
  - Updated `setup.py` to reflect new package structure.
  - Improved `ShardListDataset` to support cache directories and additional options.
  - Enhanced `wids_dl` to support file copying and improved URL handling.
  - Added `AtomicJsonUpdate` for safe JSON file updates.
  - Improved shard list extraction and URL merging in `wids_specs`.

- **Bug Fixes:**
  - Fixed MD5 sum mismatch assertion in `IndexedTarSamples`.
  - Corrected cache miss rate warning in `ShardListDataset`.
  - Ensured proper handling of transformations in `ShardListDataset`.

- **Miscellaneous:**
  - Updated Docker base image to `ubuntu:22.04`.
  - Improved handling of remote and local dataset descriptions.

## Commit: 0.2.77 -> 0.2.79

8e52bca -> efb4a1e @ 2023-11-14 14:13:38 -0800

- Made debugging output in `wids` optional by adding checks for the `WIDS_VERBOSE` environment variable.
- Updated `group_by_key` function in `wids.py` to print a warning message for ignored files.
- Enhanced `LRUShards` and `ShardListDataset` classes to conditionally print debugging information based on the `WIDS_VERBOSE` environment variable.
- Minor adjustments in `setup.py` and `__init__.py` to reflect the latest changes.

## Commit: 0.2.76 -> 0.2.77

09db99d -> 8e52bca @ 2023-11-14 14:12:56 -0800

- Added `testdata/testgz.tar` binary file.
- Modified `setup.py` and `webdataset/__init__.py` to reflect new version.
- Adjusted file paths and imports in `webdataset/__init__.py`.

## Commit: 0.2.75 -> 0.2.76

3959abb -> 09db99d @ 2023-11-09 12:31:00 -0800

- Added decompression support for `.gz` files in `webdataset/wids/wids.py`.
- Introduced a new test class `TestGz` in `webdataset/tests/test_wids.py` to verify the decompression functionality.
- Updated the `default_decoder` function to handle `.gz` file extensions, decompressing them and processing the underlying content.
- Modified `webdataset/__init__.py` to reflect the new changes.

## Commit: 0.2.74 -> 0.2.75

d46f93a -> 3959abb @ 2023-11-08 10:26:20 -0800

- `TarWriter` now automatically compresses files ending in `.gz`.
- Added a new test `test_writer_gz` in `webdataset/tests/test_writer.py` to verify the automatic compression feature.
- Updated `encode_based_on_extension1` in `webdataset/writer.py` to handle `.gz` file compression using the `gzip` module.

## Commit: 0.2.73 -> 0.2.74

039f70f -> d46f93a @ 2023-11-01 12:18:20 -0700

- Introduced a context manager to the `WebDataset` class, allowing it to be used with `with` statements for automatic resource management.
- Added a `close` method to the `DataPipeline` class to ensure proper cleanup of pipeline stages.
- Updated tests to include a context manager usage example for `WebDataset`.
- Enhanced `webdataset/compat.py` and `webdataset/pipeline.py` to support the new context manager functionality.

## Commit: 0.2.72 -> 0.2.73

e7507c9 -> 039f70f @ 2023-10-31 11:02:27 -0700

- **Test Cases**: Added new test cases in `webdataset/tests/test_wids.py` to cover specifications parsing and validation.
- **Functionality**: Moved shard list loading and validation functions to a new file `webdataset/wids/wids_specs.py` for better modularity.
- **Code Refactoring**: Refactored `ShardListDataset` class in `webdataset/wids/wids.py` to use the new `load_remote_shardlist` function from `wids_specs`.
- **Bug Fixes**: Fixed issues related to shard list extraction and validation, ensuring proper handling of nested datasets and remote sources.

## Commit: 0.2.71 -> 0.2.72

3c40a3e -> e7507c9 @ 2023-10-30 10:55:47 -0700

- Fixed the indexing script to handle file input from stdin and expanded brace expressions in filenames.
- Improved `ShardListDataset` class to correctly load and print shard lists, and return the dataset object when adding transformations.
- Enhanced `load_remote_shardlist` function to handle string inputs for file paths.
- Added JSON import to `wids.py` for better handling of dataset descriptions.
- Updated `wids_index.py` to include `wids_version` in the result dictionary and conditionally add the dataset name.

## Commit: 0.2.69 -> 0.2.71

a02f440 -> 3c40a3e @ 2023-10-30 10:15:54 -0700

- Improved handling of default decoders in `webdataset/wids/wids.py`:
  - Introduced `functools.partial` to streamline decoder format selection.
  - Added error handling for unknown formats in the `default_decoder` function.
  - Updated `ShardListDataset` to support `PIL` and `numpy` transformations directly.
- Modified `setup.py` and `webdataset/__init__.py` to reflect the new changes.
- Enhanced image decoding logic to raise errors for unknown formats and ensure proper format handling.

## Commit: 0.2.68 -> 0.2.69

4bb1a6b -> a02f440 @ 2023-10-28 20:09:23 -0700

- Removed a debugging statement from `webdataset/wids/wids_mmtar.py`.
- Updated `setup.py` and `webdataset/__init__.py` to reflect the latest changes.
- Minor adjustments in `VERSION` and `setup.py` files.
- Overall, the changes include 3 insertions and 4 deletions across 4 files.

## Commit: 0.2.65 -> 0.2.68

5dc6332 -> 4bb1a6b @ 2023-10-25 13:04:31 -0700

- Introduced `wids_mmtar.py` and `wids_tar.py` to factor out and enhance tar file handling with memory-mapped tar file support.
- Updated `tasks.py` to remove unnecessary git status check in `newversion` function.
- Modified `webdataset/wids/wids.py` to integrate new tar file handling classes and improve tar file indexing and reading.
- Changed `webdataset/autodecode.py` to ensure compatibility with `torch` tensor conversion.
- Updated `webdataset/tests/test_decode.py` to use `imageio.v3.imread` for image reading.
- Enhanced `ShardListDataset` and `ShardedSampler` classes in `webdataset/wids/wids.py` for better shard handling and sampling.

## Commit: 0.2.63 -> 0.2.65

c0e388d -> 5dc6332 @ 2023-10-20 13:25:46 -0700

- Introduced a `default_decoder` function to handle common file extensions in `webdataset`.
- Added index file caching in `TarFileReader` to improve performance.
- Enhanced `IndexedTarSamples` and `ShardListDataset` to support index files and transformations.
- Fixed bugs in `wids` and improved the handling of tar file indexing.
- Updated `tasks.py` to automate version tagging and pushing to GitHub.
- Added a new notebook `wids_mnist.ipynb` for demonstration purposes.

## Commit: 0.2.61 -> 0.2.63

977ee91 -> c0e388d @ 2023-10-18 11:54:21 -0700

- **Enhancements:**
  - Improved the docstring and argument list of all curried functions in `webdataset/filters.py`.
  - Added `functools.update_wrapper` to the `pipelinefilter` decorator for better function wrapping.

- **Bug Fixes:**
  - Fixed issues in `setup.py` and `webdataset/__init__.py` to ensure proper package inclusion and functionality.

## Commit: 0.2.60 -> 0.2.61

fe15d64 -> 977ee91 @ 2023-10-17 22:48:40 -0700

- Updated `setup.py` to reflect changes in the package.
- Modified `webdataset/__init__.py` to ensure consistency with the new version.


## Commit: 0.2.58 -> 0.2.60

b7be4da -> fe15d64 @ 2023-10-17 21:26:27 -0700

- Updated the `publish` action in `.github/workflows/pypi.yml`.
- Enhanced `webdataset/wids/wids.py` with additional docstrings for better code documentation and understanding.
- Introduced new methods in `ShardListDataset` and `ShardedSampler` classes to improve functionality and provide detailed descriptions of their purposes.
- Added a new function `check_shards` to validate the structure of shard lists.
- Improved the `ShardListDataset` class with methods for cache management, shard retrieval, and sample access, ensuring efficient data handling and locality preservation.

## Commit: 0.2.59 -> 0.2.58

b092eb6 -> b7be4da @ 2023-09-21 16:17:57 -0700

- Merged the `main` branch from the remote repository.
- Refactored the `wids` module, including moving several files to a new `wids` directory.
- Added functionality for reading JSON `wids` in `webdataset`.
- Improved the `tasks.py` script with better formatting and additional functionality.
- Enhanced the `FluidInterface` class in `webdataset/compat.py` with better handling of batch processing and decoding.
- Updated various test files to reflect changes in the `wids` module structure.
- Introduced new functions for loading remote shard lists and extracting shard lists from dataset descriptions in `webdataset/wids/wids.py`.
- Added a new `__init__.py` file in the `wids` directory to facilitate module imports.
- Improved the `wids_bench.py` script for better handling of dataset descriptions and command-line arguments.
- Updated the `wids_index.py` script to enhance shard indexing functionality.

## Commit: 0.2.57 -> 0.2.59

abc1a5d -> b092eb6 @ 2023-09-21 15:54:07 -0700

- **ShardWriter**: Added a `verbose` parameter to the `ShardWriter` class initializer, allowing users to set the verbosity level. The default value is set to `1`. This change provides more control over the logging output during the shard writing process.

## Commit: 0.2.53 -> 0.2.57

e54effd -> abc1a5d @ 2023-09-12 16:23:51 -0700

- Improved `pipe_cleaner` function in `webdataset/cache.py` to handle `hdfs` URLs.
- Made `DataPipeline.compose` method in `webdataset/pipeline.py` non-destructive by copying the pipeline stages.
- Updated `ShardWriter` in `webdataset/writer.py` to use `TarWriter` directly with the filename, ensuring proper handling of tar files.

## Commit: 0.2.51 -> 0.2.53

faa774e -> e54effd @ 2023-06-11 21:08:54 -0700

- Introduced new functionality for handling indexed web datasets, including classes like `IndexedTarSamples`, `LRUShards`, and `ShardListDataset`.
- Added comprehensive unit tests for the new classes and functionalities, ensuring robust testing coverage.
- Implemented a `ConcurrentDownloader` class to manage concurrent downloads across multiple processes, ensuring only a single download per file.
- Added utility functions for computing MD5 checksums and sample counts in tar files.
- Introduced a `wids_index.py` script for creating shard indices for datasets.
- Enhanced exception handling in `tariterators.py` to provide more informative error messages.
- Added new test files for `wids`, `wids_dl`, and `wids_lru` to validate the new functionalities.
- Included a benchmarking script `wids_bench.py` for performance testing of the new dataset handling mechanisms.

## Commit: 0.2.50 -> 0.2.51

e4c30ef -> faa774e @ 2023-06-11 09:47:53 -0700

- Added a missing `rename_files` argument to the `cached_tarfile_samples` function in `webdataset/cache.py`.
- Updated the `tar_file_expander` call within `cached_tarfile_samples` to include the new `rename_files` argument.
- Minor adjustments to `setup.py` and `webdataset/__init__.py` to reflect these changes.

## Commit: 0.2.49 -> 0.2.50

039d743 -> e4c30ef @ 2023-03-20 22:08:02 -0700

- Introduced a new `rename_files` argument to the `WebDataset` class, allowing for file renaming during dataset processing.
- Simplified the `FluidInterface` methods by consolidating multi-line function definitions into single lines.
- Enhanced the `WebDataset` initialization to handle YAML file inputs more efficiently.
- Improved the `tarfile_to_samples` function to support the new `rename_files` argument, providing more flexibility in dataset handling.

## Commit: 0.2.47 -> 0.2.49

352089f -> 039d743 @ 2023-03-18 16:40:58 -0700

- **Enhancements to Tests:**
  - Added detailed docstrings to test functions in `webdataset/tests/test_fluid.py` for better understanding and documentation.
  - Improved error handling and assertions in various test cases.
  - Introduced new test cases to cover additional scenarios and edge cases.
  - Updated `test_loaders.py` to include decoding steps in data pipelines and loaders.
  - Enhanced `test_webloader_repeat` and `test_webloader_unbatched` to include decoding steps.

- **Bug Fixes:**
  - Fixed issues in `test_fluid.py` related to dataset length and sample counting.
  - Corrected the `test_webloader` function to ensure proper sample counting and batching.

- **Code Cleanup:**
  - Removed obsolete and untested code sections, marked with `@pytest.mark.skip`.
  - Refactored import statements in `test_loaders.py` for better readability and maintainability.

## Commit: 0.2.46 -> 0.2.47

d05d8ff -> 352089f @ 2023-03-14 16:46:58 -0700

- **Library and Code Cleanup**: Refactored and cleaned up various libraries and code files, including `webdataset`, `autodecode`, `cache`, `cborsiterators`, `compat`, `extradatasets`, `filters`, `gopen`, `mix`, `pipeline`, `pytorch`, `shardlists`, `tariterators`, `tests`, `utils`, and `writer`.
- **New Features**:
  - Added new tasks in `tasks.py` for `black` and `autoflake` to format and clean up code.
  - Introduced new tests for cache, decode, handlers, loaders, mix, pipeline, shuffles, and writer functionalities.
  - Implemented file selection and renaming capabilities in `tar_file_iterator` and `tar_file_expander`.
  - Enhanced `torch_loads` function with type annotations and detailed docstrings.
- **Bug Fixes and Improvements**:
  - Fixed various issues related to imports, unused variables, and exception handling.
  - Improved the handling of tar file samples and grouping by keys.
  - Enhanced the `test_pipeline.py` and `test_fluid.py` with additional test cases and better structure.
  - Updated `webdataset` to support new decoding and caching mechanisms.
- **Testing Enhancements**:
  - Added comprehensive tests for new features and existing functionalities to ensure robustness and reliability.
  - Improved test coverage for various modules, including `cache`, `decode`, `handlers`, `loaders`, `mix`, `pipeline`, `shuffles`, and `writer`.
- **Documentation**:
  - Updated docstrings and comments across multiple files to provide better clarity and understanding of the codebase.

## Commit: 0.2.43 -> 0.2.46

dfa3895 -> d05d8ff @ 2023-03-14 12:29:47 -0700

- **Setup.py**: Minor adjustments to dependencies and configurations.
- **Webdataset Cache**: Added `time` module import and improved the `get_filetype` function to suppress output from the `file` command.
- **Webdataset Init**: Updated imports and module references for consistency and functionality.

## Commit: 0.2.39 -> 0.2.43

f2b64d8 -> dfa3895 @ 2023-03-09 12:01:27 -0800

- **Environment Variable Substitution**: Added functionality to substitute environment variables in URLs using the `WDS_` prefix in `webdataset/shardlists.py`.
- **Cache Directory Validation**: Implemented validation for cache directory existence in `webdataset/compat.py`.
- **Tests**: Updated tests to reflect changes in environment variable handling in `webdataset/tests/test_pipeline.py`.

## Commit: 0.2.35 -> 0.2.39

fa40da2 -> f2b64d8 @ 2023-03-03 17:21:19 -0800

- Implemented `expandvars` for URLs in `SimpleShardList` to support environment variable expansion.
- Fixed a redundant assignment in `ImageHandler` class in `autodecode.py`.
- Added a test case for `expandvars` in `test_pipeline.py`.
- Refactored imports in `shardlists.py` for better readability.
- Improved error handling and validation in `writer.py` for image encoding and sample writing.
- Enhanced `make_handlers` and `encode_based_on_extension` functions for better encoding of data samples.

## Commit: 0.2.34 -> 0.2.35

cb1aa32 -> fa40da2 @ 2023-02-01 10:16:51 -0800

- Merged multiple branches and pull requests to address various issues and improvements.
- Added `-f` flag to `curl` commands in `gopen.py` to handle failing file opens more gracefully.
- Enhanced `lru_cleanup` function in `cache.py` to handle `OSError` and `FileNotFoundError` exceptions, ensuring robust file deletion.
- Fixed `round_robin_longest` function in `mix.py` and added corresponding tests to ensure correct functionality.
- Improved error handling in `gopen.py` by raising `IOError` instead of a generic `Exception`.
- Added tests for handling missing files in `test_pipeline.py` to ensure proper exception raising.

## Commit: 0.2.31 -> 0.2.34

9bc1eb5 -> cb1aa32 @ 2022-11-29 14:31:29 -0800

- Cleaned up documentation and added `with_length` documentation.
- Fixed filename guessing and forward pipe-to-file user functions.
- Added a license file to the install process.
- Added a check for the `file` command.
- Added `invoke` to requirements and checked `readme.ipynb` for runnability.
- Improved version handling and added an option to make the `mtime` fixed for reproducibility.
- Merged changes from `ShardList` to `SimpleShardList` in `README.md`.
- Deleted `wordtrain.py` and other unsupported notebooks.
- Added `url_to_name` parameter in `WebDataset` class.
- Enhanced `group_by_keys` function to handle exceptions and added `mtime` parameter to `TarWriter` for reproducible tar files.

## Commit: 0.2.30 -> 0.2.31

8458543 -> 9bc1eb5 @ 2022-11-04 12:49:07 -0700

- **Enhanced `TarWriter`**: Added a new format option to `TarWriter` and changed the default format to `USTAR_FORMAT`.
- **Image Handling**: Expanded the list of supported image extensions in `autodecode.py` to include a comprehensive set of formats supported by Pillow.
- **Error Handling**: Improved error handling in `Decoder` to print a message when UTF-8 decoding fails.
- **Function Updates**: Updated `tar_file_iterator` to pass the handler parameter, and modified `imageencoder` to support TIFF format.
- **Bug Fixes**: Fixed test errors in `test_gopen` and addressed minor issues in the README.

## Commit: 0.2.26 -> 0.2.30

6864382 -> 8458543 @ 2022-09-16 21:20:12 -0700

- **Fixed Import Errors**: Added missing `import os` statements in `webdataset/multi.py` and `webdataset/gopen.py` to resolve import errors.
- **Enhanced `gopen` Functionality**: Improved `gopen_file` to handle `file:` URLs correctly and updated `gopen_curl` to use the correct `curl` command for PUT requests.
- **Updated Tests**: Modified tests in `webdataset/tests/test_gopen.py` to use the updated `gopen` function.
- **Improved Version Handling**: Enhanced version handling in `tasks.py` to include running tests before committing changes.
- **Minor Fixes**: Made small changes to `gopen` and other files to improve functionality and reliability.

## Commit: 0.2.25 -> 0.2.26

9f9b0e3 -> 6864382 @ 2022-09-14 15:56:36 -0700

- Improved image handling in `webdataset/autodecode.py` by adding support for different image modes (`L`, `RGB`, `RGBA`) and ensuring proper conversion between modes.
- Enhanced numpy and torch array handling for images, including proper dtype conversion and shape assertions.
- Added comprehensive tests in `webdataset/tests/test_pipeline.py` to validate the new image decoding functionality, ensuring compatibility with various image specifications and formats.
- Updated dependencies in `setup.py` to ensure compatibility with the new features and improvements.

## Commit: 0.2.22 -> 0.2.25

bcbb408 -> 9f9b0e3 @ 2022-09-14 15:30:12 -0700

- Fixed a typo in `webdataset/__init__.py` by correcting `__vesion__` to `__version__`.
- Corrected an environment variable typo in `webdataset/cache.py` from `os.envrion` to `os.environ`.
- Added functionality in `tasks.py` to write the updated version to `webdataset/__init__.py` during the `newversion` task.
- Implemented a bug fix in the codebase.

## Commit: 0.2.20 -> 0.2.22

89a905f -> bcbb408 @ 2022-08-30 20:56:51 -0700

- Introduced support for `GOPEN_VERBOSE` environment variable to control verbosity in `webdataset/cache.py`.
- Enhanced cache functionality to respect the `GOPEN_VERBOSE` setting.
- Minor adjustments in `setup.py` and `webdataset/cache.py` to improve functionality and maintain consistency.

## Commit: 0.2.18 -> 0.2.20

85c6524 -> 89a905f @ 2022-08-21 21:33:52 -0700

- Improved cache handling by adding `maybe_cached_tarfile_to_samples` function.
- Switched from using `hub` to `gh` for release creation in `tasks.py`.
- Enhanced `FluidInterface` class with better handling for `batched` and `decode` methods.
- Updated `WebDataset` class to handle cache size and directory more effectively.
- Refined various functions in `filters.py` for better error handling and code clarity.
- Enhanced `LMDBCached` class to ensure proper handling of cached samples.
- Improved `MultiShardSample` and `ResampledShards` classes for better shard handling and error reporting.
- Added new tests in `test_pipeline.py` to validate `LMDBCached` functionality.

## Commit: 0.2.5 -> 0.2.18

a562182 -> 85c6524 @ 2022-03-25 10:42:57 -0700

- Added new caching mechanisms with `Cached` and `LMDBCached` classes in `webdataset/filters.py` and integrated them into the pipeline.
- Introduced new functions `extract_keys`, `rename_keys`, and `xdecode` in `webdataset/filters.py` for enhanced data manipulation and decoding capabilities.
- Implemented `gopen_ais` function in `webdataset/gopen.py` to support AIS URL scheme and added environment variable handling for URL rewriting.
- Enhanced `ResampledShards` class in `webdataset/shardlists.py` for better deterministic and non-deterministic shard sampling.
- Updated `tasks.py` to improve version incrementing logic and added better error handling.
- Added new tests in `webdataset/tests/test_pipeline.py` to cover the new caching mechanisms, key extraction, and renaming functionalities.
- Improved `PipelineStage` class in `webdataset/utils.py` with a new `make_seed` function for better seed generation.

## Commit: 0.2.4 -> 0.2.5

0df0460 -> a562182 @ 2022-03-16 17:19:51 -0700

- **Updated `setup.py`**:
  - Changed the URL to `http://github.com/webdataset/webdataset`.
  
- **Modified `tasks.py`**:
  - Commented out the installation of specific versions of `torch` and `torchvision`.
  - Added a conditional check for a clean working tree before creating a GitHub release.
  - Removed `mkdocs.yml` from the list of required files.

- **Enhanced `webdataset/cache.py`**:
  - Increased the default cache size to `1e18`.
  - Replaced the walrus operator with a traditional `while` loop for compatibility.

- **Updated `webdataset/shardlists.py`**:
  - Changed the random seed source from `time.time_ns()` to `time.time()`.

## Commit: 0.1.103 -> 0.2.4

2eaa96e -> 0df0460 @ 2021-11-04 13:21:47 -0700

- Introduced a new `DataPipeline` class to streamline the creation and management of data processing pipelines.
- Added support for deterministic shuffling with the `detshuffle` function.
- Enhanced the `WebDataset` class with additional methods for handling epochs and repetitions.
- Introduced `cached_tarfile_to_samples` and `cached_url_opener` for efficient caching of tar files.
- Added new handlers for decoding various data formats, including `tenbin`, `msgpack`, `npy`, and `cbor`.
- Improved the `MultiShardSample` class to support more flexible shard specifications using YAML.
- Added new classes `RoundRobin` and `RandomMix` for mixing samples from multiple sources.
- Enhanced the `filters` module with new functions like `pipelinefilter`, `getfirst`, and `transform_with`.
- Improved error handling and logging capabilities across various modules.
- Added extensive test coverage for new features and functionalities.

## Commit: 0.1.96 -> 0.1.103

36ebfbd -> 2eaa96e @ 2021-10-24 04:24:12 -0700

- **Enhancements:**
  - Added `slice` function to `filters.py` using `itertools.islice`.
  - Improved `decode` function in `iterators.py` to handle string-based decoding.
  - Added debug output to `resampled_` function in `shardlists.py`.
  - Added new tests for `slice` and `resampled` functions in `test_pipeline.py`.

- **Bug Fixes:**
  - Fixed import issue in `pipeline.py` by changing `torch.utils.data.IterableDataset` to local import.
  - Renamed nested functions in `writer.py` to avoid type errors.
  - Corrected function name from `tarfile_sampler` to `tarfile_to_samples` in `tariterators.py`.

- **New Features:**
  - Introduced `tarfile_to_samples` function in `tariterators.py`.
  - Added `tarfile_to_samples` import in `__init__.py`.

- **Miscellaneous:**
  - Added missing export in `dbcache.py` and `extradatasets.py`.

## Commit: 0.1.87 -> 0.1.96

008bfd6 -> 36ebfbd @ 2021-10-15 02:41:28 -0700

- Introduced a new `DataPipeline` class in `webdataset/pipeline.py` to streamline the creation and management of data processing pipelines.
- Added new functions and classes such as `stage`, `split_by_node`, `split_by_worker`, `resampled`, and `non_empty` to enhance data handling capabilities.
- Updated `SimpleShardList` to support shuffling with a seed for reproducibility.
- Enhanced the `shuffle` function to improve data shuffling logic.
- Added comprehensive tests in `webdataset/tests/test_pipeline.py` to ensure the functionality of the new pipeline and data handling features.
- Improved the `WebDataset` function to include a `shardshuffle` parameter for better control over shard shuffling.
- Added `tarfile_samples` function to simplify the process of reading samples from tar files.

## Commit: 0.1.86 -> 0.1.87

0fc4e54 -> 008bfd6 @ 2021-10-15 02:35:15 -0700

- Updated `setup.py` to reflect the latest changes in the project description and metadata.

## Commit: 0.1.85 -> 0.1.86

dd6f67b -> 0fc4e54 @ 2021-10-15 02:32:16 -0700

- Removed extra output for waiting in `webdataset/gopen.py` by commenting out a debug print statement.
- Ensured that the `status` variable is checked and handled correctly without unnecessary verbose output.

## Commit: 0.1.84 -> 0.1.85

cea6299 -> dd6f67b @ 2021-10-15 01:56:38 -0700

- Improved output messages in `webdataset/gopen.py` to include process status and IDs for better debugging.
- Fixed an issue in `webdataset/gopen.py` related to the status check and verbose output.
- Minor fixes in `VERSION` and `setup.py` files.
- Updated `webdataset/gopen.py` to enhance the handling of pipe exit status and verbose logging.

## Commit: 0.1.76 -> 0.1.84

6f94aa5 -> cea6299 @ 2021-09-05 22:26:24 -0700

- Added `RoundRobin` class to `webdataset.dsspecs` with methods for adding datasets and string representation.
- Enhanced `Pipe` class in `webdataset.gopen` to handle subprocess status more robustly.
- Introduced `shuffle_rng` in `webdataset.iterators` for better random seed management.
- Updated `ResampledShards` class in `webdataset.shardlists` to include environment and random seed initialization.
- Modified `tar_file_iterator` in `webdataset.tariterators` to reset `stream.members` after yielding results.
- Added missing dependency to `setup.py`.
- Improved diagnostics for closing in `GOPEN` output.
- Added `__str__` method and comment to `RoundRobin` class.
- Fixed `StopIteration` issue and deprecated `Dataset` in `webdataset`.
- Added support for `npz` writing and decoding.

## Commit: 0.1.75 -> 0.1.76

bba3fbe -> 6f94aa5 @ 2021-09-05 22:15:56 -0700

- Introduced a new test `test_dataset_resampled` in `webdataset/tests/test_dataset.py` to verify the functionality of resampled datasets.
- Modified the `WebDataset` function in `webdataset/dataset.py` to correctly handle `resampled` URLs by assigning `ResampledShards(urls)` to `result`.

## Commit: 0.1.74 -> 0.1.75

17e429b -> bba3fbe @ 2021-09-05 21:12:55 -0700

- Introduced a `resampled` option to the `WebDataset` function in `webdataset/dataset.py`.
- Added support for shard resampling by importing `ResampledShards` from `shardlists`.
- Modified the `WebDataset` function to handle the `resampled` parameter, allowing for shard resampling when set to `True`.
- Updated the `urls` parameter handling to incorporate the new `ResampledShards` class when `resampled` is enabled.

## Commit: 0.1.73 -> 0.1.74

f8430bc -> 17e429b @ 2021-08-19 12:55:48 -0700

- **Fixed setup**: Updated `setup.py` to include `pyyaml` in the `install_requires` list, ensuring that the necessary dependencies are installed.

## Commit: 0.1.65 -> 0.1.73

9d823a1 -> f8430bc @ 2021-05-12 20:45:29 -0700

- Introduced new classes and functions for handling datasets, including `Composable`, `Shorthands`, `Processor`, `MockDataset`, `Repeatedly`, `DatasetTest`, `ChoppedDataset`, and `FakeLength`.
- Added support for YAML-based dataset specifications with `construct_dataset` and `MultiShardSample`.
- Enhanced shard handling with `PytorchShardList`, `SimpleShardList`, and `ResampledShards`.
- Improved error handling and logging with new handlers in `handlers.py`.
- Added new encoding functions for `numpy` arrays and `torch` tensors in `writer.py`.
- Implemented caching mechanisms for shards in `shardcache.py`.
- Updated `tariterators.py` to include additional metadata handling and improved error reporting.
- Enhanced the `WebDataset` and `WebLoader` classes with additional methods for dataset manipulation and transformation.
- Added comprehensive unit tests for new functionalities and dataset handling methods.

## Commit: 0.1.62 -> 0.1.65

f69d879 -> 9d823a1 @ 2021-05-01 12:08:03 -0700

- **New Features:**
  - Introduced `MockDataset` for generating mock data.
  - Added `node_equalize` method for equalizing dataset length across nodes.
  - Implemented `.test` method for easy mock data and sample verification.
  - Added `DatasetTest` class for performing final checks on datasets and supporting mock tests.
  - Introduced `split_by_node` and `split_by_worker` functions for shard selection based on node and worker information.

- **Enhancements:**
  - Refactored `MultiLoader` to use datasets directly.
  - Improved length handling in batched datasets.
  - Enhanced error handling and warnings for shard and worker distribution.

- **Bug Fixes:**
  - Fixed issues with `WorkerEnvironment` fallback and group handling.
  - Corrected length calculations in batched datasets.

- **Testing:**
  - Updated test cases to remove fluid interface and use new dataset methods.
  - Added tests for `MockDataset`, `node_equalize`, and `.test` method.

## Commit: 0.1.61 -> 0.1.62

16622f3 -> f69d879 @ 2021-04-29 21:28:11 -0700

- **Enhancements:**
  - Made `torch` optional by adding a mock implementation for `IterableDataset` and `DataLoader` in `webdataset/mock.py`.
  - Introduced `ZMQ`-based multi-loader in `webdataset/multi.py` for parallel data loading.
  - Updated `tasks.py` to streamline virtual environment setup and testing process.

- **Bug Fixes:**
  - Fixed missing module imports in `webdataset/dataset.py`, `webdataset/dbcache.py`, and `webdataset/fluid.py` by adding conditional imports for `torch`.

- **Code Refactoring:**
  - Refactored `tasks.py` to use the `venv` function within the `virtualenv` and `test` tasks for consistency.

## Commit: 0.1.59 -> 0.1.61

9d85dbd -> 16622f3 @ 2021-04-21 00:32:50 -0700

- Removed `torch` from `setup.py` and `requirements.txt` to streamline dependencies.
- Updated `tasks.py` to remove the installation of Jupyter Lab extensions.
- Adjusted `requirements.dev.txt` to reflect the removal of `torch`.

## Commit: 0.1.58 -> 0.1.59

6d2f2da -> 9d85dbd @ 2021-04-12 17:48:48 -0700

- **Enhanced `ImageHandler`**: Made image extensions configurable.
- **Improved `SplitByNode`**: Better handling of default group in distributed settings.
- **Updated `Shorthands`**: Added `collation_fn` argument to `batched` method and exposed `only` argument in `decode` shorthand.
- **Refined `DBCache`**: Added `source_` method and improved logging for database operations.

## Commit: 0.1.54 -> 0.1.58

398cf67 -> 6d2f2da @ 2021-03-16 16:47:17 -0700

- **Enhancements:**
  - Introduced `SplitByNode` class for node-based URL splitting in distributed environments.
  - Added `only` parameter to `Decoder` class to filter specific keys during decoding.
  - Enhanced `ShardList` and `WebDataset` classes to support node splitting by default.
  - Improved verbose output in `gopen` to include additional node information.
  - Added missing `__len__` method to `Repeatedly` class for better compatibility.

- **Bug Fixes:**
  - Fixed issues in `nodesplitter` and `split_by_worker` functions to handle worker and node information correctly.
  - Corrected attribute handling in `Dataset` class to ensure proper delegation to the underlying dataset.

- **Refactoring:**
  - Consolidated imports and improved modularity in `webdataset/fluid.py`.
  - Streamlined `batched` function parameters for better readability and maintainability.

## Commit: 0.1.51 -> 0.1.54

773c98d -> 398cf67 @ 2021-03-16 09:45:06 -0700

- Removed debug print statement from `webdataset/utils.py`.
- Added `WebLoader` export to `webdataset/__init__.py`.
- Fixed `setup.py` classifier to correctly list supported Python versions.
- Incremented version in `setup.py` to reflect changes.

## Commit: 0.1.49 -> 0.1.51

291f016 -> 773c98d @ 2021-02-17 21:48:28 -0800

- Added a `WebLoader` wrapper for `DataLoader` to facilitate repeated loading of datasets.
- Introduced new test cases for `WebLoader` and `repeat` functionality.
- Enhanced the `repeatedly` function to support repeating based on epochs, batches, or samples.
- Fixed an issue with `torchvision.io.read_video` by adding a missing argument.
- Included `pillow` in `requirements.txt`.
- Added new utility functions and test cases in `webdataset/utils.py` and `webdataset/tests/test_utils.py`.
- Improved dataset handling with new methods in `webdataset/dataset.py`, including `source_`, `repeat`, and `WebLoader`.
- Minor bug fixes and enhancements in various modules.

## Commit: 0.1.48 -> 0.1.49

10ab6df -> 291f016 @ 2021-02-12 00:13:35 -0800

- Improved Docker build process by adding tags and ensuring base container is built before tests.
- Updated `tasks.py` to handle Jupyter labextension installation more robustly.
- Changed Docker base image from `ubuntu:19.10` to `ubuntu:20.04`.
- Modified `ShardList` class in `webdataset/dataset.py` to use `shuffle=False` by default and updated the `shuffle` method to handle size less than 1.
- Enhanced Docker test scripts to clone the repository and copy test data for `pypi_test`.
- Refined `docker_build` function to accept a tag parameter and apply it during the build process.

## Commit: 0.1.41 -> 0.1.48

d7321fc -> 10ab6df @ 2020-12-19 00:19:58 -0800

- Enhanced `tasks.py` to ensure virtual environment activation for Jupyter lab extensions and added a print statement for completion.
- Introduced a `slice` method in `Shorthands` class within `dataset.py` for slicing datasets.
- Modified `ShardList` class to accept a callable for shuffling URLs.
- Added a new test `test_slice` in `test_dataset.py` to verify dataset slicing functionality.
- Updated `utils.py` to include `itertools` and added a `repeatedly` function for iterating over DataLoader batches.
- Enhanced `ShardWriter` class in `writer.py` to support starting from a specified shard number.

## Commit: 0.1.40 -> 0.1.41

759da05 -> d7321fc @ 2020-09-17 00:47:30 -0700

- Introduced a new `fluid` interface for constructing datasets, replacing the older `Dataset` class.
- Added support for database-based caching with the `DBCache` class.
- Enhanced the `autodecode` module with new handlers and improved the `Decoder` class.
- Refactored the `filters` module to use functions from the new `iterators` module.
- Introduced `ShardList`, `Processor`, and `WebDataset` classes for better dataset handling and processing.
- Added `shardcache` module for caching shards locally.
- Improved error handling and logging across various modules.
- Updated tests to reflect changes in dataset handling and processing.

## Commit: 0.1.38 -> 0.1.40

14b2315 -> 759da05 @ 2020-09-08 22:35:40 -0700

- **Enhancements:**
  - Added support for decompression of individually compressed files using `gzfilter`.
  - Introduced `Continue` class for handling continued decoding.
  - Improved `decode` method in `Pipeline` to include pre and post handlers.
  - Added `imagehandler`, `torch_video`, and `torch_audio` functions for better handling of image, video, and audio data.
  - Introduced `MultiDataset` class as an experimental alternative to `DataLoader`.

- **New Features:**
  - Added a benchmarking script `bench.py` for performance testing.
  - Added new tests for compressed files and writer functionalities.

- **Bug Fixes:**
  - Fixed issues with `tenbin` format in `ShardWriter`.
  - Corrected handling of non-byte data in `TarWriter`.

- **Miscellaneous:**
  - Added small test/benchmarking script.
  - Updated `ytsamples-split` example and comments.

## Commit: 0.1.37 -> 0.1.38

c30a2d6 -> 14b2315 @ 2020-08-18 08:31:11 -0700

- Introduced a new `Decoder` class to handle sample decoding using a list of handlers.
- Added `basichandlers` function for handling basic data types like text, JSON, and integers.
- Implemented `ImageHandler` class for decoding image data based on specified image specifications.
- Added support for decoding Torch video and audio files using `torchvideo` and `torchaudio` functions.
- Updated the `Pipeline.decode` method to accept multiple handlers and ensure backward compatibility with image decoding.
- Enhanced the `autodecode` module by removing the `default_handlers` dictionary and replacing it with more flexible handler functions.
- Improved error handling and decoding logic in the `decode_sample_based_on_extensions` function.
- Updated test cases to reflect changes in the decoding mechanism and ensure proper functionality.

## Commit: 0.1.34 -> 0.1.37

9c58006 -> c30a2d6 @ 2020-06-13 22:26:15 -0700

- Added support for `.pth` files and various video and audio formats in `webdataset/autodecode.py` with new handlers for `torch` and `torchvision`.
- Introduced `TorchVideoLoader` and `TorchAudioLoader` classes for handling video and audio data.
- Enhanced `webdataset/writer.py` to include a `torch_save_object` function for saving `torch` objects.
- Fixed a bug in `webdataset/filters.py` related to combining `numpy` arrays.
- Added new tests in `webdataset/tests/test_writer.py` to verify the functionality of writing and reading `.pth` files and other data types.

## Commit: 0.1.33 -> 0.1.34

2216314 -> 9c58006 @ 2020-06-11 22:41:54 -0700

- **Fixed**: Small fix in export functionality.
- **Modified**: `__all__` in `webdataset/gopen.py` to include `gopen_schemes`.
- **Updated**: Various documentation files (`README.ipynb`, `README.md`, `docs/index.md`, `docs/pydoc.md`) with significant changes.
- **Added**: New dependency in `requirements.txt`.

## Commit: 0.1.25 -> 0.1.33

8d0d9fc -> 2216314 @ 2020-05-20 23:18:16 -0700

- **New Features:**
  - Introduced `MultiDataset` and `MultiDatasetIterator` classes for parallel data loading using multiple workers.
  - Added `SampleIterator` class for iterating over samples with a given processing pipeline.
  - Implemented `Pipeline` class for building fluid data processing pipelines.
  - Added `Curried` and `Curried2` helper classes for currying pipeline stages.
  - Introduced `unbatched` function to reverse the batching process.

- **Enhancements:**
  - Improved shard selection with `worker_urls` and `all_urls` functions.
  - Enhanced `gopen` function to support additional options and verbose output.
  - Refactored `Dataset` class to use new `Pipeline` and `SampleIterator` classes.
  - Updated `filters` module with curried versions of functions like `map_stream`, `info`, `shuffle`, `select`, `decode`, `map`, `rename`, `associate`, `map_dict`, `to_tuple`, `map_tuple`, `batched`, and `unbatched`.

- **Bug Fixes:**
  - Fixed issues with tensor handling in `autodecode.py` by ensuring proper array conversion and type casting.
  - Addressed potential memory issues by adding garbage collection triggers in `tardata`.

- **Testing:**
  - Added new test cases in `test_dataset.py` and `test_multi.py` to cover new functionalities and ensure robustness.

## Commit: 0.1.24 -> 0.1.25

b31f90a -> 8d0d9fc @ 2020-05-20 23:06:51 -0700

- **Added**: `ResizedDataset` to the `webdataset/__init__.py` file.
- **Fixed**: Missing export in the `webdataset` module.

## Commit: 0.1.23 -> 0.1.24

f8460e9 -> b31f90a @ 2020-05-19 18:21:10 -0700

- Introduced batching functionality in the `Dataset` class with a new `batched` method.
- Added `batch_tensors` and `samples_to_batch` functions in `webdataset/filters.py` to handle tensor and scalar batching.
- Implemented a `batched` function in `webdataset/filters.py` to create batches of a specified size.
- Added a new test `test_batched` in `webdataset/tests/test_dataset.py` to verify the batching functionality.
- Updated `webdataset/dataset.py` to include the new batching method in the data processing pipeline.

## Commit: 0.1.22 -> 0.1.23

1ba422f -> f8460e9 @ 2020-05-11 21:56:56 -0700

- Renamed the class `ChoppedDataset` to `ResizedDataset` in `webdataset/dataset.py`.
- Updated the class docstring to reflect the new name.
- Modified the `__init__` method and other relevant parts of the class to use the new name.
- Added an alias `ChoppedDataset = ResizedDataset` for backward compatibility.

## Commit: 0.1.21 -> 0.1.22

4aa6230 -> 1ba422f @ 2020-05-07 10:46:48 -0700

- Removed the deprecated `WebDataset` class and its associated test file `test_webdataset.py`.
- Updated references from `WebDataset` to `Dataset` in `webdataset/dataset.py` and `webdataset/tests/test_writer.py`.
- Simplified the `__all__` list in `webdataset/dataset.py` by removing `WebDataset`.
- Deleted the `webdataset/webdataset.py` file, which contained the deprecated `WebDataset` class.
- Reduced the overall codebase by 642 lines, focusing on removing outdated and redundant code.

## Commit: 0.1.20 -> 0.1.21

6a3a17d -> 4aa6230 @ 2020-05-06 15:12:20 -0700

- **Dataset Enhancements:**
  - Introduced a random number generator (`rng`) to the `Dataset` class for improved shuffling.
  - Updated `shuffle` function in `filters.py` to accept a custom `rng` parameter.
  - Modified dataset tests to use OpenImages dataset instead of ImageNet.
  - Adjusted dataset tests to reflect changes in data source and structure.

- **Test Suite Adjustments:**
  - Updated test cases to align with the new dataset structure and data sources.
  - Commented out or removed redundant tests related to the old dataset format.
  - Ensured compatibility with the new dataset by modifying test parameters and expected outputs.

## Commit: 0.1.19 -> 0.1.20

322bef4 -> 6a3a17d @ 2020-05-05 21:46:01 -0700

- Introduced the `GOPEN_BUFFER` environment variable to control the buffer size for file operations in `gopen`.
- Modified `gopen` function in `webdataset/gopen.py` to use the `GOPEN_BUFFER` environment variable for setting the buffer size when opening files.

## Commit: 0.1.18 -> 0.1.19

e8d4ee7 -> 322bef4 @ 2020-03-20 01:30:41 -0700

- Introduced `ChoppedDataset` class to handle datasets with custom length and nominal length, improving flexibility in dataset iteration and epoch boundaries.
- Enhanced `Dataset` class by moving length logic to the new `ChoppedDataset` class.
- Improved length handling for multi-worker datasets.
- Added tests for `ChoppedDataset` to ensure correct functionality with various dataset sizes and configurations.
- Minor fixes and improvements in dataset export functionality.

## Commit: 0.1.17 -> 0.1.18

52d8e3e -> e8d4ee7 @ 2020-03-19 20:32:23 -0700

- Fixed a bug in `webdataset/filters.py` by modifying the `select` function to remove the `invert` parameter and correctly apply the `predicate` to each `sample`.
- Updated `setup.py` to reflect the changes.

## Commit: 0.1.16 -> 0.1.17

147b2d9 -> 52d8e3e @ 2020-03-18 21:40:25 -0700

- Added a `select` method to the `Dataset` class for filtering samples based on a predicate.
- Enhanced the `shuffle` method in `Dataset` to accept additional keyword arguments.
- Modified `getfirst` function to raise an error by default if a key is missing, controlled by `missing_is_error` parameter.
- Introduced a `select` filter in `filters.py` to yield samples based on a predicate.
- Added tests to ensure exceptions are raised for missing fields in `to_tuple` and `rename` methods in `Dataset`.
- Updated `map_tuple` and `to_tuple` functions to handle missing fields more robustly.

## Commit: 0.1.15 -> 0.1.16

e70675e -> 147b2d9 @ 2020-03-18 20:43:18 -0700

- **Enhancements:**
  - Updated `webdataset/gopen.py` to ignore additional curl status codes (23 and 26) for improved error handling during read and write operations.

## Commit: 0.1.14 -> 0.1.15

bd82a85 -> e70675e @ 2020-03-17 16:30:55 -0700

- **Refactored IO Module**: Renamed `io` module to `gopen` to avoid name conflicts and updated all references accordingly.
- **Documentation Enhancements**: Improved documentation generation in `tasks.py` by adding conversion of IPython Notebooks to Markdown and generating help text for each command.
- **New Functionality**: Added `info` function in `filters.py` for logging sample information.
- **Enhanced `gopen` Functionality**: Introduced multiple handlers for different URL schemes (`pipe`, `http`, `https`, `sftp`, `ftps`, `scp`) in `gopen.py`.
- **Test Updates**: Renamed and updated tests to reflect changes in the `gopen` module.
- **Deprecation Notice**: Added a deprecation notice in `webdataset.py` indicating that the code will be removed soon and suggesting the use of `webdataset.Dataset` instead.

## Commit: 0.1.13 -> 0.1.14

70f01ca -> bd82a85 @ 2020-03-16 23:50:05 -0700

- Renamed the `add_stage` method to `pipe` in the `Dataset` class to better reflect its functionality.
- Added a new script `run-jupyterlab` with 46 lines of code.
- Removed the entire `Dataset` class from `webdataset/webdataset.py`, resulting in a significant reduction of 221 lines.
- Minor adjustments in `setup.py` and `webdataset/dataset.py` to reflect the renaming of the method.

## Commit: 0.1.10 -> 0.1.13

71556b1 -> 70f01ca @ 2020-03-13 00:56:50 -0700

- Introduced a PyPI publishing workflow in `.github/workflows/pypi.yml`.
- Renamed exception handling functions in `webdataset`:
  - `ignore_exception` to `ignore_and_continue`
  - `warn_exception` to `warn_and_continue`
  - `ignore_and_finish` to `ignore_and_stop`
- Updated exception handling references in `webdataset/__init__.py`, `webdataset/dataset.py`, `webdataset/tests/test_dataset.py`, and `webdataset/webdataset.py` to reflect the new function names.

## Commit: 0.1.5 -> 0.1.10

77025f3 -> 71556b1 @ 2020-03-09 00:46:52 -0700

- **Refactoring and Modularization**:
  - Refactored the `WebDataset` class into a more modular `Dataset` class, allowing for more flexible pipeline stages.
  - Introduced new functions and classes in `filters.py` to handle common data transformations and decoding tasks.
  - Moved decoding logic to a separate `autodecode.py` file for better separation of concerns.

- **Error Handling Enhancements**:
  - Added various error handling strategies (`reraise_exception`, `ignore_exception`, `warn_exception`, `ignore_and_stop`, `warn_and_stop`) to improve robustness during data processing.

- **Pipeline Enhancements**:
  - Added methods to the `Dataset` class for adding pipeline stages (`add_stage`, `shuffle`, `decode`, `map`, `rename`, `map_dict`, `to_tuple`, `map_tuple`).
  - Improved the `tariterator` function to handle errors more gracefully and to support custom decoding.

- **Testing Improvements**:
  - Updated and expanded test cases to cover new functionalities and ensure robustness.
  - Added new test files `test_webdataset.py` and updated `test_dataset.py` to reflect changes in the dataset handling and pipeline processing.

- **Dependency Updates**:
  - Updated dependencies in `setup.py` to reflect changes in the codebase, ensuring compatibility and stability.

