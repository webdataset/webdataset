## Commit: v0.2.109 -> v0.2.109-2-g9869af7

4c6a74e -> HEAD @ 2025-02-10 17:04:08 -0500

- **Makefile**: Enhanced with additional logic, resulting in improved build processes.
- **README.md**: Minor adjustments to improve clarity and accuracy.
- **VERSIONS.md**: Comprehensive updates to reflect recent changes and enhancements.
- **readme.ipynb**: Small correction to ensure consistency with other documentation.

## Commit: v0.2.108 -> v0.2.109

6abb239 -> 4c6a74e @ 2025-02-10 16:56:50 -0500

- Updated the `webdataset/__init__.py` file to reflect the latest changes in the codebase.
- Adjusted configuration and documentation files to align with the latest code updates.
- Improved consistency across the project files by synchronizing changes in `pyproject.toml` and `.bumpversion.cfg`.
- Enhanced the `VERSIONS.md` file with additional details and clarifications, ensuring comprehensive documentation of changes.

## Commit: v0.2.107 -> v0.2.108

e41fa8c -> 6abb239 @ 2024-09-27 13:08:27 -0400

- **Enhancements and Refactoring:**
  - Improved the `SequentialDataset` class with new features like `repeats` and logging capabilities.
  - Enhanced WebDataset filters and test suites to support batching for both tuples and dictionaries.
  - Updated data loading and validation in the `wsds` package, including new configuration outputs and improved assertions.
  - Refactored code for clarity and functionality, including updates to dependencies and error handling.
  - Added Hugging Face support in `gopen` via `curl`.

- **Code Cleanup and Removal:**
  - Removed obsolete functions and tests, including `cached_tarfile_samples` and `test_obsolete.py`.
  - Deleted unused files and directories, such as `helpers/docker.py` and `old/setup.py`.
  - Simplified and restructured the codebase by removing deprecated features and redundant code.

- **Documentation and Task Management:**
  - Updated `mkdocs.yml` to organize example documents under new categories.
  - Enhanced task management scripts for generating documentation and managing releases.
  - Improved the FAQ generation process with new helper functions and command-line interface using `typer`.

## Commit: v0.2.106 -> v0.2.107

e004be0 -> e41fa8c @ 2024-09-27 13:03:14 -0400

- Enhanced the release process by modifying the `release` function in `tasks.py` to include tagging with `bump2version` and pushing tags with `git push --follow-tags`.
- Updated the `release` function to ensure the creation of a GitHub release with the appropriate tag and release notes.
- Adjusted the version declaration in `webdataset/__init__.py` to reflect the latest changes.

## Commit: v0.2.105 -> v0.2.106

82095f9 -> e004be0 @ 2024-09-27 13:02:11 -0400

- Added a `git push` command to the `release` function in `tasks.py` to ensure changes are pushed to the repository during the release process.
- Updated the `__version__` in `webdataset/__init__.py` to reflect the latest changes.
- Minor adjustments in configuration files to align with the latest updates.

## Commit: v0.2.104 -> v0.2.105

9934cef -> 82095f9 @ 2024-09-27 13:01:58 -0400

- Updated the version number in the `__version__` attribute within the `webdataset/__init__.py` file.
- Made corresponding updates in the `.bumpversion.cfg` and `pyproject.toml` files to reflect the new version.

## Commit: v0.2.103 -> v0.2.104

39fccc6 -> 9934cef @ 2024-09-27 13:01:25 -0400

- Fixed an issue in `tasks.py` by removing unnecessary input handling in the GitHub release creation command.
- Updated the `__version__` attribute in `webdataset/__init__.py` to reflect the latest changes.
- Minor adjustments in configuration files `.bumpversion.cfg` and `pyproject.toml` to align with the latest project updates.

## Commit: v0.2.102 -> v0.2.103

3847b90 -> 39fccc6 @ 2024-09-27 13:00:37 -0400

- Updated `tasks.py` to read the version number directly from `pyproject.toml`, improving the automation of version management.
- Enhanced the `release` task to utilize the new `read_version` function, ensuring consistency in version tagging.
- Adjusted the `__version__` attribute in `webdataset/__init__.py` to reflect the latest changes.
- Improved the process of generating release notes by ensuring the correct version is tagged and released on GitHub.

## Commit: v0.2.101 -> v0.2.102

08dc51b -> 3847b90 @ 2024-09-21 11:39:07 -0400

- **Refactoring and Code Organization**: The codebase has been significantly refactored to improve organization and readability. This includes moving test files into more structured directories, renaming files for clarity, and updating import statements.
- **New Features and Enhancements**: Introduced new features such as `DecodingError` for more specific error handling, added support for `bz2` and `xz` compression in `TarWriter`, and implemented new functions for decoding and handling datasets.
- **Testing and Documentation**: Enhanced testing with new test cases and improved test organization. Documentation generation and deployment processes have been streamlined, with updates to `mkdocs` and `FAQ` generation.
- **Pipeline and Dataset Improvements**: Improved dataset handling with new classes and functions, such as `SimpleDataset` and `ChunkedSampler`, to enhance data processing capabilities.
- **Bug Fixes and Optimizations**: Addressed various bugs and optimized code for better performance, including fixes for path handling on Windows and improvements in dataset shuffling and sampling.

## Commit: 0.2.100 -> v0.2.101

c67bab8 -> 08dc51b @ 2024-09-01 00:36:42 -0400

- Enhanced the `WebDataset` library with improved documentation, including detailed docstrings for functions and classes, to provide better guidance on usage and behavior.
- Introduced a new environment variable `WDS_PYTORCH_WEIGHTS_ONLY` for selective loading of PyTorch weights, allowing for weights-only loading within the `torch_loads` function.
- Refactored the `tasks.py` script to streamline task definitions, improve coverage reporting, and enhance documentation generation processes.
- Improved exception handling and error messages across various modules, including `autodecode.py`, `cache.py`, and `handlers.py`.
- Updated the `WebDataset` class to issue warnings for certain configurations and to handle cache directories more robustly.
- Enhanced the `TarWriter` and `ShardWriter` classes with additional options and improved handling of tar file writing.
- Improved the `MultiLoader` class to provide an alternative to PyTorch's DataLoader using ZMQ for inter-process communication.
- Added new functions and classes for handling and processing data streams, including `detshuffle`, `Cached`, and `LMDBCached`.
- Improved the handling of URL opening and rewriting in

## Commit: 0.2.99 -> 0.2.100

defd638 -> c67bab8 @ 2024-09-01 00:34:22 -0400

- Improved the release process in `tasks.py` by setting the upstream branch during `git push`, tagging the version, and pushing tags to the remote repository.
- Enhanced the release creation command to include a title and description for the GitHub release.
- Fixed tagging issues to ensure proper version tracking and release management.
- Updated `webdataset/__init__.py` to reflect the latest changes in the codebase.

## Commit: 0.2.97 -> 0.2.99

9e397e3 -> defd638 @ 2024-09-01 00:14:14 -0400

- Enhanced the `tasks.py` script with functions to read, increment, and write version numbers, improving modularity and maintainability.
- Improved the `release` task to automate version number updates and GitHub release creation, ensuring a smoother release process.
- Fixed a minor issue in `wids.py` by specifying the `path` parameter in the `IndexedTarSamples` function, enhancing code clarity and functionality.
- Updated the `summarize_version` function to handle large diffs more effectively, ensuring better performance and reliability.
- Added error handling in the `release` task to manage subprocess errors gracefully, improving robustness.

## Commit: 0.2.96 -> 0.2.97

7e33c40 -> 9e397e3 @ 2024-08-12 20:36:05 -0400

- Implemented a small fix in `wids/wids_index.py` by replacing the `SimpleDownloader` with a direct call to `wids_dl.download_file`, streamlining the file download process.
- Adjusted the logic in `wids_index.py` to remove the release of downloaded files, simplifying the code.
- Updated the `setup.py` and `webdataset/__init__.py` to reflect the latest changes in the project.

## Commit: 0.2.95 -> 0.2.96

dedf42e -> 7e33c40 @ 2024-08-12 20:21:55 -0400

- Updated the `setup.py` and `webdataset/__init__.py` files to reflect the latest changes in the project.
- Ensured consistency across the project files by aligning the version information in the `setup.py` and `__init__.py` files.

## Commit: 0.2.90 -> 0.2.95

be4f3d8 -> dedf42e @ 2024-05-13 14:57:35 -0400

- **WebDataset Enhancements**: Improved the `WebDataset` class by adding a warning when `shardshuffle` is set to `None`, advising users to set it explicitly to `False` or a number. This change enhances user awareness and control over dataset shuffling behavior.
- **Decoder Class Update**: Added an assertion to ensure the `handlers` parameter is a list, improving input validation and robustness.
- **FileCache Class Fix**: Corrected file handling to ensure each cached file is returned only once, enhancing the reliability of file streaming.
- **FluidInterface Method Flexibility**: Updated the `to_tuple` method to accept additional keyword arguments, increasing its flexibility and alignment with other methods.
- **Tariterators Enhancement**: Introduced an EOF signal at the end of each tarfile to prevent mixing samples from different shards, improving data integrity during processing.

## Commit: 0.2.88 -> 0.2.90

f11fd66 -> be4f3d8 @ 2024-03-13 14:39:27 -0700

- Introduced an `empty_check` option in `WebDataset` and `ResampledShards` to handle cases where no samples or shards are available, raising a `ValueError` if enabled and no data is found.
- Enhanced the `DataPipeline` to prevent infinite loops when no samples are generated by breaking the loop if the dataset is empty.
- Updated tests in `test_loaders.py` to include scenarios for the `empty_check` feature, ensuring proper exception handling.
- Added a `pytest.mark.skip` decorator to `test_fluid.py` to skip tests with inaccessible remote data.
- Made minor adjustments to argument orders in `webdataset/shardlists.py` for improved functionality.

