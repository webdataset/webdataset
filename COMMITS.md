
# Commit Summaries

## Commit: 0.2.88 -> 0.2.88

f11fd66 -> HEAD @   

No changes.                                                                                                                                           

## Commit: 0.2.86 -> 0.2.88

457c4ce -> f11fd66 @ 2024-03-13 14:39:27 -0700

The README.md for the WebDataset project has been updated with additional information and examples. The WebDataset format description has been        
expanded to clarify the naming convention for tar file shards and how files within a tar file are associated with each other. The documentation now   
includes a Python code snippet demonstrating how to access and list the contents of a dataset using curl and tar. The section on related projects has 
been replaced with a new section on WebDataset libraries, listing several libraries and tools that support the WebDataset format, including a Julia   
implementation and a Golang command line tool. The webdataset library section has been elaborated to describe its features, limitations, and usage    
with PyTorch, including an example of how to create a dataset instance and check its type. Finally, a new code example has been added to show how to  
preprocess and augment data using the webdataset library and PyTorch.                                                                                 

## Commit: 0.2.85 -> 0.2.86

404b538 -> 457c4ce @ 2023-12-06 21:21:25 -0800

The .sourcery.yaml configuration file has been added with specific rule settings for code refactoring. In tasks.py, the autoflake task has been       
updated to include additional directories, and new tasks isort and cleanup have been introduced for code formatting. The test task now explicitly     
targets the tests directory. Imports in test_cache.py, test_decode.py, and test_fluid.py have been reorganized, with some redundant imports removed.  
In test_fluid.py, the order of import statements has been changed. A redundant import has been removed from test_gopen.py. In test_loaders.py, import 
statements have been consolidated into a single line. Only formatting changes.                                                                        

## Commit: 0.2.84 -> 0.2.85

065208d -> 404b538 @ 2023-12-06 00:09:39 -0800

The test_length assertion in TestIndexedTarSamples has been simplified to a single line. A blank line has been added before the TestLRUShards and     
TestGz classes. In TestShardListDataset, the instantiation of ShardListDataset has been condensed into a single line. The assertions comparing the    
cache contents have been simplified by removing unnecessary line breaks. The test_spec_parsing method in TestSpecs has been duplicated, likely by     
mistake. Overall, the changes are minor and primarily involve code cleanup and formatting, with no significant functional alterations.                

## Commit: 0.2.83 -> 0.2.84

81bdb5c -> 065208d @ 2023-12-05 16:28:26 -0800

The setup.py script has been modified to update the version field and change the entry point for widsindex from webdataset.wids.wids_index:main to    
wids.wids_index:main. In the webdataset/__init__.py file, the __version__ string has been updated accordingly. The wids/wids.py file has seen         
significant changes, including the addition of imports for base64 and sqlite3, the creation of a new hash_localname function that creates a local     
cache directory and a SQLite database to store cache information, and modifications to the ShardListDataset class to handle dataset names and base    
URLs, as well as to print verbose information about the dataset. The default_localname function has been updated to use URL quoting for the shard     
name. A new hash_dataset_name function has been added to generate a hash for the dataset name.                                                        

## Commit: 0.2.79 -> 0.2.83

efb4a1e -> 81bdb5c @ 2023-12-05 02:28:06 -0800

The webdataset package has been updated with several changes. The setup.py file has been modified to include a new entry point for a console script   
named widsindex, which points to the webdataset.wids.wids_index:main function. Additionally, the package directory for webdataset.wids has been       
changed to wids. The tasks.py file has seen the addition of a new task pipx, which installs the package and injects the torch dependency. The Docker  
base container has been updated to use Ubuntu 22.04, and several commented-out lines related to installing JupyterLab, NumPy, and nbconvert have been 
added. The test_cache.py and test_decode.py files have been moved from the webdataset/tests directory to the tests directory. Only formatting changes.

## Commit: 0.2.77 -> 0.2.79

8e52bca -> efb4a1e @ 2023-11-22 08:43:16 -0800

The webdataset package has undergone several updates. In wids.py, when encountering files not in a subdirectory, a warning is now printed with the    
file name. The LRUShards class has been modified to print a download message including the URL, local file path, and download status to stderr when   
the WIDS_VERBOSE environment variable is set. Similarly, the ShardListDataset class now prints shard information when WIDS_VERBOSE is set. These      
changes are aimed at improving the verbosity and logging for debugging purposes.                                                                      

## Commit: 0.2.76 -> 0.2.77

09db99d -> 8e52bca @ 2023-11-14 14:13:38 -0800

A new binary file testdata/testgz.tar has been added to the repository. The webdataset/__init__.py file has been updated, but the specific changes    
within the file are not detailed in the provided diff output.                                                                                         

## Commit: 0.2.75 -> 0.2.76

3959abb -> 09db99d @ 2023-11-14 14:12:56 -0800

Added gzip support in the default_decoder function within wids.py, allowing for the decompression of .gz files and handling of streams accordingly.   
Enhanced the TestGz class in test_wids.py with a new test case test_gz to verify the functionality of reading gzipped files, ensuring that the sample 
dictionary contains the expected keys and values after processing a gzipped text file.                                                                

## Commit: 0.2.74 -> 0.2.75

d46f93a -> 3959abb @ 2023-11-09 12:31:00 -0800

Added gzip compression support to the encode_based_on_extension1 function in writer.py, which now compresses data if the target name ends with .gz.   
The function handles both bytes and strings, compressing them before returning. A new test case test_writer_gz was added to test_writer.py to verify  
that a .txt.gz file is correctly written and read back, ensuring the data is compressed and matches the expected content.                             

## Commit: 0.2.73 -> 0.2.74

039f70f -> d46f93a @ 2023-11-08 10:26:20 -0800

The webdataset package has been updated with new functionality. The WebDataset class in webdataset/compat.py now supports context management, allowing
it to be used with Python's with statement for better resource management. This is facilitated by the addition of __enter__ and __exit__ methods.     
Additionally, a close method has been implemented in the DataPipeline class in webdataset/pipeline.py to close the pipeline properly. A test case     
test_dataset_context has been added to webdataset/tests/test_fluid.py to verify the correct behavior of the WebDataset class when used in a context   
manager.                                                                                                                                              

## Commit: 0.2.72 -> 0.2.73

e7507c9 -> 039f70f @ 2023-11-01 12:18:20 -0700

The webdataset library has undergone a minor update. The setup.py and webdataset/__init__.py files reflect an increment in the patch version number.  
In the webdataset/tests/test_wids.py file, a new test class TestSpecs has been added with two methods for testing the parsing of dataset              
specifications from JSON. Additionally, the webdataset/wids/wids.py file has been refactored to remove the load_remote_shardlist function and related 
helper functions, as they have been moved to a separate module wids_specs. The wids_specs module is now responsible for handling the loading and      
parsing of remote shard lists.                                                                                                                        

## Commit: 0.2.71 -> 0.2.72

3c40a3e -> e7507c9 @ 2023-10-31 11:02:27 -0700

The webdataset package has undergone several updates. In the wids.py file, the load_remote_shardlist function has been modified to handle a string    
source by opening it as a stream before loading JSON data. Additionally, a print statement has been added to the ShardListDataset constructor, and the
lengths attribute is now correctly derived from the self.shards attribute. The add_transform method in the same class now returns self, allowing for  
method chaining. In the wids_index.py file, the main function has been updated to read file names from stdin if the input is a single hyphen. The     
result dictionary now includes a wids_version key, and the name key is conditionally added only if it is not an empty string. The files key in the    
result dictionary has been renamed to shardlist.                                                                                                      

## Commit: 0.2.69 -> 0.2.71

a02f440 -> 3c40a3e @ 2023-10-30 10:55:47 -0700

The setup.py and webdataset/__init__.py files have been updated to reflect a new version. In webdataset/wids/wids.py, a partial function from the     
functools module has been introduced to specify the format for the default decoder. The ShardListDataset class's constructor now accepts a            
transformations parameter that can be set to "PIL" or "numpy" to determine the format of the images. If transformations is not a list, it is converted
into a list, and each transformation is verified to be callable. Additionally, the default_decoder function has been updated to raise a ValueError if 
an unknown format is specified for image extensions.                                                                                                  

## Commit: 0.2.68 -> 0.2.69

4bb1a6b -> a02f440 @ 2023-10-30 10:15:54 -0700

The webdataset package has been updated with a minor change in the webdataset/wids/wids_mmtar.py file, where a debug print statement has been removed 
from the MMIndexedTar class. This change is likely to clean up the console output during the usage of the library. No other functional changes or     
feature additions are indicated by the diff provided.                                                                                                 

## Commit: 0.2.65 -> 0.2.68

5dc6332 -> 4bb1a6b @ 2023-10-28 20:09:23 -0700

The webdataset library has undergone several updates. The setup.py script no longer requires a clean working tree to increment the version number. In 
the autodecode.py module, a .copy() method has been added to ensure the numpy array is copied when converting to a PyTorch tensor. The test_decode.py 
module now imports imread from imageio.v3 instead of imageio. The wids.py module has been refactored, with the TarFileReader and find_index_file      
functions being moved to separate modules, and the ShardListDataset constructor now supports an additional use_mmap parameter. Additionally, the      
ShardedSampler class has been documented to clarify its purpose in sampling from a ShardListDataset.                                                  

## Commit: 0.2.63 -> 0.2.65

c0e388d -> 5dc6332 @ 2023-10-25 13:04:31 -0700

The setup.py and webdataset/__init__.py files have been updated to reflect new functionality. In tasks.py, the release function has been modified to  
include a new version commit and push to the repository. The webdataset/wids/wids.py file has seen significant additions, including a default_decoder 
function for handling common file extensions and a find_index_file function to locate index files associated with datasets. The TarFileReader class   
constructor now accepts an index_file parameter and a verbose flag.                                                                                   

## Commit: 0.2.61 -> 0.2.63

977ee91 -> c0e388d @ 2023-10-20 13:25:46 -0700

The webdataset package has been updated to include a new subpackage webdataset.wids as part of its distribution. Additionally, the pipelinefilter     
decorator function within webdataset/filters.py has been enhanced with functools.update_wrapper to properly update the wrapper function's metadata.   

## Commit: 0.2.60 -> 0.2.61

fe15d64 -> 977ee91 @ 2023-10-18 11:54:21 -0700

Only formatting changes.                                                                                                                              

## Commit: 0.2.58 -> 0.2.60

b7be4da -> fe15d64 @ 2023-10-17 22:48:40 -0700

The GitHub Actions workflow for PyPI publishing has been updated to use a specific release version of the gh-action-pypi-publish action instead of the
master branch. In the webdataset package, the __init__.py and setup.py files have been modified to reflect a new package version. Additionally, the   
wids.py module within the webdataset package has received several updates: new docstrings have been added to explain the purpose and usage of         
functions and classes, a new ShardListDataset class has been introduced for indexable datasets based on a list of shards, and a new ShardedSampler    
class has been added for sampling in a way that preserves locality. The ShardListDataset class also includes methods for cache management and         
accessing dataset samples.                                                                                                                            

## Commit: 0.2.59 -> 0.2.58

b092eb6 -> b7be4da @ 2023-10-17 21:26:27 -0700

The update includes enhancements to the Python codebase, focusing on improving code readability and maintainability. Changes include the refactoring  
of multiline function calls into more readable formats, ensuring consistent use of whitespace, and the removal of unnecessary parentheses in          
for-loops. Additionally, the FluidInterface class has been updated with new methods for batch processing and decoding, with an emphasis on flexibility
and error handling. The WebDataset class has been modified to handle YAML configuration files more robustly. There are also minor adjustments to the  
test suite to refine the testing of dataset renaming functionality.                                                                                   

## Commit: 0.2.57 -> 0.2.59

abc1a5d -> b092eb6 @ 2023-09-21 16:17:57 -0700

The ShardWriter class in webdataset/writer.py has been updated to allow the verbosity level to be set when creating an instance. The verbose parameter
has been added to the constructor with a default value of 1, and the internal self.verbose attribute is now initialized with the value of the verbose 
argument instead of being hardcoded to 1.                                                                                                             

## Commit: 0.2.53 -> 0.2.57

e54effd -> abc1a5d @ 2023-09-21 15:54:07 -0700

The GitHub Actions workflows for PyPI (pypi.yml) and testing (test.yml) have been updated to run on Ubuntu 22.04 and use Python 3.10 instead of Python
3.9.                                                                                                                                                  

## Commit: 0.2.51 -> 0.2.53

faa774e -> e54effd @ 2023-09-12 16:23:51 -0700

This update includes a new test file testdata/ixtest.tar and a corresponding test suite TestIndexedTarSamples, TestLRUShards, and TestShardListDataset
in webdataset/tests/test_wids.py. Modifications in webdataset/tariterators.py improve exception handling by ensuring that the file object and URL are 
included in the exception arguments. The webdataset/__init__.py file is updated to reflect the new version of the library.                            

## Commit: 0.2.50 -> 0.2.51

e4c30ef -> faa774e @ 2023-06-11 21:08:54 -0700

Added a new rename_files parameter to the cached_tarfile_samples function in webdataset/cache.py, allowing for renaming files during the caching      
process. This parameter is also passed to the tar_file_expander function within the same file.                                                        

## Commit: 0.2.49 -> 0.2.50

039d743 -> e4c30ef @ 2023-06-11 09:47:53 -0700

The webdataset package has been updated with functional changes in the compat.py and __init__.py files. In compat.py, the FluidInterface class's      
batched method has been refactored for better readability without altering its functionality. The WebDataset class in the same file has been enhanced 
with a new rename_files parameter in the constructor and in the tariterators.tarfile_to_samples and caching.cached_tarfile_to_samples methods,        
allowing for file renaming during dataset processing. Additionally, the WebDataset constructor now supports YAML file URLs directly, streamlining the 
process of loading dataset specifications.                                                                                                            

## Commit: 0.2.47 -> 0.2.49

352089f -> 039d743 @ 2023-03-20 22:08:02 -0700

The webdataset package has undergone several updates to its test suite. New tests have been added to verify the expected number of samples in a       
WebDataset object, both for datasets created from locally hosted data and resampled data. The test_yaml3 function has been introduced to create a     
WebDataset from a YAML specification. Additional tests have been implemented to check the functionality of the with_length, repeat, slice, and rsample
methods. Error handling has been addressed with tests for EOF errors, custom decoder errors, and missing keys in to_tuple and rename methods. The     
rename method has been tested for its keep option functionality. The test suite also includes checks for the shuffle, decode, map, and map_dict       
methods, as well as for decoding to RGB8 numpy arrays, PIL images, and handling compressed data. The test_rgb8_np_vs_torch function has been added to 
compare numpy and PyTorch tensor outputs.                                                                                                             

## Commit: 0.2.46 -> 0.2.47

d05d8ff -> 352089f @ 2023-03-18 16:40:58 -0700

The git diff introduces a new FAQ.md file that addresses various issues and questions related to the use of WebDataset with PyTorch, torch-xla, and   
PyTorch Lightning. It provides solutions and workarounds for problems such as incorrect file path decoding on Windows, random stopping of training    
programs, incomplete dataset shuffling, and handling of small embeddings. It also discusses fixes for streaming images from an s3 bucket, handling the
"gopen handler not defined" error on Windows, and using tensorcom with WebDataset. Additionally, it covers the application of online filtering,       
support for horovod, changes to the length argument, corrections to the PytorchShardList docstring, the status of the WebDataset documentation        
website, and the internal handling of multiple workers when transforming data.                                                                        

## Commit: 0.2.43 -> 0.2.46

dfa3895 -> d05d8ff @ 2023-03-14 16:46:58 -0700

The webdataset package has been updated with a minor change in the cache.py module. The get_filetype function has been modified to redirect the output
of the file command to /dev/null, ensuring that the command's output does not clutter the console. Additionally, the import statement in the same file
now includes the time module.                                                                                                                         

## Commit: 0.2.39 -> 0.2.43

f2b64d8 -> dfa3895 @ 2023-03-14 12:29:47 -0700

The setup.py and webdataset/__init__.py files have been updated to reflect a new version. In webdataset/compat.py, the WebDataset class now reads     
WDS_CACHE_SIZE and WDS_CACHE environment variables to set cache_size and cache_dir respectively, and checks if the cache_dir exists. The              
webdataset/shardlists.py file introduces new functions envlookup and envsubst to substitute environment variables with a WDS_ prefix in URLs, and     
modifies the expand_urls function to use these substitutions and perform additional checks. Lastly, webdataset/tests/test_pipeline.py has been updated
to reflect the changes in environment variable handling, using the WDS_ prefix for the TESTDATA variable.                                             

## Commit: 0.2.35 -> 0.2.39

fa40da2 -> f2b64d8 @ 2023-03-09 12:01:27 -0800

This update includes a bug fix in webdataset/autodecode.py where a redundant assignment to the result variable was removed. In                        
webdataset/shardlists.py, environment variable expansion was added to URL processing. The webdataset/tests/test_pipeline.py was updated with a new    
test case to verify environment variable expansion in shard list URLs. Additionally, imports in multiple files were reorganized for clarity, and some 
error messages in webdataset/writer.py were streamlined.                                                                                              

## Commit: 0.2.34 -> 0.2.35

cb1aa32 -> fa40da2 @ 2023-03-03 17:21:19 -0800

The Python version metadata in readme.ipynb was simplified. In webdataset/cache.py, error handling was improved to account for potential race         
conditions where files might be deleted by other processes during cleanup operations. An exception type in webdataset/gopen.py was changed from       
Exception to IOError to more accurately reflect errors related to I/O operations.                                                                     

## Commit: 0.2.31 -> 0.2.34

9bc1eb5 -> cb1aa32 @ 2023-02-01 10:16:51 -0800

The webdataset documentation HTML has been streamlined, with several meta tags and links to external resources being updated or removed. The          
JavaScript references have been modified, including an update to jQuery and the addition of a conditional comment for supporting HTML5 elements in    
older versions of Internet Explorer. The navigation structure within the HTML has been simplified, with some elements being removed and others        
reorganized for clarity. Additionally, the Python code snippet in the README has been corrected to use loader instead of dataset when obtaining a     
batch. New documentation sections have been added to describe the Length Properties and Tar Header Overhead for the WebDataset instances, providing   
guidance on setting dataset length and handling tar file overhead. Only formatting changes.                                                           

## Commit: 0.2.30 -> 0.2.31

8458543 -> 9bc1eb5 @ 2022-11-29 14:31:29 -0800

The .gitignore file has been updated to exclude additional files such as openimages.py, rust-test.ipynb, test.cbors, test.py, and several others      
within the notebooks and webdataset/tests directories. The requirements.txt file now includes the dataclasses package as a dependency. In             
webdataset/autodecode.py, the image_extensions list has been replaced with a more comprehensive IMAGE_EXTENSIONS list, which includes a wider range of
image file extensions. Additionally, the ImageHandler and imagehandler functions have been updated to use the new IMAGE_EXTENSIONS list. The Decoder  
class has been modified to handle keys starting with double underscores and to attempt decoding byte values to UTF-8, with an error message if        
decoding fails.                                                                                                                                       

## Commit: 0.2.26 -> 0.2.30

6864382 -> 8458543 @ 2022-11-04 12:49:07 -0700

The setup.py script has been updated to reflect a new version. The tasks.py script has been modified to include a virtual environment activation and a
pytest command before committing changes. In the webdataset/__init__.py, an import statement for gopen and gopen_schemes has been added. The gopen.py 
module has seen several changes: the set_options function definition has been reformatted, the gopen_file function now strips the "file:" prefix from 
URLs, and the gopen_curl function has been updated to use the -X PUT flag for write mode. Additionally, an import os statement has been added to      
multi.py. The test_gopen.py file has been updated to use the gopen function directly instead of gopen.gopen.                                          

## Commit: 0.2.25 -> 0.2.26

9f9b0e3 -> 6864382 @ 2022-09-16 21:20:12 -0700

Fixed a bug in autodecode.py related to image decoding, ensuring proper handling of different image modes (L, RGB, RGBA) and data types (uint8,       
float). The ImageHandler class now includes checks for the number of dimensions and corrects the shape of the array based on the specified mode. It   
also includes assertions to verify the mode and array type, and converts images to the appropriate format for both NumPy and PyTorch. Additionally,   
the setup.py file was updated to include a trailing comma in the install_requires list.                                                               

