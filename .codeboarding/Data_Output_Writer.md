```mermaid

graph LR

    TarWriter["TarWriter"]

    ShardWriter["ShardWriter"]

    Generalized_I_O_Handler["Generalized I/O Handler"]

    ShardWriter -- "uses" --> TarWriter

    TarWriter -- "uses" --> Generalized_I_O_Handler

```



[![CodeBoarding](https://img.shields.io/badge/Generated%20by-CodeBoarding-9cf?style=flat-square)](https://github.com/CodeBoarding/GeneratedOnBoardings)[![Demo](https://img.shields.io/badge/Try%20our-Demo-blue?style=flat-square)](https://www.codeboarding.org/demo)[![Contact](https://img.shields.io/badge/Contact%20us%20-%20contact@codeboarding.org-lightgrey?style=flat-square)](mailto:contact@codeboarding.org)



## Details



The `webdataset.writer` module is central to the `Data Output/Writer` component, handling the serialization of processed data into WebDataset (TAR) format.



### TarWriter

This component is responsible for the low-level writing of individual data samples (represented as dictionaries) into a single `.tar` or compressed `.tar.gz` file. It manages the encoding of various data types (e.g., images, audio, text) into byte streams suitable for TAR archiving, including handling metadata and file properties. It acts as the core serialization engine for a single archive.





**Related Classes/Methods**:



- <a href="https://github.com/webdataset/webdataset/blob/main/src/webdataset/writer.py#L329-L484" target="_blank" rel="noopener noreferrer">`webdataset.writer.TarWriter` (329:484)</a>





### ShardWriter

This component orchestrates the creation of multiple sharded `.tar` files. It manages the logic for splitting the output data into new archives based on configurable limits, such as the maximum number of records or the maximum file size per shard. It utilizes the `TarWriter` for the actual writing operations to each individual shard.





**Related Classes/Methods**:



- <a href="https://github.com/webdataset/webdataset/blob/main/src/webdataset/writer.py#L487-L600" target="_blank" rel="noopener noreferrer">`webdataset.writer.ShardWriter` (487:600)</a>





### Generalized I/O Handler

This component (represented by `webdataset.gopen`) handles opening output file streams, allowing writing to local files, network streams, or other supported destinations. It provides a unified interface for various file system operations, abstracting away the underlying storage mechanism.





**Related Classes/Methods**:



- <a href="https://github.com/webdataset/webdataset/blob/main/src/webdataset/gopen.py#L523-L590" target="_blank" rel="noopener noreferrer">`webdataset.gopen` (523:590)</a>









### [FAQ](https://github.com/CodeBoarding/GeneratedOnBoardings/tree/main?tab=readme-ov-file#faq)