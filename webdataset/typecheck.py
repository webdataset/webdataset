modules = [
    "tenbin",
    "autodecode",
    "cache",
    "cborsiterators",
    "compat",
    "extradatasets",
    "filters",
    "gopen",
    "handlers",
    "mix",
    "pipeline",
    "shardlists",
    "tariterators",
    "utils",
    "writer",
]

try:
    from typeguard.importhook import install_import_hook

    for module in modules:
        install_import_hook("webdataset." + module)
except ImportError:
    pass
