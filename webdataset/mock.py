"""Mock implementations of torch interfaces when torch is not available."""


class IterableDataset:
    """Empty implementation of IterableDataset when torch is not available."""

    pass


class DataLoader:
    """Empty implementation of DataLoader when torch is not available."""

    pass
