class DictToObj:
    """Convert dictionary configs to objects with dot notation support."""
    def __init__(self, d):
        for key, value in d.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObj(value))
            else:
                setattr(self, key, value)

    def get(self, key, default=None):
        """Support dict-like .get() method"""
        return getattr(self, key, default)

    def __getitem__(self, key):
        return getattr(self, key)

    def __contains__(self, key):
        return hasattr(self, key)
