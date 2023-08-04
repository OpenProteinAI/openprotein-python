try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version # py 3.7

__version__ = version("openprotein-python")