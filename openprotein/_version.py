try:
    from importlib.metadata import version
except ModuleNotFoundError:
    from importlib_metadata import version # py37

try:
    __version__ = version("openprotein-python")
except: 
    __version__ = "None"

