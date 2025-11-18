from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("fraud-detection-pipeline")
except PackageNotFoundError:
    __version__ = "unknown"
