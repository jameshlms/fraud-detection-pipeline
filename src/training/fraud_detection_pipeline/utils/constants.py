from pathlib import Path
from typing import Final
from os import cpu_count


CURRENT_DIR: Final[Path] = Path(".")
PARENT_DIR: Final[Path] = Path("..")
RAW_DIRECTORY: Final[Path] = Path("data/raw")
PROCESSED_DIRECTORY: Final[Path] = Path("data/processed")
MODEL_DIRECTORY: Final[Path] = Path("models")
SCHEMA_DIRECTORY: Final[Path] = Path("models/schemas")

TARGET_COLUMN: Final[str] = "is_fraud"

CPU_COUNT: Final[int] = cpu_count() or 1
