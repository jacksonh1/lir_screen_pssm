from pathlib import Path

from dotenv import load_dotenv
# from loguru import logger
from importlib_resources import files
import yaml
import pandas as pd
from dataclasses import dataclass

# Load environment variables from .env file if it exists
load_dotenv()

# ==============================================================================
# // main project paths
# ==============================================================================
PROJ_ROOT = Path(__file__).resolve().parents[1]
# logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# ==============================================================================
# // data loading
# ==============================================================================

@dataclass
class FilePaths:
    """Class to hold file paths for the project."""
    ilir_table: Path
    screening_hits_table: Path
    lir_central_table: Path
    full_screening_table: Path


RAWFILEPATHS = FilePaths(
    ilir_table=RAW_DATA_DIR / "iLIR_27.csv",
    screening_hits_table=RAW_DATA_DIR / "liradj_peptides_250411.csv",
    lir_central_table=RAW_DATA_DIR / "LIRCentral_filtered_sequences_250508.csv",
    full_screening_table=RAW_DATA_DIR / "231209_completedata_JK.csv"
)


# ==============================================================================
# // other environment variables
# ==============================================================================

STANDARD_AMINO_ACIDS = sorted(
    [
        "V",
        "E",
        "K",
        "I",
        "H",
        "L",
        "G",
        "T",
        "M",
        "N",
        "S",
        "P",
        "A",
        "F",
        "W",
        "Y",
        "Q",
        "R",
        "C",
        "D",
    ]
)
