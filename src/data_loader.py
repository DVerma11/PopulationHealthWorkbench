from pathlib import Path
import pandas as pd

DATA_DIR = Path(
    r"C:\AI Projects\PopulationHealthWorkbench\data\raw"
)


def load_tables() -> dict[str, pd.DataFrame]:
    """Load all CSV files from the raw-data directory."""
    if not DATA_DIR.exists():
        raise FileNotFoundError(
            f"Raw data directory was not found: {DATA_DIR}"
        )

    tables = {
        csv_file.stem: pd.read_csv(csv_file)
        for csv_file in DATA_DIR.glob("*.csv")
    }

    if not tables:
        raise FileNotFoundError(
            f"No CSV files were found in: {DATA_DIR}"
        )

    return tables


