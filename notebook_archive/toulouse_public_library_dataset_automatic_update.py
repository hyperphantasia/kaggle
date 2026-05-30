# %% [code]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Toulouse public library top borrowed items dataset pipeline
===========================================================

ETL pipeline that fetches yearly borrowing statistics from the Toulouse Métropole 
Open Data API, repairs malformed records, cleans and normalizes the data 
and prepares the dataset for publication on the Kaggle platform.

Script scheduled to run monthly through the Kaggle UI.

Pipeline overview:
    1. Load existing dataset state.
    2. Detect missing years.
    3. Fetch new records from the API.
    4. Repair misaligned records.
    5. Clean and normalize data.
    6. Merge with existing dataset.
    7. Export dataset artifacts & publish metadata and updated dataset on Kaggle.

Author: brooks
Version: 1.0
Created: 2026-05-30
Last readings:  - Le Duc (Matteo Melchiorre)
                - Les Dieux ont soif (Anatole France)
                - Le mont Analogue (René Daumal)
Last play: La nuit des rois ou Tout ce que vous voulez
Listening: Dark Star (Beck)

License:
    CC0-1.0 (Public Domain)
"""

# ============================================================================
# IMPORTS
# ============================================================================

from __future__ import annotations

import csv
import logging
from pathlib import Path
import json
import subprocess

from dataclasses import dataclass, field
from typing import Any

import pandas as pd
import requests


# ============================================================================
# LOGGING
# ============================================================================

VERBOSE = False


def configure_logger(verbose: bool = False) -> logging.Logger:
    """Configure and return the application logger.

    Args:
        verbose: Whether debug logging should be enabled.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("toulouse_pipeline")

    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG if verbose else logging.INFO)

    formatter = logging.Formatter(
        "[%(levelname)s] %(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


logger = configure_logger(VERBOSE)


# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class APIConfig:
    """Configuration for Toulouse Open Data API access."""

    base_records_template: str = (
        "https://data.toulouse-metropole.fr/api/explore/v2.1/"
        "catalog/datasets/{}/records"
        "?lang=fr&timezone=Europe%2FBerlin"
    )

    media_types: list[str] = field(
        default_factory=lambda: ["films", "imprimes", "cds"]
    )

    timeout_seconds: int = 60

    def build_dataset_id(self, media_type: str) -> str:
        """Build dataset identifier for a media type.

        Args:
            media_type: Media category.

        Returns:
            Dataset identifier used by the API.
        """
        return (
            f"top-500-des-{media_type}-les-plus-empruntes-"
            f"a-la-bibliotheque-de-toulouse"
        )

    def get_records_url(
        self,
        media_type: str,
        year: int | None = None,
        limit: int = 100,
        offset: int = 0,
        full_columns: bool = False,
    ) -> str:
        """Build a records API URL.

        Args:
            media_type: Dataset category.
            year: Optional year filter.
            limit: Number of rows to fetch.
            offset: Pagination offset.
            full_columns: Whether all columns should be returned.

        Returns:
            Complete API URL.
        """
        dataset_id = self.build_dataset_id(media_type)
        base_url = self.base_records_template.format(dataset_id)

        params: dict[str, Any] = {
            "limit": limit,
            "offset": offset,
        }

        if not full_columns:
            params["select"] = "annee"

        if year is not None:
            params["where"] = f"annee = {year}"

        query_string = "&".join(
            f"{key}={value}"
            for key, value in params.items()
        )

        return f"{base_url}&{query_string}"


@dataclass
class ProcessingConfig:
    """Configuration for data cleaning and transformation."""

    mojibake_replacements: dict[str, str] = field(
        default_factory=lambda: {
            "ãa": "â",
            "ãe": "ê",
            "ãi": "î",
            "ão": "ô",
            "ãu": "û",
            "âe": "é",
            "áa": "à",
            "áe": "è",
            "ðc": "ç",
        }
    )

    column_mapping: dict[str, str] = field(
        default_factory=lambda: {
            "annee": "year",
            "nbre_de_prets": "nb_loans",
            "titre": "title",
            "auteur": "author",
            "editeur": "publisher",
            "indice": "classification",
            "bib": "library",
            "cote": "spine_label",
            "cat_1": "audience",
            "cat_2": "media_subtype",
        }
    )


@dataclass
class KaggleConfig:
    """Configuration for Kaggle dataset publishing."""

    dataset_slug: str = "grimespoint/toulouse-public-library-mdiathque-dataset"
    input_csv: str = f"/kaggle/input/datasets/{dataset_slug}/toulouse_public_library_loans.csv"

    output_dir: Path = Path("/kaggle/working/kaggle_dataset")
    output_filename: str = "toulouse_public_library_loans.csv"
    metadata_filename: str = "dataset-metadata.json"

    title: str = "Toulouse public library loans dataset"
    keywords: list[str] = field(default_factory=lambda: [
        "tabular",
        "literature",
        "music",
        "movies and tv shows",
        "culture and humanities",
    ])

    geospatial_coverage: str = "Toulouse, France"
    update_frequency: str = "Yearly"
    license_name: str = "CC0-1.0"

    def __post_init__(self) -> None:
        """Initialize derived filesystem paths using pathlib."""
        self.output_dir = Path(self.output_dir)

    @property
    def output_csv_path(self) -> Path:
        """Full path to output CSV file."""
        return self.output_dir / self.output_filename

    @property
    def metadata_path(self) -> Path:
        """Full path to metadata JSON file."""
        return self.output_dir / self.metadata_filename


# ============================================================================
# CONFIGURATION INSTANCES
# ============================================================================

API_CONFIG = APIConfig()
PROCESSING_CONFIG = ProcessingConfig()
KAGGLE_CONFIG = KaggleConfig()


# ============================================================================
# DOMAIN CONSTANTS
# ============================================================================

FIRST_DATA_YEAR = 2011

VALID_AUDIENCES: set[str] = {
    "A",
    "E",
    "BB",
    "TP",
}

EXPECTED_KEYS: list[str] = [
    "annee",
    "nbre_de_prets",
    "titre",
    "auteur",
    "editeur",
    "indice",
    "bib",
    "cote",
    "cat_1",
    "cat_2",
]

OUTPUT_COLUMNS: list[str] = [
    "year",
    "nb_loans",
    "title",
    "author",
    "publisher",
    "classification",
    "library",
    "spine_label",
    "audience",
    "media_subtype",
    "media_type",
]


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================


def is_valid_alignment(values: list[str]) -> bool:
    """Check whether a record appears correctly aligned.

    Args:
        values: Record values ordered according to EXPECTED_KEYS.

    Returns:
        True if the record alignment appears valid.
    """
    if len(values) != len(EXPECTED_KEYS):
        return False

    cat_1 = str(
        values[EXPECTED_KEYS.index("cat_1")]
    ).strip()

    return cat_1 in VALID_AUDIENCES


def fix_mojibake(
    value: object,
    mojibake_replacements: dict[str, str],
) -> object:
    """Repair known encoding issues.

    Args:
        value: Input value.
        mojibake_replacements: Replacement mapping.

    Returns:
        Cleaned value.
    """
    if not isinstance(value, str):
        return value

    original = value

    for wrong, correct in mojibake_replacements.items():
        if wrong in value:
            value = value.replace(wrong, correct)

            logger.debug(
                "Mojibake fix applied: '%s' → '%s' in '%s'",
                wrong,
                correct,
                original,
            )

    return value


# ============================================================================
# API CLIENT
# ============================================================================


def check_year_exists_in_api(
    media_type: str,
    year: int,
) -> bool:
    """Determine whether a year exists in the remote dataset.

    Args:
        media_type: Dataset category.
        year: Year to verify.

    Returns:
        True if at least one record exists.
    """
    url = API_CONFIG.get_records_url(
        media_type=media_type,
        year=year,
        limit=1,
        full_columns=False,
    )

    try:
        response = requests.get(
            url,
            timeout=API_CONFIG.timeout_seconds,
        )

        response.raise_for_status()

        return (
            response.json()
            .get("total_count", 0)
            > 0
        )

    except requests.RequestException as exc:
        logger.warning(
            "Error checking %s/%s: %s",
            media_type,
            year,
            exc,
        )
        return False


def fetch_records_by_year(
    media_type: str,
    year: int,
    limit: int = 100,
    full_columns: bool = True,
) -> pd.DataFrame:
    """Fetch all records for a media type and year.

    Args:
        media_type: Dataset category.
        year: Year to retrieve.
        limit: Records per API page.
        full_columns: Whether all columns should be requested.

    Returns:
        DataFrame containing all fetched records.
    """
    all_records: list[dict[str, Any]] = []

    offset = 0
    pagination_limit = 10_000

    while True:
        url = API_CONFIG.get_records_url(
            media_type=media_type,
            year=year,
            limit=limit,
            offset=offset,
            full_columns=full_columns,
        )

        try:
            response = requests.get(
                url,
                timeout=API_CONFIG.timeout_seconds,
            )

            response.raise_for_status()

            data = response.json()

            records = data.get("results", [])

            if not records:
                break
            # Fix misalignment at the source
            if full_columns:
                records = repair_misaligned_records(records)

            all_records.extend(records)

            if len(all_records) >= data.get(
                "total_count",
                0,
            ):
                break

            offset += limit

            if offset + limit >= pagination_limit:
                logger.warning(
                    (
                        "Approaching pagination limit. "
                        "Fetched %s/%s records."
                    ),
                    len(all_records),
                    data.get("total_count", 0),
                )
                break

        except requests.RequestException:
            logger.exception(
                "Error fetching %s/%s at offset %s",
                media_type,
                year,
                offset,
            )
            break

    if not all_records:
        return pd.DataFrame()

    df = pd.DataFrame(all_records)

    logger.info(
        "Fetched %s records for %s/%s",
        len(df),
        media_type,
        year,
    )

    return df


# ============================================================================
# DATA REPAIR
# ============================================================================


def repair_misaligned_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Repair records affected by column misalignment.

    Args:
        records: Raw API records.

    Returns:
        Repaired records.
    """
    repaired: list[dict[str, Any]] = []

    for record in records:
        values = [
            str(record.get(key, ""))
            for key in EXPECTED_KEYS
        ]

        if is_valid_alignment(values):
            repaired.append(record)
            continue

        fixed = False

        for index, value in enumerate(values):
            if value.strip() != "-":
                continue

            candidate = values[:index] + values[index + 1:]

            candidate.extend(
                [""] * (
                    len(EXPECTED_KEYS)
                    - len(candidate)
                )
            )

            candidate = candidate[: len(EXPECTED_KEYS)]

            if is_valid_alignment(candidate):
                fixed_record = {
                    key: (
                        candidate[position]
                        if candidate[position]
                        else None
                    )
                    for position, key in enumerate(EXPECTED_KEYS)
                }

                logger.debug(
                    "Repaired '%s' (removed stray '-' at %s)",
                    fixed_record.get("titre"),
                    EXPECTED_KEYS[index],
                )

                repaired.append(fixed_record)
                fixed = True
                break

        if not fixed:
            logger.warning(
                "Could not repair '%s' (cat_1='%s')",
                record.get("titre"),
                values[EXPECTED_KEYS.index("cat_1")],
            )

            repaired.append(record)

    return repaired

# ============================================================================
# DATA CLEANING & TRANSFORMATION
# ============================================================================


def clean_dataframe(df: pd.DataFrame, media_type: str) -> pd.DataFrame:
    """Clean and normalize a single dataset.

    Steps:
        - Drop incomplete rows
        - Fix encoding issues (mojibake)
        - Normalize category values
        - Convert types
        - Rename columns
        - Select final schema
        - Sort and deduplicate

    Args:
        df: Raw input DataFrame.
        media_type: Media category.

    Returns:
        Cleaned DataFrame.
    """
    if df.empty:
        return df

    df = df.copy()
    logger.debug("Starting cleanup: %s rows", len(df))

    # Drop missing critical values
    required_cols = [
        col for col in ["annee", "nbre_de_prets"]
        if col in df.columns
    ]

    if required_cols:
        before = len(df)
        df = df.dropna(subset=required_cols)
        logger.debug(
            "After dropna: %s rows (removed %s)",
            len(df),
            before - len(df),
        )

    # Fix encoding issues
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].apply(
            lambda v: fix_mojibake(
                v,
                PROCESSING_CONFIG.mojibake_replacements,
            )
        )

    # Normalize audience
    if "cat_1" in df.columns:

        def extract_category(value: str) -> str | None:
            text = str(value).strip()

            if text in VALID_AUDIENCES:
                return text

            for code in sorted(VALID_AUDIENCES, key=len, reverse=True):
                if code in text:
                    return code

            return None

        cat_series = df["cat_1"]
        df["cat_1"] = cat_series.apply(extract_category)

    # Type conversions
    for col in ["annee", "nbre_de_prets"]:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .astype("Int32")
            )

    # Rename columns
    df = df.rename(columns=PROCESSING_CONFIG.column_mapping)

    # Add metadata column
    df["media_type"] = media_type

    # Select output schema
    df = df[
        [
            col for col in OUTPUT_COLUMNS
            if col in df.columns
        ]
    ]

    # Final cleanup
    before = len(df)
    df = df.dropna(subset=["year"])
    logger.debug(
        "Final cleanup: %s rows (removed %s)",
        len(df),
        before - len(df),
    )

    sort_cols = [
        col for col in ["year", "nb_loans", "title"]
        if col in df.columns
    ]

    if sort_cols:
        df = df.sort_values(
            by=sort_cols,
            na_position="last",
        ).reset_index(drop=True)

    return df


def process_all_datasets(
    datasets: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Process and combine multiple media datasets.

    Args:
        datasets: Mapping of media type to DataFrame.

    Returns:
        combined_data cleaned DataFrame.
    """
    frames: list[pd.DataFrame] = []

    for media_type, df in datasets.items():
        if df.empty:
            logger.debug("Skipping %s (empty dataset)", media_type)
            continue

        logger.info("Processing %s", media_type)
        logger.info("Raw rows: %s", len(df))

        cleaned = clean_dataframe(df, media_type)

        logger.info(
            "Rows after cleaning: %s",
            len(cleaned),
        )

        if not cleaned.empty:
            frames.append(cleaned)

    if not frames:
        logger.warning(
            "No datasets produced output",
        )
        return pd.DataFrame()

    combined_data = pd.concat(frames, ignore_index=True)

    logger.info(
        "Total combined_data rows: %s",
        len(combined_data),
    )

    logger.info(
        "Distribution by media type:\n%s",
        combined_data["media_type"].value_counts(),
    )

    return combined_data


# ============================================================================
# DATA PLANNING (UPDATE LOGIC)
# ============================================================================

def get_latest_years_by_media_type(
    df: pd.DataFrame,
) -> dict[str, int | None]:
    """Get most recent year per media type.

    Args:
        df: existing_dataset dataset.

    Returns:
        Mapping of media type to latest year.
    """
    if df.empty or "year" not in df.columns:
        return {m: None for m in API_CONFIG.media_types}

    latest: dict[str, int | None] = {}

    for media_type in API_CONFIG.media_types:
        years = df.loc[
            df["media_type"] == media_type,
            "year",
        ].dropna()

        latest[media_type] = (
            int(years.max())
            if not years.empty
            else None
        )

    return latest


def compute_missing_years(
    latest_years: dict[str, int | None],
    first_year: int,
) -> tuple[dict[str, list[int]], bool]:
    """Compute which years need fetching.

    Args:
        latest_years: Latest known year per media type.
        first_year: Starting year if dataset is empty.

    Returns:
        Tuple of (years_to_fetch, has_updates).
    """
    updates: dict[str, list[int]] = {}

    for media_type, latest_year in latest_years.items():

        start_year = (
            latest_year + 1
            if latest_year is not None
            else first_year
        )

        years: list[int] = []
        year = start_year

        while check_year_exists_in_api(media_type, year):
            years.append(year)
            year += 1

        updates[media_type] = years

    logger.info("Update plan:")
    for media_type, years in updates.items():
        logger.info("  %s -> %s", media_type, years or "none")

    return updates, any(updates.values())


def fetch_updates(
    updates_needed: dict[str, list[int]],
) -> dict[str, pd.DataFrame]:
    """Fetch missing years from API.

    Args:
        updates_needed: Mapping of media type to years.

    Returns:
        Raw fetched datasets.
    """
    results: dict[str, pd.DataFrame] = {}

    for media_type, years in updates_needed.items():
        if not years:
            continue

        frames: list[pd.DataFrame] = []

        for year in years:
            logger.info("Fetching %s/%s", media_type, year)

            df = fetch_records_by_year(
                media_type=media_type,
                year=year,
                full_columns=True,
            )

            if not df.empty:
                frames.append(df)

        if frames:
            results[media_type] = pd.concat(
                frames,
                ignore_index=True,
            )

    return results


# ============================================================================
# STORAGE
# ============================================================================

def load_csv(path: str | Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Load CSV file into a DataFrame.

    Args:
        path: Path to CSV file.
        columns: Optional subset of columns.

    Returns:
        Loaded DataFrame or empty DataFrame if file does not exist.
    """
    path = Path(path)

    if not path.exists():
        return pd.DataFrame()

    try:
        kwargs: dict[str, object] = {
            "delimiter": ",",
            "na_values": ["", "NaN", "nan"],
        }

        if columns:
            kwargs["usecols"] = columns
        else:
            kwargs["dtype"] = {
                "year": "Int32",
                "nb_loans": "Int32",
            }

        return pd.read_csv(path, **kwargs)

    except Exception:
        logger.exception("Error loading CSV: %s", path)
        return pd.DataFrame()


def save_dataset(df: pd.DataFrame) -> None:
    """Save dataset as CSV and Parquet using pathlib paths.

    Args:
        df: Final processed dataset.
    """
    KAGGLE_CONFIG.output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        KAGGLE_CONFIG.output_csv_path,
        index=False,
        quoting=csv.QUOTE_NONNUMERIC,
        encoding="utf-8",
    )

    logger.info("CSV saved: %s", KAGGLE_CONFIG.output_csv_path)

    parquet_path = KAGGLE_CONFIG.output_csv_path.with_suffix(".parquet")

    df.to_parquet(
        parquet_path,
        index=False,
        compression="snappy",
    )

    logger.info("Parquet saved: %s", parquet_path)


# ============================================================================
# METADATA & PUBLISHING
# ============================================================================

def create_metadata(df: pd.DataFrame) -> dict:
    """Generate Kaggle dataset metadata.

    Args:
        df: Dataset.

    Returns:
        Metadata dictionary.
    """
    if df.empty or "year" not in df.columns:
        year_range = ("unknown", "unknown")
    else:
        years = df["year"].dropna()

        year_range = (
            (str(int(years.min())), str(int(years.max())))
            if not years.empty
            else ("unknown", "unknown")
        )

    return {
        "title": KAGGLE_CONFIG.title,
        "id": KAGGLE_CONFIG.dataset_slug,
        "licenses": [{"name": KAGGLE_CONFIG.license_name}],
        "keywords": KAGGLE_CONFIG.keywords,
        "temporalCoverage": {
            "startYear": FIRST_DATA_YEAR,
            "endYear": year_range[1],
        },
        "geospatialCoverage": KAGGLE_CONFIG.geospatial_coverage,
        "updateFrequency": KAGGLE_CONFIG.update_frequency,
    }


def write_metadata(df: pd.DataFrame) -> None:
    """Write dataset metadata to JSON file.

    Args:
        df: Dataset used to compute metadata.
    """
    KAGGLE_CONFIG.output_dir.mkdir(parents=True, exist_ok=True)

    metadata = create_metadata(df)

    with KAGGLE_CONFIG.metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info("Metadata written: %s", KAGGLE_CONFIG.metadata_path)


def publish_to_kaggle() -> None:
    """Publish dataset to Kaggle via CLI."""
    timestamp = pd.Timestamp.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    cmd = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        str(KAGGLE_CONFIG.output_dir),
        "-m",
        f"Yearly update: {timestamp}",
        "--dir-mode",
        "zip",
    ]

    logger.info("Publishing to Kaggle...")

    try:
        subprocess.run(cmd, check=True)
        logger.info("Kaggle publish successful")

    except subprocess.CalledProcessError as exc:
        logger.error("Kaggle publish failed: %s", exc)
        raise


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main() -> None:
    """Execute full data pipeline."""
    logger.info("Starting pipeline")

    existing_dataset = load_csv(KAGGLE_CONFIG.input_csv)
    logger.info("Loaded rows: %s", len(existing_dataset))

    latest = get_latest_years_by_media_type(existing_dataset)

    updates, has_updates = compute_missing_years(
        latest,
        FIRST_DATA_YEAR,
    )

    if not has_updates:
        logger.info("Dataset is already up to date")
        return

    raw_updates = fetch_updates(updates)

    new_data = process_all_datasets(raw_updates)

    if new_data.empty:
        logger.warning("No new data after processing")
        return

    combined_data = pd.concat(
        [existing_dataset, new_data],
        ignore_index=True,
    )

    combined_data = combined_data.drop_duplicates(
        subset=[
            "year",
            "title",
            "author",
            "media_type",
            "nb_loans",
        ],
        keep="last",
    )

    logger.info(
        "Final dataset size: %s (was %s)",
        len(combined_data),
        len(existing_dataset),
    )

    save_dataset(combined_data)
    write_metadata(combined_data)
    publish_to_kaggle(combined_data)

    logger.info("Pipeline completed successfully")


# ============================================================================
# ENTRYPOINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.critical("Fatal pipeline error", exc_info=True)
        raise
