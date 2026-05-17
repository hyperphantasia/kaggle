# %% [code]
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
################################################################################
#
#  PARIS FLOOD DATASET UPDATER
#
#  Description:
#    Automated data pipeline for collecting, processing, and publishing daily water
#    level monitoring data from French HubEau API to Kaggle Datasets for public access.
#    Script scheduled to run weekly via the Kaggle UI.
#
#  Author: Brooks (44°48′01″N 1°37′07″E)
#
#  Created: 2026-05-16
#
#  Version: 1.0
#
#  Dependencies:
#    - pandas >= 2.2.3
#    - requests >= 2.32.3
#    - kaggle >= 2.0.2 (for publishing)
#
#  Usage:
#    python paris_flood_updater.py
#
#  Configuration:
#    Modify the following dataclass instances to adjust behavior:
#      - APIConfig: API endpoint, metric, pagination
#      - StationConfig: Monitoring station codes, flood threshold
#      - KaggleConfig: Dataset slug, file paths
#
#  Output files:
#    - {output_dir}/paris_flood_dataset.csv
#      The main dataset with processed water level observations
#    - {output_dir}/dataset-metadata.json
#      Kaggle dataset metadata (title, license, tags, etc.)
#
#  Data sources:
#    Primary: HubEau API (French water data service)
#      Endpoint: https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab
#      Metric: HIXnJ (daily water level elaborated observations)
#
#  Publishing: Kaggle Datasets
#      Dataset: https://www.kaggle.com/datasets/grimespoint/paris-flood-dataset
#
#  Transformations applied:
#    1. Rename columns (from French) to English
#    2. Standardize datetime and numeric types
#    3. Map categorical values (validation status, quality, method)
#    4. Deduplicate records to prevent double-counting
#    5. Add derived metrics (flood_alert flag)
#    6. Sort columns
#
#  Error Handling:
#    - Network errors in API fetching are caught and logged
#    - Missing or malformed data is handled gracefully with coercion
#    - Fatal errors trigger script exit with error code 1
#
#  License:
#    CC0-1.0 (Public Domain)
#
#  Notes:
#    - Data fetching is incremental (only new days are fetched)
#    - The script skips updates if the dataset already covers yesterday
#    - Deduplication key: (station_code, date, water_level_mm)
#    - Flood alert threshold: 6000 mm (configurable via StationConfig)
#    - All dates handled in UTC/date-only format for consistency
#
#  Reading and listening: Thème de la couleur de la mer (the Offline)
#                         The castle in the forest (N.M) | The prestige (C.P.)
#
################################################################################

import os
import json
import sys
import requests
import subprocess
import pandas as pd
from datetime import date, timedelta
from typing import Optional, List, Dict, Tuple, Set
from dataclasses import dataclass, field

# ============================================================================
# CONFIGURATION
# ============================================================================


@dataclass
class APIConfig:
    """API configuration for HubEau data fetching.

    Attributes:
        base_url (str): The base URL for the HubEau API endpoint.
        metric (str): The hydrometric metric code to fetch ('HIXnJ' for aggregated max daily water level).
        max_per_page (int): Maximum number of records per API request page.
        timeout_seconds (int): Request timeout in seconds.
    """
    base_url: str = "https://hubeau.eaufrance.fr/api/v2/hydrometrie/obs_elab"
    metric: str = "HIXnJ"
    max_per_page: int = 20000
    timeout_seconds: int = 60


@dataclass
class StationConfig:
    """Station monitoring configuration.

    Attributes:
        codes (List[str]): List of station codes to monitor (Hydroportail/Vigicrues identifiers).
        flood_threshold_mm (int): Water level threshold in millimeters above which 
            flood_alert flag is set.
        earliest_date (str): Earliest date to fetch data from (ISO format YYYY-MM-DD).
    """
    codes: List[str] = None
    flood_threshold_mm: int = 6000
    earliest_date: str = "1900-01-01"

    def __post_init__(self):
        if self.codes is None:
            self.codes = [
                "F700000109",
                "F700000110",
                "F700000111",
                "F700000102",
                "F700000103",
            ]


@dataclass
class KaggleConfig:
    """Kaggle dataset publishing configuration.

    Combines dataset publication paths and metadata settings.

    Attributes:
        dataset_slug (str): Kaggle dataset identifier in format 'username/dataset-name'.
        input_csv (str): Path to input CSV file with {slug} placeholder for dynamic substitution.
        output_dir (str): Directory where processed files will be saved.
        output_filename (str): Name of the output CSV file.
        metadata_filename (str): Name of the dataset metadata JSON file.
        title (str): Human-readable title for the Kaggle dataset.
        keywords (list[str]): Search keywords/tags for dataset discovery.
        geospatial_coverage (str): Geographic location described in metadata.
        update_frequency (str): How often the dataset is updated (e.g., 'Weekly').
        license_name (str): SPDX license identifier (e.g., 'CC0-1.0').
        output_csv_path (str): Full path to output CSV (computed in __post_init__).
        metadata_path (str): Full path to metadata file (computed in __post_init__).
    """
    dataset_slug: str = "grimespoint/paris-flood-dataset"
    input_csv: str = "/kaggle/input/datasets/{slug}/paris_flood_dataset.csv"
    output_dir: str = "/kaggle/working/kaggle_dataset"
    output_filename: str = "paris_flood_dataset.csv"
    metadata_filename: str = "dataset-metadata.json"

    # Metadata fields
    title: str = "Paris flood dataset"
    keywords: list[str] = field(default_factory=lambda: [
        "tabular",
        "weather and climate",
        "environment",
        "europe",
        "time series analysis"
    ])
    geospatial_coverage: str = "Paris, France"
    update_frequency: str = "Weekly"
    license_name: str = "CC0-1.0"

    def __post_init__(self):
        """Initialize computed path attributes after dataclass initialization.

        Substitutes the dataset slug placeholder in input_csv and constructs
        full paths for output CSV and metadata files.
        """
        self.input_csv = self.input_csv.format(slug=self.dataset_slug)
        self.output_csv_path = os.path.join(
            self.output_dir, self.output_filename)
        self.metadata_path = os.path.join(
            self.output_dir, self.metadata_filename)


# Merge configs for easy access
API_CONFIG = APIConfig()
STATION_CONFIG = StationConfig()
KAGGLE_CONFIG = KaggleConfig()

# ============================================================================
# COLUMN MAPPINGS & TRANSFORMATIONS
# ============================================================================

API_TO_EN = {
    "code_site": "location_code",
    "code_station": "station_code",
    "date_obs_elab": "record_date",
    "resultat_obs_elab": "water_level_mm",
    "date_prod": "data_production_date",
    "code_statut": "validation_status_code",
    "libelle_statut": "validation_status",
    "code_methode": "production_method_code",
    "libelle_methode": "production_method",
    "code_qualification": "quality_code",
    "libelle_qualification": "quality_assessment",
    "longitude": "longitude",
    "latitude": "latitude",
    "grandeur_hydro_elab": "hubeau_elab_code",
}

EN_TO_API = {v: k for k, v in API_TO_EN.items()}

CATEGORICAL_MAPPINGS = {
    "validation_status": {
        "Donnée validée": "validated",
        "Donnée brute": "raw",
        "Donnée pré-validée": "pre-validated",
    },
    "quality_assessment": {
        "Bonne": "good",
        "Non qualifiée": "unqualified",
        "Douteuse": "dubious",
    },
    "production_method": {
        "Calculée": "calculated",
        "Mesurée": "measured",
        "Expertisée": "expert-reviewed",
    },
}

COLUMN_ORDER = [
    "location_code", "station_code", "record_date", "water_level_mm", "flood_alert",
    "data_production_date", "validation_status_code", "validation_status",
    "production_method_code", "production_method", "quality_code", "quality_assessment",
    "longitude", "latitude", "hubeau_elab_code",
]

# ============================================================================
# DATA LOADING & BASIC I/O
# ============================================================================


def load_csv(path: str) -> pd.DataFrame:
    """Load CSV file or return empty DataFrame if file doesn't exist.

    Args:
        path (str): Full path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data or empty DataFrame if file not found.

    Raises:
        pd.errors.ParserError: If CSV is malformed.
    """
    if os.path.exists(path):
        return pd.read_csv(path, low_memory=False)
    return pd.DataFrame()


def create_output_dir() -> None:
    """Create output directory if it doesn't exist.

    Creates the directory specified in KAGGLE_CONFIG.output_dir with parent
    directories as needed.
    """
    os.makedirs(KAGGLE_CONFIG.output_dir, exist_ok=True)


# ============================================================================
# COLUMN RENAMING & SCHEMA CONVERSION
# ============================================================================

def rename_to_api_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Rename English column names to API schema names.

    Args:
        df (pd.DataFrame): DataFrame with English column names.

    Returns:
        pd.DataFrame: New DataFrame with API column names, original unchanged.
    """
    if df.empty:
        return df.copy()
    columns_to_rename = {k: v for k, v in EN_TO_API.items() if k in df.columns}
    return df.rename(columns=columns_to_rename)


def rename_to_english(df: pd.DataFrame) -> pd.DataFrame:
    """Rename API column names to English schema names.

    Args:
        df (pd.DataFrame): DataFrame with API column names.

    Returns:
        pd.DataFrame: New DataFrame with English column names, original unchanged.
    """
    if df.empty:
        return df.copy()
    columns_to_rename = {k: v for k, v in API_TO_EN.items() if k in df.columns}
    return df.rename(columns=columns_to_rename)


# ============================================================================
# DATA NORMALIZATION & TYPE CONVERSION
# ============================================================================

def normalize_datetime_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert specified columns to pandas datetime type.

    Handles coercion of unparseable values to NaT.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): Column names to convert.

    Returns:
        pd.DataFrame: New DataFrame with normalized datetime columns.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def normalize_numeric_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert specified columns to numeric type.

    Handles coercion of unparseable values to NaN.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns (List[str]): Column names to convert.

    Returns:
        pd.DataFrame: New DataFrame with normalized numeric columns.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def standardize_for_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame to API schema with normalized types for comparison.

    Performs the following:
    - Renames columns to API schema
    - Normalizes datetime and numeric columns
    - Adds '_day_normalized' column for grouping by date
    - Converts station codes to strings

    Args:
        df (pd.DataFrame): Input DataFrame in any schema.

    Returns:
        pd.DataFrame: Standardized DataFrame ready for deduplication/comparison.
    """
    if df.empty:
        return df.copy()

    df = rename_to_api_schema(df)
    df = normalize_datetime_columns(df, ["date_obs_elab"])
    df = normalize_numeric_columns(df, ["resultat_obs_elab"])

    # Add normalized day column for grouping
    if "date_obs_elab" in df.columns:
        df["_day_normalized"] = df["date_obs_elab"].dt.normalize()

    if "code_station" in df.columns:
        df["code_station"] = df["code_station"].astype(str)

    return df


# ============================================================================
# DEDUPLICATION
# ============================================================================

def create_dedup_key(df: pd.DataFrame) -> pd.Series:
    """Create unique deduplication key from station, day, and water level value.

    Key format: "station_code_YYYY-MM-DD_value"

    Args:
        df (pd.DataFrame): DataFrame with standardized columns (from standardize_for_comparison).

    Returns:
        pd.Series: Deduplication keys, NaN for rows with missing components.
    """
    parts = []

    if "code_station" in df.columns:
        parts.append(df["code_station"].astype(str))

    if "_day_normalized" in df.columns:
        parts.append(df["_day_normalized"].dt.strftime("%Y-%m-%d"))

    if "resultat_obs_elab" in df.columns:
        parts.append(df["resultat_obs_elab"].astype(str))

    if not parts:
        return pd.Series(index=df.index, dtype="object")

    return "_".join(parts) if len(parts) == 1 else pd.Series(
        ["_".join(row) for row in zip(*parts)], index=df.index
    )


def remove_duplicates(existing: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Remove rows from 'new' that already exist in 'existing'.

    Uses station code, date, and water level value to identify duplicates.

    Args:
        existing (pd.DataFrame): Existing dataset to check against.
        new (pd.DataFrame): New data to filter.

    Returns:
        pd.DataFrame: Copy of 'new' with duplicate rows removed.
    """
    if existing.empty or new.empty:
        return new.copy()

    existing_std = standardize_for_comparison(existing)
    new_std = standardize_for_comparison(new)

    existing_keys = set(create_dedup_key(existing_std).dropna())
    new_keys = create_dedup_key(new_std)

    mask = ~new_keys.isin(existing_keys)
    result = new.iloc[new_keys[mask].index].copy()

    return result


# ============================================================================
# API FETCHING
# ============================================================================

def fetch_single_station_data(station_code: str, start_date: str) -> pd.DataFrame:
    """Fetch all hydrometric data for a single station from API.

    Fetches MAX_PER_PAGE records per request.
    Continues until no new data is found or yesterday's date is reached.

    Args:
        station_code (str): Hydroportail/Vigicrues station code (e.g., 'F700000109').
        start_date (str): Start date for data retrieval (ISO format YYYY-MM-DD).

    Returns:
        pd.DataFrame: Combined data from all pages with normalized datetime column.

    Raises:
        requests.RequestException: If API request fails (caught and logged).
    """
    session = requests.Session()
    frames = []
    cursor = start_date

    while True:
        params = {
            "code_entite": station_code,
            "grandeur_hydro_elab": API_CONFIG.metric,
            "date_debut_obs_elab": cursor,
            "size": API_CONFIG.max_per_page,
        }

        try:
            response = session.get(
                API_CONFIG.base_url,
                params=params,
                timeout=API_CONFIG.timeout_seconds
            )
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"Error fetching data for station {station_code}: {e}")
            break

        data = response.json().get("data", [])
        if not data:
            break

        page_df = pd.DataFrame(data)
        page_df = normalize_datetime_columns(page_df, ["date_obs_elab"])
        frames.append(page_df)

        last_page_date = page_df["date_obs_elab"].max()
        yesterday = date.today() - timedelta(days=1)

        if pd.isna(last_page_date) or last_page_date.date() >= yesterday:
            break

        cursor = (last_page_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        if len(data) < API_CONFIG.max_per_page:
            break

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


def fetch_all_data(start_date: str) -> pd.DataFrame:
    """ Orchestrator that fetches data for all configured stations starting 
    from a given date.

    Iterates through STATION_CONFIG.codes and calls fetch_single_station_data for each.

    Args:
        start_date (str): Start date for data retrieval (ISO format YYYY-MM-DD).

    Returns:
        pd.DataFrame: Combined data from all stations, empty if no data retrieved.
    """
    frames = []

    for station_code in STATION_CONFIG.codes:
        print(f"Fetching data for station {station_code}...")
        df_station = fetch_single_station_data(station_code, start_date)

        if not df_station.empty:
            frames.append(df_station)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


# ============================================================================
# UPDATE LOGIC
# ============================================================================

def determine_update_range(existing: pd.DataFrame) -> Tuple[bool, Optional[str]]:
    """Determine whether an update is needed and from what date to fetch.

    Checks if the existing dataset covers yesterday's date. If not, returns
    the next day after the last available data.

    Args:
        existing (pd.DataFrame): Existing dataset (any schema).

    Returns:
        Tuple[bool, Optional[str]]: (should_update, start_date_iso_string)
            - should_update: False if dataset is current, True if update needed
            - start_date_iso_string: ISO date string to start fetching from, 
              None if no update needed
    """
    existing_std = standardize_for_comparison(existing)

    if existing_std.empty or "_day_normalized" not in existing_std.columns:
        print("No existing data found. Will fetch all data from earliest date.")
        return True, STATION_CONFIG.earliest_date

    last_day = existing_std["_day_normalized"].max()
    yesterday = date.today() - timedelta(days=1)

    print(f"Last day in dataset: {last_day.date()}")

    if last_day.date() >= yesterday:
        print("Dataset already covers yesterday or later. No update needed.")
        return False, None

    next_day = (last_day + pd.Timedelta(days=1)).date()
    print(f"Will retrieve data starting from: {next_day}")

    return True, next_day.isoformat()


# ============================================================================
# POST-PROCESSING & TRANSFORMATIONS
# ============================================================================

def map_categorical_tolerant(series: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    """Map categorical values with case-insensitive fallback (categorical values translation).

    Attempts exact match first, then case-insensitive match for unmatched values.
    Preserves original values if no match is found.

    Args:
        series (pd.Series): Series with categorical values.
        mapping (Dict[str, str]): Dictionary mapping original to standardized values.

    Returns:
        pd.Series: Series with mapped values, original values preserved if unmapped.
    """
    if series is None or series.empty:
        return series.copy()

    result = series.copy()

    # Handle null/NaN values (keep them)
    null_mask = result.isna()

    # Convert non-null values to string for mapping
    non_null_series = result[~null_mask].astype(str)

    # Try exact match first
    mapped = non_null_series.map(mapping)

    # Case-insensitive fallback for unmapped values
    lower_mapping = {k.lower(): v for k, v in mapping.items()}
    unmapped_mask = mapped.isna()

    if unmapped_mask.any():
        lower_values = non_null_series[unmapped_mask].str.lower()
        case_insensitive_mapped = lower_values.map(lower_mapping)

        # For values that still don't match, keep original
        still_unmapped = case_insensitive_mapped.isna()
        case_insensitive_mapped[still_unmapped] = non_null_series[unmapped_mask][still_unmapped]

        mapped[unmapped_mask] = case_insensitive_mapped

    # Assign mapped values back to result
    result.loc[~null_mask] = mapped

    return result


def apply_categorical_mappings(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all categorical mappings from CATEGORICAL_MAPPINGS.

    Processes validation_status, quality_assessment, and production_method columns.

    Args:
        df (pd.DataFrame): DataFrame with English column names.

    Returns:
        pd.DataFrame: New DataFrame with mapped categorical columns.
    """
    df = df.copy()

    for col_name, mapping in CATEGORICAL_MAPPINGS.items():
        if col_name in df.columns:
            df[col_name] = map_categorical_tolerant(df[col_name], mapping)

    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add computed columns based on raw data.

    Currently adds:
    - flood_alert: Boolean flag if water_level_mm exceeds flood_threshold_mm

    Args:
        df (pd.DataFrame): DataFrame with English column names.

    Returns:
        pd.DataFrame: New DataFrame with derived columns appended.
    """
    df = df.copy()

    if "water_level_mm" in df.columns:
        df["flood_alert"] = df["water_level_mm"] > STATION_CONFIG.flood_threshold_mm

    return df


def order_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder columns to preferred order, appending any remaining columns.

    Uses COLUMN_ORDER list for preferred column sequence.

    Args:
        df (pd.DataFrame): DataFrame with any column order.

    Returns:
        pd.DataFrame: DataFrame with reordered columns.
    """
    present_cols = [c for c in COLUMN_ORDER if c in df.columns]
    other_cols = [c for c in df.columns if c not in present_cols]
    return df[present_cols + other_cols]


def postprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all post-processing transformations to combined dataset.

    Performs in order:
    1. Normalize datetime and numeric columns
    2. Rename to English schema
    3. Apply categorical mappings (translates values to english)
    4. Add derived columns
    5. Reorder columns to preferred order
    6. Sort by record_date and station_code
    7. Reset index

    Args:
        df (pd.DataFrame): Combined raw data in API schema.

    Returns:
        pd.DataFrame: Fully processed dataset ready for export.
    """
    if df.empty:
        return df

    df = df.copy()

    # Type conversions
    df = normalize_datetime_columns(df, ["date_obs_elab", "date_prod"])
    df = normalize_numeric_columns(df, ["resultat_obs_elab"])

    # Rename columns
    df = rename_to_english(df)

    # Apply mappings and derived columns
    df = apply_categorical_mappings(df)
    df = add_derived_columns(df)

    # Final organization
    df = order_columns(df)
    df = df.sort_values(["record_date", "station_code"]).reset_index(drop=True)

    return df


# ============================================================================
# METADATA & PUBLISHING
# ============================================================================

def create_metadata(df: pd.DataFrame, config: KaggleConfig) -> Dict:
    """Generate Kaggle dataset metadata dictionary.

    Constructs metadata from configuration and DataFrame temporal coverage,
    formatted for Kaggle dataset publishing.

    Args:
        df (pd.DataFrame): Processed DataFrame with 'record_date' column.
        config (KaggleConfig): Kaggle configuration containing metadata settings.

    Returns:
        Dict: Metadata dictionary in Kaggle-compatible format.
    """
    if df.empty or "record_date" not in df.columns:
        first_date = "unknown"
        last_date = "unknown"
    else:
        first_date = df["record_date"].min().strftime("%Y-%m-%d")
        last_date = df["record_date"].max().strftime("%Y-%m-%d")

    return {
        "title": config.title,
        "id": config.dataset_slug,
        "licenses": [{"name": config.license_name}],
        "keywords": config.keywords,
        "temporalCoverage": {
            "startDate": first_date,
            "endDate": last_date,
        },
        "geospatialCoverage": config.geospatial_coverage,
        "updateFrequency": config.update_frequency,
    }


def write_metadata(df: pd.DataFrame) -> None:
    """Write dataset metadata to JSON file.

    Calls create_metadata to generate content and writes to path specified
    in KAGGLE_CONFIG.metadata_path.

    Args:
        df (pd.DataFrame): Processed DataFrame to extract temporal coverage from.

    Raises:
        IOError: If file writing fails.
    """
    metadata = create_metadata(df, KAGGLE_CONFIG)

    with open(KAGGLE_CONFIG.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"Metadata written to {KAGGLE_CONFIG.metadata_path}")


def publish_to_kaggle() -> None:
    """Publish the updated dataset to Kaggle using the Kaggle CLI.

    Runs 'kaggle datasets version' command with current output directory.
    Includes timestamp in the version message.

    Raises:
        subprocess.CalledProcessError: If Kaggle CLI command fails.
    """
    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    message = f"Weekly update: {timestamp}"

    cmd = [
        "kaggle",
        "datasets",
        "version",
        "-p",
        KAGGLE_CONFIG.output_dir,
        "-m",
        message,
        "--dir-mode",
        "zip",
    ]

    print("Publishing to Kaggle...")
    print("Command:", " ".join(cmd))

    try:
        subprocess.run(cmd, check=True)
        print("Successfully published to Kaggle.")
    except subprocess.CalledProcessError as e:
        print(f"Error publishing to Kaggle: {e}")
        raise


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main() -> None:
    """Execute the main data update workflow.

    Performs the following steps:
    1. Setup output directory
    2. Load existing dataset from Kaggle
    3. Check if update is needed
    4. Fetch new data from HubEau API
    5. Combine and deduplicate with existing data
    6. Post-process dataset
    7. Write metadata and publish to Kaggle

    Exits early if no update is needed or no new data is available.
    Logs progress at each step.

    Raises:
        Exception: Any unhandled errors are caught and logged before exiting.
    """
    print("=" * 80)
    print("Paris Flood Dataset Update")
    print("=" * 80)

    # Setup
    create_output_dir()

    # Load existing data
    print("\n[1/6] Loading existing dataset...")
    existing_data = load_csv(KAGGLE_CONFIG.input_csv)
    print(f"  Loaded {len(existing_data)} rows")

    # Determine if update is needed
    print("\n[2/6] Checking if update is needed...")
    should_update, start_date = determine_update_range(existing_data)

    if not should_update:
        print("  No update needed. Exiting.")
        return

    # Fetch new data
    print("\n[3/6] Fetching new data from API...")
    new_data = fetch_all_data(start_date)

    if new_data.empty:
        print("  No new data retrieved. Exiting.")
        return

    print(f"  Fetched {len(new_data)} rows")

    # Combine and deduplicate
    print("\n[4/6] Processing data...")
    new_data_deduped = remove_duplicates(existing_data, new_data)
    print(f"  Adding {len(new_data_deduped)} new unique rows")

    # Convert existing to API schema and combine
    existing_api_schema = rename_to_api_schema(existing_data)
    if existing_api_schema.empty:
        combined_data = new_data_deduped.copy()
    else:
        combined_data = pd.concat(
            [existing_api_schema, new_data_deduped], ignore_index=True)

    if combined_data.empty:
        print("  No data to process. Exiting.")
        return

    print(f"  Total combined rows: {len(combined_data)}")

    # Post-process
    print("\n[5/6] Post-processing data...")
    final_data = postprocess(combined_data)
    final_data.to_csv(KAGGLE_CONFIG.output_csv_path, index=False)
    print(f"  Saved to {KAGGLE_CONFIG.output_csv_path}")

    # Write metadata and publish
    print("\n[6/6] Writing metadata and publishing a new dataset version on Kaggle...")
    write_metadata(final_data)
    publish_to_kaggle()

    print("\n" + "=" * 80)
    print("Update completed successfully!")
    print("=" * 80)


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}", file=sys.stderr)
        sys.exit(1)
