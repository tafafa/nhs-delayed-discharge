"""
preprocessing.py
----------------
Data cleaning and feature engineering for NHS Delayed Discharge project.

Three datasets are processed and merged into a single modelling-ready DataFrame:
  - df1: NHS Delayed Discharge by Health Board (monthly, 2016–2024)
  - df2: Care Inspectorate Care Home dataset (active services snapshot)
  - df3: Mid-Year Population Estimates by NHS Health Board (annual)
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Dataset 1 — NHS Delayed Discharge
# ---------------------------------------------------------------------------

def load_nhs_delayed_discharge(filepath: str) -> pd.DataFrame:
    """Load and return the raw NHS Delayed Discharge CSV."""
    return pd.read_csv(filepath)


def preprocess_nhs_delayed_discharge(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and engineer features for the NHS Delayed Discharge dataset.

    Steps:
        1. Drop irrelevant / quality-flag columns
        2. Remove aggregated rows ('18plus', 'All Delay Reasons')
        3. Parse MonthOfDelay as datetime
        4. Pivot ReasonForDelay into separate columns
        5. Rename pivoted columns
        6. Compute TotalDelayedBedDays and lag feature PrevMonthDelay
        7. Extract Year, Month, Quarter from MonthOfDelay
        8. Filter to 2019–2024
        9. Compute delay-reason rate features
        10. Drop raw delay-reason count columns
        11. Standardise column names to snake_case
    """
    df = df.copy()

    # 1. Drop quality flag and unused columns
    drop_cols = [
        "_id", "HBTQF", "AgeGroupQF", "ReasonForDelayQF",
        "AverageDailyNumberOfDelayedBeds"
    ]
    df.drop(columns=drop_cols, errors="ignore", inplace=True)

    # 2. Remove aggregated rows
    df = df[df["AgeGroup"] != "18plus"]
    df = df[df["ReasonForDelay"] != "All Delay Reasons"]

    # 3. Parse datetime
    df["MonthOfDelay"] = pd.to_datetime(df["MonthOfDelay"].astype(str), format="%Y%m")

    # 4. Pivot delay reasons into columns
    df = df.pivot_table(
        index=["HBT", "MonthOfDelay"],
        columns="ReasonForDelay",
        values="NumberOfDelayedBedDays",
        aggfunc="sum"
    ).reset_index()
    df.columns.name = None

    # 5. Rename pivoted columns
    df = df.rename(columns={
        "Health and Social Care Reasons": "HealthSocial",
        "Patient and Family Related Reasons": "PatientFamily",
        "Code 9 AWI": "AWI",
        "Code 9 Non-AWI": "NonAWI"
    })

    # 6. Total delayed bed days and lag feature
    df["TotalDelayedBedDays"] = (
        df["HealthSocial"] + df["PatientFamily"] + df["AWI"] + df["NonAWI"]
    )
    df["PrevMonthDelay"] = df.groupby("HBT")["TotalDelayedBedDays"].shift(1)

    # 7. Temporal features
    df["Year"] = df["MonthOfDelay"].dt.year
    df["Month"] = df["MonthOfDelay"].dt.month
    df["Quarter"] = df["MonthOfDelay"].dt.quarter

    # 8. Filter to study period
    df = df[df["Year"].between(2019, 2024)]

    # 9. Delay-reason rate features
    df["HealthSocialRate"] = df["HealthSocial"] / df["TotalDelayedBedDays"]
    df["PatientFamilyRate"] = df["PatientFamily"] / df["TotalDelayedBedDays"]
    df["AWIRate"] = df["AWI"] / df["TotalDelayedBedDays"]
    df["NonAWIRate"] = df["NonAWI"] / df["TotalDelayedBedDays"]
    df = df.round(2)

    # 10. Drop raw count columns
    df.drop(columns=["AWI", "NonAWI", "HealthSocial", "PatientFamily"],
            errors="ignore", inplace=True)

    # 11. Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    return df


# ---------------------------------------------------------------------------
# Dataset 2 — Care Inspectorate Care Home Data
# ---------------------------------------------------------------------------

def load_care_inspectorate(filepath: str) -> pd.DataFrame:
    """Load and return the raw Care Inspectorate XLS file."""
    return pd.read_excel(filepath)


def preprocess_care_inspectorate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean Care Inspectorate data to produce care home capacity per Health Board.

    Steps:
        1. Keep only required columns
        2. Filter to active Care Home Services
        3. Aggregate TotalBeds by Health Board
        4. Rename column
        5. Standardise column names
    """
    df = df.copy()

    # 1. Keep relevant columns
    df = df[["CareService", "ServiceType", "ServiceStatus", "Health_Board_Name", "TotalBeds"]]

    # 2. Filter active care home services
    df = df[df["CareService"] == "Care Home Service"]
    df = df[df["ServiceStatus"] == "Active"]

    # 3. Aggregate
    df = df.groupby("Health_Board_Name")["TotalBeds"].sum().reset_index()

    # 4. Rename
    df = df.rename(columns={"TotalBeds": "CareHomePlaces"})

    # 5. Standardise column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

    return df


# ---------------------------------------------------------------------------
# Dataset 3 — Mid-Year Population Estimates
# ---------------------------------------------------------------------------

def load_population_estimates(filepath: str) -> pd.DataFrame:
    """Load and return the raw NRS Mid-Year Population Estimates Excel file."""
    return pd.read_excel(filepath)


def preprocess_population_estimates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean population estimates data to produce elderly population rate per Health Board per year.

    Steps:
        1. Filter to Sex == 'Persons'
        2. Standardise column names
    """
    df = df.copy()
    df = df[df["Sex"] == "Persons"]
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


# ---------------------------------------------------------------------------
# Merge all three datasets
# ---------------------------------------------------------------------------

def merge_datasets(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    df3: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge all three preprocessed datasets into a single modelling DataFrame.

    Merge order:
        df3 (population) LEFT JOIN df2 (care homes) → df2_3
        df2_3: compute carehomecapacityrate (carehomeplaces / pop75plus)
        df1 (NHS delays) LEFT JOIN df2_3 → final df
    """
    # Align key column names
    df2 = df2.rename(columns={"health_board_name": "healthboard"})
    df3 = df3.rename(columns={"area_name": "healthboard", "area_code": "hbt"})

    # Merge population + care homes
    df2_3 = pd.merge(df3, df2, on="healthboard", how="left")
    df2_3["carehomecapacityrate"] = (
        df2_3["carehomeplaces"] / df2_3["pop75plus"]
    ).round(2)

    # Merge with NHS discharge data
    df_final = pd.merge(df1, df2_3, on=["hbt", "year"], how="left")

    return df_final


def save_processed_data(df: pd.DataFrame, output_path: str) -> None:
    """Save the final merged DataFrame to CSV."""
    df.to_csv(output_path, index=False)
    print(f"Processed dataset saved to: {output_path}")
    print(f"Shape: {df.shape}")
