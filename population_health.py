"""
Reusable population health analytics functions.

This module supports patient-level disease burden,
utilization, cost, and risk-stratification analyses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Patient condition burden
def calculate_patient_condition_burden(
    conditions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate unique condition burden for each patient.
    """
    required_columns = {
        "PATIENT",
        "CODE",
    }

    missing_columns = required_columns - set(conditions.columns)

    if missing_columns:
        raise KeyError(
            "conditions is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    return (
        conditions
        .groupby("PATIENT", as_index=False)
        .agg(
            Unique_Conditions=("CODE", "nunique"),
            Condition_Records=("CODE", "size"),
        )
        .rename(columns={"PATIENT": "PATIENTID"})
        .sort_values(
            "Unique_Conditions",
            ascending=False,
        )
        .reset_index(drop=True)
    )


# population-level condition prevalence

def calculate_condition_prevalence(
    conditions: pd.DataFrame,
    total_patients: int,
) -> pd.DataFrame:
    """
    Calculate condition prevalence across the patient population.
    """
    required_columns = {
        "PATIENT",
        "CODE",
        "DESCRIPTION",
    }

    missing_columns = required_columns - set(conditions.columns)

    if missing_columns:
        raise KeyError(
            "conditions is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    if total_patients <= 0:
        raise ValueError(
            "total_patients must be greater than zero."
        )

    condition_prevalence = (
        conditions
        .groupby(
            ["CODE", "DESCRIPTION"],
            as_index=False,
        )
        .agg(
            Patients_With_Condition=("PATIENT", "nunique"),
            Condition_Records=("PATIENT", "size"),
        )
    )

    condition_prevalence["Prevalence_Rate"] = (
        condition_prevalence["Patients_With_Condition"]
        / total_patients
    )

    return (
        condition_prevalence
        .sort_values(
            [
                "Patients_With_Condition",
                "Condition_Records",
            ],
            ascending=False,
        )
        .reset_index(drop=True)
    )

# Patient utilization
def calculate_patient_utilization(
    encounters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate encounter utilization by patient.
    """
    required_columns = {
        "PATIENT",
        "Id",
        "START",
    }

    missing_columns = required_columns - set(encounters.columns)

    if missing_columns:
        raise KeyError(
            "encounters is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    working = encounters.copy()

    working["START"] = pd.to_datetime(
        working["START"],
        errors="coerce",
    )

    utilization = (
        working
        .groupby("PATIENT", as_index=False)
        .agg(
            Total_Encounters=("Id", "nunique"),
            First_Encounter=("START", "min"),
            Last_Encounter=("START", "max"),
        )
        .rename(columns={"PATIENT": "PATIENTID"})
    )

    if "ENCOUNTERCLASS" in working.columns:
        encounter_counts = (
            working
            .pivot_table(
                index="PATIENT",
                columns="ENCOUNTERCLASS",
                values="Id",
                aggfunc="nunique",
                fill_value=0,
            )
            .reset_index()
            .rename(columns={"PATIENT": "PATIENTID"})
        )

        encounter_counts.columns = [
            (
                column
                if column == "PATIENTID"
                else f"{str(column).title()}_Encounters"
            )
            for column in encounter_counts.columns
        ]

        utilization = utilization.merge(
            encounter_counts,
            on="PATIENTID",
            how="left",
        )

    utilization_columns = [
        column
        for column in utilization.columns
        if column.endswith("_Encounters")
    ]

    utilization[utilization_columns] = (
        utilization[utilization_columns]
        .fillna(0)
        .astype(int)
    )

    return (
        utilization
        .sort_values(
            "Total_Encounters",
            ascending=False,
        )
        .reset_index(drop=True)
    )

# Build patient population profile

def build_patient_population_profile(
    patients: pd.DataFrame,
    condition_burden: pd.DataFrame,
    patient_utilization: pd.DataFrame,
    patient_costs: pd.DataFrame,
) -> pd.DataFrame:
    """
    Combine demographics, clinical burden, utilization,
    and financial data into one patient-level table.
    """
    if "Id" not in patients.columns:
        raise KeyError(
            "patients is missing required column: Id"
        )

    required_cost_columns = {
        "PATIENTID",
        "Total_Charges",
        "Number_of_Claims",
    }

    missing_cost_columns = (
        required_cost_columns
        - set(patient_costs.columns)
    )

    if missing_cost_columns:
        raise KeyError(
            "patient_costs is missing required columns: "
            f"{sorted(missing_cost_columns)}"
        )

    profile = patients.copy().rename(
        columns={"Id": "PATIENTID"}
    )

    profile = (
        profile
        .merge(
            condition_burden,
            on="PATIENTID",
            how="left",
        )
        .merge(
            patient_utilization,
            on="PATIENTID",
            how="left",
        )
        .merge(
            patient_costs[
                [
                    "PATIENTID",
                    "Total_Charges",
                    "Number_of_Claims",
                ]
            ],
            on="PATIENTID",
            how="left",
        )
    )

    numeric_columns = [
        "Unique_Conditions",
        "Condition_Records",
        "Total_Encounters",
        "Total_Charges",
        "Number_of_Claims",
    ]

    encounter_class_columns = [
        column
        for column in profile.columns
        if (
            column.endswith("_Encounters")
            and column != "Total_Encounters"
        )
    ]

    numeric_columns.extend(encounter_class_columns)

    existing_numeric_columns = [
        column
        for column in numeric_columns
        if column in profile.columns
    ]

    profile[existing_numeric_columns] = (
        profile[existing_numeric_columns]
        .fillna(0)
    )

    return profile

# Initial rule-based risk stratification
def stratify_population_risk(
    population_profile: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a transparent, rule-based population health
    risk score.

    This is a demonstration framework and not a validated
    clinical risk-prediction model.
    """
    required_columns = {
        "Total_Charges",
        "Total_Encounters",
        "Unique_Conditions",
    }

    missing_columns = (
        required_columns
        - set(population_profile.columns)
    )

    if missing_columns:
        raise KeyError(
            "population_profile is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    profile = population_profile.copy()

    profile["Cost_Percentile"] = (
        profile["Total_Charges"].rank(
            pct=True,
            method="average",
        )
    )

    profile["Utilization_Percentile"] = (
        profile["Total_Encounters"].rank(
            pct=True,
            method="average",
        )
    )

    profile["Condition_Percentile"] = (
        profile["Unique_Conditions"].rank(
            pct=True,
            method="average",
        )
    )

    profile["Risk_Score"] = (
        0.40 * profile["Cost_Percentile"]
        + 0.35 * profile["Utilization_Percentile"]
        + 0.25 * profile["Condition_Percentile"]
    )

    profile["Risk_Tier"] = pd.cut(
        profile["Risk_Score"],
        bins=[
            -np.inf,
            0.50,
            0.75,
            np.inf,
        ],
        labels=[
            "Low",
            "Moderate",
            "High",
        ],
        include_lowest=True,
    ).astype(str)

    return (
        profile
        .sort_values(
            "Risk_Score",
            ascending=False,
        )
        .reset_index(drop=True)
    )

