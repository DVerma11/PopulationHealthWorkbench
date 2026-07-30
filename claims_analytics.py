"""
Reusable claims analytics functions for the
Population Health Workbench.

The functions in this module transform Synthea
claims transaction data into claim-, patient-,
encounter-, and diagnosis-level analytical tables.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
# ==========================================================
# DATA PREPARATION
# ==========================================================

def prepare_claim_transactions(
    claims_transactions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare Synthea claims transaction data.

    Converts financial columns to numeric values and returns
    a copy so that the source DataFrame is not modified.
    """
    required_columns = {
        "CLAIMID",
        "PATIENTID",
        "TYPE",
        "AMOUNT",
        "PAYMENTS",
    }

    missing_columns = required_columns - set(claims_transactions.columns)

    if missing_columns:
        raise KeyError(
            "claims_transactions is missing required columns: "
            f"{sorted(missing_columns)}"
        )

    prepared = claims_transactions.copy()

    prepared["AMOUNT"] = pd.to_numeric(
        prepared["AMOUNT"],
        errors="coerce",
    )

    prepared["PAYMENTS"] = pd.to_numeric(
        prepared["PAYMENTS"],
        errors="coerce",
    )

    return prepared

# ==========================================================
# CLAIM FINANCIALS
# ==========================================================

def build_claim_financials(
    claims_transactions: pd.DataFrame,
) -> pd.DataFrame:

    """
    Aggregate transaction-level charges and payments to one
    record per claim.
    """
    transactions = prepare_claim_transactions(
        claims_transactions
    )

    charge_rows = transactions.loc[
        transactions["TYPE"].eq("CHARGE")
    ].copy()

    payment_rows = transactions.loc[
        transactions["TYPE"].eq("PAYMENT")
    ].copy()

    charge_aggregations: dict[str, tuple[str, str]] = {
        "Total_Charge": ("AMOUNT", "sum"),
    }

    payment_aggregations: dict[str, tuple[str, str]] = {
        "Total_Payment": ("PAYMENTS", "sum"),
    }

    if "CHARGEID" in charge_rows.columns:
        charge_aggregations["Charge_Transactions"] = (
            "CHARGEID",
            "count",
        )
    else:
        charge_aggregations["Charge_Transactions"] = (
            "TYPE",
            "size",
        )

    if "CHARGEID" in payment_rows.columns:
        payment_aggregations["Payment_Transactions"] = (
            "CHARGEID",
            "count",
        )
    else:
        payment_aggregations["Payment_Transactions"] = (
            "TYPE",
            "size",
        )

    claim_charges = (
        charge_rows
        .groupby("CLAIMID", as_index=False)
        .agg(**charge_aggregations)
    )

    claim_payments = (
        payment_rows
        .groupby("CLAIMID", as_index=False)
        .agg(**payment_aggregations)
    )

    claim_financials = (
        claim_charges
        .merge(
            claim_payments,
            on="CLAIMID",
            how="outer",
        )
        .fillna(
            {
                "Total_Charge": 0,
                "Charge_Transactions": 0,
                "Total_Payment": 0,
                "Payment_Transactions": 0,
            }
        )
    )

    return claim_financials

# Claims KPIs
def calculate_claim_kpis(
    claims: pd.DataFrame,
    claim_financials: pd.DataFrame,
) -> dict[str, Any]:
    """
    Calculate executive claims and payment KPIs.
    """
    total_claims = claims["Id"].nunique()

    positive_charges = claim_financials.loc[
        claim_financials["Total_Charge"] > 0,
        "Total_Charge",
    ]

    positive_payments = claim_financials.loc[
        claim_financials["Total_Payment"] > 0,
        "Total_Payment",
    ]

    total_charges = claim_financials["Total_Charge"].sum()
    total_payments = claim_financials["Total_Payment"].sum()

    payment_to_charge_ratio = (
        total_payments / total_charges
        if total_charges > 0
        else np.nan
    )

    return {
        "total_claims": int(total_claims),
        "claims_with_charges": int(
            claim_financials.loc[
                claim_financials["Total_Charge"] > 0,
                "CLAIMID",
            ].nunique()
        ),
        "total_charges": float(total_charges),
        "total_payments": float(total_payments),
        "average_claim_charge": float(
            positive_charges.mean()
        ),
        "median_claim_charge": float(
            positive_charges.median()
        ),
        "largest_claim_charge": float(
            claim_financials["Total_Charge"].max()
        ),
        "average_claim_payment": float(
            positive_payments.mean()
        ),
        "largest_claim_payment": float(
            claim_financials["Total_Payment"].max()
        ),
        "payment_to_charge_ratio": float(
            payment_to_charge_ratio
        ),
        "unpaid_balance": float(
            total_charges - total_payments
        ),
    }

#KPI table
def build_claim_kpi_table(
    metrics: dict[str, Any],
) -> pd.DataFrame:
    """
    Convert the claims KPI dictionary into a report-ready table.
    """
    labels = {
        "total_claims": "Total Claims",
        "claims_with_charges": "Claims with Charges",
        "total_charges": "Total Charges",
        "total_payments": "Total Payments",
        "average_claim_charge": "Average Claim Charge",
        "median_claim_charge": "Median Claim Charge",
        "largest_claim_charge": "Largest Claim Charge",
        "average_claim_payment": "Average Claim Payment",
        "largest_claim_payment": "Largest Claim Payment",
        "payment_to_charge_ratio": "Payment-to-Charge Ratio",
        "unpaid_balance": "Aggregate Unpaid Balance",
    }

    currency_metrics = {
        "total_charges",
        "total_payments",
        "average_claim_charge",
        "median_claim_charge",
        "largest_claim_charge",
        "average_claim_payment",
        "largest_claim_payment",
        "unpaid_balance",
    }

    count_metrics = {
        "total_claims",
        "claims_with_charges",
    }

    rows = []

    for metric_name, label in labels.items():
        value = metrics.get(metric_name)

        if value is None or pd.isna(value):
            formatted_value = "Not available"
        elif metric_name in currency_metrics:
            formatted_value = f"${value:,.2f}"
        elif metric_name == "payment_to_charge_ratio":
            formatted_value = f"{value:.1%}"
        elif metric_name in count_metrics:
            formatted_value = f"{int(value):,}"
        else:
            formatted_value = f"{value:,.2f}"

        rows.append(
            {
                "Metric": label,
                "Value": value,
                "Formatted Value": formatted_value,
            }
        )

    return pd.DataFrame(rows)

# Highest-cost claims
def get_highest_cost_claims(
    claim_financials: pd.DataFrame,
    claims: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Return the highest-cost claims with patient identifiers.
    """
    return (
        claim_financials
        .merge(
            claims[["Id", "PATIENTID"]],
            left_on="CLAIMID",
            right_on="Id",
            how="left",
        )
        .sort_values(
            "Total_Charge",
            ascending=False,
        )
        .head(top_n)
        .reset_index(drop=True)
    )

# Patient costs
def calculate_patient_costs(
    claims_transactions: pd.DataFrame,
    patients: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate charge concentration and claim volume by patient.
    """
    transactions = prepare_claim_transactions(
        claims_transactions
    )

    charge_rows = transactions.loc[
        transactions["TYPE"].eq("CHARGE")
    ].copy()

    transaction_count_column = (
        "CHARGEID"
        if "CHARGEID" in charge_rows.columns
        else "TYPE"
    )

    patient_costs = (
        charge_rows
        .groupby("PATIENTID", as_index=False)
        .agg(
            Total_Charges=("AMOUNT", "sum"),
            Number_of_Claims=("CLAIMID", "nunique"),
            Charge_Transactions=(
                transaction_count_column,
                "count",
            ),
        )
    )

    patient_columns = [
        column
        for column in [
            "Id",
            "FIRST",
            "LAST",
            "GENDER",
            "BIRTHDATE",
        ]
        if column in patients.columns
    ]

    patient_costs = patient_costs.merge(
        patients[patient_columns],
        left_on="PATIENTID",
        right_on="Id",
        how="left",
    )

    patient_costs = (
        patient_costs
        .sort_values(
            "Total_Charges",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    patient_costs["Patient"] = [
        f"Patient {index:03d}"
        for index in range(1, len(patient_costs) + 1)
    ]

    return patient_costs

# Cost by encounter type
def calculate_encounter_costs(
    claim_financials: pd.DataFrame,
    claims_transactions: pd.DataFrame,
    encounters: pd.DataFrame,
) -> pd.DataFrame:
    """
    Link claims to encounters through APPOINTMENTID and summarize
    cost by encounter class.
    """
    required_columns = {
        "CLAIMID",
        "PATIENTID",
        "APPOINTMENTID",
    }

    missing_columns = (
        required_columns
        - set(claims_transactions.columns)
    )

    if missing_columns:
        raise KeyError(
            "claims_transactions is missing encounter-linking "
            f"columns: {sorted(missing_columns)}"
        )

    claim_links = (
        claims_transactions[
            [
                "CLAIMID",
                "PATIENTID",
                "APPOINTMENTID",
            ]
        ]
        .drop_duplicates(subset="CLAIMID")
    )

    claim_costs = (
        claim_financials
        .merge(
            claim_links,
            on="CLAIMID",
            how="left",
        )
        .merge(
            encounters[
                [
                    "Id",
                    "ENCOUNTERCLASS",
                    "START",
                ]
            ],
            left_on="APPOINTMENTID",
            right_on="Id",
            how="left",
        )
    )

    encounter_costs = (
        claim_costs
        .groupby(
            "ENCOUNTERCLASS",
            dropna=False,
        )
        .agg(
            Total_Charges=("Total_Charge", "sum"),
            Total_Payments=("Total_Payment", "sum"),
            Number_of_Claims=("CLAIMID", "nunique"),
            Average_Charge_per_Claim=(
                "Total_Charge",
                "mean",
            ),
            Median_Charge_per_Claim=(
                "Total_Charge",
                "median",
            ),
        )
        .reset_index()
    )

    total_charges = encounter_costs[
        "Total_Charges"
    ].sum()

    encounter_costs["Share_of_Total_Charges"] = (
        encounter_costs["Total_Charges"]
        / total_charges
        if total_charges > 0
        else np.nan
    )

    return encounter_costs.sort_values(
        "Total_Charges",
        ascending=False,
    ).reset_index(drop=True)

# ==========================================================
# DIAGNOSIS ANALYSIS
# ==========================================================

# Cost by diagnosis
def calculate_diagnosis_costs(
    claim_financials: pd.DataFrame,
    claims: pd.DataFrame,
    conditions: pd.DataFrame,
    snomed_manual_map: dict[str, str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Attach primary diagnosis descriptions to claim-level financials.

    Returns
    -------
    diagnosis_costs:
        Claim-level clinical-financial records.

    diagnosis_summary:
        Diagnosis-level financial summary.
    """
    diagnosis_lookup = (
        conditions[["CODE", "DESCRIPTION"]]
        .drop_duplicates(subset="CODE")
        .copy()
    )

    diagnosis_lookup["CODE"] = (
        diagnosis_lookup["CODE"]
        .astype("string")
    )

    claims_for_merge = claims.copy()

    claims_for_merge["DIAGNOSIS1"] = (
        claims_for_merge["DIAGNOSIS1"]
        .astype("string")
    )

    claims_with_diagnosis = claims_for_merge.merge(
        diagnosis_lookup,
        left_on="DIAGNOSIS1",
        right_on="CODE",
        how="left",
    )

    diagnosis_costs = claim_financials.merge(
        claims_with_diagnosis[
            [
                "Id",
                "PATIENTID",
                "DIAGNOSIS1",
                "DESCRIPTION",
            ]
        ],
        left_on="CLAIMID",
        right_on="Id",
        how="left",
    )

    if snomed_manual_map:
        diagnosis_costs["DESCRIPTION"] = (
            diagnosis_costs["DESCRIPTION"]
            .fillna(
                diagnosis_costs["DIAGNOSIS1"]
                .astype(str)
                .map(snomed_manual_map)
            )
        )

    diagnosis_costs["DESCRIPTION"] = (
        diagnosis_costs["DESCRIPTION"]
        .fillna(
            "Unmapped SNOMED ("
            + diagnosis_costs["DIAGNOSIS1"].astype(str)
            + ")"
        )
    )

    diagnosis_summary = (
        diagnosis_costs
        .groupby(
            "DESCRIPTION",
            dropna=False,
        )
        .agg(
            Total_Charges=("Total_Charge", "sum"),
            Total_Payments=("Total_Payment", "sum"),
            Number_of_Claims=("CLAIMID", "nunique"),
            Unique_Patients=("PATIENTID", "nunique"),
            Average_Claim_Cost=(
                "Total_Charge",
                "mean",
            ),
        )
        .reset_index()
    )

    total_charges = diagnosis_summary[
        "Total_Charges"
    ].sum()

    diagnosis_summary["Share_of_Total_Charges"] = (
        diagnosis_summary["Total_Charges"]
        / total_charges
        if total_charges > 0
        else np.nan
    )

    diagnosis_summary = diagnosis_summary.sort_values(
        "Total_Charges",
        ascending=False,
    ).reset_index(drop=True)

    return diagnosis_costs, diagnosis_summary

# Curate clinical diagnosis view
def create_clinical_diagnosis_summary(
    diagnosis_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a disease-focused diagnosis view by excluding obvious
    findings, situations, procedures, and social concepts.
    """
    non_disease_patterns = (
        r"\(finding\)|"
        r"\(situation\)|"
        r"\(procedure\)|"
        r"\(observable entity\)|"
        r"\(social concept\)"
    )

    clinical_summary = diagnosis_summary.loc[
        ~diagnosis_summary["DESCRIPTION"]
        .str.contains(
            non_disease_patterns,
            case=False,
            na=False,
            regex=True,
        )
    ].copy()

    clinical_summary = clinical_summary.loc[
        ~clinical_summary["DESCRIPTION"]
        .str.startswith(
            "Unmapped SNOMED",
            na=False,
        )
    ]

    return clinical_summary.sort_values(
        "Total_Charges",
        ascending=False,
    ).reset_index(drop=True)

# Unmapped terminology summary
def create_unmapped_diagnosis_summary(
    diagnosis_summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return all explicitly labeled unmapped SNOMED concepts.
    """
    return (
        diagnosis_summary.loc[
            diagnosis_summary["DESCRIPTION"]
            .str.startswith(
                "Unmapped SNOMED",
                na=False,
            )
        ]
        .sort_values(
            "Total_Charges",
            ascending=False,
        )
        .reset_index(drop=True)
    )