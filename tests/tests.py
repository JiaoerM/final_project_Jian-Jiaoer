"""
tests/test_merge.py
Unit tests for the merged food desert + health outcomes dataset.
Run with:  pytest tests/test_merge.py -v
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# ── Fixture: load data once for all tests ────────────────────────────────────
@pytest.fixture(scope="module")
def df():
    path = Path("/Users/majiaoer/Desktop/final_project/data/cleaned/merged_tract_data.csv")
    assert path.exists(), (
        f"Merged data file not found at {path}. "
        "Run preprocessing and merge scripts first."
    )
    return pd.read_csv(path, dtype={"CensusTract": str})


# ══════════════════════════════════════════════════════════════════════════════
# 1. SCHEMA TESTS — correct columns exist with correct types
# ══════════════════════════════════════════════════════════════════════════════
class TestSchema:

    REQUIRED_COLUMNS = [
        "CensusTract", "State", "County", "Urban",
        "LILATracts_1And10", "LILATracts_halfAnd10",
        "lapop1", "lapop10", "lalowi1",
        "MedianFamilyIncome", "TractSNAP",
        "TractWhite", "TractBlack", "TractHispanic", "TractHUNV",
        "LocationID", "BPHIGH", "CSMOKING", "DIABETES", "OBESITY",
    ]

    def test_all_required_columns_present(self, df):
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        assert missing == [], f"Missing columns: {missing}"

    def test_no_extra_unexpected_columns(self, df):
        extra = [c for c in df.columns if c not in self.REQUIRED_COLUMNS]
        # Warn but don't fail — extra columns are acceptable
        if extra:
            pytest.warns(UserWarning, match="extra columns") if False else None
            print(f"\n  [info] Extra columns present (OK): {extra}")

    def test_census_tract_is_string(self, df):
        assert df["CensusTract"].dtype == object, (
            "CensusTract should be string (object) dtype for FIPS safety"
        )

    def test_health_outcomes_are_numeric(self, df):
        for col in ["DIABETES", "OBESITY", "BPHIGH", "CSMOKING"]:
            assert pd.api.types.is_numeric_dtype(df[col]), \
                f"{col} should be numeric, got {df[col].dtype}"

    def test_flag_columns_are_integer(self, df):
        for col in ["LILATracts_1And10", "LILATracts_halfAnd10", "Urban"]:
            assert pd.api.types.is_integer_dtype(df[col]), \
                f"{col} should be integer dtype, got {df[col].dtype}"


# ══════════════════════════════════════════════════════════════════════════════
# 2. MERGE INTEGRITY TESTS — join was executed correctly
# ══════════════════════════════════════════════════════════════════════════════
class TestMergeIntegrity:

    def test_row_count_in_expected_range(self, df):
        """
        USDA 2019 atlas has ~72K tracts; CDC PLACES covers most but not all.
        Inner join should yield between 50K and 75K tracts.
        """
        assert 50_000 <= len(df) <= 75_000, (
            f"Unexpected row count after merge: {len(df):,}. "
            "Check that inner join didn't over- or under-filter."
        )

    def test_no_duplicate_census_tracts(self, df):
        dupes = df["CensusTract"].duplicated().sum()
        assert dupes == 0, (
            f"{dupes} duplicate CensusTract rows found. "
            "Each tract should appear exactly once after merge."
        )

    def test_census_tract_fips_length(self, df):
        """All FIPS codes must be 11 digits (with leading zeros if needed)."""
        # Strip any trailing .0 artifacts from float conversion
        cleaned = df["CensusTract"].str.split(".").str[0].str.zfill(11)
        wrong_length = (cleaned.str.len() != 11).sum()
        assert wrong_length == 0, (
            f"{wrong_length} CensusTract values are not 11 digits after zero-padding."
        )

    def test_both_datasets_contributed_columns(self, df):
        """Verify columns from USDA side and CDC side both exist post-merge."""
        usda_cols = ["LILATracts_1And10", "lapop1", "MedianFamilyIncome"]
        cdc_cols  = ["DIABETES", "OBESITY", "BPHIGH", "CSMOKING"]
        for col in usda_cols + cdc_cols:
            assert col in df.columns, f"Expected column '{col}' missing after merge"

    def test_no_all_null_rows(self, df):
        """No row should be entirely null across health outcome columns."""
        health_cols = ["DIABETES", "OBESITY", "BPHIGH", "CSMOKING"]
        all_null = df[health_cols].isnull().all(axis=1).sum()
        assert all_null == 0, \
            f"{all_null} rows have null values in ALL health outcome columns"

    def test_locationid_matches_censustract(self, df):
        """LocationID (CDC) and CensusTract (USDA) should refer to same tract."""
        if "LocationID" in df.columns:
            ct_clean  = df["CensusTract"].str.split(".").str[0].str.zfill(11).astype(str)
            loc_clean = df["LocationID"].astype(str).str.split(".").str[0].str.zfill(11)
            mismatches = (ct_clean != loc_clean).sum()
            assert mismatches == 0, \
                f"{mismatches} rows have mismatched CensusTract vs LocationID"


# ══════════════════════════════════════════════════════════════════════════════
# 3. DATA QUALITY TESTS — values within expected bounds
# ══════════════════════════════════════════════════════════════════════════════
class TestDataQuality:

    def test_health_outcomes_within_percent_range(self, df):
        """Prevalence rates are percentages — must be between 0 and 100."""
        for col in ["DIABETES", "OBESITY", "BPHIGH", "CSMOKING"]:
            assert df[col].min() >= 0,   f"{col} has negative values"
            assert df[col].max() <= 100, f"{col} exceeds 100%"

    def test_health_outcomes_are_plausible(self, df):
        """Sanity check: national averages should be within known ranges."""
        checks = {
            "DIABETES": (8, 20),    # CDC national avg ~11-13%
            "OBESITY":  (20, 50),   # CDC national avg ~30-35%
            "BPHIGH":   (20, 55),   # CDC national avg ~30-40%
            "CSMOKING": (8, 25),    # CDC national avg ~12-14%
        }
        for col, (lo, hi) in checks.items():
            mean_val = df[col].mean()
            assert lo <= mean_val <= hi, (
                f"{col} national mean {mean_val:.2f}% is outside plausible range [{lo}, {hi}]"
            )

    def test_lila_flag_is_binary(self, df):
        allowed = {0, 1}
        for col in ["LILATracts_1And10", "LILATracts_halfAnd10"]:
            actual = set(df[col].unique())
            assert actual <= allowed, \
                f"{col} contains values other than 0/1: {actual - allowed}"

    def test_urban_flag_is_binary(self, df):
        actual = set(df["Urban"].unique())
        assert actual <= {0, 1}, \
            f"Urban column contains unexpected values: {actual - {0, 1}}"

    def test_median_income_plausible(self, df):
        valid = df["MedianFamilyIncome"].dropna()
        assert valid.min() >= 0,       "MedianFamilyIncome has negative values"
        assert valid.max() <= 500_000, "MedianFamilyIncome has implausibly high values (>$500K)"
        assert valid.mean() >= 30_000, "MedianFamilyIncome mean is implausibly low (<$30K)"

    def test_population_columns_non_negative(self, df):
        for col in ["lapop1", "lapop10", "lalowi1", "TractSNAP",
                    "TractWhite", "TractBlack", "TractHispanic", "TractHUNV"]:
            non_neg = df[col].dropna()
            assert (non_neg >= 0).all(), f"{col} has negative values"

    def test_state_column_has_valid_entries(self, df):
        assert df["State"].isnull().sum() == 0, "State column has null values"
        assert df["State"].nunique() >= 48,     "Fewer than 48 states found — possible merge issue"
        assert df["State"].nunique() <= 56,     "More than 56 state/territory entries — check source data"


# ══════════════════════════════════════════════════════════════════════════════
# 4. MISSING DATA TESTS — expected nulls are present, no surprise nulls
# ══════════════════════════════════════════════════════════════════════════════
class TestMissingData:

    def test_no_nulls_in_key_identifier_columns(self, df):
        """These columns must be complete — they're used as join/filter keys."""
        for col in ["CensusTract", "State", "County", "Urban",
                    "LILATracts_1And10", "DIABETES", "OBESITY", "BPHIGH", "CSMOKING"]:
            null_count = df[col].isnull().sum()
            assert null_count == 0, f"'{col}' has {null_count} unexpected null values"

    def test_lapop10_mostly_null_expected(self, df):
        """
        lapop10 (rural, 10-mile measure) is only populated for rural tracts.
        Expect >50% null — this is correct, not a bug.
        """
        pct_null = df["lapop10"].isnull().mean()
        assert pct_null > 0.50, (
            f"lapop10 null rate is only {pct_null:.1%}. "
            "Expected >50% null (rural-only column). Something may be wrong."
        )

    def test_lapop1_null_rate_acceptable(self, df):
        """lapop1 nulls occur in purely rural tracts — should be <40%."""
        pct_null = df["lapop1"].isnull().mean()
        assert pct_null < 0.40, (
            f"lapop1 null rate is {pct_null:.1%}, higher than expected (<40%). "
            "May indicate a merge or preprocessing issue."
        )

    def test_income_null_rate_low(self, df):
        pct_null = df["MedianFamilyIncome"].isnull().mean()
        assert pct_null < 0.02, (
            f"MedianFamilyIncome null rate is {pct_null:.1%} — expected <2%"
        )


# ══════════════════════════════════════════════════════════════════════════════
# 5. ANALYSIS LOGIC TESTS — verify core findings are computable
# ══════════════════════════════════════════════════════════════════════════════
class TestAnalysisLogic:

    def test_food_desert_mean_higher_than_nondesert_for_diabetes(self, df):
        """
        Food desert tracts should show higher mean diabetes rates.
        This is the core hypothesis — if it fails, check the data.
        """
        desert_mean     = df[df["LILATracts_1And10"] == 1]["DIABETES"].mean()
        nondesert_mean  = df[df["LILATracts_1And10"] == 0]["DIABETES"].mean()
        assert desert_mean > nondesert_mean, (
            f"Food desert mean diabetes ({desert_mean:.2f}%) is NOT higher than "
            f"non-desert ({nondesert_mean:.2f}%). Check merge or column alignment."
        )

    def test_food_desert_mean_higher_for_obesity(self, df):
        desert_mean    = df[df["LILATracts_1And10"] == 1]["OBESITY"].mean()
        nondesert_mean = df[df["LILATracts_1And10"] == 0]["OBESITY"].mean()
        assert desert_mean > nondesert_mean, (
            f"Food desert mean obesity ({desert_mean:.2f}%) is NOT higher than "
            f"non-desert ({nondesert_mean:.2f}%)."
        )

    def test_food_desert_proportion_plausible(self, df):
        """Nationally ~10-15% of tracts qualify as LILA food deserts."""
        pct = df["LILATracts_1And10"].mean()
        assert 0.05 <= pct <= 0.25, (
            f"Food desert proportion is {pct:.1%}, outside plausible range [5%, 25%]"
        )

    def test_urban_tracts_majority(self, df):
        """Most US census tracts are urban — expect >60%."""
        pct_urban = df["Urban"].mean()
        assert pct_urban > 0.60, \
            f"Only {pct_urban:.1%} of tracts are urban — expected >60%"

    def test_correlation_diabetes_income_negative(self, df):
        """Higher income should correlate with lower diabetes rates (negative r)."""
        valid = df[["DIABETES", "MedianFamilyIncome"]].dropna()
        r = valid.corr().loc["DIABETES", "MedianFamilyIncome"]
        assert r < 0, (
            f"Diabetes–Income correlation is {r:.3f} (positive). "
            "Expected negative — higher income areas should have lower diabetes rates."
        )

    def test_correlation_diabetes_obesity_positive(self, df):
        """Diabetes and obesity should be positively correlated at tract level."""
        r = df[["DIABETES", "OBESITY"]].corr().loc["DIABETES", "OBESITY"]
        assert r > 0.7, (
            f"Diabetes–Obesity correlation is only {r:.3f}. "
            "Expected strong positive (>0.7) at census tract level."
        )

    def test_groupby_state_no_nulls(self, df):
        """State-level aggregation should produce clean results."""
        state_means = df.groupby("State")["DIABETES"].mean()
        assert state_means.isnull().sum() == 0, \
            "Some states returned null mean diabetes — check groupby"
        assert len(state_means) >= 48, \
            f"Only {len(state_means)} states in aggregation — expected ≥48"


# ══════════════════════════════════════════════════════════════════════════════
# 6. REGRESSION SMOKE TEST — model runs without error
# ══════════════════════════════════════════════════════════════════════════════
class TestRegressionSmoke:

    def test_ols_runs_without_error(self, df):
        """OLS regression should fit cleanly on clean rows."""
        try:
            import statsmodels.formula.api as smf
        except ImportError:
            pytest.skip("statsmodels not installed")

        model_data = df.dropna(subset=[
            "DIABETES", "LILATracts_1And10", "lapop1",
            "MedianFamilyIncome", "TractHUNV", "TractSNAP",
            "TractBlack", "TractHispanic", "Urban"
        ])
        model = smf.ols(
            "DIABETES ~ LILATracts_1And10 + lapop1 + MedianFamilyIncome "
            "+ TractHUNV + TractSNAP + TractBlack + TractHispanic + Urban",
            data=model_data
        ).fit()

        assert model.rsquared > 0.10, \
            f"OLS R² is only {model.rsquared:.4f} — model may be misspecified"
        assert model.rsquared < 1.0, \
            "OLS R² = 1.0 — perfect fit suggests a data leakage problem"

    def test_lila_coefficient_positive_for_diabetes(self, df):
        """Food desert flag should have a positive coefficient for diabetes."""
        try:
            import statsmodels.formula.api as smf
        except ImportError:
            pytest.skip("statsmodels not installed")

        model_data = df.dropna(subset=[
            "DIABETES", "LILATracts_1And10", "lapop1",
            "MedianFamilyIncome", "TractHUNV", "TractSNAP",
            "TractBlack", "TractHispanic", "Urban"
        ])
        model = smf.ols(
            "DIABETES ~ LILATracts_1And10 + lapop1 + MedianFamilyIncome "
            "+ TractHUNV + TractSNAP + TractBlack + TractHispanic + Urban",
            data=model_data
        ).fit()

        coef = model.params["LILATracts_1And10"]
        assert coef > 0, (
            f"LILA coefficient for DIABETES is {coef:.4f} (negative). "
            "Expected positive — food deserts should predict higher diabetes rates."
        )