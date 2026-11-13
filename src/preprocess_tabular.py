"""Preprocess transaction and social-profile CSVs for Formative 2.

Run from the repository root:
    python src/preprocess_tabular.py

Expected input locations:
    data/raw/customer_transactions.csv
    data/raw/customer_social_profiles.csv

The social ID normalization (A151 -> 151) is a candidate mapping and must be
confirmed against the dataset documentation before it is treated as authoritative.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TX_CANDIDATES = [
    ROOT / "data/raw/customer_transactions.csv",
    ROOT / "customer_transactions.csv",
    ROOT / "customer_transactions - customer_transactions.csv",
]
SOCIAL_CANDIDATES = [
    ROOT / "data/raw/customer_social_profiles.csv",
    ROOT / "customer_social_profiles.csv",
    ROOT / "customer_social_profiles - customer_social_profiles.csv",
]

def locate(candidates):
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError("Missing input file. Checked:\n" + "\n".join(map(str, candidates)))

def stable_mode(values):
    values = values.dropna().astype(str)
    if values.empty:
        return pd.NA
    counts = values.value_counts()
    return sorted(counts[counts == counts.max()].index)[0]

def main():
    tx = pd.read_csv(locate(TX_CANDIDATES))
    social = pd.read_csv(locate(SOCIAL_CANDIDATES))

    required_tx = {"customer_id_legacy", "transaction_id", "purchase_amount",
                   "purchase_date", "product_category", "customer_rating"}
    required_social = {"customer_id_new", "social_media_platform",
                       "engagement_score", "purchase_interest_score", "review_sentiment"}
    if missing := required_tx - set(tx.columns):
        raise ValueError(f"Transactions file missing columns: {sorted(missing)}")
    if missing := required_social - set(social.columns):
        raise ValueError(f"Social file missing columns: {sorted(missing)}")

    tx["customer_id_legacy"] = tx["customer_id_legacy"].astype("string").str.strip()
    social["customer_id_new"] = social["customer_id_new"].astype("string").str.strip()
    tx["purchase_date"] = pd.to_datetime(tx["purchase_date"], errors="coerce")
    tx["product_category"] = tx["product_category"].astype("string").str.strip()
    for col in ["purchase_amount", "customer_rating"]:
        tx[col] = pd.to_numeric(tx[col], errors="coerce")
    for col in ["engagement_score", "purchase_interest_score"]:
        social[col] = pd.to_numeric(social[col], errors="coerce")

    tx = tx.dropna(subset=["customer_id_legacy", "product_category"]).copy()
    social["customer_id_candidate"] = social["customer_id_new"].str.replace(r"^A", "", regex=True)

    tx_ids = set(tx["customer_id_legacy"].dropna().astype(str))
    social_ids = set(social["customer_id_candidate"].dropna().astype(str))
    overlap = tx_ids & social_ids
    print(f"Transaction rows: {len(tx)}")
    print(f"Social rows: {len(social)}")
    print(f"Unique transaction customers: {len(tx_ids)}")
    print(f"Unique social customers after candidate normalization: {len(social_ids)}")
    print(f"Overlapping customer IDs: {len(overlap)}")
    print(f"Transaction customers without social match: {len(tx_ids - social_ids)}")
    print(f"Social customers without transaction match: {len(social_ids - tx_ids)}")

    social_summary = social.groupby("customer_id_candidate").agg(
        social_record_count=("customer_id_new", "size"),
        distinct_platform_count=("social_media_platform", "nunique"),
        engagement_mean=("engagement_score", "mean"),
        engagement_max=("engagement_score", "max"),
        engagement_std=("engagement_score", "std"),
        purchase_interest_mean=("purchase_interest_score", "mean"),
        purchase_interest_max=("purchase_interest_score", "max"),
        purchase_interest_std=("purchase_interest_score", "std"),
        primary_platform=("social_media_platform", stable_mode),
        dominant_sentiment=("review_sentiment", stable_mode),
    ).reset_index()

    platform_flags = pd.crosstab(
        social["customer_id_candidate"], social["social_media_platform"]
    ).gt(0).astype("int8")
    platform_flags.columns = [
        "platform_" + str(c).strip().lower().replace(" ", "_")
        for c in platform_flags.columns
    ]
    platform_flags = platform_flags.reset_index()
    social_summary = social_summary.merge(
        platform_flags, on="customer_id_candidate", validate="one_to_one"
    )

    merged = tx.merge(
        social_summary,
        left_on="customer_id_legacy",
        right_on="customer_id_candidate",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    assert len(merged) == len(tx), "Merge changed transaction row count."
    assert merged["transaction_id"].is_unique, "Transaction IDs duplicated after merge."

    matched = int((merged["_merge"] == "both").sum())
    unmatched = int((merged["_merge"] == "left_only").sum())
    print(f"Matched transaction rows: {matched}")
    print(f"Unmatched transaction rows: {unmatched}")
    merged = merged.drop(columns=["customer_id_candidate", "_merge"])
    merged["purchase_year"] = merged["purchase_date"].dt.year
    merged["purchase_month"] = merged["purchase_date"].dt.month
    merged["purchase_dayofweek"] = merged["purchase_date"].dt.dayofweek

    out = ROOT / "data/processed"
    out.mkdir(parents=True, exist_ok=True)
    tx.to_csv(out / "transactions_clean.csv", index=False)
    social_summary.to_csv(out / "social_profiles_customer_level.csv", index=False)
    merged.to_csv(out / "customer_transactions_social_merged.csv", index=False)

    excluded = {"transaction_id", "customer_id_legacy", "purchase_date",
                "purchase_amount", "customer_rating", "product_category"}
    features = [c for c in merged.columns if c not in excluded]
    merged[features + ["product_category"]].to_csv(
        out / "tabular_model_features.csv", index=False
    )
    print("Saved processed files to:", out.resolve())
    print("Target distribution:")
    print(merged["product_category"].value_counts(dropna=False))

if __name__ == "__main__":
    main()
