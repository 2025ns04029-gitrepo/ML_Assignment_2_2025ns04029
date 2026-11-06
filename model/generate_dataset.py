"""Reproduce the bundled synthetic campaign dataset in a separate CSV."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
TARGET = "booked_with_card_7d"


def generate_campaign_csv(path, rows=2000):
    """Create one synthetic campaign impression and outcome per customer."""
    rng = np.random.default_rng(SEED)
    card_type = rng.choice(["credit", "debit", "prepaid"], rows, p=[0.47, 0.45, 0.08])
    offer_views = rng.poisson(5, rows)
    # Historical bookings can never exceed the number of historical views.
    offer_bookings = rng.binomial(offer_views, 0.20)
    recently_active = rng.random(rows) < 0.90
    transaction_count = np.where(recently_active, 1 + rng.poisson(11, rows), 0)
    purchase_recency = np.where(recently_active, rng.integers(0, 30, rows),
                                rng.integers(30, 91, rows))
    average_amount = np.where(recently_active, rng.lognormal(7.4, 0.55, rows), 0)

    # All columns in this table are known when the current offer is shown.
    data = pd.DataFrame({
        "is_credit_card": (card_type == "credit").astype(int),
        "is_debit_card": (card_type == "debit").astype(int),
        "is_prepaid_card": (card_type == "prepaid").astype(int),
        "card_tenure_months": rng.integers(1, 121, rows),
        "transactions_last_30d": transaction_count,
        "avg_transaction_inr_30d": average_amount.round(2),
        "days_since_last_purchase": purchase_recency,
        "offer_views_last_90d": offer_views,
        "offer_bookings_last_90d": offer_bookings,
        "travel_txns_last_90d": rng.poisson(2, rows),
        "dining_txns_last_90d": rng.poisson(5, rows),
        "historical_email_open_rate": rng.beta(2, 4, rows).round(3),
        "offered_discount_pct": rng.choice([5, 10, 15, 20, 25, 30], rows),
        "minimum_spend_inr": rng.choice([500, 1000, 2000, 3500, 5000], rows),
        "offer_validity_days": rng.choice([7, 10, 14, 21, 30], rows),
        "shown_on_weekend": rng.binomial(1, 2 / 7, rows),
        "shown_in_app": rng.binomial(1, 0.60, rows),
    })

    # These teaching assumptions shape booking probability. They are simulated.
    # The random draw means that customers with the same inputs can differ.
    booking_score = (
        -2.8
        + 0.07 * data["offered_discount_pct"]
        + 0.47 * data["offer_bookings_last_90d"]
        + 0.035 * data["transactions_last_30d"]
        + 0.85 * data["historical_email_open_rate"]
        + 0.35 * data["shown_in_app"]
        - 0.022 * data["days_since_last_purchase"]
        - 0.00013 * data["minimum_spend_inr"]
    )
    booking_probability = 1 / (1 + np.exp(-booking_score))
    data[TARGET] = rng.binomial(1, booking_probability)
    data.to_csv(path, index=False)
    print(f"Created {path.name} with {len(data):,} synthetic records.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new file path to keep the existing CSV.")
    generate_campaign_csv(args.output)
