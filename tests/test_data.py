import pandas as pd
import numpy as np
from src.data import align_prices, validate_prices, drop_sparse_tickers, to_processed_csv

def _mk_raw():
    # Two tickers with some NaNs and non-positive junk row
    idx = pd.date_range("2024-01-01", periods=8, freq="D")
    a = pd.Series([0.0, 100, 101, np.nan, 103, 104, 105, 106], index=idx, name="Adj Close")
    b = pd.Series([0.0,  50,  np.nan,  51,  52,  53,   0.0, 54], index=idx, name="Adj Close")
    df_a = pd.concat([a, pd.Series(1_000, index=idx, name="Volume")], axis=1)
    df_b = pd.concat([b, pd.Series(2_000, index=idx, name="Volume")], axis=1)
    return {"AAA": df_a, "BBB": df_b}

def test_align_validate_and_drop_sparse(tmp_path):
    raw = _mk_raw()
    wide = align_prices(raw)
    assert set(wide.columns) == {"AAA", "BBB"}
    # Cap forward-fill to 1 day to avoid long imputations
    clean = validate_prices(wide, max_ffill_days=1)
    # After dropping non-positive-all row(s), index should start at first fully valid row
    assert clean.index.is_monotonic_increasing
    # Drop sparse tickers with >20% NaNs
    filtered = drop_sparse_tickers(clean, max_nan_frac=0.2)
    assert set(filtered.columns).issubset({"AAA", "BBB"})
    # Persist to CSV
    out = tmp_path / "adj_close.csv"
    to_processed_csv(filtered, str(out))
    assert out.exists()
