#!/usr/bin/env python3
"""
Estimate NVIDIA monthly revenue by day-weighting SEC 10-Q quarterly revenue
----------------------------------------------------------------------------

What this does
 - Calls the SEC "companyfacts" API for a given CIK (defaults to NVIDIA)
 - Extracts quarterly revenue (10-Q only) from preferred us-gaap concepts
 - Spreads each quarter's revenue evenly across its calendar days
 - Sums daily values into monthly estimates and writes a CSV

Quick usage
 1) Provide a descriptive SEC User-Agent (required by the SEC). Use your
    name/email or organization per
    https://www.sec.gov/os/accessing-edgar-data

        Example:
            python3 nvidiaSales.py \
                --user-agent "Hector Ramirez hector@example.com" \
                --start 2023-01 --end 2023-12 \
                --out nvidia_monthly_revenue_estimate.csv

 2) Optional filters:
        --start YYYY-MM  First month to include
        --end   YYYY-MM  Last month to include
        --cik            Override the default NVIDIA CIK

Outputs
 - CSV with columns: month (YYYY-MM), estimated_revenue_usd, source

Notes
 - We only use 10-Q (quarterly) to avoid double-counting with 10-K values.
 - The day-weighting is a simple uniform allocation across each quarter.
 - Respect SEC rate limits and include a proper User-Agent.
"""
import argparse
import sys
from typing import List, Optional

import pandas as pd
import requests
import traceback
import json
from pathlib import Path


SEC_COMPANYFACTS_URL = (
    "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json"
)
# NVIDIA CIK with leading zeros
NVIDIA_CIK = "0001045810"

# Revenue concepts to try, in order of preference
REVENUE_CONCEPTS = [
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "SalesRevenueNet",
    "Revenues",
    "Revenue",
]

# Accept both quarterly and annual; we'll derive Q4 from 10-K when needed
ACCEPT_FORMS = {"10-Q", "10-K", "10-K/A"}
# Accept durations roughly a quarter (13 weeks ~ 91 days).
# Include a tolerance for calendar variations.
MIN_Q_DAYS = 80
MAX_Q_DAYS = 105
MIN_Y_DAYS = 330
MAX_Y_DAYS = 380
DEBUG = False


def fetch_companyfacts(cik: str, user_agent: str) -> dict:
    """Fetch the SEC companyfacts JSON for a CIK.

    The SEC requires a descriptive User-Agent string that identifies you.
    See: https://www.sec.gov/os/accessing-edgar-data

    Args:
        cik: A 10-digit string CIK with leading zeros.
        user_agent: Descriptive user agent (e.g., "Your Name your@email").

    Returns:
        Parsed JSON (dict) from the companyfacts API.
    """
    headers = {
        "User-Agent": user_agent,
        "Accept-Encoding": "gzip, deflate",
        "Host": "data.sec.gov",
    }
    url = SEC_COMPANYFACTS_URL.format(cik=cik)
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def get_companyfacts(
    cik: str,
    user_agent: str,
    cache_dir: Optional[str] = None,
    refresh: bool = False,
) -> dict:
    """Fetch companyfacts with optional on-disk cache.

    If cache_dir is provided, store/load JSON at
    <cache_dir>/companyfacts_CIK<cik>.json.
    """
    if cache_dir:
        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        cache_file = cache_path / f"companyfacts_CIK{cik}.json"
        if cache_file.exists() and not refresh:
            log_debug(f"Loading companyfacts from cache: {cache_file}")
            with cache_file.open("r", encoding="utf-8") as f:
                return json.load(f)
        data = fetch_companyfacts(cik, user_agent)
        try:
            with cache_file.open("w", encoding="utf-8") as f:
                json.dump(data, f)
            log_debug(f"Cached companyfacts to: {cache_file}")
        except Exception:
            # Best-effort cache write
            pass
        return data
    return fetch_companyfacts(cik, user_agent)


def select_revenue_items(facts: dict) -> pd.DataFrame:
    """Build a DataFrame of quarterly revenue facts (10-Q; duration items).

    Strategy:
    - Prefer a short list of us-gaap revenue concepts (ordered by specificity)
    - Restrict to 10-Q forms
    - Exclude YTD/cumulative frames
    - Keep only duration facts with start/end dates in a quarter-like length
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    records: List[dict] = []

    for concept in REVENUE_CONCEPTS:
        if concept not in us_gaap:
            continue
        units = us_gaap[concept].get("units", {})
        # Prefer USD values for revenue
        for unit_key in ["USD"]:
            if unit_key not in units:
                continue
            for item in units[unit_key]:
                # We want duration facts with explicit period start and end
                start = item.get("start")
                end = item.get("end")
                form = item.get("form")
                val = item.get("val")
                frame = item.get("frame", "")
                if not (start and end and form and val is not None):
                    continue
                # Filter to accepted forms (10-Q/10-K)
                if form not in ACCEPT_FORMS:
                    continue
                # Exclude YTD or other cumulative frames if indicated
                if isinstance(frame, str) and "YTD" in frame.upper():
                    continue
                # Parse dates and ensure duration is quarter-like
                try:
                    start_d = pd.to_datetime(start).date()
                    end_d = pd.to_datetime(end).date()
                except (ValueError, TypeError):
                    # Skip rows with unparsable dates
                    continue
                dur_days = (end_d - start_d).days + 1
                # Keep only quarter-like 10-Q durations here
                if not str(form).startswith("10-Q"):
                    continue
                if dur_days < MIN_Q_DAYS or dur_days > MAX_Q_DAYS:
                    continue

                records.append(
                    {
                        "concept": concept,
                        "unit": unit_key,
                        "form": form,
                        "start": start_d,
                        "end": end_d,
                        "days": dur_days,
                        "value_usd": float(val),
                        "fy": item.get("fy"),
                        "fp": item.get("fp"),
                        "accn": item.get("accn"),
                    }
                )
        # If we collected records for this concept, prefer it and stop
        if records:
            break

    if not records:
        raise RuntimeError("No suitable quarterly revenue facts found.")

    df = pd.DataFrame.from_records(records)
    # Prefer amended 10-Q/A over original 10-Q for the same period, then drop
    # duplicates by period so only one row per [start,end] remains.
    form_rank = {"10-Q": 0, "10-Q/A": 1}
    df["_form_rank"] = df["form"].map(form_rank).fillna(0).astype(int)
    df = df.sort_values(["start", "end", "_form_rank"]).drop_duplicates(
        subset=["start", "end"], keep="last"
    )
    df = df.drop(columns=["_form_rank"])  # cleanup helper column
    # Normalize dtypes to avoid categorical ordering issues
    df = _normalize_types(
        df,
        date_cols=["start", "end"],
        num_cols=["days", "value_usd", "fy"],
        str_cols=["concept", "unit", "form", "fp", "accn"],
    )
    return df


def select_annual_items(facts: dict) -> pd.DataFrame:
    """Build a DataFrame of annual totals from 10-K forms (duration items).

    Filters:
    - Concepts limited to REVENUE_CONCEPTS
    - USD units
    - 10-K / 10-K/A only
    - Excludes YTD/cumulative frames
    - Duration roughly a year (330–380 days)
    """
    us_gaap = facts.get("facts", {}).get("us-gaap", {})
    records: List[dict] = []

    for concept in REVENUE_CONCEPTS:
        if concept not in us_gaap:
            continue
        units = us_gaap[concept].get("units", {})
        for unit_key in ["USD"]:
            if unit_key not in units:
                continue
            for item in units[unit_key]:
                start = item.get("start")
                end = item.get("end")
                form = item.get("form")
                val = item.get("val")
                frame = item.get("frame", "")
                if not (start and end and form and val is not None):
                    continue
                if not (form.startswith("10-K")):
                    continue
                if isinstance(frame, str) and "YTD" in frame.upper():
                    continue
                try:
                    start_d = pd.to_datetime(start).date()
                    end_d = pd.to_datetime(end).date()
                except (ValueError, TypeError):
                    continue
                dur_days = (end_d - start_d).days + 1
                if dur_days < MIN_Y_DAYS or dur_days > MAX_Y_DAYS:
                    continue

                records.append(
                    {
                        "concept": concept,
                        "unit": unit_key,
                        "form": form,
                        "start": start_d,
                        "end": end_d,
                        "days": dur_days,
                        "value_usd": float(val),
                        "fy": item.get("fy"),
                        "fp": item.get("fp"),
                        "accn": item.get("accn"),
                    }
                )
        if records:
            break

    if not records:
        return pd.DataFrame(
            columns=[
                "concept",
                "unit",
                "form",
                "start",
                "end",
                "days",
                "value_usd",
                "fy",
                "fp",
                "accn",
            ]
        )

    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["start", "end"]).drop_duplicates(
        subset=["start", "end", "accn"], keep="last"
    )
    df = _normalize_types(
        df,
        date_cols=["start", "end"],
        num_cols=["days", "value_usd", "fy"],
        str_cols=["concept", "unit", "form", "fp", "accn"],
    )
    return df


def infer_q4_from_10k(
    qtr_df: pd.DataFrame, ann_df: pd.DataFrame
) -> pd.DataFrame:
    """Fill in Q4 by subtracting Q1–Q3 from the annual 10-K total.

    We treat daily allocation as end-exclusive ranges. Derived Q4 is
    (last_Q_end, K_end], i.e. starts the day after the last 10-Q end
    and ends on the 10-K end date.
    """
    if ann_df.empty or qtr_df.empty:
        return qtr_df

    q = qtr_df.copy()
    k = ann_df.copy()
    # Normalize types to avoid Categorical/ordering issues
    q["start"] = pd.to_datetime(q["start"]).dt.tz_localize(None)
    q["end"] = pd.to_datetime(q["end"]).dt.tz_localize(None)
    q["fy"] = pd.to_numeric(q["fy"], errors="coerce").astype("Int64")
    q["form"] = q["form"].astype(str)
    k["start"] = pd.to_datetime(k["start"]).dt.tz_localize(None)
    k["end"] = pd.to_datetime(k["end"]).dt.tz_localize(None)
    k["fy"] = pd.to_numeric(k["fy"], errors="coerce").astype("Int64")
    k["form"] = k["form"].astype(str)

    out_rows: List[dict] = []
    # prefer latest 10-K by fiscal year
    k = k.sort_values(["fy", "end"]).drop_duplicates(
        subset=["fy"], keep="last"
    )

    for _, krecd in k.iterrows():
        fy = krecd.get("fy")
        if pd.isna(fy):
            continue
        k_total = (
            float(krecd["value_usd"]) if "value_usd" in krecd
            else float(krecd["value"])
        )
        k_end = krecd["end"]
        # Qs for same fiscal year on/before K end
        qs = q[(q["fy"] == fy) & (q["end"] <= k_end)].sort_values("end")
        if qs.empty:
            continue
        # keep only 10-Q rows
        qs = qs[qs["form"].str.startswith("10-Q")]
        sum_q = (
            float(qs["value_usd"].sum())
            if "value_usd" in qs
            else float(qs["value"].sum())
        )
        q4_val = k_total - sum_q
        if q4_val <= 0:
            continue
        last_q_end = qs["end"].max()
    # Derived Q4 starts the day after last_q_end to keep sets disjoint
        q4_start = last_q_end + pd.Timedelta(days=1)
        q4_end = k_end
        dur = (q4_end - q4_start).days + 1
        form_label = (
            "10-K-derived"
            if 60 <= dur <= 120
            else "10-K-derived (unusual duration)"
        )
        out_rows.append({
            "concept": qs.iloc[-1]["concept"],
            "unit": "USD",
            "form": form_label,
            # Keep as Timestamps to avoid mixing with datetime.date
            "start": q4_start,
            "end": q4_end,
            "days": dur,
            "value_usd": float(q4_val),
            "fy": int(fy),
            "fp": "Q4",
            "accn": None,
        })

    if out_rows:
        q = pd.concat([q, pd.DataFrame(out_rows)], ignore_index=True)
        # Normalize types BEFORE sorting to avoid mixed dtype ordering issues
        q = _normalize_types(
            q,
            date_cols=["start", "end"],
            num_cols=["days", "value_usd", "fy"],
            str_cols=["concept", "unit", "form", "fp", "accn"],
        )
        q = q.sort_values(["start", "end"]).reset_index(drop=True)
    return q


def allocate_quarter_to_daily(df_quarters: pd.DataFrame) -> pd.DataFrame:
    """Evenly allocate each quarter's revenue to its calendar days.

    For each quarterly row, compute a constant per-day revenue and produce a
    long daily DataFrame. Overlaps (if any) are summed.
    """
    daily_parts: List[pd.DataFrame] = []
    for _, row in df_quarters.iterrows():
        start = row["start"]
        end = row["end"]
        val = float(row["value_usd"])
        # Use shared helper for end-exclusive period days
        dr = days_in_period(start, end)
        if len(dr) == 0:
            continue
        per_day = val / len(dr)
        part = pd.DataFrame({"date": dr, "revenue_usd": per_day})
        daily_parts.append(part)
    daily = pd.concat(daily_parts, ignore_index=True)
    # If overlapping periods exist (rare with 10-Q), sum them safely
    daily = daily.groupby("date", as_index=False)["revenue_usd"].sum()
    return daily


def daily_to_monthly(
    daily: pd.DataFrame,
    start_month: Optional[str],
    end_month: Optional[str],
) -> pd.DataFrame:
    """Aggregate daily revenue to monthly totals with optional month bounds.

    Args:
        daily: DataFrame with columns [date, revenue_usd]
        start_month: Optional YYYY-MM inclusive lower bound
        end_month: Optional YYYY-MM inclusive upper bound
    """
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.set_index("date").sort_index()

    # Optionally bound by user-specified months
    if start_month:
        start_dt = pd.to_datetime(start_month + "-01")
        daily = daily[daily.index >= start_dt]
    if end_month:
        # end_month is YYYY-MM; include the entire month
        end_dt = (pd.to_datetime(end_month + "-01") + pd.offsets.MonthEnd(1))
        daily = daily[daily.index <= end_dt]

    monthly = (
        daily.resample("ME")["revenue_usd"].sum().to_frame().reset_index()
    )
    monthly["month"] = monthly["date"].dt.strftime("%Y-%m")
    monthly = monthly[["month", "revenue_usd"]]
    monthly = monthly.rename(columns={"revenue_usd": "estimated_revenue_usd"})
    return monthly


def days_in_period(start, end) -> pd.DatetimeIndex:
    # End-exclusive to avoid double-counting boundary days between periods
    return pd.date_range(start, end, inclusive="left", freq="D")


def _normalize_types(
    df: pd.DataFrame,
    *,
    date_cols: list[str] = None,
    num_cols: list[str] = None,
    str_cols: list[str] = None,
) -> pd.DataFrame:
    """Coerce dtypes and strip categoricals to avoid ordering errors.

    - Convert categorical columns to their underlying values (object/string)
    - Ensure date cols are tz-naive Timestamps
    - Ensure numeric cols are numeric (nullable Int64/float)
    - Ensure string cols are string
    """
    out = df.copy()
    # Decategorize any categorical columns
    for col in out.select_dtypes(include="category").columns:
        out[col] = out[col].astype(object)

    if date_cols:
        for c in date_cols:
            if c in out.columns:
                dt = pd.to_datetime(out[c], errors="coerce")
                out[c] = dt.dt.tz_localize(None)
    if num_cols:
        for c in num_cols:
            if c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
    if str_cols:
        for c in str_cols:
            if c in out.columns:
                out[c] = out[c].astype(str)
    return out


def assert_no_daily_overlaps(periods_df: pd.DataFrame) -> None:
    """Warn (do not raise) if end-exclusive daily overlaps are detected.

    We use end-exclusive allocation (left-inclusive), so adjacent periods that
    share a boundary should not overlap. Some issuances (e.g., 10-Q vs 10-Q/A)
    can still produce overlaps; we issue a warning in debug mode and continue,
    since allocation groups by day and safely sums values.
    """
    seen = set()
    overlaps = set()
    for _, r in periods_df.iterrows():
        for d in days_in_period(r["start"], r["end"]):
            if d in seen:
                overlaps.add(d)
            else:
                seen.add(d)
    if overlaps:
        sample = sorted(list(overlaps))[:10]
        log_debug(
            f"[WARN] Overlapping periods detected (sample days): {sample}"
        )


def log_debug(msg: str) -> None:
    if DEBUG:
        print(f"[DEBUG] {msg}")


def df_debug_info(
    df: pd.DataFrame, *, head_rows: int = 5, label: str = "df"
) -> None:
    if not DEBUG:
        return
    try:
        print(f"[DEBUG] {label}: shape={df.shape}")
        print(f"[DEBUG] {label} dtypes:\n{df.dtypes}")
        print(f"[DEBUG] {label} head:\n{df.head(head_rows)}")
    except Exception:
        # best-effort debug; do not fail due to debug printing
        pass


def main():
    """CLI entry point: parse args, run pipeline, write CSV, print summary."""
    parser = argparse.ArgumentParser(
        description=(
            "Estimate NVIDIA monthly revenue by allocating SEC quarterly "
            "revenue across days."
        )
    )
    parser.add_argument(
        "--user-agent",
        required=True,
        help=(
            "SEC API User-Agent header, e.g. "
            "\"Your Name your.email@example.com\""
        ),
    )
    parser.add_argument(
        "--cik",
        default=NVIDIA_CIK,
        help="Company CIK with leading zeros (default NVIDIA).",
    )
    parser.add_argument(
        "--start",
        dest="start_month",
        default=None,
        help="Start month filter, format YYYY-MM (optional).",
    )
    parser.add_argument(
        "--end",
        dest="end_month",
        default=None,
        help="End month filter, format YYYY-MM (optional).",
    )
    parser.add_argument(
        "--out",
        default="nvidia_monthly_revenue_estimate.csv",
        help="Output CSV filename.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose debug logging and full tracebacks.",
    )
    parser.add_argument(
        "--cache-dir",
        default=".cache/nvidiaSales",
        help="Directory to cache SEC companyfacts JSON (optional).",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Ignore cache and re-fetch companyfacts from SEC.",
    )
    parser.add_argument(
        "--tmp-dir",
        default=".tmp/nvidiaSales",
        help="Directory to dump intermediate CSVs (used with --dump).",
    )
    parser.add_argument(
        "--dump",
        action="store_true",
        help="Dump raw JSON and intermediate DataFrames to tmp dir.",
    )
    args = parser.parse_args()
    global DEBUG
    DEBUG = bool(args.debug)

    try:
        log_debug("Fetching companyfacts…")
        facts = get_companyfacts(
            args.cik,
            args.user_agent,
            cache_dir=args.cache_dir,
            refresh=args.refresh_cache,
        )
        log_debug("Fetched companyfacts")

        # Optionally dump raw JSON
        if args.dump:
            tmp_path = Path(args.tmp_dir)
            tmp_path.mkdir(parents=True, exist_ok=True)
            raw_file = tmp_path / f"companyfacts_CIK{args.cik}.json"
            try:
                with raw_file.open("w", encoding="utf-8") as f:
                    json.dump(facts, f)
                log_debug(f"Dumped raw facts JSON to: {raw_file}")
            except Exception:
                pass

        log_debug("Selecting quarterly (10-Q) items…")
        qdf = select_revenue_items(facts)
        df_debug_info(qdf, label="10-Q df", head_rows=3)
        if args.dump:
            try:
                (Path(args.tmp_dir) / "qdf.csv").parent.mkdir(
                    parents=True, exist_ok=True
                )
                qdf.to_csv(Path(args.tmp_dir) / "qdf.csv", index=False)
            except Exception:
                pass

        log_debug("Selecting annual (10-K) items…")
        kdf = select_annual_items(facts)
        df_debug_info(kdf, label="10-K df", head_rows=3)
        if args.dump:
            try:
                kdf.to_csv(Path(args.tmp_dir) / "kdf.csv", index=False)
            except Exception:
                pass

        log_debug("Inferring Q4 from 10-K…")
        # Incorporate a derived Q4 (from 10-K) if present
        qdf = infer_q4_from_10k(qdf, kdf)
        df_debug_info(qdf, label="10-Q (+Q4) df", head_rows=3)
        if args.dump:
            try:
                qdf.to_csv(Path(args.tmp_dir) / "qdf_with_q4.csv", index=False)
            except Exception:
                pass

        log_debug("Checking overlaps…")
        # Sanity check: periods should not overlap on a calendar day basis
        assert_no_daily_overlaps(qdf)

        log_debug("Allocating to daily…")
        daily = allocate_quarter_to_daily(qdf)
        df_debug_info(daily, label="daily df", head_rows=3)
        if args.dump:
            try:
                daily.to_csv(Path(args.tmp_dir) / "daily.csv", index=False)
            except Exception:
                pass

        log_debug("Aggregating monthly…")
        monthly = daily_to_monthly(daily, args.start_month, args.end_month)
        monthly["source"] = (
            "SEC 10-Q + 10-K (Q4 derived); day-weighted allocation"
        )
        df_debug_info(monthly, label="monthly df", head_rows=5)
        if args.dump:
            try:
                monthly.to_csv(Path(args.tmp_dir) / "monthly.csv", index=False)
            except Exception:
                pass
        monthly.to_csv(args.out, index=False)
        print(f"Wrote {len(monthly)} rows to {args.out}")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error from SEC API: {e}", file=sys.stderr)
        if DEBUG:
            traceback.print_exc()
        sys.exit(1)
    except requests.exceptions.RequestException as e:
        print(f"Network error contacting SEC API: {e}", file=sys.stderr)
        if DEBUG:
            traceback.print_exc()
        sys.exit(1)
    except (ValueError, TypeError, KeyError) as e:
        print(f"Data error: {e}", file=sys.stderr)
        if DEBUG:
            traceback.print_exc()
        sys.exit(1)
    # Intentionally avoid a broad bare "except Exception" to keep errors
    # explicit and actionable.


if __name__ == "__main__":
    main()
