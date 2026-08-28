"""Download public daily prices or, if every source fails, a labelled TOY panel.

Price source order per ticker: yfinance (auto-adjusted), then the Yahoo Chart
v8 API used by yfinance. Macro tickers are optional; investable tickers that
fail are dropped. If the remaining investable set is empty, a synthetic TOY
panel is returned and every downstream artefact must be labelled TOY.
"""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

USER_AGENT = (
    "xgboost-return-ranking/0.1 "
    "(research; Hossein Tabasi; https://github.com/hosseinTabasi)"
)
YAHOO_UA = "Mozilla/5.0 (compatible; research-xgb-rank/0.1)"
SSL_CTX = ssl.create_default_context()


@dataclass
class DownloadMeta:
    """Provenance for the price panel actually used."""

    data_mode: str  # "FULL-public" or "TOY"
    start: str
    end: str
    investable: list[str]
    macro: list[str]
    crypto: list[str]
    failed: dict[str, str] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "data_mode": self.data_mode,
            "start": self.start,
            "end": self.end,
            "investable": self.investable,
            "macro": self.macro,
            "crypto": self.crypto,
            "failed": self.failed,
            "sources": self.sources,
            "notes": self.notes,
        }


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def flatten_investable(cfg: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    groups = cfg["investable"]
    crypto = list(groups.get("crypto", []))
    inv: list[str] = []
    for key in ("us_large_cap", "sector_etfs", "gold", "bonds", "crypto"):
        inv.extend(groups.get(key, []))
    # Preserve order, drop duplicates.
    seen: set[str] = set()
    ordered = []
    for t in inv:
        if t not in seen:
            seen.add(t)
            ordered.append(t)
    macro = list(cfg.get("macro", []))
    return ordered, crypto, macro


def http_get(
    url: str,
    *,
    timeout: int = 40,
    headers: dict[str, str] | None = None,
    retries: int = 2,
    sleep_s: float = 1.2,
) -> tuple[bytes | None, str]:
    hdrs = {"User-Agent": USER_AGENT}
    if headers:
        hdrs.update(headers)
    last_err = "no-attempt"
    for attempt in range(retries + 1):
        req = urllib.request.Request(url, headers=hdrs)
        try:
            with urllib.request.urlopen(req, timeout=timeout, context=SSL_CTX) as resp:
                return resp.read(), "ok"
        except urllib.error.HTTPError as exc:
            last_err = f"HTTP {exc.code}"
            try:
                last_err = f"HTTP {exc.code}: {exc.read()[:180]!r}"
            except Exception:
                pass
            if exc.code in {429, 500, 502, 503} and attempt < retries:
                time.sleep(sleep_s * (attempt + 1))
                continue
            return None, last_err
        except Exception as exc:  # noqa: BLE001 — network surface
            last_err = f"{type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(sleep_s * (attempt + 1))
                continue
            return None, last_err
    return None, last_err


def _normalise_daily(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.index = pd.to_datetime(out.index, utc=True).tz_convert("UTC").tz_localize(None).normalize()
    out = out[~out.index.duplicated(keep="last")].sort_index()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["close"], how="any")


def download_yfinance_daily(symbol: str, start: str, end: str) -> tuple[pd.DataFrame | None, str]:
    try:
        import yfinance as yf
    except Exception as exc:  # noqa: BLE001
        return None, f"yfinance_import_failed: {exc}"
    try:
        raw = yf.download(
            symbol,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            threads=False,
        )
    except Exception as exc:  # noqa: BLE001
        return None, f"yfinance_failed: {exc}"
    if raw is None or raw.empty:
        return None, "yfinance_failed: empty"
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = [c[0] for c in raw.columns]
    cols = {c.lower(): c for c in raw.columns}
    close_col = cols.get("close") or cols.get("adj close")
    if close_col is None:
        return None, f"yfinance_failed: columns {list(raw.columns)}"
    vol_col = cols.get("volume")
    frame = pd.DataFrame({"close": raw[close_col]})
    if vol_col is not None:
        frame["volume"] = raw[vol_col]
    frame = _normalise_daily(frame)
    if frame.empty:
        return None, "yfinance_failed: empty after clean"
    return frame, f"yfinance ok n={len(frame)}"


def download_yahoo_chart_daily(symbol: str, start: str, end: str) -> tuple[pd.DataFrame | None, str]:
    start_ts = int(datetime.fromisoformat(start).replace(tzinfo=UTC).timestamp())
    # Inclusive end-of-day: add one day so the end date itself is requested.
    end_ts = int(datetime.fromisoformat(end).replace(tzinfo=UTC).timestamp()) + 86400
    url = (
        "https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(symbol, safe='')}?interval=1d"
        f"&period1={start_ts}&period2={end_ts}&events=history"
    )
    body, note = http_get(url, timeout=45, headers={"User-Agent": YAHOO_UA}, retries=2)
    if body is None:
        return None, f"yahoo_chart_failed: {note}"
    try:
        payload = json.loads(body)
        result = ((payload.get("chart") or {}).get("result") or [None])[0]
        if not result:
            err = (payload.get("chart") or {}).get("error")
            return None, f"yahoo_chart_failed: {err}"
        ts = result.get("timestamp") or []
        quote = ((result.get("indicators") or {}).get("quote") or [{}])[0]
        adj = ((result.get("indicators") or {}).get("adjclose") or [{}])[0]
        close = adj.get("adjclose") or quote.get("close")
        df = pd.DataFrame(
            {
                "close": close,
                "volume": quote.get("volume"),
            },
            index=pd.to_datetime(ts, unit="s", utc=True),
        )
        df = _normalise_daily(df)
        if df.empty:
            return None, "yahoo_chart_failed: empty after dropna"
        return df, f"yahoo_chart ok n={len(df)}"
    except Exception as exc:  # noqa: BLE001
        return None, f"yahoo_chart_failed: parse {exc}"


def fetch_one(symbol: str, start: str, end: str) -> tuple[pd.DataFrame | None, str]:
    df, note = download_yfinance_daily(symbol, start, end)
    if df is not None and len(df) >= 200:
        return df, note
    yf_note = note
    df2, note2 = download_yahoo_chart_daily(symbol, start, end)
    if df2 is not None and len(df2) >= 200:
        return df2, note2 + f" (after {yf_note})"
    return None, f"{yf_note}; {note2}"


def cache_path(raw_dir: Path, symbol: str) -> Path:
    safe = symbol.replace("^", "IDX_").replace("/", "_")
    return raw_dir / f"{safe}.csv"


def load_or_download_symbol(
    symbol: str,
    start: str,
    end: str,
    raw_dir: Path,
) -> tuple[pd.DataFrame | None, str]:
    path = cache_path(raw_dir, symbol)
    if path.exists():
        try:
            cached = pd.read_csv(path, index_col=0, parse_dates=True)
            cached = _normalise_daily(cached)
            if len(cached) >= 200:
                return cached, f"cache n={len(cached)}"
        except Exception:
            pass
    df, note = fetch_one(symbol, start, end)
    if df is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=True)
    return df, note


def make_toy_prices(
    investable: list[str],
    macro: list[str],
    crypto: list[str],
    start: str,
    end: str,
    seed: int = 42,
) -> pd.DataFrame:
    """Synthetic daily panel. Every result built from this must be labelled TOY."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, end, freq="C")
    n = len(dates)
    market = rng.normal(0.00035, 0.009, size=n)
    # Mild AR(1) in the common factor so momentum is not pure noise.
    for i in range(1, n):
        market[i] = 0.08 * market[i - 1] + 0.92 * market[i]
    cols: dict[str, np.ndarray] = {}
    for i, tkr in enumerate(investable):
        beta = 0.4 + 0.08 * i
        if tkr in crypto:
            beta = 1.6
            vol = 0.035
        elif tkr in {"TLT", "IEF", "LQD"}:
            beta = -0.15
            vol = 0.006
        elif tkr == "GLD":
            beta = 0.1
            vol = 0.009
        else:
            vol = 0.012
        idio = rng.normal(0.0, vol, size=n)
        r = 0.00015 + beta * market + idio
        cols[tkr] = 100.0 * np.exp(np.cumsum(r))
    if "^VIX" in macro:
        cols["^VIX"] = 12.0 + np.abs(rng.normal(0, 4, size=n) - 8 * market)
    if "^TNX" in macro:
        cols["^TNX"] = 2.0 + np.cumsum(rng.normal(0, 0.01, size=n))
    return pd.DataFrame(cols, index=dates)


def assemble_prices(
    cfg: dict[str, Any],
    root: Path,
) -> tuple[pd.DataFrame, DownloadMeta]:
    start = str(cfg["start"])
    end = str(cfg["end"])
    investable, crypto, macro = flatten_investable(cfg)
    raw_dir = root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.Series] = {}
    failed: dict[str, str] = {}
    sources: dict[str, str] = {}
    notes: list[str] = []

    for symbol in investable + macro:
        print(f"[download] {symbol} ...", flush=True)
        df, note = load_or_download_symbol(symbol, start, end, raw_dir)
        sources[symbol] = note
        if df is None:
            failed[symbol] = note
            notes.append(f"DROP {symbol}: {note}")
            print(f"  FAIL {note}", flush=True)
            continue
        frames[symbol] = df["close"].rename(symbol)
        print(f"  {note}  {df.index.min().date()} -> {df.index.max().date()}", flush=True)
        time.sleep(0.15)

    prices = pd.DataFrame(frames).sort_index()
    got_inv = [t for t in investable if t in prices.columns]
    got_macro = [t for t in macro if t in prices.columns]
    got_crypto = [t for t in crypto if t in got_inv]

    data_mode = "FULL-public"
    if len(got_inv) < 8:
        notes.append(
            f"Too few investable downloads ({len(got_inv)}); switching to TOY synthetic panel."
        )
        prices = make_toy_prices(investable, macro, crypto, start, end, seed=int(cfg.get("seed", 42)))
        got_inv = list(investable)
        got_macro = [t for t in macro if t in prices.columns]
        got_crypto = list(crypto)
        data_mode = "TOY"

    # Clip to requested window using actual file dates.
    prices = prices.loc[(prices.index >= pd.Timestamp(start)) & (prices.index <= pd.Timestamp(end))]
    act_start = str(prices.dropna(how="all").index.min().date())
    act_end = str(prices.dropna(how="all").index.max().date())
    meta = DownloadMeta(
        data_mode=data_mode,
        start=act_start,
        end=act_end,
        investable=got_inv,
        macro=got_macro,
        crypto=got_crypto,
        failed=failed,
        sources=sources,
        notes=notes,
    )
    return prices, meta


def save_prices(prices: pd.DataFrame, root: Path) -> None:
    proc = root / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)
    prices.to_csv(proc / "prices.csv")
    daily_ret = prices.pct_change()
    daily_ret.to_csv(proc / "returns_daily.csv")
    try:
        prices.to_parquet(proc / "prices.parquet")
        daily_ret.to_parquet(proc / "returns_daily.parquet")
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] parquet write failed ({exc}); csv kept", flush=True)
