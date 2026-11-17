#!/usr/bin/env python3
"""Hour-by-hour quantile forecaster for a game launch's Twitch viewers.

For each hour 2..24 of the release day (UTC) this program forecasts five
quantiles (5%, 25%, 50%, 75%, 95%) of the category's concurrent viewers,
given everything recorded up to the end of hour 1.

Method
------
Every case ships a set of comparables: past big launches with their full
hourly viewer curves around release day. For each forecast hour we fit a
small weighted regression in log space over those launches:

    log(viewers at hour h) ~ a + b * log(hour-1 viewers)
                              + c * log(pre-release peak)
                              + d * weekend flag

The regression is kernel-weighted: launches whose hour-1 audience is close
to the target's (Gaussian kernel, bandwidth 1.5 in log space) count most,
because a launch's shape depends on its scale -- a game that already drew
800k viewers in hour 1 ramps very differently from one that drew 400.
The pre-release peak (the busiest hour in the week before release) is a
second scale signal, and the weekend flag captures the fact that weekend
releases draw far more daytime/evening viewing than weekday ones.

The spread comes from the same fit: the weighted residual standard
deviation at that hour, so hours where past launches disagree get honestly
wider bands. Quantiles are the fitted median shifted by the standard
normal quantiles times that spread.

Fallbacks keep the output valid when data is thin: a global (unweighted)
fit, then the pooled median hour-to-hour-1 ratio, then flat persistence
with the house baseline's widening spread.

Standard library only, no network, deterministic: the same case always
produces the same bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone

# Standard normal quantiles at the five scored levels.
Z_SCORES = [-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722]

HOURS = list(range(2, 25))
BANDWIDTH = 1.5          # kernel bandwidth in log hour-1 viewers
MIN_SD = 0.20            # floor on the log spread (avoids overconfidence)
MAX_SD = 1.50            # cap on the log spread (avoids absurd bands)
MIN_LOCAL_POINTS = 8     # need this many launches for a kernel-weighted fit
MIN_GLOBAL_POINTS = 3    # need this many launches for any regression fit
PRE_WINDOW_DAYS = 7      # pre-release window for the peak-viewers signal
MAX_VIEWERS = 1_000_000_000


def parse_ts(text):
    """Parse an ISO 8601 UTC timestamp into an aware datetime."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def hour_index(day_start, moment):
    """Which release-day hour (1-based) a timestamp falls in, or None."""
    delta = (moment - day_start).total_seconds()
    if delta < 0 or delta >= 24 * 3600:
        return None
    return int(delta // 3600) + 1


def pre_release_peak(points, day_start, days=PRE_WINDOW_DAYS):
    """Highest viewer count in the window before the release day.

    points: iterable of (timestamp, viewers). Returns the peak, or None
    when the window holds no usable point.
    """
    start = day_start - timedelta(days=days)
    best = None
    for moment, viewers in points:
        if start <= moment < day_start and viewers:
            if best is None or viewers > best:
                best = viewers
    return best


def target_features(case):
    """Scale signals for the launch being forecast.

    Returns (log hour-1 viewers, log pre-release peak, weekend flag,
    hour-1 print). The pre-release peak pools the case's own hourly
    history with the pre-release curve of the target's own category when
    the comparables carry it (the final case does).
    """
    day_start = parse_ts(case["day_start"])
    known = case.get("known") or []
    hour1 = None
    if known:
        hour1 = known[0].get("viewers")
    hour1_known = bool(hour1 and hour1 > 0)
    if not hour1_known:
        hour1 = 1
    weekend = 1.0 if day_start.weekday() >= 4 else 0.0

    points = []
    for row in case.get("history_hourly") or []:
        try:
            points.append((parse_ts(row["from"]), row.get("viewers")))
        except (KeyError, ValueError):
            continue
    game_name = (case.get("game") or {}).get("name") or ""
    if game_name:
        for comp in case.get("comparables") or []:
            if game_name.lower() not in (comp.get("name") or "").lower():
                continue
            for row in comp.get("hourly") or []:
                try:
                    points.append((parse_ts(row["from"]), row.get("viewers")))
                except (KeyError, ValueError):
                    continue
    peak = pre_release_peak(points, day_start)
    log_peak = math.log(peak) if peak and peak > 0 else math.log(hour1)
    if hour1_known:
        log_h1 = math.log(hour1)
    else:
        # No hour-1 print: anchor the scale on the pre-release peak.
        log_h1 = log_peak
        hour1 = peak if peak and peak > 0 else 1
    return log_h1, log_peak, weekend, hour1


def comparable_rows(case):
    """One training row per comparable launch with a usable hour 1.

    Each row: (log hour-1, log pre-release peak, weekend flag,
    {hour: log viewers for hours 1..24}). Rows come back sorted by
    launch name so the fit is independent of input order.
    """
    rows = []
    for comp in sorted(case.get("comparables") or [],
                       key=lambda c: c.get("name") or ""):
        try:
            day_start = parse_ts(comp["day_start"])
        except (KeyError, ValueError):
            continue
        hourly = {}
        points = []
        for row in comp.get("hourly") or []:
            try:
                moment = parse_ts(row["from"])
            except (KeyError, ValueError):
                continue
            viewers = row.get("viewers")
            if not viewers or viewers <= 0:
                continue
            points.append((moment, viewers))
            h = hour_index(day_start, moment)
            if h is not None:
                hourly[h] = math.log(viewers)
        if 1 not in hourly:
            continue
        peak = pre_release_peak(points, day_start)
        log_peak = math.log(peak) if peak and peak > 0 else hourly[1]
        weekend = 1.0 if day_start.weekday() >= 4 else 0.0
        rows.append((hourly[1], log_peak, weekend, hourly))
    return rows


def solve_linear(matrix, vector):
    """Solve a small dense linear system by Gaussian elimination.

    Returns the solution list, or None when the system is singular.
    """
    n = len(vector)
    aug = [row[:] + [vector[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            return None
        aug[col], aug[pivot] = aug[pivot], aug[col]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col] / aug[col][col]
            if factor == 0.0:
                continue
            for cc in range(col, n + 1):
                aug[r][cc] -= factor * aug[col][cc]
    return [aug[i][n] / aug[i][i] for i in range(n)]


def weighted_fit(samples, weights):
    """Weighted least squares with an intercept.

    samples: list of (features tuple, target). weights: parallel list.
    Returns (coefficients, residuals) or None when unsolvable.
    """
    width = len(samples[0][0]) + 1
    ata = [[0.0] * width for _ in range(width)]
    aty = [0.0] * width
    for (feats, target), w in zip(samples, weights):
        row = [1.0] + list(feats)
        for i in range(width):
            aty[i] += w * row[i] * target
            for j in range(width):
                ata[i][j] += w * row[i] * row[j]
    beta = solve_linear(ata, aty)
    if beta is None:
        return None
    residuals = []
    for (feats, target), w in zip(samples, weights):
        fitted = beta[0] + sum(beta[i + 1] * feats[i]
                               for i in range(len(feats)))
        residuals.append((target - fitted, w))
    return beta, residuals


def residual_sd(residuals):
    """Weighted standard deviation of residuals, floored and capped."""
    total_w = sum(w for _, w in residuals)
    if total_w <= 0:
        return MIN_SD
    var = sum(w * r * r for r, w in residuals) / total_w
    return min(max(math.sqrt(var), MIN_SD), MAX_SD)


def predict_hour(rows, target, hour):
    """Median log viewers and log spread for one forecast hour."""
    log_h1, log_peak, weekend, _ = target
    samples = []
    for row_h1, row_peak, row_wknd, hourly in rows:
        if hour in hourly:
            samples.append(((row_h1, row_peak, row_wknd), hourly[hour]))
    if len(samples) >= MIN_LOCAL_POINTS:
        weights = [math.exp(-0.5 * ((feats[0] - log_h1) / BANDWIDTH) ** 2)
                   for feats, _ in samples]
        fitted = weighted_fit(samples, weights)
        if fitted is not None:
            beta, residuals = fitted
            median = (beta[0] + beta[1] * log_h1 + beta[2] * log_peak
                      + beta[3] * weekend)
            return median, residual_sd(residuals)
    if len(samples) >= MIN_GLOBAL_POINTS:
        fitted = weighted_fit(samples, [1.0] * len(samples))
        if fitted is not None:
            beta, residuals = fitted
            median = (beta[0] + beta[1] * log_h1 + beta[2] * log_peak
                      + beta[3] * weekend)
            return median, residual_sd(residuals)
    if samples:
        ratios = sorted(y - feats[0] for feats, y in samples)
        n = len(ratios)
        median_ratio = ratios[n // 2] if n % 2 else 0.5 * (
            ratios[n // 2 - 1] + ratios[n // 2])
        mean_ratio = sum(ratios) / n
        var = sum((r - mean_ratio) ** 2 for r in ratios) / n
        sd = min(max(math.sqrt(var), MIN_SD), MAX_SD)
        return log_h1 + median_ratio, sd
    # No launch to learn from: flat persistence, baseline-style spread.
    return log_h1, 0.15 * math.sqrt(hour - 1)


def forecast_case(case):
    """Forecast one case: {"hours": [{"hour": h, "q": [5 quantiles]}]}."""
    target = target_features(case)
    rows = comparable_rows(case)
    out_hours = []
    for hour in HOURS:
        median_log, sd = predict_hour(rows, target, hour)
        quants = []
        for z in Z_SCORES:
            value = math.exp(median_log + z * sd)
            value = min(max(value, 1.0), float(MAX_VIEWERS))
            quants.append(round(value, 2))
        # Guarantee the ordering the format requires.
        for i in range(1, len(quants)):
            if quants[i] < quants[i - 1]:
                quants[i] = quants[i - 1]
        out_hours.append({"hour": hour, "q": quants})
    return {"hours": out_hours}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(n for n in os.listdir(input_dir) if n.endswith(".in"))
    for name in names:
        case_id = name[:-3]
        with open(os.path.join(input_dir, name), "r",
                  encoding="utf-8") as fh:
            case = json.load(fh)
        result = forecast_case(case)
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(result, fh)
            fh.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
