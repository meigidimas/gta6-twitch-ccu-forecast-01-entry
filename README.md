[![make.fun competition](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/meigidimas/gta6-twitch-ccu-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/meigidimas/gta6-twitch-ccu-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01)

# GTA VI launch-day Twitch viewers — hour-by-hour quantile forecaster

Entry for the [make.fun bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01):
forecast how many people watch Grand Theft Auto VI live on Twitch in each
hour of its release day (19 November 2026, UTC), hours 2–24, as five
quantiles (5%, 25%, 50%, 75%, 95%) per hour, given only the data recorded
up to the end of hour 1.

## How it works

`main.py` (Python 3.13, standard library only, no network) learns the
launch-day shape from the case's comparables — past big launches with
their full hourly viewer curves around release day. For each forecast
hour it fits a small kernel-weighted regression in log space:

```
log(viewers at hour h) ~ a + b * log(hour-1 viewers)
                          + c * log(pre-release peak)
                          + d * weekend flag
```

- **Kernel weighting:** launches whose hour-1 audience is close to the
  target's count most (Gaussian kernel, bandwidth 1.5 in log space),
  because a launch's shape depends on its scale — a game that drew
  800k viewers in hour 1 ramps very differently from one that drew 400.
- **Pre-release peak:** the busiest hour in the week before release, a
  second scale signal (for the final this pools the case's own hourly
  history with the GTA VI category's pre-release curve).
- **Weekend flag:** weekend releases draw far more daytime/evening
  viewing than weekday ones; the fitted effect is about 1.6× on the
  evening hours.

The spread is honest, not tuned: the weighted residual standard deviation
at each hour, so hours where past launches disagree get wider bands.
Quantiles are the fitted median shifted by the standard normal quantiles
times that spread. Fallbacks (global fit → pooled median ratio → flat
persistence) keep the output valid when data is thin. The program is
deterministic: the same case always produces the same bytes.

## Run it locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=<kit>/input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

It reads every `*.in` case in the input directory and writes one
`<id>.out` forecast per case. About 0.07 s per case.

## Local validation

No local grader ships with the kit (the server grades every entry), so
the model was validated by leave-one-out cross-validation over the 37
comparables of practice case p00: each launch in turn acted as the
target, trained on the other 36, and scored with the exact pinball loss
from the rules.

| model | mean pinball (all 37) | mean pinball (hour-1 > 50k) |
| --- | --- | --- |
| house baseline (persistence) | 0.9296 | 0.2509 |
| this entry | **0.3501** | **0.2074** |

Typical miss 82.9%, 90% coverage 0.86. The big-launch regime is the one
that matters for the final: GTA VI's hour-1 audience will be at the top
of the comparable range, where the model beats the baseline by about 17%.
For reference, the rules state the house baseline scores about 0.387 on
the four practice cases; the leave-one-out figures above are the closest
local proxy for this entry's practice score.

## License

MIT — see [LICENSE](LICENSE).
