# MAIN48 rev3 rescore (PSEM-SMALL-MODEL-PROBE-v1-rev3, evaluator_revision=3)

> frozen taus from cal/results_repaired_v3/thresholds.json (applied as-is, no MAIN48 retuning)

## Headlines (500 ms primary; 300 ms sensitivity diagnostic)

| model | regime | tau | false_cuts/24 | missed/16 | contam s/h | src p50/p90 | dec p50/p90 | total p50/p90 | cuts@tau | sens@tau |
|---|---|---|---|---|---|---|---|---|---|---|
| firered | O | 0.85 | 22/24 | 3/16 | 282.30763058653616 | 540.0/1668.0 | 500.0/500.0 | 1040.0/2168.0 | 205 | 4802 |
| firered | C | 0.05 | 14/24 | 12/16 | 749.405440102704 | 1635.0/1786.0 | 500.0/500.0 | 2135.0/2286.0 | 44 | 1294 |
| neovad | O | 0.05 | 8/24 | 13/16 | 783.9910133996631 | 1820.0/1940.0 | 500.0/500.0 | 2320.0/2440.0 | 21 | 620 |
| neovad | C | 0.05 | 1/24 | 16/16 | 867.4893685308514 | None/None | None/None | None/None | 1 | 20 |

## Stratum breakdown (detected = CUT-role episodes with a valid CUT)

| model | regime | C1 fc/8 | C2 det/8 | C3 fc/8 | C4 det/8 | C5 fc/8 | C6 n | CUT det/16 |
|---|---|---|---|---|---|---|---|---|
| firered | O | 7/8 | 6/8 | 7/8 | 7/8 | 8/8 | 8 | 13/16 |
| firered | C | 5/8 | 2/8 | 3/8 | 2/8 | 6/8 | 8 | 4/16 |
| neovad | O | 1/8 | 1/8 | 3/8 | 2/8 | 4/8 | 8 | 3/16 |
| neovad | C | 0/8 | 0/8 | 1/8 | 0/8 | 0/8 | 8 | 0/16 |
