# CAL12 rev3 threshold freeze (PSEM-SMALL-MODEL-PROBE-v1-rev3, evaluator_revision=3)

Fixed priority: (a) zero KEEP false cuts when achievable; (b) lowest C2-clean-transfer miss rate (C4 NEVER in miss objective); (c) lowest C2 median total delay.

| model | regime | tau | keep_false/6 | c2_missed/2 | c2_rate | cut(all)/4 | contam s/h | cuts@tau | sens@tau |
|---|---|---|---|---|---|---|---|---|---|
| firered | O | 0.85 | 5/6 | 0/2 | 0.0 | 0/4 | 185.33180915709045 | 71 | 1516 |
| firered | C | 0.05 | 4/6 | 2/2 | 1.0 | 3/4 | 803.1285722880693 | 21 | 586 |
| neovad | O | 0.05 | 1/6 | 2/2 | 1.0 | 2/4 | 875.3263943204381 | 5 | 122 |
| neovad | C | 0.05 | 0/6 | 2/2 | 1.0 | 3/4 | 931.6406955056856 | 1 | 20 |
