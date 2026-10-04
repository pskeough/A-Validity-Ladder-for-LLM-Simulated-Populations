## A per model

| model | gate k=10 min / rec | k for 0.25 SD (max) | SD ratio range | L1 pass | misfit ratio max | overfit ratio range | R1 pass | R2 pass | phi lower limit range | RMSD range |
|---|---|---|---|---|---|---|---|---|---|---|
| claude-haiku-4.5 | 5/5, 5/5 | 3 | 0.14 to 0.37 | 0/5 | 0.00 | 0.01 to 18.95 | 5/5 | 0/5 | 0.965 to 0.997 | 0.257 to 0.374 |
| claude-sonnet-4.6 | 5/5, 5/5 | 1 | 0.01 to 0.11 | 0/5 | 0.00 | 0.01 to 19.90 | 0/5 | 0/5 | -0.026 to 0.598 | 0.568 to 0.805 |
| deepseek-v3.2 | 4/5, 1/5 | 14 | 0.38 to 0.92 | 0/5 | 0.13 | 1.27 to 5.62 | 5/5 | 0/5 | 0.980 to 0.997 | 0.113 to 0.239 |
| gemini-3-flash-preview | 5/5, 4/5 | 6 | 0.11 to 0.58 | 0/5 | 0.00 | 0.29 to 14.64 | 2/5 | 0/5 | 0.733 to 0.994 | 0.248 to 0.485 |
| gemini-3.1-flash-lite-preview | 5/5, 4/5 | 8 | 0.21 to 0.69 | 0/5 | 0.00 | 2.80 to 16.30 | 5/5 | 0/5 | 0.972 to 0.996 | 0.256 to 0.307 |
| gpt-5.4 | 4/5, 3/5 | 12 | 0.17 to 0.86 | 0/5 | 0.00 | 0.81 to 4.32 | 4/5 | 0/5 | 0.953 to 0.997 | 0.260 to 0.330 |
| gpt-5.4-mini | 5/5, 4/5 | 4 | 0.17 to 0.44 | 0/5 | 0.01 | 9.87 to 18.86 | 5/5 | 0/5 | 0.981 to 0.997 | 0.222 to 0.377 |
| gpt-5.4-nano | 5/5, 3/5 | 8 | 0.12 to 0.69 | 0/5 | 0.02 | 5.38 to 13.59 | 2/5 | 0/5 | 0.643 to 0.989 | 0.195 to 0.432 |


## A L1 verdict categories per model (counts of 5 scales)

| model | compressed (both tails in deficit) | excess overfit and misfit deficit | misfit deficit |
|---|---|---|---|
| claude-haiku-4.5 | 1 | 3 | 1 |
| claude-sonnet-4.6 | 2 | 3 | 0 |
| deepseek-v3.2 | 0 | 3 | 2 |
| gemini-3-flash-preview | 0 | 2 | 3 |
| gemini-3.1-flash-lite-preview | 0 | 4 | 1 |
| gpt-5.4 | 0 | 2 | 3 |
| gpt-5.4-mini | 0 | 5 | 0 |
| gpt-5.4-nano | 0 | 5 | 0 |


## B per population and model

| population | model | n range | L1 pass / undetermined / fail | R1 pass | R2 pass / unresolved / fail | SD ratio range | RMSD range | mean inter-item r (sim vs hum) |
|---|---|---|---|---|---|---|---|---|
| generic | GPT-3.5 | 147 to 149 | 0/1/4 | 2/5 | 0/1/4 | 0.30 to 0.59 | 0.132 to 0.603 | 0.26 vs 0.44 |
| generic | GPT-4 | 121 to 142 | 0/4/1 | 5/5 | 0/3/2 | 0.49 to 0.90 | 0.087 to 0.210 | 0.44 vs 0.44 |
| silicon | GPT-3.5 | 942 to 997 | 0/0/5 | 0/5 | 0/0/5 | 0.18 to 0.33 | 0.249 to 0.835 | 0.12 vs 0.44 |
| silicon | GPT-4 | 790 to 997 | 0/0/5 | 1/5 | 0/0/5 | 0.16 to 0.38 | 0.133 to 0.427 | 0.22 vs 0.44 |
