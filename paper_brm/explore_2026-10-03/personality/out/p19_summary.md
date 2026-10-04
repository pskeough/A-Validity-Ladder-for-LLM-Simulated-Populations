## A. IPIP-50, 8 LLMs, one persona, 10,000 draws each: tally over 40 model x scale cells

                             rung  False  True  fail
gate (single draw within 0.25 SD)   20.0  20.0   0.0
                 L1 at frozen tau    0.0   0.0  40.0
                    L1 at tau 1.5    0.0   0.0  40.0
                               R1   12.0  28.0   0.0
                               R2    0.0   0.0  40.0
           L4 (R1 and R2; R4 n/a)    0.0   0.0  40.0

Verdict grid (L1 at frozen tau / R1 / R2): 
scale                                     E            ES             A             C             O
model                                                                                              
claude-haiku-4.5               fail/R1p/R2f  fail/R1p/R2f  fail/R1p/R2f  fail/R1p/R2f  fail/R1p/R2f
claude-sonnet-4.6              fail/R1f/R2f  fail/R1f/R2f  fail/R1f/R2f  fail/R1f/R2f  fail/R1f/R2f
deepseek-v3.2                  fail/R1p/R2f  fail/R1p/R2f  fail/R1p/R2f  fail/R1p/R2f  fail/R1p/R2f
gemini-3-flash-preview         fail/R1p/R2f  fail/R1p/R2f  fail/R1f/R2f  fail/R1f/R2f  fail/R1f/R2f
gemini-3.1-flash-lite-preview  fail/R1p/R2f  fail/R1p/R2f  fail/R1p/R2f  fail/R1p/R2f  fail/R1p/R2f
gpt-5.4                        fail/R1p/R2f  fail/R1p/R2f  fail/R1f/R2f  fail/R1p/R2f  fail/R1p/R2f
gpt-5.4-mini                   fail/R1p/R2f  fail/R1p/R2f  fail/R1p/R2f  fail/R1p/R2f  fail/R1p/R2f
gpt-5.4-nano                   fail/R1p/R2f  fail/R1p/R2f  fail/R1f/R2f  fail/R1f/R2f  fail/R1f/R2f

## A. tau per scale and the subgroup that sets it
| scale | tau | group | n | resolved_departure | point_departure |
|---|---|---|---|---|---|
| Agreeableness | 2.00 | country=IT | 112 | 1.80 | 5.68 |
| Conscientiousness | 3.00 | country=SG | 153 | 2.42 | 21.86 |
| Emotional_Stability | 2.00 | country=IN | 279 | 1.63 | 2.93 |
| Extraversion | 3.00 | country=IN | 279 | 2.70 | 5.48 |
| Openness | 1.50 | country=IT | 112 | 1.18 | 2.55 |

## A. Closest cells to the R2 size limit (RMSD lower 90% limit must be <= 0.10)
| scale | model | R1 | ev_ratio | load_min | phi | phi_lo90 | loading_rmsd | loading_rmsd_lo90 | loading_rmsd_hi90 | R2 |
|---|---|---|---|---|---|---|---|---|---|---|
| Openness | deepseek-v3.2 | True | 4.047 | 0.504 | 0.998 | 0.997 | 0.113 | 0.107 | 0.120 | fail |
| Extraversion | deepseek-v3.2 | True | 11.147 | 0.774 | 0.995 | 0.995 | 0.121 | 0.117 | 0.126 | fail |
| Agreeableness | deepseek-v3.2 | True | 4.154 | 0.552 | 0.982 | 0.980 | 0.159 | 0.152 | 0.168 | fail |
| Conscientiousness | deepseek-v3.2 | True | 5.785 | 0.670 | 0.994 | 0.993 | 0.174 | 0.167 | 0.181 | fail |
| Extraversion | gpt-5.4-nano | True | 9.837 | 0.493 | 0.988 | 0.987 | 0.195 | 0.192 | 0.199 | fail |
| Emotional_Stability | gpt-5.4-nano | True | 11.142 | 0.590 | 0.990 | 0.989 | 0.216 | 0.212 | 0.220 | fail |

RMSD point range over 40 cells: 0.113 to 0.805; median 0.306. Congruence phi range 0.171 to 0.998.

## A. Per model, averaged over the five scales
| model | sd_ratio_mean | sd_ratio_min | sd_ratio_max | mean_diff_sd | item_sd_ratio | inter_item_r | hum_inter_item_r | item_profile_corr |
|---|---|---|---|---|---|---|---|---|
| claude-haiku-4.5 | 0.24 | 0.14 | 0.37 | 0.31 | 0.21 | 0.56 | 0.39 | 0.45 |
| claude-sonnet-4.6 | 0.06 | 0.01 | 0.11 | 0.22 | 0.06 |  | 0.39 | 0.61 |
| deepseek-v3.2 | 0.63 | 0.38 | 0.92 | 0.84 | 0.57 | 0.48 | 0.39 | 0.78 |
| gemini-3-flash-preview | 0.25 | 0.11 | 0.58 | 1.08 | 0.22 | 0.43 | 0.39 | 0.64 |
| gemini-3.1-flash-lite-preview | 0.38 | 0.21 | 0.69 | 0.55 | 0.34 | 0.52 | 0.39 | 0.49 |
| gpt-5.4 | 0.41 | 0.17 | 0.86 | 0.46 | 0.35 | 0.52 | 0.39 | 0.58 |
| gpt-5.4-mini | 0.28 | 0.17 | 0.44 | 0.69 | 0.24 | 0.62 | 0.39 | 0.47 |
| gpt-5.4-nano | 0.34 | 0.12 | 0.69 | 0.31 | 0.35 | 0.27 | 0.39 | 0.30 |

Mean L1 ratios per model (point values; band is [1/tau, tau]):
| model | misfit_ratio | overfit_ratio |
|---|---|---|
| claude-haiku-4.5 | 0.00 | 10.93 |
| claude-sonnet-4.6 | 0.00 | 11.82 |
| deepseek-v3.2 | 0.04 | 3.77 |
| gemini-3-flash-preview | 0.00 | 3.91 |
| gemini-3.1-flash-lite-preview | 0.00 | 10.37 |
| gpt-5.4 | 0.00 | 2.60 |
| gpt-5.4-mini | 0.00 | 13.70 |
| gpt-5.4-nano | 0.00 | 8.95 |

Profile concentration (10-item scale profiles, 10,000 draws per model; humans: 20,000 reference sample, unique profiles 13448 to 17844):
| model | unique_profiles_10items | modal_profile_share | share_within_two_adjacent_cats | hum_share_within_two_adjacent_cats |
|---|---|---|---|---|
| claude-haiku-4.5 | 40.800 | 0.669 | 0.869 | 0.156 |
| claude-sonnet-4.6 | 6.400 | 0.742 | 0.570 | 0.156 |
| deepseek-v3.2 | 1682.200 | 0.073 | 0.519 | 0.156 |
| gemini-3-flash-preview | 35.200 | 0.581 | 0.783 | 0.156 |
| gemini-3.1-flash-lite-preview | 77.800 | 0.466 | 0.819 | 0.156 |
| gpt-5.4 | 39.200 | 0.380 | 0.541 | 0.156 |
| gpt-5.4-mini | 63.800 | 0.603 | 0.949 | 0.156 |
| gpt-5.4-nano | 320.400 | 0.236 | 0.679 | 0.156 |

## A. Gate (one persona; draw variance only): single-draw SD over human SD, draws k needed for SE <= 0.25 SD (max over scales)
| model | single_draw_sd_over_ref_sd | k_min | k_rec | single_pass |
|---|---|---|---|---|
| claude-haiku-4.5 | 0.24 | 3.00 | 9.00 | 3 |
| claude-sonnet-4.6 | 0.06 | 1.00 | 1.00 | 5 |
| deepseek-v3.2 | 0.63 | 14.00 | 55.00 | 0 |
| gemini-3-flash-preview | 0.25 | 6.00 | 22.00 | 4 |
| gemini-3.1-flash-lite-preview | 0.38 | 8.00 | 31.00 | 1 |
| gpt-5.4 | 0.41 | 12.00 | 48.00 | 2 |
| gpt-5.4-mini | 0.28 | 4.00 | 13.00 | 2 |
| gpt-5.4-nano | 0.34 | 8.00 | 31.00 | 3 |

## A. Pooled level shift (not a persona grid): TOST at 0.5 SD: {'fail': 22, 'pass': 17, 'unresolved': 1}; at 0.2 SD: {'fail': 31, 'pass': 9}

## A. Positive control, 10,000 held-out humans through L1
| scale | tau_rule | tau | misfit_ratio | misfit_ratio_ci90_lo | misfit_ratio_ci90_hi | overfit_ratio | overfit_ratio_ci90_lo | overfit_ratio_ci90_hi | verdict |
|---|---|---|---|---|---|---|---|---|---|
| Extraversion | frozen | 3.000 | 1.042 | 0.973 | 1.158 | 0.997 | 0.936 | 1.116 | pass |
| Extraversion | tau1.5 | 1.500 | 1.042 | 0.973 | 1.158 | 0.997 | 0.936 | 1.116 | pass |
| Emotional_Stability | frozen | 2.000 | 0.942 | 0.876 | 1.055 | 0.962 | 0.900 | 1.074 | pass |
| Emotional_Stability | tau1.5 | 1.500 | 0.942 | 0.876 | 1.055 | 0.962 | 0.900 | 1.074 | pass |
| Agreeableness | frozen | 2.000 | 0.954 | 0.890 | 1.075 | 1.061 | 0.979 | 1.162 | pass |
| Agreeableness | tau1.5 | 1.500 | 0.954 | 0.890 | 1.075 | 1.061 | 0.979 | 1.162 | pass |
| Conscientiousness | frozen | 3.000 | 1.110 | 1.029 | 1.215 | 1.041 | 0.962 | 1.139 | pass |
| Conscientiousness | tau1.5 | 1.500 | 1.110 | 1.029 | 1.215 | 1.041 | 0.962 | 1.139 | pass |
| Openness | frozen | 1.500 | 1.119 | 1.035 | 1.227 | 1.018 | 0.959 | 1.128 | pass |
| Openness | tau1.5 | 1.500 | 1.119 | 1.035 | 1.227 | 1.018 | 0.959 | 1.128 | pass |

## A. Positive control, 10,000 held-out humans through L4
| scale | R1 | phi | phi_lo90 | loading_rmsd | loading_rmsd_lo90 | loading_rmsd_hi90 | R2 |
|---|---|---|---|---|---|---|---|
| Extraversion | True | 1.000 | 1.000 | 0.006 | 0.006 | 0.014 | pass |
| Emotional_Stability | True | 1.000 | 1.000 | 0.008 | 0.007 | 0.016 | pass |
| Agreeableness | True | 1.000 | 1.000 | 0.008 | 0.008 | 0.017 | pass |
| Conscientiousness | True | 1.000 | 1.000 | 0.007 | 0.008 | 0.017 | pass |
| Openness | True | 1.000 | 1.000 | 0.007 | 0.008 | 0.018 | pass |

## B. Petrov et al.: level 1
| scale | population | model | n | tau | misfit_ratio | misfit_ratio_ci90_lo | misfit_ratio_ci90_hi | overfit_ratio | overfit_ratio_ci90_lo | overfit_ratio_ci90_hi | verdict | verdict_tau1p5 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Extraversion | generic | GPT-3.5 | 149 | 1.50 | 0.27 | 0.00 | 0.67 | 0.09 | 0.00 | 0.40 | fail | fail |
| Extraversion | generic | GPT-4 | 142 | 1.50 | 0.14 | 0.00 | 0.56 | 1.05 | 0.31 | 1.56 | fail | fail |
| Extraversion | silicon | GPT-3.5 | 993 | 1.50 | 0.00 | 0.00 | 0.00 | 13.25 | 9.64 | 14.44 | fail | fail |
| Extraversion | silicon | GPT-4 | 996 | 1.50 | 0.00 | 0.00 | 0.00 | 4.54 | 2.98 | 5.20 | fail | fail |
| Agreeableness | generic | GPT-3.5 | 149 | 1.50 | 0.54 | 0.00 | 1.07 | 0.13 | 0.00 | 0.40 | fail | fail |
| Agreeableness | generic | GPT-4 | 121 | 1.50 | 0.83 | 0.33 | 1.65 | 1.38 | 0.70 | 2.40 | undetermined | undetermined |
| Agreeableness | silicon | GPT-3.5 | 997 | 1.50 | 0.00 | 0.00 | 0.00 | 3.35 | 2.03 | 4.39 | fail | fail |
| Agreeableness | silicon | GPT-4 | 996 | 1.50 | 0.00 | 0.00 | 0.00 | 0.23 | 0.10 | 0.51 | fail | fail |
| Conscientiousness | generic | GPT-3.5 | 147 | 1.50 | 0.95 | 0.41 | 1.90 | 0.27 | 0.00 | 0.82 | undetermined | undetermined |
| Conscientiousness | generic | GPT-4 | 142 | 1.50 | 0.85 | 0.42 | 1.84 | 0.60 | 0.23 | 1.70 | undetermined | undetermined |
| Conscientiousness | silicon | GPT-3.5 | 991 | 1.50 | 0.00 | 0.00 | 0.00 | 0.23 | 0.14 | 0.51 | fail | fail |
| Conscientiousness | silicon | GPT-4 | 997 | 1.50 | 0.02 | 0.00 | 0.06 | 0.53 | 0.26 | 0.71 | fail | fail |
| Neuroticism | generic | GPT-3.5 | 148 | 1.50 | 0.14 | 0.00 | 0.41 | 0.00 | 0.00 | 0.00 | fail | fail |
| Neuroticism | generic | GPT-4 | 129 | 1.50 | 0.62 | 0.16 | 1.40 | 0.82 | 0.31 | 2.06 | undetermined | undetermined |
| Neuroticism | silicon | GPT-3.5 | 942 | 1.50 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.38 | fail | fail |
| Neuroticism | silicon | GPT-4 | 790 | 1.50 | 0.00 | 0.00 | 0.00 | 0.13 | 0.00 | 0.66 | fail | fail |
| Openness | generic | GPT-3.5 | 149 | 1.50 | 0.54 | 0.13 | 0.94 | 0.13 | 0.00 | 0.42 | fail | fail |
| Openness | generic | GPT-4 | 142 | 1.50 | 0.42 | 0.00 | 0.85 | 1.09 | 0.48 | 1.97 | undetermined | undetermined |
| Openness | silicon | GPT-3.5 | 993 | 1.50 | 0.00 | 0.00 | 0.00 | 0.42 | 0.16 | 0.80 | fail | fail |
| Openness | silicon | GPT-4 | 996 | 1.50 | 0.00 | 0.00 | 0.00 | 0.50 | 0.16 | 1.30 | fail | fail |

## B. Petrov et al.: level 4
| scale | population | model | n | R1 | ev_ratio | load_min | phi | phi_lo90 | loading_rmsd | loading_rmsd_lo90 | loading_rmsd_hi90 | R2 | R3 | human_ev_ratio | human_load_min |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Extraversion | generic | GPT-3.5 | 149 | False | 1.904 | -0.190 | 0.613 | 0.421 | 0.603 | 0.464 | 0.748 | fail | False | 4.353 | 0.584 |
| Extraversion | generic | GPT-4 | 142 | True | 6.539 | 0.697 | 0.994 | 0.987 | 0.105 | 0.085 | 0.146 | unresolved | False | 4.353 | 0.584 |
| Extraversion | silicon | GPT-3.5 | 993 | False | 1.928 | 0.277 | 0.914 | 0.847 | 0.315 | 0.257 | 0.399 | fail | False | 4.353 | 0.584 |
| Extraversion | silicon | GPT-4 | 996 | False | 2.929 | 0.397 | 0.959 | 0.941 | 0.216 | 0.190 | 0.251 | fail | False | 4.353 | 0.584 |
| Agreeableness | generic | GPT-3.5 | 149 | True | 4.854 | 0.580 | 0.991 | 0.967 | 0.132 | 0.127 | 0.210 | fail | False | 4.104 | 0.591 |
| Agreeableness | generic | GPT-4 | 121 | True | 5.234 | 0.415 | 0.979 | 0.958 | 0.168 | 0.143 | 0.220 | fail | False | 4.104 | 0.591 |
| Agreeableness | silicon | GPT-3.5 | 997 | False | 2.069 | 0.136 | 0.928 | 0.896 | 0.249 | 0.208 | 0.293 | fail | False | 4.104 | 0.591 |
| Agreeableness | silicon | GPT-4 | 996 | False | 4.146 | 0.270 | 0.974 | 0.957 | 0.151 | 0.128 | 0.196 | fail | False | 4.104 | 0.591 |
| Conscientiousness | generic | GPT-3.5 | 147 | False | 2.539 | 0.442 | 0.983 | 0.915 | 0.158 | 0.099 | 0.307 | unresolved | False | 5.137 | 0.676 |
| Conscientiousness | generic | GPT-4 | 142 | True | 5.248 | 0.606 | 0.993 | 0.981 | 0.087 | 0.077 | 0.147 | unresolved | False | 5.137 | 0.676 |
| Conscientiousness | silicon | GPT-3.5 | 991 | False | 2.690 | -0.060 | 0.930 | 0.744 | 0.271 | 0.152 | 0.553 | fail | False | 5.137 | 0.676 |
| Conscientiousness | silicon | GPT-4 | 997 | True | 9.342 | 0.722 | 0.995 | 0.992 | 0.133 | 0.128 | 0.157 | fail | False | 5.137 | 0.676 |
| Neuroticism | generic | GPT-3.5 | 148 | True | 3.250 | 0.351 | 0.961 | 0.897 | 0.242 | 0.195 | 0.362 | fail | False | 7.004 | 0.713 |
| Neuroticism | generic | GPT-4 | 129 | True | 3.157 | 0.481 | 0.982 | 0.941 | 0.210 | 0.163 | 0.304 | fail | False | 7.004 | 0.713 |
| Neuroticism | silicon | GPT-3.5 | 942 | False | 1.511 | 0.264 | 0.936 | 0.798 | 0.296 | 0.245 | 0.471 | fail | False | 7.004 | 0.713 |
| Neuroticism | silicon | GPT-4 | 790 | False | 1.501 | -0.118 | 0.853 | 0.256 | 0.427 | 0.295 | 0.832 | fail | False | 7.004 | 0.713 |
| Openness | generic | GPT-3.5 | 149 | False | 4.753 | -0.468 | 0.883 | 0.830 | 0.334 | 0.291 | 0.404 | fail | False | 4.985 | 0.285 |
| Openness | generic | GPT-4 | 142 | True | 4.984 | 0.390 | 0.996 | 0.988 | 0.096 | 0.083 | 0.142 | unresolved | False | 4.985 | 0.285 |
| Openness | silicon | GPT-3.5 | 993 | False | 1.570 | -0.999 | 0.248 | 0.175 | 0.835 | 0.724 | 0.879 | fail | False | 4.985 | 0.285 |
| Openness | silicon | GPT-4 | 996 | False | 4.362 | 0.126 | 0.945 | 0.928 | 0.248 | 0.228 | 0.280 | fail | False | 4.985 | 0.285 |

## B. Petrov et al.: description
| scale | population | model | n | sd_ratio | mean_diff_sd | item_sd_ratio_mean | item_profile_corr | sim_mean_inter_item_r | hum_mean_inter_item_r | sim_unique_profiles | hum_unique_profiles |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Extraversion | generic | GPT-3.5 | 149 | 0.308 | 0.226 | 0.449 | 0.985 | 0.136 | 0.480 | 76 | 1671 |
| Extraversion | generic | GPT-4 | 142 | 0.750 | 0.708 | 0.707 | 0.846 | 0.563 | 0.480 | 122 | 1671 |
| Extraversion | silicon | GPT-3.5 | 993 | 0.179 | 0.624 | 0.259 | 0.927 | 0.127 | 0.480 | 77 | 1671 |
| Extraversion | silicon | GPT-4 | 996 | 0.266 | 0.516 | 0.344 | 0.974 | 0.230 | 0.480 | 133 | 1671 |
| Agreeableness | generic | GPT-3.5 | 149 | 0.593 | -0.837 | 0.588 | 0.544 | 0.387 | 0.348 | 69 | 1634 |
| Agreeableness | generic | GPT-4 | 121 | 0.899 | 0.042 | 0.853 | 0.592 | 0.446 | 0.348 | 98 | 1634 |
| Agreeableness | silicon | GPT-3.5 | 997 | 0.328 | 0.229 | 0.445 | 0.550 | 0.124 | 0.348 | 173 | 1634 |
| Agreeableness | silicon | GPT-4 | 996 | 0.339 | -0.020 | 0.398 | 0.871 | 0.206 | 0.348 | 131 | 1634 |
| Conscientiousness | generic | GPT-3.5 | 147 | 0.552 | -0.644 | 0.676 | 0.797 | 0.307 | 0.452 | 110 | 1485 |
| Conscientiousness | generic | GPT-4 | 142 | 0.757 | 0.021 | 0.805 | 0.725 | 0.445 | 0.452 | 119 | 1485 |
| Conscientiousness | silicon | GPT-3.5 | 991 | 0.265 | 0.130 | 0.344 | 0.771 | 0.183 | 0.452 | 81 | 1485 |
| Conscientiousness | silicon | GPT-4 | 997 | 0.384 | 0.043 | 0.424 | 0.904 | 0.375 | 0.452 | 106 | 1485 |
| Neuroticism | generic | GPT-3.5 | 148 | 0.303 | 0.113 | 0.396 | 0.795 | 0.253 | 0.542 | 59 | 1631 |
| Neuroticism | generic | GPT-4 | 129 | 0.489 | -0.147 | 0.610 | 0.788 | 0.299 | 0.542 | 108 | 1631 |
| Neuroticism | silicon | GPT-3.5 | 942 | 0.189 | -0.135 | 0.237 | 0.839 | 0.106 | 0.542 | 64 | 1631 |
| Neuroticism | silicon | GPT-4 | 790 | 0.160 | -0.088 | 0.245 | 0.911 | 0.076 | 0.542 | 66 | 1631 |
| Openness | generic | GPT-3.5 | 149 | 0.520 | -0.673 | 0.621 | 0.665 | 0.233 | 0.365 | 106 | 1793 |
| Openness | generic | GPT-4 | 142 | 0.878 | 0.022 | 0.811 | 0.665 | 0.461 | 0.365 | 128 | 1793 |
| Openness | silicon | GPT-3.5 | 993 | 0.208 | -0.072 | 0.273 | 0.508 | 0.050 | 0.365 | 82 | 1793 |
| Openness | silicon | GPT-4 | 996 | 0.341 | -0.230 | 0.377 | 0.652 | 0.234 | 0.365 | 126 | 1793 |

## B. Petrov et al.: level 3 (twin-matched residual)
| scale | model | group | n | hum_sd | sim_mean | hum_mean | twin_r | resid | resid_sd | ci90_lo_sd | ci90_hi_sd | verdict_0p5sd | verdict_0p2sd |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Extraversion | GPT-3.5 | all | 993 | 0.85 | 3.46 | 3.24 | 0.13 | 0.22 | 0.26 | 0.21 | 0.31 | pass | fail |
| Extraversion | GPT-3.5 | sex=0 | 651 | 0.85 | 3.46 | 3.28 | 0.10 | 0.17 | 0.20 | 0.14 | 0.27 | pass | unresolved |
| Extraversion | GPT-3.5 | sex=1 | 342 | 0.85 | 3.45 | 3.15 | 0.19 | 0.30 | 0.36 | 0.27 | 0.45 | pass | fail |
| Extraversion | GPT-3.5 | age<30 | 369 | 0.85 | 3.46 | 3.23 | 0.09 | 0.23 | 0.27 | 0.18 | 0.35 | pass | unresolved |
| Extraversion | GPT-3.5 | age30-44 | 428 | 0.85 | 3.49 | 3.26 | 0.14 | 0.23 | 0.27 | 0.19 | 0.35 | pass | unresolved |
| Extraversion | GPT-3.5 | age45+ | 196 | 0.85 | 3.37 | 3.19 | 0.16 | 0.19 | 0.22 | 0.10 | 0.34 | pass | unresolved |
| Extraversion | GPT-3.5 | country=GB | 888 | 0.85 | 3.46 | 3.24 | 0.15 | 0.22 | 0.26 | 0.20 | 0.31 | pass | fail |
| Extraversion | GPT-3.5 | country!=GB | 105 | 0.85 | 3.44 | 3.23 | -0.03 | 0.21 | 0.25 | 0.09 | 0.40 | pass | unresolved |
| Agreeableness | GPT-3.5 | all | 997 | 0.60 | 4.06 | 3.76 | 0.02 | 0.30 | 0.50 | 0.44 | 0.56 | unresolved | fail |
| Agreeableness | GPT-3.5 | sex=0 | 655 | 0.60 | 4.09 | 3.82 | 0.05 | 0.27 | 0.45 | 0.38 | 0.51 | unresolved | fail |
| Agreeableness | GPT-3.5 | sex=1 | 342 | 0.60 | 4.01 | 3.64 | -0.11 | 0.36 | 0.60 | 0.50 | 0.70 | fail | fail |
| Agreeableness | GPT-3.5 | age<30 | 370 | 0.60 | 3.94 | 3.75 | 0.04 | 0.18 | 0.30 | 0.22 | 0.39 | pass | fail |
| Agreeableness | GPT-3.5 | age30-44 | 429 | 0.60 | 4.09 | 3.76 | -0.01 | 0.33 | 0.55 | 0.46 | 0.63 | unresolved | fail |
| Agreeableness | GPT-3.5 | age45+ | 198 | 0.60 | 4.23 | 3.77 | 0.05 | 0.46 | 0.77 | 0.64 | 0.89 | fail | fail |
| Agreeableness | GPT-3.5 | country=GB | 891 | 0.60 | 4.06 | 3.77 | 0.01 | 0.29 | 0.49 | 0.43 | 0.55 | unresolved | fail |
| Agreeableness | GPT-3.5 | country!=GB | 106 | 0.60 | 4.07 | 3.71 | 0.12 | 0.36 | 0.60 | 0.43 | 0.77 | unresolved | fail |
| Conscientiousness | GPT-3.5 | all | 991 | 0.71 | 4.03 | 3.70 | 0.14 | 0.34 | 0.48 | 0.43 | 0.53 | unresolved | fail |
| Conscientiousness | GPT-3.5 | sex=0 | 649 | 0.71 | 4.06 | 3.76 | 0.11 | 0.30 | 0.43 | 0.36 | 0.49 | pass | fail |
| Conscientiousness | GPT-3.5 | sex=1 | 342 | 0.71 | 3.98 | 3.58 | 0.13 | 0.41 | 0.58 | 0.49 | 0.67 | unresolved | fail |
| Conscientiousness | GPT-3.5 | age<30 | 369 | 0.71 | 3.92 | 3.57 | 0.28 | 0.36 | 0.50 | 0.42 | 0.59 | unresolved | fail |
| Conscientiousness | GPT-3.5 | age30-44 | 426 | 0.71 | 4.07 | 3.74 | -0.09 | 0.33 | 0.47 | 0.39 | 0.56 | unresolved | fail |
| Conscientiousness | GPT-3.5 | age45+ | 196 | 0.71 | 4.16 | 3.86 | 0.09 | 0.31 | 0.43 | 0.32 | 0.55 | unresolved | fail |
| Conscientiousness | GPT-3.5 | country=GB | 885 | 0.71 | 4.03 | 3.71 | 0.15 | 0.32 | 0.46 | 0.41 | 0.52 | unresolved | fail |
| Conscientiousness | GPT-3.5 | country!=GB | 106 | 0.71 | 4.05 | 3.61 | 0.05 | 0.44 | 0.62 | 0.45 | 0.79 | unresolved | fail |
| Neuroticism | GPT-3.5 | all | 942 | 0.81 | 2.58 | 2.98 | 0.20 | -0.40 | -0.49 | -0.55 | -0.44 | unresolved | fail |
| Neuroticism | GPT-3.5 | sex=0 | 611 | 0.81 | 2.62 | 3.11 | 0.15 | -0.48 | -0.60 | -0.66 | -0.53 | fail | fail |
| Neuroticism | GPT-3.5 | sex=1 | 331 | 0.81 | 2.50 | 2.75 | 0.13 | -0.24 | -0.30 | -0.39 | -0.21 | pass | fail |
| Neuroticism | GPT-3.5 | age<30 | 362 | 0.81 | 2.63 | 3.13 | 0.17 | -0.50 | -0.62 | -0.70 | -0.53 | fail | fail |
| Neuroticism | GPT-3.5 | age30-44 | 414 | 0.81 | 2.57 | 2.91 | 0.20 | -0.34 | -0.42 | -0.50 | -0.34 | pass | fail |
| Neuroticism | GPT-3.5 | age45+ | 166 | 0.81 | 2.50 | 2.83 | 0.12 | -0.33 | -0.41 | -0.53 | -0.28 | unresolved | fail |
| Neuroticism | GPT-3.5 | country=GB | 841 | 0.81 | 2.58 | 2.98 | 0.20 | -0.40 | -0.49 | -0.55 | -0.44 | unresolved | fail |
| Neuroticism | GPT-3.5 | country!=GB | 101 | 0.81 | 2.55 | 2.95 | 0.21 | -0.40 | -0.49 | -0.65 | -0.33 | unresolved | fail |
| Openness | GPT-3.5 | all | 993 | 0.67 | 3.70 | 3.65 | 0.07 | 0.05 | 0.07 | 0.02 | 0.12 | pass | pass |
| Openness | GPT-3.5 | sex=0 | 649 | 0.67 | 3.72 | 3.60 | 0.05 | 0.12 | 0.18 | 0.11 | 0.24 | pass | unresolved |
| Openness | GPT-3.5 | sex=1 | 344 | 0.67 | 3.67 | 3.76 | 0.18 | -0.09 | -0.14 | -0.22 | -0.05 | pass | unresolved |
| Openness | GPT-3.5 | age<30 | 370 | 0.67 | 3.74 | 3.64 | 0.04 | 0.10 | 0.15 | 0.06 | 0.23 | pass | unresolved |
| Openness | GPT-3.5 | age30-44 | 428 | 0.67 | 3.68 | 3.61 | 0.09 | 0.07 | 0.10 | 0.02 | 0.19 | pass | pass |
| Openness | GPT-3.5 | age45+ | 195 | 0.67 | 3.67 | 3.77 | 0.14 | -0.11 | -0.16 | -0.28 | -0.04 | pass | unresolved |
| Openness | GPT-3.5 | country=GB | 888 | 0.67 | 3.70 | 3.64 | 0.09 | 0.06 | 0.09 | 0.04 | 0.15 | pass | pass |
| Openness | GPT-3.5 | country!=GB | 105 | 0.67 | 3.71 | 3.81 | -0.10 | -0.10 | -0.15 | -0.33 | 0.03 | pass | unresolved |
| Extraversion | GPT-4 | all | 996 | 0.85 | 3.36 | 3.23 | 0.14 | 0.12 | 0.15 | 0.09 | 0.20 | pass | pass |
| Extraversion | GPT-4 | sex=0 | 653 | 0.85 | 3.34 | 3.28 | 0.09 | 0.06 | 0.07 | 0.00 | 0.13 | pass | pass |
| Extraversion | GPT-4 | sex=1 | 343 | 0.85 | 3.39 | 3.14 | 0.25 | 0.25 | 0.29 | 0.20 | 0.38 | pass | fail |
| Extraversion | GPT-4 | age<30 | 370 | 0.85 | 3.37 | 3.23 | 0.08 | 0.14 | 0.16 | 0.07 | 0.24 | pass | unresolved |
| Extraversion | GPT-4 | age30-44 | 429 | 0.85 | 3.39 | 3.26 | 0.12 | 0.13 | 0.15 | 0.07 | 0.23 | pass | unresolved |
| Extraversion | GPT-4 | age45+ | 197 | 0.85 | 3.26 | 3.17 | 0.23 | 0.10 | 0.11 | -0.00 | 0.23 | pass | unresolved |
| Extraversion | GPT-4 | country=GB | 891 | 0.85 | 3.35 | 3.23 | 0.15 | 0.12 | 0.14 | 0.09 | 0.20 | pass | pass |
| Extraversion | GPT-4 | country!=GB | 105 | 0.85 | 3.37 | 3.23 | -0.04 | 0.15 | 0.17 | 0.01 | 0.33 | pass | unresolved |
| Agreeableness | GPT-4 | all | 996 | 0.60 | 3.89 | 3.76 | 0.12 | 0.13 | 0.22 | 0.16 | 0.27 | pass | unresolved |
| Agreeableness | GPT-4 | sex=0 | 653 | 0.60 | 3.95 | 3.82 | 0.07 | 0.13 | 0.22 | 0.15 | 0.29 | pass | unresolved |
| Agreeableness | GPT-4 | sex=1 | 343 | 0.60 | 3.77 | 3.64 | 0.07 | 0.13 | 0.21 | 0.12 | 0.31 | pass | unresolved |
| Agreeableness | GPT-4 | age<30 | 368 | 0.60 | 3.81 | 3.75 | 0.14 | 0.06 | 0.10 | 0.02 | 0.19 | pass | pass |
| Agreeableness | GPT-4 | age30-44 | 429 | 0.60 | 3.91 | 3.76 | 0.09 | 0.15 | 0.25 | 0.16 | 0.33 | pass | unresolved |
| Agreeableness | GPT-4 | age45+ | 199 | 0.60 | 3.99 | 3.77 | 0.16 | 0.22 | 0.37 | 0.24 | 0.49 | pass | fail |
| Agreeableness | GPT-4 | country=GB | 891 | 0.60 | 3.89 | 3.76 | 0.11 | 0.12 | 0.21 | 0.15 | 0.26 | pass | unresolved |
| Agreeableness | GPT-4 | country!=GB | 105 | 0.60 | 3.89 | 3.71 | 0.20 | 0.18 | 0.30 | 0.14 | 0.47 | pass | unresolved |
| Conscientiousness | GPT-4 | all | 997 | 0.70 | 3.97 | 3.70 | 0.13 | 0.27 | 0.39 | 0.33 | 0.44 | pass | fail |
| Conscientiousness | GPT-4 | sex=0 | 656 | 0.70 | 3.97 | 3.76 | 0.11 | 0.21 | 0.29 | 0.23 | 0.36 | pass | fail |
| Conscientiousness | GPT-4 | sex=1 | 341 | 0.70 | 3.97 | 3.58 | 0.15 | 0.39 | 0.56 | 0.47 | 0.65 | unresolved | fail |
| Conscientiousness | GPT-4 | age<30 | 370 | 0.70 | 3.82 | 3.56 | 0.21 | 0.25 | 0.36 | 0.28 | 0.45 | pass | fail |
| Conscientiousness | GPT-4 | age30-44 | 429 | 0.70 | 4.04 | 3.73 | -0.03 | 0.31 | 0.44 | 0.36 | 0.53 | unresolved | fail |
| Conscientiousness | GPT-4 | age45+ | 198 | 0.70 | 4.08 | 3.87 | 0.10 | 0.21 | 0.30 | 0.19 | 0.42 | pass | unresolved |
| Conscientiousness | GPT-4 | country=GB | 891 | 0.70 | 3.97 | 3.71 | 0.14 | 0.26 | 0.37 | 0.31 | 0.43 | pass | fail |
| Conscientiousness | GPT-4 | country!=GB | 106 | 0.70 | 3.97 | 3.61 | 0.05 | 0.36 | 0.52 | 0.35 | 0.69 | unresolved | fail |
| Neuroticism | GPT-4 | all | 790 | 0.81 | 2.63 | 2.99 | 0.10 | -0.36 | -0.44 | -0.50 | -0.38 | pass | fail |
| Neuroticism | GPT-4 | sex=0 | 499 | 0.81 | 2.65 | 3.11 | 0.09 | -0.46 | -0.56 | -0.64 | -0.49 | unresolved | fail |
| Neuroticism | GPT-4 | sex=1 | 291 | 0.81 | 2.59 | 2.78 | 0.04 | -0.19 | -0.23 | -0.33 | -0.13 | pass | unresolved |
| Neuroticism | GPT-4 | age<30 | 278 | 0.81 | 2.63 | 3.14 | 0.13 | -0.51 | -0.63 | -0.72 | -0.53 | fail | fail |
| Neuroticism | GPT-4 | age30-44 | 352 | 0.81 | 2.63 | 2.90 | 0.05 | -0.27 | -0.33 | -0.42 | -0.24 | pass | fail |
| Neuroticism | GPT-4 | age45+ | 160 | 0.81 | 2.62 | 2.91 | 0.15 | -0.29 | -0.36 | -0.49 | -0.23 | pass | fail |
| Neuroticism | GPT-4 | country=GB | 712 | 0.81 | 2.63 | 2.99 | 0.09 | -0.36 | -0.45 | -0.51 | -0.38 | unresolved | fail |
| Neuroticism | GPT-4 | country!=GB | 78 | 0.81 | 2.62 | 2.93 | 0.16 | -0.32 | -0.39 | -0.56 | -0.21 | unresolved | fail |
| Openness | GPT-4 | all | 996 | 0.67 | 3.59 | 3.65 | 0.15 | -0.06 | -0.10 | -0.15 | -0.04 | pass | pass |
| Openness | GPT-4 | sex=0 | 653 | 0.67 | 3.59 | 3.60 | 0.14 | -0.01 | -0.01 | -0.08 | 0.05 | pass | pass |
| Openness | GPT-4 | sex=1 | 343 | 0.67 | 3.59 | 3.76 | 0.17 | -0.17 | -0.25 | -0.34 | -0.17 | pass | unresolved |
| Openness | GPT-4 | age<30 | 370 | 0.67 | 3.62 | 3.64 | 0.08 | -0.01 | -0.02 | -0.10 | 0.07 | pass | pass |
| Openness | GPT-4 | age30-44 | 425 | 0.67 | 3.58 | 3.62 | 0.17 | -0.04 | -0.06 | -0.14 | 0.02 | pass | pass |
| Openness | GPT-4 | age45+ | 201 | 0.67 | 3.55 | 3.77 | 0.25 | -0.21 | -0.32 | -0.43 | -0.20 | pass | fail |
| Openness | GPT-4 | country=GB | 890 | 0.67 | 3.58 | 3.64 | 0.16 | -0.05 | -0.08 | -0.13 | -0.02 | pass | pass |
| Openness | GPT-4 | country!=GB | 106 | 0.67 | 3.64 | 3.82 | 0.03 | -0.18 | -0.26 | -0.44 | -0.08 | pass | unresolved |

## B. Petrov et al.: level 2 (twin-matched gaps)
| scale | model | contrast | n_a | n_b | g | se_g | gamma | se_gamma | cov | ratio | ci_lo | ci_hi | reading | verdict | label | k_rel_halfwidth | family_size |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Extraversion | GPT-3.5 | sex=1 minus sex=0 | 342 | 651 | -0.00 | 0.01 | -0.13 | 0.06 | 0.00 | 0.04 |  |  | undetermined | no population gap | not read | 1.09 | 10 |
| Extraversion | GPT-3.5 | age<30 minus age45+ | 369 | 196 | 0.08 | 0.02 | 0.04 | 0.07 | 0.00 | 1.89 |  |  | undetermined | no population gap | not read | 4.34 | 10 |
| Agreeableness | GPT-3.5 | sex=1 minus sex=0 | 342 | 655 | -0.08 | 0.01 | -0.18 | 0.04 | -0.00 | 0.47 | 0.22 | 1.20 | undetermined | reference too imprecise | not read | 0.58 | 10 |
| Agreeableness | GPT-3.5 | age<30 minus age45+ | 370 | 198 | -0.29 | 0.02 | -0.02 | 0.05 | 0.00 | 18.39 |  |  | steepened | no population gap | not read | 8.73 | 10 |
| Conscientiousness | GPT-3.5 | sex=1 minus sex=0 | 342 | 649 | -0.08 | 0.01 | -0.18 | 0.05 | 0.00 | 0.41 | 0.20 | 1.26 | undetermined | reference too imprecise | not read | 0.68 | 10 |
| Conscientiousness | GPT-3.5 | age<30 minus age45+ | 369 | 196 | -0.24 | 0.02 | -0.29 | 0.06 | 0.00 | 0.83 | 0.53 | 1.81 | undetermined | reference too imprecise | not read | 0.55 | 10 |
| Neuroticism | GPT-3.5 | sex=1 minus sex=0 | 331 | 611 | -0.12 | 0.01 | -0.36 | 0.05 | 0.00 | 0.33 | 0.21 | 0.54 | missing or attenuated | missing or attenuated | not kept | 0.39 | 10 |
| Neuroticism | GPT-3.5 | age<30 minus age45+ | 362 | 166 | 0.12 | 0.02 | 0.29 | 0.07 | 0.00 | 0.42 | 0.21 | 1.14 | undetermined | reference too imprecise | not read | 0.64 | 10 |
| Openness | GPT-3.5 | sex=1 minus sex=0 | 344 | 649 | -0.05 | 0.01 | 0.16 | 0.04 | 0.00 | -0.28 | -0.96 | -0.11 | reversed or missing | reversed or missing | not kept | 0.68 | 10 |
| Openness | GPT-3.5 | age<30 minus age45+ | 370 | 195 | 0.07 | 0.01 | -0.14 | 0.06 | 0.00 | -0.49 |  |  | reversed or missing | no population gap | not read | 1.14 | 10 |
| Extraversion | GPT-4 | sex=1 minus sex=0 | 343 | 653 | 0.05 | 0.02 | -0.14 | 0.06 | 0.00 | -0.38 |  |  | reversed or missing | no population gap | not read | 1.07 | 10 |
| Extraversion | GPT-4 | age<30 minus age45+ | 370 | 197 | 0.10 | 0.02 | 0.06 | 0.07 | 0.00 | 1.60 |  |  | undetermined | no population gap | not read | 2.91 | 10 |
| Agreeableness | GPT-4 | sex=1 minus sex=0 | 343 | 653 | -0.18 | 0.01 | -0.18 | 0.04 | 0.00 | 1.02 | 0.62 | 2.42 | undetermined | reference too imprecise | not read | 0.58 | 10 |
| Agreeableness | GPT-4 | age<30 minus age45+ | 368 | 199 | -0.17 | 0.02 | -0.02 | 0.05 | 0.00 | 10.47 |  |  | kept or steepened | no population gap | not read | 8.16 | 10 |
| Conscientiousness | GPT-4 | sex=1 minus sex=0 | 341 | 656 | 0.01 | 0.02 | -0.18 | 0.05 | 0.00 | -0.03 | -0.50 | 0.31 | reversed to attenuated | reversed to attenuated | not kept | 0.65 | 10 |
| Conscientiousness | GPT-4 | age<30 minus age45+ | 370 | 198 | -0.26 | 0.02 | -0.30 | 0.06 | 0.00 | 0.86 | 0.54 | 1.74 | undetermined | reference too imprecise | not read | 0.50 | 10 |
| Neuroticism | GPT-4 | sex=1 minus sex=0 | 291 | 499 | -0.06 | 0.01 | -0.33 | 0.06 | 0.00 | 0.17 | 0.07 | 0.36 | missing or attenuated | missing or attenuated | not kept | 0.47 | 10 |
| Neuroticism | GPT-4 | age<30 minus age45+ | 278 | 160 | 0.01 | 0.02 | 0.23 | 0.08 | 0.00 | 0.05 | -0.29 | 0.59 | reversed to attenuated | reversed to attenuated | not kept | 0.90 | 10 |
| Openness | GPT-4 | sex=1 minus sex=0 | 343 | 653 | 0.01 | 0.02 | 0.17 | 0.04 | 0.00 | 0.04 | -0.29 | 0.32 | reversed to attenuated | reversed to attenuated | not kept | 0.64 | 10 |
| Openness | GPT-4 | age<30 minus age45+ | 370 | 201 | 0.07 | 0.02 | -0.13 | 0.06 | 0.00 | -0.55 |  |  | reversed or missing | no population gap | not read | 1.16 | 10 |

## B. Silicon level 3 by domain (pass = every group row inside the band)
| model | scale | L3_0.5SD | L3_0.5SD_rows_pass | L3_0.2SD | L3_0.2SD_rows_pass |
|---|---|---|---|---|---|
| GPT-3.5 | Agreeableness | fail | 1/8 | fail | 0/8 |
| GPT-3.5 | Conscientiousness | unresolved | 1/8 | fail | 0/8 |
| GPT-3.5 | Extraversion | pass | 8/8 | fail | 0/8 |
| GPT-3.5 | Neuroticism | fail | 2/8 | fail | 0/8 |
| GPT-3.5 | Openness | pass | 8/8 | unresolved | 3/8 |
| GPT-4 | Agreeableness | pass | 8/8 | fail | 1/8 |
| GPT-4 | Conscientiousness | unresolved | 5/8 | fail | 0/8 |
| GPT-4 | Extraversion | pass | 8/8 | fail | 3/8 |
| GPT-4 | Neuroticism | fail | 4/8 | fail | 0/8 |
| GPT-4 | Openness | pass | 8/8 | fail | 5/8 |

## B. Silicon level 2 labels per model (10 contrasts each)
| model | not kept | not read |
|---|---|---|
| GPT-3.5 | 2 | 8 |
| GPT-4 | 4 | 6 |

Verdicts:
| model | missing or attenuated | no population gap | reference too imprecise | reversed or missing | reversed to attenuated |
|---|---|---|---|---|---|
| GPT-3.5 | 1 | 4 | 4 | 1 | 0 |
| GPT-4 | 1 | 4 | 2 | 0 | 3 |

## A. Whole-vector concentration (50 items)
| group | n | unique_vectors | unique_share | modal_vector_share | mean_item_modal_share | mean_item_entropy_bits |
|---|---|---|---|---|---|---|
| humans (20,000 sample) | 20000 | 20000 | 1.000 | 0.000 | 0.339 | 2.083 |
| claude-haiku-4.5 | 10000 | 1057 | 0.106 | 0.146 | 0.900 | 0.373 |
| claude-sonnet-4.6 | 10000 | 90 | 0.009 | 0.376 | 0.970 | 0.093 |
| deepseek-v3.2 | 10000 | 9997 | 1.000 | 0.000 | 0.603 | 1.304 |
| gemini-3-flash-preview | 10000 | 1337 | 0.134 | 0.090 | 0.880 | 0.405 |
| gemini-3.1-flash-lite-preview | 10000 | 3471 | 0.347 | 0.034 | 0.802 | 0.680 |
| gpt-5.4 | 10000 | 4225 | 0.422 | 0.041 | 0.781 | 0.656 |
| gpt-5.4-mini | 10000 | 2112 | 0.211 | 0.200 | 0.888 | 0.446 |
| gpt-5.4-nano | 10000 | 9729 | 0.973 | 0.001 | 0.778 | 0.738 |
