# Week12 efficiency summary: 15 official runs (token_update_matched_v2)

wall definition per run recorded in wall_source column; replaces week12_main_table.md wall column (summed-all-keys, double-counted)

| method | seed | full_tok | answer_tok | steps | wall_s (def) | probe_s | utility_s | overhead |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random_sameloop | 22 | 1499981 | 761674 | 818 | 1307.1 (training_wall) | 87.9 | 0.07 | 0.0673 |
| random_sameloop | 23 | 1499980 | 787886 | 822 | 1202.4 (training_wall) | 87.6 | 0.07 | 0.0729 |
| random_sameloop | 24 | 1499971 | 713268 | 815 | 1165.9 (training_wall) | 87.6 | 0.07 | 0.0752 |
| lp_only_replay | 22 | 1499945 | 787476 | 794 | 1198.4 (training_wall) | 334.4 | 0.05 | 0.2791 |
| lp_only_replay | 23 | 1499990 | 761410 | 813 | 1357.3 (training_wall) | 340.8 | 0.05 | 0.2511 |
| lp_only_replay | 24 | 1499975 | 795450 | 828 | 1570.9 (training_wall) | 344.8 | 0.05 | 0.2195 |
| lp_match_replay | 22 | 1499956 | 732197 | 804 | 1326.5 (training_wall) | 338.5 | 0.05 | 0.2552 |
| lp_match_replay | 23 | 1499993 | 719286 | 817 | 1429.4 (training_wall) | 340.8 | 0.05 | 0.2384 |
| lp_match_replay | 24 | 1499977 | 725836 | 800 | 1186.7 (training_wall) | 336.3 | 0.05 | 0.2834 |
| lpf_sparse | 22 | 1499920 | 818624 | 826 | 1275.6 (training_wall) | 88.2 | 0.07 | 0.0692 |
| lpf_sparse | 23 | 1499972 | 805535 | 855 | 1722.9 (training_wall) | 91.7 | 0.06 | 0.0533 |
| lpf_sparse | 24 | 1499990 | 791931 | 851 | 1687.8 (training_wall) | 91.8 | 0.07 | 0.0544 |
| safe_fallback | 25 | 1499965 | 724885 | 816 | 1126.4 (training_wall) | 344.6 | 0.06 | 0.3060 |
| safe_fallback | 26 | 1499971 | 731856 | 811 | 1168.1 (training_wall) | 347.9 | 0.06 | 0.2979 |
| safe_fallback | 27 | 1499968 | 697661 | 806 | 1058.4 (training_wall) | 340.5 | 0.06 | 0.3218 |
