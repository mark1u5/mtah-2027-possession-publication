# Measurement

The repository's scorer defines `spellF1` by matching same-team H/A possession spans one-to-one using temporal IoU thresholds 0.3, 0.4, 0.5, 0.6, and 0.7, then averaging F1 across thresholds. Dead-ball spans are not possessions. `macroF1` is the unweighted mean over H/A/D; a predicted X on a graded frame is a miss.

For candidate selection, score complete matches independently. Use grouped leave-one-match-out folds; do not split frames randomly. Report the three Metrica games separately, their mean, the worst game, per-threshold spell scores, macroF1, spell counts, and transition diagnostics. With three games, uncertainty estimates are descriptive and do not establish broad generalization.

The pooled IDSSE+Metrica metric is a secondary transfer check, not a substitute for Metrica-held-out performance. The public BAR-ATH slice is for alignment/format checks only.
