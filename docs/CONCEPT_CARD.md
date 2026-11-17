# Concept card — Metrica-first possession spells

## Objective

Recover complete H/A possession spells from noisy, intermittent tracking, while preserving the competition's distinction between live possession (`H`/`A`) and dead ball (`D`).

## Differentiating hypothesis

A temporal, physically informed decoder can use kinematics to reduce impossible handovers, while a source-aware learned component can adapt to the Metrica label and tracking domain. The proposed next experiment, only if source terms permit, gives substantial, match-balanced weight to labeled non-target Metrica games and compares it with the frozen IDSSE/SkillCorner-trained baseline.

## Validation contract

Use three grouped leave-one-Metrica-match-out folds. In each fold, hold out every frame from one game, including preprocessing and calibration. Report all three games and the worst case. Fit on all three only after selecting the method. Never use BAR-ATH labels for training or selection.

## Risks

Three matches are a small sample. Data-source labels may define possession differently. Metrica's public sample repository has no formal license; permission to train and redistribute derived weights is unresolved. No top-20 result can be guaranteed.

## Status

This is the plan under evaluation, not a measured claim. No final grouped Metrica results are included in this staging snapshot.
