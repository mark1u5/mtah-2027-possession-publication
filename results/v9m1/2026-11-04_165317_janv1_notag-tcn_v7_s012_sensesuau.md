# Arnès extern: janv1_notag, tcn_v7

Data 2026-11-04 16:53. Llavors [0, 1, 2]. Etiqueta principal `label_acc`, robustesa `label_dur`. Savitzky-Golay de les posicions: no; derivades amb finestra 11. Temps total 278.6 s.

spellF1 i macroF1 de `evaluation/possession_metrics.py`. '10' = mitjana sobre els 10 partits de selecció (7 d'IDSSE i 3 de Metrica) amb l'interval del 95 % per bootstrap entre partits (2000 remostrejos, llavor 0). SkillCorner és informatiu. La columna pública és informativa: no s'hi decideix res.

| model | spellF1 10 [IC95] | macroF1 10 [IC95] | IDSSE spellF1 / macroF1 | Metrica spellF1 / macroF1 | SkillCorner (informatiu) spellF1 / macroF1 | partit: poss. / durada s / % D / % X | públic spellF1 / macroF1 (no s'hi decideix) |
|---|---|---|---|---|---|---|---|
| janv1_notag | 0.478 [0.436, 0.526] | 0.761 [0.733, 0.791] | 0.434 / 0.732 | 0.582 / 0.829 | 0.034 / 0.356 | 345 / 9.98 / 45.84 / 0.0 | 0.5333 / 0.7818 |
| tcn_v7 | 0.215 [0.185, 0.243] | 0.741 [0.665, 0.806] | 0.241 / 0.813 | 0.154 / 0.574 | 0.387 / 0.897 | 1083 / 4.29 / 26.88 / 0.0 | 0.0719 / 0.4971 |
| janv1 | només partit (informatiu) | - | - | - | - | 301 / 10.55 / 50.0 / 0.0 | 0.5714 / 0.8553 |

## Comparacions aparellades (spellF1, bootstrap per partits)

| A | B | B − A | IC95 | partits on B > A | decisió |
|---|---|---|---|---|---|
| janv1_notag | tcn_v7 | -0.2639 | [-0.3391, -0.2007] | 0/10 | entra janv1_notag |

## Robustesa

Tau de Kendall entre el rànquing amb `label_acc` i amb `label_dur` (spellF1 sobre els 10): 1.0.

| model | spellF1 10 amb label_dur [IC95] | macroF1 10 amb label_dur |
|---|---|---|
| janv1_notag | 0.476 [0.433, 0.523] | 0.761 |
| tcn_v7 | 0.206 [0.183, 0.229] | 0.741 |

Rànquing per llavor (spellF1 sobre els 10):

- llavor 0: janv1_notag (0.469) > tcn_v7 (0.217)
- llavor 1: janv1_notag (0.481) > tcn_v7 (0.217)
- llavor 2: janv1_notag (0.485) > tcn_v7 (0.210)

## Per partit

Mitjana sobre les llavors. spellF1 / macroF1 / precisió per frame / ECE (l'ECE només per als models que donen probabilitats). SkillCorner, per a tcn_v7, és dins de l'entrenament.

| partit | janv1_notag | tcn_v7 |
|---|---|---|
| idsse/DFL-MAT-J03WMX | 0.491 / 0.762 / 0.779 / - | 0.275 / 0.865 / 0.871 / 0.015 |
| idsse/DFL-MAT-J03WN1 | 0.434 / 0.737 / 0.783 / - | 0.199 / 0.784 / 0.815 / 0.016 |
| idsse/DFL-MAT-J03WOH | 0.418 / 0.731 / 0.764 / - | 0.250 / 0.795 / 0.810 / 0.020 |
| idsse/DFL-MAT-J03WOY | 0.406 / 0.724 / 0.757 / - | 0.208 / 0.791 / 0.811 / 0.022 |
| idsse/DFL-MAT-J03WPY | 0.454 / 0.750 / 0.782 / - | 0.236 / 0.816 / 0.836 / 0.030 |
| idsse/DFL-MAT-J03WQQ | 0.394 / 0.688 / 0.705 / - | 0.241 / 0.814 / 0.823 / 0.025 |
| idsse/DFL-MAT-J03WR9 | 0.444 / 0.729 / 0.752 / - | 0.274 / 0.824 / 0.837 / 0.022 |
| metrica/Sample_Game_1 | 0.621 / 0.825 / 0.828 / - | 0.144 / 0.559 / 0.592 / 0.202 |
| metrica/Sample_Game_2 | 0.595 / 0.828 / 0.829 / - | 0.146 / 0.603 / 0.634 / 0.155 |
| metrica/Sample_Game_3 | 0.529 / 0.832 / 0.830 / - | 0.171 / 0.560 / 0.642 / 0.181 |
| skillcorner/1874553 | 0.035 / 0.335 / 0.266 / - | 0.402 / 0.900 / 0.906 / 0.020 |
| skillcorner/1886347 | 0.021 / 0.328 / 0.260 / - | 0.393 / 0.917 / 0.921 / 0.023 |
| skillcorner/1899585 | 0.040 / 0.362 / 0.307 / - | 0.438 / 0.898 / 0.898 / 0.026 |
| skillcorner/1925299 | 0.036 / 0.341 / 0.286 / - | 0.343 / 0.875 / 0.883 / 0.008 |
| skillcorner/1927964 | 0.015 / 0.328 / 0.191 / - | 0.418 / 0.919 / 0.933 / 0.025 |
| skillcorner/1953632 | 0.019 / 0.325 / 0.225 / - | 0.439 / 0.930 / 0.942 / 0.037 |
| skillcorner/1959846 | 0.033 / 0.354 / 0.303 / - | 0.387 / 0.922 / 0.922 / 0.039 |
| skillcorner/1986691 | 0.029 / 0.347 / 0.298 / - | 0.386 / 0.889 / 0.891 / 0.028 |
| skillcorner/1996435 | 0.019 / 0.337 / 0.260 / - | 0.395 / 0.895 / 0.902 / 0.019 |
| skillcorner/1996436 | 0.039 / 0.371 / 0.372 / - | 0.337 / 0.878 / 0.879 / 0.019 |
| skillcorner/2006229 | 0.041 / 0.346 / 0.278 / - | 0.430 / 0.912 / 0.914 / 0.025 |
| skillcorner/2006363 | 0.026 / 0.366 / 0.298 / - | 0.463 / 0.911 / 0.914 / 0.026 |
| skillcorner/2007448 | 0.022 / 0.343 / 0.317 / - | 0.368 / 0.895 / 0.895 / 0.024 |
| skillcorner/2007721 | 0.036 / 0.368 / 0.304 / - | 0.384 / 0.903 / 0.905 / 0.018 |
| skillcorner/2010085 | 0.044 / 0.353 / 0.303 / - | 0.333 / 0.883 / 0.890 / 0.005 |
| skillcorner/2011166 | 0.041 / 0.400 / 0.367 / - | 0.357 / 0.882 / 0.883 / 0.021 |
| skillcorner/2013725 | 0.037 / 0.369 / 0.330 / - | 0.318 / 0.875 / 0.873 / 0.023 |
| skillcorner/2015213 | 0.048 / 0.374 / 0.392 / - | 0.350 / 0.879 / 0.881 / 0.022 |
| skillcorner/2016236 | 0.038 / 0.370 / 0.322 / - | 0.425 / 0.898 / 0.898 / 0.025 |
| skillcorner/2017461 | 0.070 / 0.411 / 0.384 / - | 0.370 / 0.889 / 0.896 / 0.018 |
| **mitjana dels 10** | 0.478 / 0.761 / 0.781 / - | 0.215 / 0.741 / 0.767 / 0.069 |

Partit, tcn_v7: probabilitat mitjana H/A/D [0.3781, 0.3649, 0.257], % d'argmax H/A/D [37.03, 36.09, 26.88].

## Frames avaluats

| font | % dels frames vàlids avaluats | % exclosos per trams < 1 min |
|---|---|---|
| idsse | 100.0 | 0.0 |
| metrica | 100.0 | 0.0 |
| skillcorner | 74.27 | 25.73 |
