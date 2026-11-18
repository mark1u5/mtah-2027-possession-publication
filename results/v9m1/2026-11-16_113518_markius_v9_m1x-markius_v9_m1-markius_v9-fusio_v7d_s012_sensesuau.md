# Arnès extern: markius_v9_m1x, markius_v9_m1, markius_v9, fusio_v7d

Data 2026-11-16 11:35. Git HEAD `b144bece3d506609110040066bc5c8f1d90386fd`. Llavors [0, 1, 2]. Etiqueta principal `label_acc`, robustesa `label_dur`. Savitzky-Golay de les posicions: no; derivades amb finestra 11. Temps total 1526.0 s.

spellF1 i macroF1 de `evaluation/possession_metrics.py`. '10' = mitjana sobre els 10 partits de selecció (7 d'IDSSE i 3 de Metrica) amb l'interval del 95 % per bootstrap entre partits (2000 remostrejos, llavor 0). SkillCorner és informatiu. La columna pública és informativa: no s'hi decideix res.

| model | spellF1 10 [IC95] | macroF1 10 [IC95] | IDSSE spellF1 / macroF1 | Metrica spellF1 / macroF1 | SkillCorner (informatiu) spellF1 / macroF1 | partit: poss. / durada s / % D / % X | domini | públic spellF1 / macroF1 (no s'hi decideix) |
|---|---|---|---|---|---|---|---|---|
| markius_v9_m1x | 0.701 [0.669, 0.733] | 0.893 [0.877, 0.909] | 0.717 / 0.909 | 0.665 / 0.856 | 0.696 / 0.892 | 216 / 17.69 / 39.85 / 0.0 | dins | 0.4828 / 0.6836 |
| markius_v9_m1 | 0.701 [0.667, 0.735] | 0.893 [0.876, 0.909] | 0.714 / 0.909 | 0.671 / 0.856 | 0.726 / 0.900 | 211 / 18.06 / 40.03 / 0.0 | dins | 0.4828 / 0.6802 |
| markius_v9 | 0.697 [0.664, 0.727] | 0.892 [0.876, 0.907] | 0.708 / 0.908 | 0.670 / 0.856 | 0.727 / 0.900 | 208 / 18.31 / 40.06 / 0.0 | dins | 0.4828 / 0.6802 |
| fusio_v7d | 0.696 [0.664, 0.728] | 0.891 [0.875, 0.906] | 0.705 / 0.906 | 0.676 / 0.856 | 0.697 / 0.889 | 208 / 17.84 / 41.59 / 0.0 | dins | 0.4774 / 0.6742 |
| janv1 | només partit (informatiu) | - | - | - | - | 301 / 10.55 / 50.0 / 0.0 | dins | 0.5714 / 0.8553 |

Comprovació de domini (docs/criteri_seleccio.md): prior 264.4 possessions, interval [132.2, 396.6]. Un model fora de l'interval no es pot triar.

## Comparacions aparellades (spellF1, bootstrap per partits)

| A | B | B − A | IC95 | partits on B > A | decisió |
|---|---|---|---|---|---|
| markius_v9_m1x | markius_v9_m1 | -0.0000 | [-0.0068, +0.0062] | 4/10 | no es distingeixen: guanya el més simple |
| markius_v9_m1x | markius_v9 | -0.0048 | [-0.0118, +0.0022] | 3/10 | no es distingeixen: guanya el més simple |
| markius_v9_m1x | fusio_v7d | -0.0050 | [-0.0131, +0.0035] | 4/10 | no es distingeixen: guanya el més simple |
| markius_v9_m1 | markius_v9 | -0.0047 | [-0.0105, -0.0004] | 3/10 | entra markius_v9_m1 |
| markius_v9_m1 | fusio_v7d | -0.0049 | [-0.0111, +0.0014] | 3/10 | no es distingeixen: guanya el més simple |
| markius_v9 | fusio_v7d | -0.0002 | [-0.0050, +0.0041] | 5/10 | no es distingeixen: guanya el més simple |

## Robustesa

Tau de Kendall entre el rànquing amb `label_acc` i amb `label_dur` (spellF1 sobre els 10): -0.3333333333333333.

| model | spellF1 10 amb label_dur [IC95] | macroF1 10 amb label_dur |
|---|---|---|
| markius_v9_m1x | 0.687 [0.657, 0.719] | 0.892 |
| markius_v9_m1 | 0.693 [0.658, 0.728] | 0.892 |
| markius_v9 | 0.690 [0.656, 0.725] | 0.891 |
| fusio_v7d | 0.692 [0.661, 0.726] | 0.890 |

Rànquing per llavor (spellF1 sobre els 10):

- llavor 0: markius_v9_m1 (0.699) > markius_v9_m1x (0.696) > markius_v9 (0.695) > fusio_v7d (0.694)
- llavor 1: markius_v9_m1x (0.701) > markius_v9_m1 (0.695) > fusio_v7d (0.695) > markius_v9 (0.693)
- llavor 2: markius_v9_m1 (0.710) > markius_v9_m1x (0.707) > markius_v9 (0.702) > fusio_v7d (0.700)

## Per partit

Mitjana sobre les llavors. spellF1 / macroF1 / precisió per frame / ECE (l'ECE només per als models que donen probabilitats). SkillCorner, per a fusio_v7d, markius_v9, markius_v9_m1, markius_v9_m1x, és dins de l'entrenament.

| partit | markius_v9_m1x | markius_v9_m1 | markius_v9 | fusio_v7d |
|---|---|---|---|---|
| idsse/DFL-MAT-J03WMX | 0.699 / 0.904 / 0.912 / - | 0.706 / 0.904 / 0.912 / - | 0.706 / 0.904 / 0.912 / - | 0.689 / 0.900 / 0.909 / - |
| idsse/DFL-MAT-J03WN1 | 0.691 / 0.906 / 0.925 / - | 0.683 / 0.907 / 0.926 / - | 0.681 / 0.906 / 0.925 / - | 0.675 / 0.905 / 0.924 / - |
| idsse/DFL-MAT-J03WOH | 0.712 / 0.911 / 0.924 / - | 0.711 / 0.912 / 0.925 / - | 0.688 / 0.909 / 0.923 / - | 0.687 / 0.910 / 0.923 / - |
| idsse/DFL-MAT-J03WOY | 0.713 / 0.904 / 0.915 / - | 0.692 / 0.900 / 0.913 / - | 0.692 / 0.900 / 0.913 / - | 0.691 / 0.900 / 0.913 / - |
| idsse/DFL-MAT-J03WPY | 0.788 / 0.928 / 0.940 / - | 0.799 / 0.929 / 0.941 / - | 0.781 / 0.923 / 0.936 / - | 0.786 / 0.924 / 0.936 / - |
| idsse/DFL-MAT-J03WQQ | 0.754 / 0.925 / 0.931 / - | 0.751 / 0.925 / 0.931 / - | 0.750 / 0.925 / 0.931 / - | 0.756 / 0.919 / 0.925 / - |
| idsse/DFL-MAT-J03WR9 | 0.661 / 0.888 / 0.899 / - | 0.659 / 0.885 / 0.897 / - | 0.659 / 0.885 / 0.897 / - | 0.653 / 0.881 / 0.893 / - |
| metrica/Sample_Game_1 | 0.722 / 0.853 / 0.855 / - | 0.730 / 0.854 / 0.855 / - | 0.731 / 0.853 / 0.855 / - | 0.737 / 0.854 / 0.855 / - |
| metrica/Sample_Game_2 | 0.684 / 0.853 / 0.854 / - | 0.699 / 0.854 / 0.854 / - | 0.696 / 0.853 / 0.854 / - | 0.698 / 0.851 / 0.851 / - |
| metrica/Sample_Game_3 | 0.588 / 0.862 / 0.862 / - | 0.583 / 0.861 / 0.861 / - | 0.583 / 0.860 / 0.861 / - | 0.592 / 0.863 / 0.863 / - |
| skillcorner/1874553 | 0.768 / 0.900 / 0.905 / - | 0.811 / 0.912 / 0.916 / - | 0.811 / 0.912 / 0.916 / - | 0.743 / 0.895 / 0.901 / - |
| skillcorner/1886347 | 0.693 / 0.903 / 0.907 / - | 0.720 / 0.913 / 0.917 / - | 0.720 / 0.913 / 0.917 / - | 0.668 / 0.899 / 0.903 / - |
| skillcorner/1899585 | 0.679 / 0.890 / 0.889 / - | 0.707 / 0.898 / 0.898 / - | 0.707 / 0.898 / 0.898 / - | 0.678 / 0.883 / 0.883 / - |
| skillcorner/1925299 | 0.682 / 0.882 / 0.888 / - | 0.707 / 0.889 / 0.895 / - | 0.709 / 0.890 / 0.895 / - | 0.692 / 0.880 / 0.886 / - |
| skillcorner/1927964 | 0.791 / 0.919 / 0.930 / - | 0.807 / 0.924 / 0.935 / - | 0.807 / 0.924 / 0.935 / - | 0.812 / 0.921 / 0.928 / - |
| skillcorner/1953632 | 0.809 / 0.934 / 0.944 / - | 0.846 / 0.942 / 0.952 / - | 0.846 / 0.942 / 0.952 / - | 0.778 / 0.924 / 0.934 / - |
| skillcorner/1959846 | 0.774 / 0.924 / 0.923 / - | 0.795 / 0.931 / 0.930 / - | 0.795 / 0.931 / 0.930 / - | 0.764 / 0.914 / 0.913 / - |
| skillcorner/1986691 | 0.671 / 0.876 / 0.877 / - | 0.696 / 0.884 / 0.885 / - | 0.706 / 0.885 / 0.886 / - | 0.667 / 0.880 / 0.881 / - |
| skillcorner/1996435 | 0.696 / 0.883 / 0.887 / - | 0.727 / 0.890 / 0.895 / - | 0.724 / 0.887 / 0.893 / - | 0.675 / 0.871 / 0.874 / - |
| skillcorner/1996436 | 0.599 / 0.867 / 0.870 / - | 0.638 / 0.878 / 0.880 / - | 0.638 / 0.878 / 0.880 / - | 0.626 / 0.874 / 0.876 / - |
| skillcorner/2006229 | 0.670 / 0.888 / 0.887 / - | 0.713 / 0.899 / 0.899 / - | 0.711 / 0.897 / 0.898 / - | 0.701 / 0.885 / 0.884 / - |
| skillcorner/2006363 | 0.745 / 0.912 / 0.913 / - | 0.787 / 0.921 / 0.923 / - | 0.787 / 0.922 / 0.923 / - | 0.745 / 0.914 / 0.914 / - |
| skillcorner/2007448 | 0.701 / 0.890 / 0.890 / - | 0.722 / 0.897 / 0.897 / - | 0.723 / 0.897 / 0.897 / - | 0.699 / 0.890 / 0.890 / - |
| skillcorner/2007721 | 0.710 / 0.895 / 0.896 / - | 0.726 / 0.902 / 0.904 / - | 0.726 / 0.902 / 0.904 / - | 0.706 / 0.895 / 0.896 / - |
| skillcorner/2010085 | 0.637 / 0.878 / 0.884 / - | 0.650 / 0.886 / 0.892 / - | 0.650 / 0.885 / 0.892 / - | 0.657 / 0.874 / 0.880 / - |
| skillcorner/2011166 | 0.652 / 0.873 / 0.874 / - | 0.708 / 0.883 / 0.885 / - | 0.708 / 0.883 / 0.885 / - | 0.679 / 0.882 / 0.883 / - |
| skillcorner/2013725 | 0.710 / 0.877 / 0.875 / - | 0.729 / 0.886 / 0.885 / - | 0.729 / 0.886 / 0.885 / - | 0.698 / 0.866 / 0.864 / - |
| skillcorner/2015213 | 0.639 / 0.880 / 0.883 / - | 0.656 / 0.886 / 0.889 / - | 0.661 / 0.887 / 0.890 / - | 0.645 / 0.878 / 0.881 / - |
| skillcorner/2016236 | 0.678 / 0.885 / 0.885 / - | 0.704 / 0.891 / 0.892 / - | 0.703 / 0.890 / 0.891 / - | 0.654 / 0.876 / 0.876 / - |
| skillcorner/2017461 | 0.628 / 0.879 / 0.887 / - | 0.671 / 0.891 / 0.898 / - | 0.671 / 0.891 / 0.898 / - | 0.644 / 0.880 / 0.888 / - |
| **mitjana dels 10** | 0.701 / 0.893 / 0.902 / - | 0.701 / 0.893 / 0.901 / - | 0.697 / 0.892 / 0.900 / - | 0.696 / 0.891 / 0.899 / - |

## Frames avaluats

| font | % dels frames vàlids avaluats | % exclosos per trams < 1 min |
|---|---|---|
| idsse | 100.0 | 0.0 |
| metrica | 100.0 | 0.0 |
| skillcorner | 74.27 | 25.73 |
