# Arnès extern: markius_v9, markius_v9_r, fusio_v7d, porta_v7d

Data 2026-11-13 21:49. Llavors [0, 1, 2]. Etiqueta principal `label_acc`, robustesa `label_dur`. Savitzky-Golay de les posicions: no; derivades amb finestra 11. Temps total 833.4 s.

spellF1 i macroF1 de `evaluation/possession_metrics.py`. '10' = mitjana sobre els 10 partits de selecció (7 d'IDSSE i 3 de Metrica) amb l'interval del 95 % per bootstrap entre partits (2000 remostrejos, llavor 0). SkillCorner és informatiu. La columna pública és informativa: no s'hi decideix res.

| model | spellF1 10 [IC95] | macroF1 10 [IC95] | IDSSE spellF1 / macroF1 | Metrica spellF1 / macroF1 | SkillCorner (informatiu) spellF1 / macroF1 | partit: poss. / durada s / % D / % X | domini | públic spellF1 / macroF1 (no s'hi decideix) |
|---|---|---|---|---|---|---|---|---|
| markius_v9 | 0.697 [0.664, 0.727] | 0.892 [0.876, 0.907] | 0.708 / 0.908 | 0.670 / 0.856 | 0.727 / 0.900 | - | - | - |
| fusio_v7d | 0.696 [0.664, 0.728] | 0.891 [0.875, 0.906] | 0.705 / 0.906 | 0.676 / 0.856 | 0.697 / 0.889 | - | - | - |
| porta_v7d | 0.690 [0.654, 0.723] | 0.891 [0.875, 0.906] | 0.701 / 0.906 | 0.662 / 0.856 | 0.248 / 0.753 | - | - | - |
| markius_v9_r | 0.688 [0.660, 0.715] | 0.882 [0.870, 0.893] | 0.698 / 0.894 | 0.665 / 0.854 | 0.667 / 0.879 | - | - | - |

## Comparacions aparellades (spellF1, bootstrap per partits)

| A | B | B − A | IC95 | partits on B > A | decisió |
|---|---|---|---|---|---|
| markius_v9 | markius_v9_r | -0.0085 | [-0.0140, -0.0023] | 2/10 | entra markius_v9 |
| markius_v9 | fusio_v7d | -0.0002 | [-0.0050, +0.0041] | 5/10 | no es distingeixen: guanya el més simple |
| markius_v9 | porta_v7d | -0.0069 | [-0.0152, +0.0012] | 3/10 | no es distingeixen: guanya el més simple |
| markius_v9_r | fusio_v7d | +0.0083 | [-0.0002, +0.0158] | 8/10 | no es distingeixen: guanya el més simple |
| markius_v9_r | porta_v7d | +0.0016 | [-0.0084, +0.0111] | 6/10 | no es distingeixen: guanya el més simple |
| fusio_v7d | porta_v7d | -0.0067 | [-0.0170, +0.0048] | 3/10 | no es distingeixen: guanya el més simple |

## Robustesa

Tau de Kendall entre el rànquing amb `label_acc` i amb `label_dur` (spellF1 sobre els 10): 0.3333333333333333.

| model | spellF1 10 amb label_dur [IC95] | macroF1 10 amb label_dur |
|---|---|---|
| markius_v9 | 0.690 [0.656, 0.725] | 0.891 |
| markius_v9_r | 0.682 [0.654, 0.712] | 0.880 |
| fusio_v7d | 0.692 [0.661, 0.726] | 0.890 |
| porta_v7d | 0.692 [0.651, 0.730] | 0.891 |

Rànquing per llavor (spellF1 sobre els 10):

- llavor 0: markius_v9 (0.695) > fusio_v7d (0.694) > porta_v7d (0.694) > markius_v9_r (0.691)
- llavor 1: fusio_v7d (0.695) > markius_v9 (0.693) > porta_v7d (0.691) > markius_v9_r (0.685)
- llavor 2: markius_v9 (0.702) > fusio_v7d (0.700) > markius_v9_r (0.688) > porta_v7d (0.684)

## Per partit

Mitjana sobre les llavors. spellF1 / macroF1 / precisió per frame / ECE (l'ECE només per als models que donen probabilitats). SkillCorner, per a fusio_v7d, markius_v9, markius_v9_r, porta_v7d, és dins de l'entrenament.

| partit | markius_v9 | markius_v9_r | fusio_v7d | porta_v7d |
|---|---|---|---|---|
| idsse/DFL-MAT-J03WMX | 0.706 / 0.904 / 0.912 / - | 0.701 / 0.893 / 0.901 / - | 0.689 / 0.900 / 0.909 / - | 0.722 / 0.911 / 0.918 / - |
| idsse/DFL-MAT-J03WN1 | 0.681 / 0.906 / 0.925 / - | 0.667 / 0.897 / 0.917 / - | 0.675 / 0.905 / 0.924 / - | 0.650 / 0.897 / 0.918 / - |
| idsse/DFL-MAT-J03WOH | 0.688 / 0.909 / 0.923 / - | 0.679 / 0.895 / 0.909 / - | 0.687 / 0.910 / 0.923 / - | 0.687 / 0.909 / 0.922 / - |
| idsse/DFL-MAT-J03WOY | 0.692 / 0.900 / 0.913 / - | 0.677 / 0.888 / 0.900 / - | 0.691 / 0.900 / 0.913 / - | 0.696 / 0.901 / 0.914 / - |
| idsse/DFL-MAT-J03WPY | 0.781 / 0.923 / 0.936 / - | 0.761 / 0.902 / 0.914 / - | 0.786 / 0.924 / 0.936 / - | 0.775 / 0.920 / 0.932 / - |
| idsse/DFL-MAT-J03WQQ | 0.750 / 0.925 / 0.931 / - | 0.734 / 0.900 / 0.905 / - | 0.756 / 0.919 / 0.925 / - | 0.729 / 0.918 / 0.924 / - |
| idsse/DFL-MAT-J03WR9 | 0.659 / 0.885 / 0.897 / - | 0.667 / 0.881 / 0.893 / - | 0.653 / 0.881 / 0.893 / - | 0.652 / 0.889 / 0.900 / - |
| metrica/Sample_Game_1 | 0.731 / 0.853 / 0.855 / - | 0.717 / 0.852 / 0.853 / - | 0.737 / 0.854 / 0.855 / - | 0.724 / 0.851 / 0.852 / - |
| metrica/Sample_Game_2 | 0.696 / 0.853 / 0.854 / - | 0.686 / 0.850 / 0.850 / - | 0.698 / 0.851 / 0.851 / - | 0.696 / 0.855 / 0.855 / - |
| metrica/Sample_Game_3 | 0.583 / 0.860 / 0.861 / - | 0.592 / 0.861 / 0.861 / - | 0.592 / 0.863 / 0.863 / - | 0.565 / 0.862 / 0.862 / - |
| skillcorner/1874553 | 0.811 / 0.912 / 0.916 / - | 0.705 / 0.885 / 0.891 / - | 0.743 / 0.895 / 0.901 / - | 0.190 / 0.720 / 0.712 / - |
| skillcorner/1886347 | 0.720 / 0.913 / 0.917 / - | 0.645 / 0.891 / 0.895 / - | 0.668 / 0.899 / 0.903 / - | 0.203 / 0.751 / 0.743 / - |
| skillcorner/1899585 | 0.707 / 0.898 / 0.898 / - | 0.648 / 0.872 / 0.871 / - | 0.678 / 0.883 / 0.883 / - | 0.262 / 0.748 / 0.738 / - |
| skillcorner/1925299 | 0.709 / 0.890 / 0.895 / - | 0.665 / 0.874 / 0.880 / - | 0.692 / 0.880 / 0.886 / - | 0.247 / 0.754 / 0.754 / - |
| skillcorner/1927964 | 0.807 / 0.924 / 0.935 / - | 0.796 / 0.916 / 0.922 / - | 0.812 / 0.921 / 0.928 / - | 0.195 / 0.769 / 0.742 / - |
| skillcorner/1953632 | 0.846 / 0.942 / 0.952 / - | 0.758 / 0.911 / 0.923 / - | 0.778 / 0.924 / 0.934 / - | 0.235 / 0.781 / 0.779 / - |
| skillcorner/1959846 | 0.795 / 0.931 / 0.930 / - | 0.737 / 0.904 / 0.902 / - | 0.764 / 0.914 / 0.913 / - | 0.304 / 0.797 / 0.788 / - |
| skillcorner/1986691 | 0.706 / 0.885 / 0.886 / - | 0.644 / 0.872 / 0.873 / - | 0.667 / 0.880 / 0.881 / - | 0.211 / 0.737 / 0.731 / - |
| skillcorner/1996435 | 0.724 / 0.887 / 0.893 / - | 0.654 / 0.860 / 0.864 / - | 0.675 / 0.871 / 0.874 / - | 0.225 / 0.729 / 0.713 / - |
| skillcorner/1996436 | 0.638 / 0.878 / 0.880 / - | 0.577 / 0.860 / 0.862 / - | 0.626 / 0.874 / 0.876 / - | 0.243 / 0.740 / 0.749 / - |
| skillcorner/2006229 | 0.711 / 0.897 / 0.898 / - | 0.659 / 0.871 / 0.869 / - | 0.701 / 0.885 / 0.884 / - | 0.232 / 0.762 / 0.750 / - |
| skillcorner/2006363 | 0.787 / 0.922 / 0.923 / - | 0.712 / 0.905 / 0.905 / - | 0.745 / 0.914 / 0.914 / - | 0.308 / 0.784 / 0.771 / - |
| skillcorner/2007448 | 0.723 / 0.897 / 0.897 / - | 0.663 / 0.877 / 0.877 / - | 0.699 / 0.890 / 0.890 / - | 0.253 / 0.757 / 0.755 / - |
| skillcorner/2007721 | 0.726 / 0.902 / 0.904 / - | 0.679 / 0.885 / 0.885 / - | 0.706 / 0.895 / 0.896 / - | 0.220 / 0.758 / 0.745 / - |
| skillcorner/2010085 | 0.650 / 0.885 / 0.892 / - | 0.633 / 0.868 / 0.874 / - | 0.657 / 0.874 / 0.880 / - | 0.248 / 0.760 / 0.760 / - |
| skillcorner/2011166 | 0.708 / 0.883 / 0.885 / - | 0.630 / 0.873 / 0.873 / - | 0.679 / 0.882 / 0.883 / - | 0.242 / 0.713 / 0.709 / - |
| skillcorner/2013725 | 0.729 / 0.886 / 0.885 / - | 0.663 / 0.851 / 0.849 / - | 0.698 / 0.866 / 0.864 / - | 0.285 / 0.744 / 0.736 / - |
| skillcorner/2015213 | 0.661 / 0.887 / 0.890 / - | 0.615 / 0.867 / 0.870 / - | 0.645 / 0.878 / 0.881 / - | 0.298 / 0.761 / 0.776 / - |
| skillcorner/2016236 | 0.703 / 0.890 / 0.891 / - | 0.628 / 0.865 / 0.864 / - | 0.654 / 0.876 / 0.876 / - | 0.277 / 0.751 / 0.742 / - |
| skillcorner/2017461 | 0.671 / 0.891 / 0.898 / - | 0.627 / 0.877 / 0.885 / - | 0.644 / 0.880 / 0.888 / - | 0.273 / 0.748 / 0.758 / - |
| **mitjana dels 10** | 0.697 / 0.892 / 0.900 / - | 0.688 / 0.882 / 0.890 / - | 0.696 / 0.891 / 0.899 / - | 0.690 / 0.891 / 0.900 / - |

## Frames avaluats

| font | % dels frames vàlids avaluats | % exclosos per trams < 1 min |
|---|---|---|
| idsse | 100.0 | 0.0 |
| metrica | 100.0 | 0.0 |
| skillcorner | 74.27 | 25.73 |
