# Diagnosi del v9m1 (16/11/2026)

Protocol escrit i publicat abans d'executar res. Només dades externes; el partit no s'hi calcula.

## 1. Desfasament de les transicions

La spellF1 casa possessions per IoU (0,3 a 0,7), i un D talla una possessió. Un retard sistemàtic
a les vores fa perdre IoU a totes les possessions. Hi ha dues fonts possibles: la pilota morta es
detecta amb la mitjana de velocitat dels jugadors en una finestra centrada de 2,4 s, i el Viterbi
amb durades mínimes pot empènyer les vores.

`bench/diag_v9m1.py` aparella cada transició predita del v9m1 (paràmetres per plec, com a l'arnès)
amb la de `label_acc` dins de ±2 s, amb el mateix origen i destí, i dona la diferència amb signe
(predita menys real) per tipus: viu a D, D a viu i H a A. També dona la mediana per partit i, per
font, les possessions predites contra les reals i la precisió i el record per llindar d'IoU.

**Criteri per corregir-lo (fase b), fixat ara.** Un tipus de transició es corregeix si la mediana
del desfasament és d'almenys 4 frames en valor absolut a IDSSE, amb el mateix signe en 6 dels 7
partits d'IDSSE o més, i la mediana de Metrica té el mateix signe. La correcció seria un
desplaçament per tipus estimat només amb els partits d'entrenament de cada plec, i es promociona
amb la mateixa regla que les regles (`docs/regles_v9m1.md`).

## 2. HMM amb H i A simètrics

Les estadístiques del descodificador (prior, durades i transicions) separen H i A, i per tant
porten el biaix de local dels partits externs al partit. `decoder.simetritza` fa la mitjana de H i
A al prior de l'HMM, a les durades i a les transicions. La TCN es continua dividint pel prior
d'entrenament, que és el que va aprendre. No hi ha cap constant nova.

`markius_v9_m1__simetric` es compara amb `markius_v9_m1` amb els mateixos paràmetres per plec. Es
promociona si l'interval aparellat de spellF1 (`label_acc`) no inclou zero, guanya 7/10 partits o
més, `label_dur` no baixa i, en una execució final de l'arnès amb el partit, el nombre de
possessions queda dins de l'interval de domini.

## 3. Limitació coneguda (no s'arregla aquí)

Els paràmetres del descodificador per al partit es trien només amb IDSSE degradat amb la v3, que
talla els forats de pilota a la represa (`docs/degradacio.md`) i on la TCN està ben calibrada (ECE
0,02 contra 0,18 a Metrica). El descodificador pot refiar-se més de la TCN del que convindria amb
el tracking real del partit.

## Resultat (16/11/2026, `bench/results/*_diag_v9m1_s012.md`)

**Desfasament: no es corregeix.** A IDSSE la mediana és de −1 a 0 frames per als tres tipus, i el
signe per partit no és consistent. No es compleix el criteri.

Als trackers reals sí que hi ha retard a l'inici de la pilota morta, però en forma de cua i no de
desplaçament fix: a Metrica, viu a D té mediana +1, mitjana +6,4 i un 83 % de transicions tard
(H a A: mitjana +6,8, p75 +16, 66 % tard); a SkillCorner, viu a D té una mediana per partit de
−1 a +7, gairebé sempre positiva. A IDSSE degradat no passa. És coherent amb una pilota que el
tracker real perd quan surt i amb una pilota morta que llavors es detecta per la frenada dels
jugadors (finestra centrada de 2,4 s). És una prova directa de la limitació 3: IDSSE degradat no
reprodueix com falla un tracker real.

**Segmentació: perdem possessions.** Possessions predites / reals: 0,87 a IDSSE, 0,81 a Metrica i
0,80 a SkillCorner, amb la precisió per sobre del record a tots els llindars (IoU 0,5: 0,78 / 0,68 a
IDSSE i 0,74 / 0,64 a Metrica). Gairebé tot ve dels canvis de possessió: es perden un 47 % dels H a A
a IDSSE i un 51 % a Metrica, i les espúries són la meitat. La graella del v9m1 ja oferia
`s_P = 0,5` i va triar 1,0, de manera que no és la durada mínima: els canvis curts no tenen prou
suport a la TCN ni a janv1. És la línia de millora principal.

**HMM simètric: rebutjat.** −0,0003 [−0,0009, +0,0002], 3/10 partits; `label_dur` −0,0004.
