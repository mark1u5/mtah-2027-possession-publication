# Regles de janv1 sobre el v9m1 (16/11/2026)

Protocol escrit i publicat abans d'executar res. Cap número del tram públic no hi entra.

## Per què

El v9m1 descodifica la TCN v7d amb les etiquetes de janv1 (`principled_v1`, configuració `p3_notag`).
Dues regles de la definició de `CHALLENGE.md` (`p4_contest` i `p4_control`) es van descartar el
31/10 només amb el tram públic de 5 minuts (17 possessions, 2 jugades decidien), que no pot decidir
res. No s'han mesurat mai al banc extern. A més, `p3_notag` deixa en `D` els frames entre el final
d'una aturada i el primer control detectat, contra "whenever play restarts, there is a clear
possessor".

## Variants (`starter/principled_v1.py`, `NOTAG_VARIANTS`)

| regles | canvi sobre `p3_notag` | constant nova |
|---|---|---|
| `p3_notag_contest` | disputa (els dos equips a l'abast): es queda l'últim equip | cap |
| `p3_notag_control` | canvi només si la pilota es mou amb el jugador nou (mediana de velocitat relativa <= 3,5 m/s) | 3,5 m/s, heretada de markius_v1 |
| `p3_notag_control30`, `_control40` | el mateix amb 3,0 i 4,0 m/s | només sensibilitat, no es tria |
| `p3_notag_restart` | la primera possessió després d'una aturada comença on es reprèn el joc | cap |
| `p2_notag` | sense el retard al mig del vol | informatiu, no es tria |
| `p3_notag+a+b` | combinacions | les de cada part |

## Fase 1: cribratge

`markius_v9_m1__<regles>`: prediu amb les regles noves i amb els paràmetres del descodificador que el
v9m1 tria a cada plec. Mesura la regla sola, sense cap ajust nou. Arnès amb llavors 0, 1, 2 i
`--sense-partit` (no es calcula res del partit).

Passa a la fase 2 una variant amb diferència mitjana de spellF1 > 0 respecte de `markius_v9_m1` i
6/10 partits guanyats o més amb `label_acc`. `p2_notag` i les de sensibilitat no passen mai: són
informatives.

## Fase 2: ajust i promoció

Per a cada variant que passa (i per a la combinació, si n'hi ha més d'una): `tune_decoder.py
entrades` i `graella --graella v9m1` amb `--regles`, i arnès de `markius_v9_m1__<regles>__ajust`
contra `markius_v9_m1`.

Es promociona si compleix les quatre:

1. l'interval aparellat del 95 % de spellF1 (`label_acc`) no inclou zero;
2. guanya 7/10 partits o més;
3. el spellF1 amb `label_dur` no baixa;
4. el nombre de possessions al partit és dins de l'interval de domini.

Si en passen diverses, guanya la més simple (menys canvis). Si no en passa cap, s'envia el v9m1 tal
com està. La columna del tram públic que escriu l'arnès no es mira.

## Resultat (16/11/2026, `bench/results/2026-11-16_164242_*`)

Cribratge amb llavors 0, 1, 2 i `label_acc` (B − A respecte de `markius_v9_m1`):

| variant | B − A | IC95 | partits on B > A |
|---|---|---|---|
| `p3_notag_contest` | −0,0006 | [−0,0025, +0,0016] | 3/10 |
| `p3_notag_control` | −0,0009 | [−0,0028, +0,0001] | 2/10 |
| `p3_notag_restart` | −0,0002 | [−0,0005, +0,0000] | 0/10 |

Amb `label_dur`, totes entre 0,692 i 0,693 (v9m1: 0,693). Cap no passa el cribratge: no hi ha fase 2
i el v9m1 queda amb `p3_notag`. Les etiquetes de janv1 són una entrada entre d'altres del
descodificador, i aquests canvis de regla el mouen al tercer decimal. Les variants informatives
(`p2_notag`, `control30`, `control40`) no s'han executat.
