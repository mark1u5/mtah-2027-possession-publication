# Data provenance, labels, and rights

Do not commit raw tracking, event files, ground-truth labels, predictions, caches, or data-derived frame arrays. The competition rules prohibit using BAR-ATH private labels or anything derived from them; the public five-minute slice is not for training or tuning.

## External sources

- **IDSSE / DFL:** documented as CC-BY 4.0 in `docs/llicencies.md`; cite the dataset paper and DFL and describe transformations.
- **SkillCorner Open Data:** source repository carries MIT terms and requests attribution; document the exact dataset version and terms for data/model artifacts.
- **Metrica sample data:** three sample matches and event files are available from the Metrica Sports sample-data repository, but that repository has no formal license. Current internal audit marks training and publication of derived weights as unresolved. Do not redistribute Metrica-trained weights or derived data until rights are clarified.

## Label protocols

The code constructs source-specific H/A/D labels from event and tracking fields. IDSSE, SkillCorner, and Metrica use different control/restart signals; do not claim they represent identical possession semantics. The final technical document must state each source's label producer, criteria, alignment, amount, and version.
