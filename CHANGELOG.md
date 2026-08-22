# Changelog

## 0.17.0 (2026-08-22)


### ⚠ BREAKING CHANGES

* **molecules:** Complex no longer sorts chains by ID. Chains iterate and serialize in insertion order, so code relying on get_chains() returning alphabetical order must sort explicitly. Combining with & or | appends the new chain last rather than sorting it into place, and both now return a new Complex instead of mutating the left operand. Complex is also no longer hashable, matching its mutable-mapping semantics. Structures parsed from CIF or PDB are unaffected.

### Features

* expose unnumbered_policy and job warnings on alignments
* **prompt:** curated antibody prompts (VH, VL, VH-VL, VL-VH) as attributes on the prompt API
* **molecules:** dict-like Complex interface with insertion-ordered chains


## 0.16.2 (2026-07-20)


### Bug Fixes

* export single site future

## 0.16.1 (2026-07-14)


### Bug Fixes

* remove internal backend PR references from multichain e2e

## 0.16.0 (2026-07-14)


### Features

* expose GP kernel selection in the client
* add solublempnn model


### Bug Fixes

* **molecules:** treat B-factors &gt;100 as experimental in pLDDT heuristic
* split ':'-delimited sequences into one multichain complex
