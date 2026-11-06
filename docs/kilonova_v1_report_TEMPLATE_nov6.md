# Kilonova Hunter v1 — held-test report TEMPLATE (pre-commit before scoring)

**Status:** TEMPLATE ONLY — commit this file (or its hashed repo copy) **before** any held-test score.  
**Prepared:** 2026-11-06 11:59:37 MST (America/Denver) by Science Lead for Engineer/Codex.  
**Pipeline role:** Explicit on-disk markdown report shell that Engineer fills after QA clears pre-score gates. **Not a Canvas.**  
**This template does not authorize scoring.** No held-test numbers exist yet. Do **not** pre-label the held test PASS/FAIL.

**Sources used to draft this shell:**  
`/workspace/context/kilonova_v1_eval_context.md`,  
`/workspace/writeups/kilonova_v1_freeze_summary.md`,  
`/workspace/reviews/kilonova_freeze_model_review_nov6.md` (status addendum retained),  
`/workspace/reviews/kilonova_freeze_ruling_qa_nov6.md`.

---

## 0. Pre-score gates (must be clear before filling §7–§10 from a real run)

QA will verify each item (`kilonova_freeze_ruling_qa_nov6.md`). **No score until QA clears.**

| # | Gate | Owner fill / evidence | Cleared? |
|---|------|------------------------|----------|
| 1 | Decision Log names option (ii), current fade, cuts below, judgeability **(a)**, and the **exact scoring SHA** | Decision Log URL + SHA: `________________` | ☐ |
| 2 | CI green on the scored SHA (or QA-approved merge of it) | CI run URL + SHA: `________________` | ☐ |
| 3 | Test set **regenerated** with diffusion wrapper + SFD dust (**file and line** cited); **not** the cached pre-fix PR #18 split | Provenance note: `________________` | ☐ |
| 4 | Non-test **hard-fail dry run** shows hard block still **writes numbers** and **consumes** the pending ledger entry before nonzero exit; no rescore without approved `--rescore-decision` | Dry-run path / log: `________________` | ☐ |
| 5 | **This report template** is committed (hashed) **before** scoring | Commit SHA of template: `________________` | ☐ |
| 6 | Tuning-only sensitivity results on option (ii) exist (reported, not acted on) — see §11 | Artifact paths: `________________` | ☐ |

**PM-reported code state (attributed; not independently verified here):** PM reports **main = `a2bbb12`** (merge). Full **scoring SHA** for the single held-test run is left for **Engineer/QA verification** below — do not assume `a2bbb12` equals the scored commit without their check.

- Scoring SHA (full, Engineer/QA): `________________________________________`
- Relation to PM-reported main `a2bbb12`: `________________` (same / ancestor / other — with evidence)
- CI green on scoring SHA: ☐ yes / ☐ no — URL: `________________`

---

## 1. Identity of the frozen v1 filter (verified from freeze docs)

| Field | Frozen value (from docs) | Run fill |
|-------|--------------------------|----------|
| Color reading | **Option (ii):** latest **same-night** g+r pair up to day D (Decision Log 12(22); freeze summary) | confirm unchanged: ☐ |
| Fade | **Current fade measurement** (pre-registered fade-measurement candidate **rejected** under rule 5; 12(22) bundle **used up**) | confirm unchanged: ☐ |
| Cuts | **g−r > 0.0**, **rise 0.3**, **window 2 d**, **gap 1.0 d** | confirm unchanged: ☐ |
| Judgeable definition | **(a)** kept; **(b)** dropped (12(22)) | confirm unchanged: ☐ |
| Pass bar (day 3 only) | 95% Wilson lower bound ≥ **70%** on 2017-like judgeable set | — |
| Sample rule | Target **400** judgeable at day 3; cap 20,000; <50 untestable; 50–399 underpowered (report power) | — |
| Score-once | Pending ledger entry before score; object-ID overlap refusal; crash consumes set; rescore only with approved `--rescore-decision` | ledger path: `________________` |

**Framing (required wording):** results are for **this rule on our simulated 2–4-night cadence** (min same-band gap **1.979 d**), **not** a claim that “Rubin can’t.” Footprint is a **stand-in box**, not the official Rubin file. Fink slice is a **ZTF proxy for Rubin**.

---

## 2. Protocol status vs held-test status (keep separate)

### 2.1 Known protocol / tuning outcomes (already measured — not the held test)

These are **tuning / sanity / contamination-protocol** results. They can block “Done / signed off” under the rules even before the held test is opened.

| Check | Result (sourced) | Pass/fail under protocol |
|-------|------------------|---------------------------|
| 2017-like tuning recall, day 3 | **280/410 = 68.3%**; Wilson LB **63.6%** (bar 70%) | **FAIL** (tuning) |
| 2017-like tuning, day 5 / day 7 | **58.8%** / **56.1%** (descriptive; no pass/fail) | descriptive |
| Realistic tuning, day 3 / 5 / 7 | **35.9%** / **53.9%** / **54.3%** (no pass/fail) | descriptive |
| Rule 5 on frozen candidate (tuning) | **0** caught-but-not-judgeable | **PASS** |
| Sanity copies (seed **2026110401**, D=3) | **33/52** (10 fade-only, 7 color-only, 2 both) | **FAIL** (<95%; also >2 misses under 12(16)) |
| Fink ≤10 new candidates/night | **0 flags / 648 cell-nights**; only **225/7,783** objects with ≥2 rows | **contamination limit NOT applied**; cuts **PROVISIONAL**; contamination **unmeasured** (do not call this a measured pass) |
| 12(22) serial-fix budget | One bundled revision used up | further v1 design change **not** pre-authorized |

### 2.2 Held test (single score) — **NOT YET SCORED**

Fill **only after** gates in §0 clear and one score-once run completes. Until then leave placeholders. **Do not invent numbers. Do not pre-label FAIL or PASS.**

---

## 3. Software / data provenance (fill at score time)

| Field | Value |
|-------|-------|
| Scoring git SHA (full) | `NOT YET SCORED — ________________` |
| PM-reported main (attributed) | `a2bbb12` (PM report; Engineer/QA confirm relation to scoring SHA) |
| Template commit SHA (this file, pre-score) | `________________` |
| Decision Log freeze entry ID / URL | `________________` |
| redback version (pinned) | `________________` (expect 1.20.0 per 12(6); verify) |
| Diffusion wrapper applied | ☐ yes — file:line `________________` |
| Dust mode / SFD map | ☐ SFD ×0.86 + F99 confirmed — file:line `________________` / ☐ DRY RUN |
| Population / settings hash | `________________` |
| Test-set identity hash (seeds, fake IDs, population spec) | `________________` |
| Cache / events hash | `________________` |
| Ledger claim ID / pending entry ID | `________________` |
| Ledger consumed? | ☐ yes / ☐ no |
| Python / key package versions | `________________` |
| CI run URL on scoring SHA | `________________` |

**Test-set provenance statement (required):**  
`NOT YET SCORED —` regenerate with diffusion wrapper + SFD; cite file:line; explicitly state **not** the cached pre-fix PR #18 split (old day-3 judgeable counts 156 / 222 are invalid for scoring).

---

## 4. Cadence and model limitations (carry into final report)

- Simulated cadence: one g/r pair ~30 min apart per night; nights **2–4 d** apart; min same-band gap **1.979 d**. **Not** OpSim / not proven Rubin WFD.
- Stand-in footprint box (not official file).
- Early model faintness before ~day 0.8 (no cocoon shock-cooling); day-3 recall biased **low** on detections (12(15)(a) corrected).
- Model g−r too red by about **+0.4** (±0.05; ~**+0.3** without instrument→LSST term) at 1.5–3 d vs AT2017gfo → color cut may look **optimistic**; report sensitivity, do not retune (12(19)/12(20)).
- Realistic-set temperature floors **5000/2500 K** and fixed opacities **0.5/10** are **our assumptions**, not redback defaults (12(14)).
- Realistic set: redback `two_component_kilonova_model` flat after **6 d** source-frame — day-7 numbers use data to day 6 only and say so (12(19)).
- Passing on ZTF proxy ≠ passing on Rubin.

---

## 5. Tuning recall detail (sourced; already known)

### 5.1 2017-like (pass/fail only at day 3)

| Day | Caught / judgeable | Recall | Wilson LB | Role |
|-----|--------------------|--------|-----------|------|
| **3** | **280/410** | **68.3%** | **63.6%** | **Primary — FAIL vs 70% bar** |
| 5 | — / — | **58.8%** | — | descriptive |
| 7 | — / — | **56.1%** | — | descriptive |

Catch rate (caught / all fakes) on tuning: `________________` (fill from run artifacts if not already in REPORT; do not invent).  
Distance bins (luminosity 10–100 / 100–200 / 200–300 Mpc), day 3: `________________`.

### 5.2 Realistic (descriptive only — no pass/fail)

| Day | Recall (tuning) |
|-----|-----------------|
| 3 | **35.9%** |
| 5 | **53.9%** |
| 7 | **54.3%** |

Rule 5 (caught-but-not-judgeable) at days 3/5/7 on frozen candidate (tuning): **0 / 0 / 0** (sourced freeze path).

### 5.3 Sanity copies (protocol blocker)

- Seed **2026110401**, D = 3: **33/52** caught → **FAIL**.
- Miss breakdown: **10** fade-only, **7** color-only, **2** both (color misses truly blue / early blue phase).
- Sign-off blocked under rule 7 / 12(16) regardless of held-test outcome unless a later Sage-approved protocol change says otherwise.

---

## 6. Fink contamination (protocol; unmeasured)

- Wording to use: **“contamination limit NOT applied”**; cuts remain **PROVISIONAL** on this check.
- Evidence: **0** flags in **648** cell-nights; **225/7,783** objects with ≥2 rows → almost nothing for the rules to run on; contamination **unmeasured**.
- Label: ZTF proxy for Rubin; truncated class=Unknown samples (server n=1000 cap).

---

## 7. Held-test 2017-like results — **NOT YET SCORED**

**Day 3 is primary (pass/fail). Days 5 and 7 are descriptive only.**

| Day | Judgeable N | Caught | Recall | Wilson LB | Pass/fail |
|-----|-------------|--------|--------|-----------|-----------|
| **3** | `NOT YET SCORED` | `NOT YET SCORED` | `NOT YET SCORED` | `NOT YET SCORED` | `NOT YET SCORED` (compare to 70% LB bar **only after numbers exist**) |
| 5 | `NOT YET SCORED` | `NOT YET SCORED` | `NOT YET SCORED` | — | descriptive only |
| 7 | `NOT YET SCORED` | `NOT YET SCORED` | `NOT YET SCORED` | — | descriptive only |

Power note (day 3 judgeable under (a)):  
- if N < 50 → **untestable** (never report as pass)  
- if 50 ≤ N ≤ 399 → **underpowered** — report power  
- if N ≥ 400 → powered under sample rule  

Catch rate (caught / all fakes) + Wilson: `NOT YET SCORED`  
Distance bins (10–100 / 100–200 / 200–300 Mpc): `NOT YET SCORED`  
Caught-but-not-judgeable (rule 5) days 3/5/7: `NOT YET SCORED` / `NOT YET SCORED` / `NOT YET SCORED` (must be 0 or hard-block with numbers still saved)

---

## 8. Held-test realistic results — **NOT YET SCORED** (descriptive; no pass/fail)

| Day | Judgeable N | Caught | Recall | Rule 5 count |
|-----|-------------|--------|--------|--------------|
| 3 | `NOT YET SCORED` | `NOT YET SCORED` | `NOT YET SCORED` | `NOT YET SCORED` |
| 5 | `NOT YET SCORED` | `NOT YET SCORED` | `NOT YET SCORED` | `NOT YET SCORED` |
| 7 | `NOT YET SCORED` | `NOT YET SCORED` | `NOT YET SCORED` | `NOT YET SCORED` (data-to-day-6 caveat if applicable) |

---

## 9. Held-test rule 5 / hard-block behavior — **NOT YET SCORED**

- Rule 5 counts written: `NOT YET SCORED`
- If hard block fired: ☐ yes / ☐ no  
- Numbers written before nonzero exit: ☐ verified  
- Ledger pending entry consumed: ☐ verified  
- Rescore attempted: ☐ no / ☐ yes with `--rescore-decision` `________________`

---

## 10. TNS / false-alarm follow-up (sourced status)

- Protocol expects flags/night, fraction with any TNS class, breakdown, raw counts (eval rule 13).
- **Current sourced status:** TNS lookup **not done** — raises **`NotImplementedError`** (`kilonova_v1_eval_context.md` / revision reports).
- Held-test TNS table: `NOT YET SCORED` **or** `NOT IMPLEMENTED` (state which; do not invent classes).

| Metric | Value |
|--------|-------|
| New flags / night (test slice, if scored) | `NOT YET SCORED` |
| Fraction with any TNS class | `NOT YET SCORED` / `NOT IMPLEMENTED` |
| Breakdown by class | `NOT YET SCORED` / `NOT IMPLEMENTED` |
| Raw counts | `NOT YET SCORED` / `NOT IMPLEMENTED` |

---

## 11. Tuning-only sensitivities on option (ii) (pre-score gate)

**Required before scoring** (QA gate 6). Run on **tuning data only**; **report, do not act on** (no cut changes).

| Sensitivity | Status | Artifact / result |
|-------------|--------|-------------------|
| g−r shifted bluer by **+0.4** (and **+0.3**) on frozen option (ii) cuts | **PENDING** unless Engineer pastes sourced numbers here | `________________` |
| Temperature floors **674 / 3745 K** (wind/dynamical) | **PENDING** unless sourced | `________________` |
| Opacities drawn from redback prior (extra) | **PENDING** unless sourced | `________________` |

Note: older +0.3/+0.4 / floor runs on **pre–option (ii)** or placeholder definitions **do not** satisfy this gate (QA / Claude checklist). Fill only with option-(ii) tuning-only results.

---

## 12. Headline (fill only after held-test numbers exist)

- Tuning day-3 bar: **FAIL** (280/410, LB 63.6% < 70%) — already known.  
- Copies: **FAIL** (33/52) — already known.  
- Contamination: **limit NOT applied / PROVISIONAL / unmeasured** — already known.  
- Held-test day-3 2017-like: `NOT YET SCORED` — **do not pre-label**.  
- v1 sign-off / “Done”: blocked by protocol failures above unless Sage changes the protocol; held-test outcome does not erase copy FAIL or unmeasured contamination by itself.
- Frozen, scored (when filled), documented result — **not** signed off by default.

---

## 13. Engineer checklist after QA go (ordering)

1. Confirm §0 all ☐ → QA clear.  
2. Commit **this template** (hash recorded) if not already on the scoring SHA lineage.  
3. Finish §11 sensitivities (tuning-only) and paste sourced results.  
4. Confirm diffusion+SFD test provenance (§3).  
5. Confirm hard-fail dry run (§0.4).  
6. **Then** single held-test score-once; fill §7–§10 from artifacts only.  
7. Hand filled report + ledger + hashes to QA.

---

## 14. Change log for this template file

| When | Who | What |
|------|-----|------|
| 2026-11-06 11:59:37 MST | Science Lead | Initial TEMPLATE for pre-score commit; held-test placeholders only; PM-reported main `a2bbb12` attributed; scoring SHA left for Engineer/QA. |

