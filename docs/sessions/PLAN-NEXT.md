# PLAN-NEXT — next session after 2026-04-29 v0.2.0 milestone

**Last session:** 2026-04-29 — Qwen 3.6 27B Phase 2 sweep + universal
reporting infrastructure. 8 commits, branch `phase1-qwen36-27b`, ready
to merge.

**Current state:**
- v0.1.0 released (DOI 10.5281/zenodo.19851347).
- v0.2.0 work-in-progress on `phase1-qwen36-27b` branch (not yet merged
  to `main`, not yet tagged, not yet pushed to GitHub at session close —
  push happens at start of next session).
- METHODOLOGY.md v1.0 in place — authoritative for all subsequent work.
- 13-model suite on disk; only 2 models analyzed so far (Qwen 7B retrofit
  + Qwen 27B v0.2.0 single-shot).
- Statistical decision: Tier A only, n=10 reruns + Holm-Bonferroni.

---

## Next session objectives (in order)

### 1. Close v0.2.0 release loop (~30 min)

Resume operations on `phase1-qwen36-27b` branch:

```bash
cd ~/navimed-umb
git status                                  # confirm clean tree
git push -u origin phase1-qwen36-27b        # push branch to GitHub
```

Then create PR and merge to `main` via GitHub UI (reviewing the 9 commits
as a coherent narrative). After merge:

```bash
git checkout main && git pull
git tag -a v0.2.0 -m "v0.2.0 — Phase 2 scaling sweep (Qwen 3.6 27B BF16) + universal reporting infrastructure"
git push origin v0.2.0
```

Zenodo will auto-mint a DOI v2 from the GitHub release. Update README.md
DOI badge in a follow-up commit on `main` (separate atomic change).

GitHub Release notes draft: paste session log narrative for v0.2.0.

### 2. Decide on next benchmark target (~15 min discussion)

Three credible candidates, ordered by paper-impact / time:

**Option A — Bielik 11B FP16 + AWQ (Polish model, white-space)**
- ~6h compute total: Phase 1 envelope (~30 min) + Phase 2 sweep BF16 + AWQ
  (~3h each), n=1 per cell.
- Paper impact: HIGH. First public R9700 benchmark of Polish model.
- Risk: LOW — model fits 1× R9700 comfortably (TP=1 + TP=2 comparable).

**Option B — Qwen 2.5 72B AWQ (international, completes parameter sweep)**
- ~5h compute: Phase 1 (~30 min) + Phase 2 sweep at one config (~4h),
  n=1 per cell.
- Paper impact: MEDIUM. Validates TP=2-mandatory regime (model >32 GB
  single GPU), completes 7B → 27B → 72B story arc.
- Risk: MEDIUM — first time loading 72B; possible OOM at high N due to
  smaller per-request KV budget.

**Option C — Tier A n=10 reruns of Qwen 3.6 27B (statistical rigor)**
- ~6h compute: 9 additional reruns of BF16 max2048 util097 sweep.
- Paper impact: HIGH for any single statistical claim about the 27B
  numbers; LOW for cross-model breadth.
- Risk: LOW — repetition of known-good sweep.

**Recommended: Option A (Bielik)** — extends suite into Polish-model
white space which is the platform's main differentiator. Tier A reruns
deferred to dedicated weekend session covering 2-3 key configs across
multiple models in one batch.

### 3. Execute next benchmark (~3-6h)

Standard workflow per METHODOLOGY:

```bash
# Phase 1 envelope (sanity load all candidate configs)
python3 benchmarks/scripts/runners/sanity_<model>.py \
    | tee logbook/raw_logs/phase1_<model>_$(date +%Y%m%d_%H%M%S).log

# Analyze + select Phase 2 config
python3 benchmarks/scripts/analysis/analyze_phase1_envelope.py --verbose

# Phase 2 sweep
PYTHONUNBUFFERED=1 \
ONLY_QUANT=<quant> \
<MODEL>_MAX_LEN=<from-phase1> \
<MODEL>_UTIL=<from-phase1> \
bash benchmarks/scripts/orchestrators/run_<model>_sweep.sh \
    2>&1 | tee logbook/raw_logs/phase2_<model>_$(date +%Y%m%d_%H%M%S).log

# Analyze + plot
python3 benchmarks/scripts/analysis/analyze_phase2_sweep.py \
    --thermal-runs-dir benchmarks/results/<model>/thermal-runs \
    --output-dir benchmarks/results/<model> \
    --model "<HF-id>" --tuning stock

python3 benchmarks/scripts/plotting/plot_phase2_sweep.py \
    --csv benchmarks/results/<model>/results_table.csv \
    --output-dir benchmarks/results/<model>/
```

For Bielik: `<model>` = `bielik-11b-v23` and `bielik-11b-v23-awq`,
sweeping each as separate config.

### 4. Lab log + commits (~30 min)

Per METHODOLOGY §7.3:
- `docs/sessions/YYYY-MM-DD-<model>-sweep.md` — same template as today's.
- Atomic commits: `data(phase1)`, `data(phase2)` (aggregated artefacts;
  raw thermal-runs gitignored automatically).
- Embargo classification per artefact in commit message.

---

## Open questions for next session

1. **Plotter v2 with multi-config support** — needed for the Bielik
   comparison (BF16 vs AWQ on same plot) and for the deferred 7B TP=1 vs
   TP=2 official figure. Estimate: ~1h to extend `plot_phase2_sweep.py`.
   Should be done before Bielik to keep figures consistent.

2. **Should we attempt Bielik FP16 on TP=1 single GPU?** Phase 1 envelope
   will tell us. If 11B FP16 fits 32 GB (~22 GB weights + KV), TP=1 vs
   TP=2 comparison is publishable as 7B-style story. If not, only TP=2
   works. Decide based on Phase 1 result.

3. **AMD AI Developer Program application** — separate from benchmark
   sessions. ~30 min to fill the form, link account to existing
   GitHub navimed-umb repo, request initial 25 hours MI300X credit.
   Goal: pilot study Polish models subset on MI300X for separate
   "sovereign vs cloud" paper (memory-tracked future work). Can be
   triggered any time, no compute conflict with R9700 sessions.

4. **HF strategy** — defer all uploads until after paper publication
   (decided 2026-04-29). No action this session.

5. **CRLF warning** — git keeps emitting `LF will be replaced by CRLF`
   warnings due to `core.autocrlf=true` (Windows-mode line endings).
   On Linux this is wrong default. One-time fix at start of next session:
   ```bash
   git config core.autocrlf input
   echo "* text=auto eol=lf" >> .gitattributes
   git add --renormalize . && git status
   # commit if anything changed: chore(git): enforce LF line endings
   ```

---

## Known unknowns (to revisit periodically)

- Tier A n=10 schedule — which models get full statistical treatment vs
  which stay single-shot. Decision deferred until 4-5 models have v0.2-style
  exploratory data, then prioritize Tier A by paper impact.
- Multi-config plotter v2 specs — colors per config, legend placement,
  log-x with two TP curves, knee annotation per curve.
- Lemonade Server v0.4 phase — start date depends on vLLM v0.2-v0.3
  completion. Earliest realistic: post Bielik + Qwen 72B (~2 weekend
  sessions out).
- Cross-model figure design for the paper — needed before Tier A reruns
  to know which configs are most informative to repeat.

---

## Embargo policy reminder

Active per METHODOLOGY §11. At every result-generating step, label outputs
explicitly:
- **PUBLIC — engineering note for repo:** load status, VRAM, env vars,
  knee position observation, scheduler robustness statement.
- **EMBARGO — paper figure:** concrete throughput@N, latency P50/P95/P99,
  KV cache curves, cross-model comparative numbers.

Stricter for Polish models (Bielik, PLLuM) — Polish AI community small,
scoop risk higher than Qwen/Mistral. PLLuM 70B in particular: paper-bound
numbers stay local-only until co-author review.
