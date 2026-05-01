# PLAN-NEXT — Bielik 11B v2.3 weekend (2026-05-02 / 2026-05-03)

**Last session:** 2026-05-01 — repository consistency cleanup, 8 atomic
commits + logbook. See `logbook/2026-05-01.md`.

**Current state (post-cleanup):**

* v0.1.0 (DOI 10.5281/zenodo.19851347) and v0.2.0 released.
* `main` clean, all branches merged, no open PRs.
* Repo audit complete: licenses SPDX-recognized, README at v0.2.0,
  CITATION.cff valid, AI_USAGE_DISCLOSURE.md v1.1, mailmap consolidated.
* Bielik 11B v2.3 Instruct + AWQ already in `~/models/` and HF cache.
* METHODOLOGY.md v1.0 in place — authoritative for Bielik runs.

---

## Saturday 2026-05-02 — Bielik 11B FP16 (~5-6h)

### Pre-flight (~30 min)

```
cd ~/navimed-umb
source ~/venvs/vllm/bin/activate
unset PYTORCH_ALLOC_CONF
export VLLM_ROCM_USE_AITER=0
export AMD_SERIALIZE_KERNEL=1
export HIP_LAUNCH_BLOCKING=1
export ROCR_VISIBLE_DEVICES=0,1

python3 -c "import torch; print(torch.cuda.device_count())"
ls ~/models/bielik-11b-v23/
```

Plotter v2 multi-config (~1h) — extend plot_phase2_sweep.py to accept
--csv csv1.csv,csv2.csv --labels FP16,AWQ for cross-quant overlay.
Required before AWQ on Sunday for single comparison plot.

### Phase 1 envelope (~30 min)

Adapt sanity_qwen36_27b.py to sanity_bielik_11b.py. Bielik is not
hybrid attention, so enforce_eager=False (CUDA graphs) is a real
candidate config — first model in suite where eager may not be
mandatory. Test 8-10 configs spanning TP={1,2}, max_len={2048,4096,8192},
util={0.85,0.95}, eager={True,False}, plus 1-2 KV fp8e4m3 sanity points.

**Pre-specified decision criteria** (write into runner comment before
launch, no HARKing):

> Phase 2 config = highest tok/s_out among loaded configs with
> max_concurrency >= 30x scheduler default and VRAM_peak <= 30 GiB.
> Tie-breaker: lowest VRAM peak.

### Phase 2 sweep (~3-4h)

Best config from Phase 1, N={10,25,50,100,200,500,1000}, n_runs=10
with first run discarded (cold cache). Tier A statistics per
METHODOLOGY section 7.4: Holm-Bonferroni FWER on the 7 N values.
Output schema identical to Qwen 3.6 27B sweep for cross-model
comparability.

### Lab log + commits

Atomic commits per METHODOLOGY section 7.3:

* data(phase1): Bielik 11B v2.3 FP16 envelope on R9700 (10 configs)
* data(phase2): Bielik 11B FP16 sweep N=10..1000 (n_runs=10)
* docs(sessions): 2026-05-02 Bielik 11B v2.3 FP16 sweep

Embargo classification per artefact in commit message. Polish-model
embargo is **stricter** than Qwen — keep concrete numbers EMBARGOED
until paper acceptance per scoop-risk policy.

---

## Sunday 2026-05-03 — Bielik 11B AWQ (~4-5h)

Same workflow as Saturday, model = bielik-11b-v23-awq. Phase 1 may be
shorter (4-6 configs vs 10 for FP16) since many variables already
known from FP16 sweep — skip KV fp8 retest, skip eager=False if FP16
showed CUDA graphs work.

**Cross-quant comparison plot** = the deliverable that makes this a
publishable cross-quant story. Plotter v2 must produce a single
scaling_curve.png with both FP16 and AWQ curves overlaid (not two
separate plots). This is the artefact the FP16 + AWQ pair exists for.

After Sunday: tag v0.3.0 if both quants land cleanly.

---

## Open questions (carry forward)

1. **Plotter v2 multi-config** — extend plot_phase2_sweep.py for
   cross-quant overlay. ~1h. Critical for Sunday deliverable.
2. **Bielik TP=1 vs TP=2** — Phase 1 will decide. 11B FP16 should fit
   one R9700 (~22 GB weights + KV); if so, TP=1 vs TP=2 comparison is
   publishable as 7B-style story (TP=2 harmful at single-GPU-fit).
3. **AMD AI Developer Program** — apply (~5 min form). Goal: pilot
   Polish-model subset on MI300X for separate "sovereign vs cloud"
   paper. No compute conflict with R9700 sessions.
4. **HF strategy** — defer all uploads until paper publication.

---

## Embargo policy reminder (METHODOLOGY section 11)

At every result-generating step, label outputs explicitly:

* **PUBLIC** — engineering note for repo: load status, VRAM, env vars,
  knee position observation, scheduler robustness statement.
* **EMBARGO — paper figure** — concrete throughput@N, latency
  P50/P95/P99, KV cache curves, cross-model comparative numbers.

**Stricter for Polish models** (Bielik, PLLuM): scoop risk higher than
Qwen/Mistral due to small Polish AI community. Concrete numbers stay
local-only until preprint submission.
