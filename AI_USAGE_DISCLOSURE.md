# AI Usage Disclosure

> **Document version**: 1.1 (released with `navimed-umb` v0.2.0)
> **Author**: Łukasz Minarowski, MD, PhD
> **Last updated**: 2026-05-01

This document discloses the AI tools used during the development of
this repository and the associated preprint, the roles those tools
played, and the boundaries within which their output was used.
Transparency about AI assistance is a default policy for this project.

---

## 1. Tools used

| Tool | Provider | Version / Model | Interface | Period of use |
|---|---|---|---|---|
| Claude Opus 4.7 | Anthropic | `claude-opus-4-7` | claude.ai web app | Apr 2026 |
| Claude Code (Opus 4.7) | Anthropic | `claude-opus-4-7` | command-line interface, Kubuntu 24.04 LTS | Apr 2026 |
| GPT-5.5 Deep Thinking | OpenAI | GPT-5.5 with extended reasoning | OpenAI web app | Apr 2026 |

All three tools were accessed through their official end-user
interfaces. No API integration, no automated pipelines, no autonomous
agents. Every interaction was a human-initiated session with explicit
prompts.

A relevant technical detail about Claude Opus 4.7: its knowledge cutoff
is January 2026. The primary empirical finding documented in this
release — the working envelope of Qwen 3.6 27B (released 2026-04-22)
on AMD Radeon AI PRO R9700 GPUs — is therefore strictly post-cutoff
for that model. The model could not have generated those measurements
from training data; it could only react to logs and outputs provided
by the author from local experiments.

### Commit attribution transparency

Two commits in early project history carry explicit `Co-Authored-By: Claude
Opus 4.7` trailers in their commit messages, identifying AI co-authorship at
the commit level per modern git/GitHub disclosure conventions:

- `9a02dfe` (2026-04-25): *docs(benchmarks): add 13-model overview, methodology,
  and Qwen 72B AWQ writeup*
- `426a712` (2026-04-25): *chore(benchmarks): reorganize into scripts/,
  results/<model>/, methodology/, assets/*

Both predate the v0.1.0 release tag. All commits from v0.1.0 onward
(2026-04-26 and later) are attributed solely to the author. The two
trailers are intentionally retained — removing them would require rewriting
the SHA values archived in Zenodo DOI 10.5281/zenodo.19851347 (v0.1.0) and
the v0.2.0 DOI, breaking immutability guarantees expected of citation
records. Per the principle that scientific disclosure should not be
retroactively edited to appear cleaner than it was, the trailers stay.

This is the position consistent with COPE/ICMJE: AI is *not* an author
(Section 3 below), but commit-level acknowledgement of AI assistance during
specific code/documentation operations is appropriate and is preserved here
for the historical record.

---

## 2. Roles

The three tools were used for the following classes of tasks:

### Conceptualisation and architectural discussion
- *Tools*: Claude Opus 4.7 (web), GPT-5.5 Deep Thinking (web)
- Sounding-board for repository structure decisions, project framing,
  and methodology trade-offs.
- Iterative refinement of the public-release scope (what to publish
  now vs. what to keep in private notes).
- Strategic review of long-term roadmap.

### Documentation prose
- *Tools*: Claude Opus 4.7 (web), Claude Code (CLI)
- Drafting and editing of `README.md`, `docs/sessions/*.md`,
  `benchmarks/methodology/*.md`, this disclosure, and the v0.1
  preprint.
- Reformulating empirical findings (provided by the author from
  experimental output) into structured narrative.

### Local code and repository operations
- *Tool*: Claude Code (CLI)
- Multi-file edits, `git mv` reorganisation of `benchmarks/`,
  cross-reference updates after directory renames, drafting of
  per-directory `README.md` files based on author-provided structure.

### Debugging dialogue
- *Tool*: Claude Opus 4.7 (web)
- Reading vLLM/ROCm tracebacks pasted by the author, suggesting
  hypotheses about failure modes (memory pressure, kernel
  incompatibility, environment variable mismatch), proposing follow-up
  tests.
- The author executed every test locally and reported the actual
  outcome; the tool never executed code on the workstation.

### Second-opinion review
- *Tool*: GPT-5.5 Deep Thinking (web)
- Independent review of repository structure proposals and benchmark
  strategy outlines drafted by the author.

---

## 3. Boundaries — what AI did *not* do

The following are entirely the author's responsibility and were not
delegated to any AI tool:

- **Hardware procurement, assembly, and configuration** of the dual
  R9700 workstation, including PCIe topology, power delivery, thermal
  setup, and BIOS tuning.
- **Experimental design**: selection of models, choice of
  configurations to test, definition of pass/fail criteria.
- **Empirical measurements**: every benchmark figure, latency, memory
  footprint, throughput value, and stability outcome was obtained from
  local execution by the author. No measurement was generated,
  estimated, or interpolated by an AI tool.
- **Interpretation of results**: claims about what the measurements
  mean (e.g., "BF16 outpaces FP8 by ~75% on R9700 due to missing FP8
  kernel configurations") are author conclusions formed from the data,
  not assertions sourced from AI tools.
- **Final decisions about repository structure, licensing, citation
  format, and what to include or exclude from the public release.**
- **All decisions about scientific scope** of the v0.1 preprint —
  specifically the choice to limit it to the empirical envelope and to
  defer broader strategic claims to future releases.
- **Authorship**: this work has a single author. AI tools are
  documented as research aids, not co-authors. This boundary is
  consistent with current authorship guidelines from the Committee on
  Publication Ethics (COPE) and the International Committee of Medical
  Journal Editors (ICMJE), which exclude AI tools from authorship
  eligibility on the grounds that AI cannot take accountability for
  the work nor consent to authorship.

---

## 4. Verification practice

For every artefact produced with AI assistance, the following
verification steps were applied before commit or release:

1. **Code**: read the diff manually before `git add`. Any commands
   were executed and observed by the author; AI suggestions were never
   committed unverified.
2. **Documentation prose**: every paragraph was read end-to-end and
   edited for accuracy against the author's own knowledge of the
   subject. Any statement that the author could not personally vouch
   for was either rewritten or removed.
3. **Numerical values**: every measurement, version number, and
   technical specification was either taken directly from local logs
   or cross-checked against authoritative sources (vendor
   documentation, official release notes, model cards).
4. **External claims**: cross-references to other work (e.g., the
   Digital Spaceport CUDA benchmark mentioned in the preprint) were
   verified by the author against the original source.

The author takes full responsibility for the final content of this
repository and the preprint. AI assistance does not transfer
responsibility.

---

## 5. Versioning and updates

This disclosure document will be updated for each tagged release. If
the tooling profile changes (different models, new tools, removed
tools), a new version will be issued with a changelog entry. Past
versions remain in git history.

| Document version | Repository version | Changes |
|---|---|---|
| 1.0 | v0.1.0 | Initial disclosure: Claude Opus 4.7 (web + CLI), GPT-5.5 Deep Thinking (web). |
| 1.1 | v0.2.0 | Replaced PENDING DOI placeholder with assigned concept DOI (10.5281/zenodo.19851346). Added §1 "Commit attribution transparency" documenting two pre-v0.1.0 `Co-Authored-By: Claude` commits intentionally retained for Zenodo immutability. |

---

## 6. Inline disclosure for citing this work

The following short paragraph is the canonical inline disclosure for
papers, posters, and external presentations citing this repository.
Authors may copy it verbatim or paraphrase, with attribution to this
document for full detail:

> *"This work used Claude Opus 4.7 (Anthropic, claude.ai web and
> Claude Code CLI) and GPT-5.5 Deep Thinking (OpenAI, web app) as
> research assistants for documentation, debugging dialogue, and
> sounding-board discussion. All experimental design, hardware
> configuration, empirical measurements, and scientific claims are the
> sole responsibility of the author. AI tools did not execute
> experiments, did not access the workstation, and were not given
> autonomous agency. Full disclosure: see `AI_USAGE_DISCLOSURE.md` in
> the project repository (DOI: 10.5281/zenodo.19851346)."*

---

## 7. Contact

Questions about AI usage in this project, requests for further
specificity about any particular section of the work, or concerns
about the boundaries described above:

**Łukasz Minarowski**
Department of Respiratory Physiopathology
Medical University of Białystok, Poland
ORCID: [0000-0002-2536-3508](https://orcid.org/0000-0002-2536-3508)

---

*This disclosure is licensed under CC-BY-4.0, identical to the rest of
the documentation in this repository. It may be reused as a template
for similar disclosure in other AI-assisted scientific projects, with
attribution.*
