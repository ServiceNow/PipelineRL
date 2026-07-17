# Tau2 Gemma4 PipelineRL Port

## Status

This branch starts from PipelineRL `main` at `58d3934`. It targets
`google/gemma-4-26B-A4B-it` on Tau2 through NeMo Gym.

PipelineRL is the trainer, rollout scheduler, inference owner, and weight-sync
authority. NeMo Gym is an environment backend. No experiment should launch
until the acceptance gates in this document pass.

## Session Bootstrap

- Worktree: `/home/toolkit/PipelineRL-gemma4-tau2`
- Branch and base: `gemma4-tau2` from
  `main@58d393458625ad63ed539f2dcd072c85700c557f`
- Canonical coordination: `/home/toolkit/PipelineRL/AGENT_CHAT.md`, block
  `<gemma4-tau2-port>`
- Consensus history: the canonical claim block, the reviewed Tau2/Gemma
  discussion in `AGENT_CHAT.md`, and Claude memory `project_tau2_gemma_branch.md`

Start every implementation session with:

```bash
source /opt/conda/bin/activate pipeline-rl
cd /home/toolkit/PipelineRL-gemma4-tau2
export PYTHONPATH=$PWD
```

Fast verification for each reviewed wave:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest <focused-test-files> -q
python -m py_compile <changed-python-files>
git diff --check
```

Codex implements one wave at a time and Claude reviews the exact diff before
commit. Each wave must update the canonical claim with files, status, and
verification. Do not stack an unreviewed wave on top of another.

**Workstream separation:** terminal and `dppo_05`/`dppo_06` continue in the
other session. This worktree must not edit terminal experiment files, modify
the live checkout, or control live jobs.

## Objective

Build a minimal Tau2 training path that:

- keeps PipelineRL's asynchronous training and vLLM weight propagation;
- uses NeMo Gym's Tau2 agent, resources, verifier, and frozen user simulator;
- trains every policy-generated assistant token from each Tau2 trajectory;
- preserves the exact token IDs and behavior log probabilities used at
  generation time;
- starts from Tau2's scalar outcome reward without local shaping; and
- retains enough provenance and audit data to diagnose learning failures.

## Invariants

1. Any behavior required to collect reward must reliably appear in the trained
   token stream.
2. Configuration must be verified where it executes, not only where it is
   composed.
3. PipelineRL owns policy inference. NeMo Gym must never launch a second policy
   vLLM.
4. A rollout, group, or merged training sample is dropped as a whole on a
   boundary violation. It is never silently truncated or partially retained.
5. All new behavior is config-gated and default-off outside the Tau2 recipe.

## Service Topology

PipelineRL continues to launch and update the policy vLLM endpoints. NeMo Gym
provides external head, Tau2-agent, resource, frozen-user-model, and policy
proxy services. The policy proxy forwards requests to the existing PipelineRL
vLLM URLs.

The PipelineRL Tau2 rollout policy:

1. selects a PipelineRL `TrainableLLM` endpoint under the existing actor
   admission limit;
2. invokes the Tau2 agent's `/run` endpoint with request-scoped affinity to
   that policy endpoint;
3. receives the Tau2 scalar reward and every policy model-call record;
4. reconstructs one prefix-contiguous training sequence for the rollout; and
5. returns one `TrainingText` whose labels cover all assistant-generation
   spans and mask user, tool, and environment spans.

Gym services may be launched through PipelineRL environment jobs. That does
not imply terminal-domain proot, fleet, replay, or verifier machinery.

## Trajectory And TITO Contract

For policy calls `c_0 ... c_n`, the prompt IDs for `c_i` must begin with all
IDs already consumed from earlier calls. The adapter appends only the unseen
prompt suffix and the newly generated IDs. It labels generated assistant IDs
and masks the intervening prompt suffix.

Each call must carry:

- response-side prompt token IDs;
- response-side generated token IDs;
- generated-token behavior log probabilities;
- the exact policy model version active for that call; and
- prompt and completion token counts.

The adapter fails loudly when:

- response-side token IDs are absent or malformed;
- generated IDs disagree with IDs recovered from logprob records;
- usage lengths disagree with captured token lengths;
- a later prompt is not an exact prefix extension;
- an OOV token would require rewriting; or
- the merged sample exceeds the configured training sequence length.

No local retokenization may replace captured policy IDs. Cross-turn TITO is
defined over the exact history that Tau2 sent back to the policy. History must
not be rewritten after generation.

## Reward And Credit

The baseline reward is the untouched scalar reward returned by Tau2. There is
no no-submit penalty, event credit, command-error shaping, graded verifier
reward, or other PipelineRL-local shaping.

The rollout-level group advantage is applied to every labeled assistant token
in the merged trajectory. Turn selection modes such as `uniform_one` and
`submit_aware_one` are not used.

Mixed-version trajectories retain behavior logprobs and model version per
assistant span. Rollout metadata also records the minimum, maximum, and spread
of model versions. Conservative lag uses the oldest version.

## Selective Port Manifest

These terminal-branch commits are source references, not a blanket
cherry-pick list:

| Source | Ported behavior |
| --- | --- |
| `8592024` | Gemma-compatible vLLM/Transformers versions and required generic API adaptations only |
| `8bad773` | Per-engine vLLM compilation-cache isolation |
| `daa2807` | Strict response-side TITO assertions, adapted at the Gym policy-proxy boundary |
| `45d14e3` | Atomic group envelopes, staging, padding, and queue metrics only |
| `737e054` | Typed oversized shared-memory writes and reserved-slot recovery |
| `4f5806d` | Per-call model-version provenance, adapted through the Gym proxy |
| `c49ab11`, `8856fd5` | Generic rollout audit field and stream concepts |
| `3a1e160` | DPPO as an optional, default-off policy loss |

The runtime commit is ported selectively because it also contains
model-specific parser compatibility unrelated to Gemma. The packing commit is
ported selectively because its one-turn selectors solve a terminal-specific
history-rewrite problem that Tau2 must not inherit.

## Explicit Exclusions

Do not port:

- terminal-domain proot environments, fleet camouflage, replay, resurrection,
  memory eviction, clean verifier, or tamper scanning;
- terminal event credit, no-submit penalty, command-error shaping, graded
  pytest reward, or contamination handling;
- `uniform_one`, `submit_aware_one`, or other one-turn selectors;
- Qwen-specific weight remapping or parser configuration;
- `gspo_token` before Tau2 demonstrates a need for nonuniform token credit;
  or
- run overlays from the terminal experiment series.

## Implementation Waves

Each wave is reviewed before commit.

1. **Runtime:** Gemma-compatible dependency/API changes and per-engine compile
   cache isolation.
2. **Gym bridge:** service lifecycle, prepared Tau2 config, frozen user
   simulator, policy-proxy forwarding, and request-scoped endpoint affinity.
3. **All-turn TITO:** response-side capture, prefix reconstruction, masks,
   old logprobs, context checks, and whole-rollout drops.
4. **Provenance:** per-call versions, mixed-version rollout metadata,
   conservative lag accounting, and audit stream.
5. **Buffering:** atomic group envelopes, oversize handling, counters, and
   deterministic update composition.
6. **Loss option:** DPPO default-off plus a fixed-batch comparison against
   GSPO before selecting the first-run loss.
7. **Recipe:** Tau2/Gemma config and launcher only after every pre-run gate
   passes.

## Pre-Run Acceptance Gates

1. **User separation:** use the prepared Tau2 benchmark configuration with a
   frozen user simulator. Startup asserts that its model and endpoint differ
   from the policy.
2. **Source pins:** record exact NeMo Gym and Tau2 source SHAs. Moving branch
   references are forbidden.
3. **MoE transfer:** after one optimizer step, prove that all Gemma expert,
   router, embedding, and output-head tensors reach every policy vLLM.
4. **Policy parity:** on fixed prompts, trainer and vLLM token logprobs match
   within a registered tolerance before and after a weight update.
5. **Packing isolation:** cross-rollout sequence packing is off during
   bring-up. It may be enabled only after packed and unpacked logits, loss,
   gradients, attention boundaries, positions, labels, and reductions agree.
   This does not disable all-turn merging within a rollout.
6. **Endpoint affinity:** all calls in one Tau2 rollout use its
   actor-selected PipelineRL vLLM endpoint and remain within actor admission
   limits.
7. **Version transport:** each assistant span records the policy version that
   generated it; no rollout-level inferred substitute is accepted.
8. **Mixed-version loss:** a synthetic merged trajectory spanning multiple
   behavior versions must produce finite, expected GSPO and DPPO gradients and
   clipping statistics. Loss selection follows this test.
9. **Context fit:** measure real Tau2 prompt and merged-sample lengths,
   configure a conservative generation margin, and prove that no captured call
   or training sample is silently truncated. Whole-rollout context drops are
   counted.

## Required Audit Surface

Every rollout record must retain:

- rollout, group, task, dataset, and policy-endpoint identifiers;
- the untouched Tau2 scalar reward and verifier outcome;
- explicit submitted and terminated flags plus stop reason;
- model version for every assistant span and rollout-level minimum, maximum,
  and spread;
- prompt, response, labeled-token, and total merged-sequence lengths;
- per-call endpoint affinity and the policy endpoint chosen at admission;
- enough command/action and verifier detail to reproduce the outcome without
  silently truncating audit payloads; and
- whether the rollout entered training, with an exact drop reason otherwise.

Counters must be emitted for each whole-rollout drop cause, including:

- context-fit or sequence-fit rejection;
- TITO assertion failure, missing captured token IDs, or OOV token IDs;
- endpoint-affinity or version-transport violation;
- malformed Gym response or verifier failure; and
- queue admission, atomic-envelope rejection, or stale-data filtering.

TITO assertion failures and affinity violations are fatal for the affected
rollout and visible both as counters and audit records. A zero count is not
treated as proof until the corresponding adversarial acceptance test fires the
counter intentionally.

## Pinned Sources And Lifecycle

- NeMo Gym: 5f92a73217258074b74b7be26526c69f0ce3075d
- Tau2 runtime: befd120003fb55f48b498f6549556dcaf74582d5
- Tau2 prepared-data source: ce4013b0afe03c873488878b72851414f92f458b

The v1 service boundary is one externally supervised Gym cluster per
PipelineRL run. It contains one single-endpoint policy proxy and Tau2 agent for
each PipelineRL actor vLLM endpoint, plus one shared frozen user-model proxy.
PipelineRL maps its selected actor endpoint to the matching agent and verifies
the executed Gym config, source pins, service health, and user/policy
separation at launch and periodically while collecting rollouts.

The in-repo launcher owns the generated Gym config and exact source refs. A v2
follow-up moves this cluster under PipelineRL orchestrator lifecycle management;
external supervision is a deliberate v1 waypoint, not the target architecture.

## Explicit TBDs

These inputs must be resolved and recorded before recipe implementation:

- **Gemma topology:** verify the exact `gemma-4-26B-A4B` configuration,
  expert/router tensor names, active-parameter count, tokenizer/chat template,
  and the text-only loading surface that excludes unused vision components.
- **Weights:** confirm the exact Hugging Face model ID, revision, access terms,
  checkpoint availability, and immutable artifact identity. The provisional
  `google/gemma-4-26B-A4B-it` name in this plan is not a source pin.
- **Hardware recipe:** choose trainer GPU type/count, tensor/pipeline/expert
  parallelism, vLLM actor count and tensor parallelism, memory utilization,
  frozen user-simulator placement, rollout concurrency, and expected token
  budget per optimizer step.
- **Context budget:** measure real Tau2 call-prefix and complete merged-rollout
  token distributions before setting model length, generation margin, or
  maximum episode steps.

## First Experiment Boundary

The first Tau2/Gemma experiment starts only after all nine gates pass. It uses
the untouched Tau2 reward, no local shaping, no cross-rollout sequence packing,
and the loss selected by the mixed-version synthetic test. Its purpose is to
validate end-to-end learning and observability, not to combine estimator,
packing, reward, or curriculum ablations.
