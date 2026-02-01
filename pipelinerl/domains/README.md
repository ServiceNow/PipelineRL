# Domains

This directory contains domain-specific implementations for multi-domain RL training. Each domain provides:
- Dataset loader
- Rollout generation
- Verification/reward computation

## Available Domains

| Domain | Description | Dataset | Verifier |
|--------|-------------|---------|----------|
| `math` | Mathematical reasoning | OpenReasonerZero, AIME, etc. | math-verify |
| `coding` | Code generation | TACO + APPS | SandboxFusion |
| `livecodebench` | Code generation (eval only) | LiveCodeBench v5 | SandboxFusion |
| `logic` | Logic puzzles | PrimeIntellect INTELLECT-3-RL (logic) | i3-logic |
| `fn_calling` | Function calling | BFCL v3 | bfcl-eval |
| `ifeval` | Instruction following (train) | AllenAI IF_multi_constraints_upto5 | instruction_following_eval |
| `google_ifeval` | Instruction following (eval) | Google IFEval (541 samples) | instruction_following_eval |

## Installation

### Quick Install (all domains)

```bash
pip install pipelinerl[domains]
```

### Per-Domain Installation

```bash
# Coding domain (SandboxFusion SDK)
pip install pipelinerl[coding]

# Function calling domain (BFCL verifier)
pip install pipelinerl[fn_calling]

# Logic domain (requires two steps)
pip install pipelinerl[logic]
pip install prime
prime env install primeintellect/i3-logic

# IFEval domain (instruction following)
pip install pipelinerl[ifeval]
```

### Logic Domain Setup

The logic domain uses PrimeIntellect's `i3-logic` verifier which is not on PyPI. It must be installed via the Prime CLI:

```bash
# Install the Prime CLI
pip install prime

# Authenticate (get API key from https://app.primeintellect.ai)
prime login
# or
export PRIME_API_KEY="your-api-key"

# Install i3-logic environment
prime env install primeintellect/i3-logic
```

### Coding Domain Setup

The coding domain uses [SandboxFusion](https://github.com/bytedance/SandboxFusion) for sandboxed Python execution. SandboxFusion is a Docker-based code execution service that must be deployed separately.

```bash
# Install SandboxFusion Python SDK
pip install sandbox-fusion

# Deploy SandboxFusion server (Docker required)
# See https://bytedance.github.io/SandboxFusion for deployment options
docker run -d -p 8080:8080 bytedance/sandbox-fusion:latest
```

Configuration in `conf/coding.yaml`:
```yaml
actor:
  sandbox_endpoint: http://127.0.0.1:8080
  sandbox_timeout: 10.0
  max_tests_per_problem: 5
```

## Usage in Config

Specify datasets using the format `<domain>::<dataset>[@subset]`:

```yaml
train_dataset_names:
  - math::open_reasoner_zero_57k
  - coding::coding@all
  - logic::logic@train
  - fn_calling::fn_calling@train
  - ifeval::ifeval

test_dataset_names:
  - math::aime_2025
  - livecodebench::livecodebench_v5
  - logic::logic@test
  - fn_calling::fn_calling@test
  - google_ifeval::google_ifeval
```

## Domain Details

### Math
- **Datasets**: OpenReasonerZero (57k, 72k extended), AIME 2024/2025, AMC, MATH500
- **Verifier**: `math-verify` package (included in base dependencies)
- **Reward**: Binary (correct/incorrect)

### Coding
- **Training data**: TACO + APPS datasets (combined, filtered by difficulty)
- **Eval data**: LiveCodeBench v5 (880 problems, Aug 2024 - Jan 2025)
- **Verifier**: Sandboxed Python execution via SandboxFusion (concurrent)
- **Reward**: Fraction of test cases passed

### Logic
- **Dataset**: PrimeIntellect INTELLECT-3-RL logic domain (11.6k samples, 87 task types)
- **Verifier**: `i3-logic` package (PrimeIntellect)
- **Reward**: Binary (correct/incorrect)
- **Note**: Some task types (arc_agi, buggy_tables) are skipped by default

### Function Calling
- **Dataset**: Berkeley Function Calling Leaderboard (BFCL) v3
- **Categories**: simple, multiple, parallel, parallel_multiple
- **Verifier**: `bfcl-eval` package
- **Reward**: Binary per function call, averaged

### IFEval (Instruction Following)
- **Training data**: AllenAI IF_multi_constraints_upto5 (95k samples, up to 5 constraints each)
- **Eval data**: Google IFEval benchmark (541 held-out samples)
- **Verifier**: `instruction_following_eval` package
- **Reward**: Supports partial credit (fraction of constraints satisfied)
- **Constraint types**: 25+ types including word count, formatting, keyword inclusion, etc.

## Reward Configuration

All domains use a shared `RewardTable` from `pipelinerl.domains.math.rollouts`. This ensures consistent reward ranges across domains.

### RewardTable Fields

| Field | Description |
|-------|-------------|
| `correct_answer_finished` | Reward for correct answer with EOS token |
| `correct_answer_not_finished` | Reward for correct answer without EOS (truncated) |
| `wrong_answer_finished` | Reward for wrong answer with EOS token |
| `wrong_answer_not_finished` | Reward for wrong answer without EOS |
| `no_answer_finished` | Reward when no answer could be parsed |
| `no_answer_not_finished` | Reward when no answer parsed and truncated |
| `unparsable_finished` | Reward for unparsable/error responses |
| `unparsable_not_finished` | Reward for unparsable and truncated |
| `buffer_tokens` | Tokens before max_tokens to start length penalty (0=disabled) |

### Available Reward Configs

| Config | Range | Description |
|--------|-------|-------------|
| `pure_success` | [0, 1] | Binary: correct=1.0, wrong=0.0 |
| `success_and_format` | [0, 1] | Correct + finished = 1.0, else = 0.0 |
| `success` | [0, 1] | Correct = 1.0 (0.5 if truncated), wrong = 0.0 |

### Domain-Specific Notes

- **IFEval**: Supports partial credit via `actor.ifeval_partial_credit: true` (default). Interpolates between wrong and correct rewards based on fraction of constraints satisfied.
- **Logic/Coding**: Map "error" verification status to "unparsable" rewards.

## Adding a New Domain

1. Create a new directory under `pipelinerl/domains/<domain_name>/`
2. Implement:
   - `dataset.py` - Dataset loader with `load_problems()` function
   - `rollouts.py` - Rollout generation with `generate_<domain>_rollout()` async function
   - `verifier_api.py` - Verification logic (local or RPC-based)
   - `__init__.py` - Export the rollout function
3. Register in `pipelinerl/domains/multidomain/loader.py`
4. Add rollout mapping in `conf/domain_rollouts/base.yaml`
5. Add domain config in `conf/multi_domain/base.yaml`
6. **Import `RewardTable` from `pipelinerl.domains.math.rollouts`** to use shared reward config
