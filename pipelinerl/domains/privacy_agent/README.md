# privacy_agent

`privacy_agent` trains the vendored DRBench concise-QA agent on a small seed set
of human-accepted multi-hop chains. By default it expects a separate helper service
to handle retrieval-side computation.

## Required Helper Service

Start a helper service that exposes:

- `GET /health`
- `POST /embed`
- `POST /local/search`
- `POST /browsecomp/search`

Then point the domain at that service with:

```bash
export PRIVACY_AGENT_HELPER_URL=http://<helper-host>:8012
```

The domain defaults are:

- `use_remote_embeddings: true`
- `use_remote_local_search: true`
- `use_remote_browsecomp: true`

If your helper supports task-specific curated web sidecars, `privacy_agent`
will pass `task_id` through remote BrowseComp search requests automatically.

## Build private-doc indices

Set the dataset paths in config or with environment variables such as:

- `PRIVACY_AGENT_ANNOTATIONS_PATH`
- `PRIVACY_AGENT_CURATED_PATH`
- `PRIVACY_AGENT_TASK_DATA_ROOT`
- `PRIVACY_AGENT_LOCAL_INDEX_ROOT`
- `PRIVACY_AGENT_BROWSECOMP_INDEX_GLOB`

Then build the task-local private indices once before training:

```bash
python -m pipelinerl.domains.privacy_agent.build_local_indices \
  --dataset-name seed20 \
  --index-root .privacy_agent/local_indices \
  --helper-service-url "$PRIVACY_AGENT_HELPER_URL"
```

## Run

Use the normal Hydra entrypoint:

```bash
python -m pipelinerl.launch \
  --config-name privacy_agent \
  --config-dir conf \
  privacy_agent.helper_service_url="$PRIVACY_AGENT_HELPER_URL"
```
