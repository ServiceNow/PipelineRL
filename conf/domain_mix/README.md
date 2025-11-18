# Domain mix presets

Hydra group `domain_mix` stores reusable presets for `actor.domain_mix`.

Usage examples:

```
python main.py --config-name multi_domain/base +domain_mix=math_coding_70_30
python main.py --config-name multi_domain/base +domain_mix=balanced
```

Override or extend these presets by creating new files under `conf/domain_mix/`.
