# Math Visual Reasoning (mathv)

A Vision Language Model (VLM) RL example for math reasoning over images.
Trains on [Geometry3K](https://huggingface.co/datasets/hiyouga/geometry3k)
and evaluates on [MathVista](https://huggingface.co/datasets/AI4Math/MathVista)
(`testmini` split).

## Usage

```bash
python -m pipelinerl.launch output_dir=results/mathv --config-name mathv
```
