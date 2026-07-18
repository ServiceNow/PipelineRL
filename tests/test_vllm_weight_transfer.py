import asyncio
import math
from types import SimpleNamespace

import pytest
import torch

from pipelinerl.prerun_evidence import (
    GEMMA_POLICY_IDENTITY,
    read_evidence_records,
)
from pipelinerl.finetune_loop import (
    ParameterInfo,
    WeightUpdateManager as TrainerWeightUpdateManager,
    WeightUpdateRequest,
    completion_logprobs_from_logits,
    parse_vllm_completion_logprobs,
)
from pipelinerl.vllm1 import (
    WeightUpdateManager as VllmWeightUpdateManager,
    WorkerExtension,
)


class _BroadcastGroup:
    def __init__(self):
        self.index = 0

    def broadcast(self, buffer, *, src, stream):
        assert src == 0
        buffer.copy_(
            torch.tensor(
                [self.index + 1.0, self.index + 2.0],
                dtype=buffer.dtype,
            )
        )
        self.index += 1


class _Model:
    def named_parameters(self):
        return [
            (
                "model.embed_tokens.weight",
                torch.nn.Parameter(torch.ones(2)),
            )
        ]

    def load_weights(self, weights):
        name, _ = weights[0]
        if ".experts." in name:
            return {
                "model.layers.0.moe.experts.0.w13_weight",
                "model.layers.0.moe.experts.1.w13_weight",
            }
        if "k_proj" in name:
            return {
                "model.layers.0.self_attn.k_proj.weight",
                "model.layers.0.self_attn.v_proj.weight",
            }
        return {name}


def _worker(model=None):
    return SimpleNamespace(
        rank=1,
        local_rank=1,
        device=torch.device("cpu"),
        model_runner=SimpleNamespace(
            model=model or _Model()
        ),
        model_update_group=_BroadcastGroup(),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                tie_word_embeddings=True,
            )
        ),
    )


def _request(*, collect_evidence):
    return WeightUpdateRequest(
        version=192,
        collect_evidence=collect_evidence,
        parameters_info=[
            ParameterInfo(
                name="model.embed_tokens.weight",
                shape=[2],
                dtype="torch.float32",
            ),
            ParameterInfo(
                name=(
                    "model.layers.0.experts."
                    "gate_up_proj"
                ),
                shape=[2],
                dtype="torch.float32",
            ),
            ParameterInfo(
                name=(
                    "model.layers.0.router."
                    "proj.weight"
                ),
                shape=[2],
                dtype="torch.float32",
            ),
            ParameterInfo(
                name=(
                    "model.layers.0.self_attn."
                    "k_proj.weight"
                ),
                shape=[2],
                dtype="torch.float32",
            ),
        ],
    )


def test_worker_receipts_cover_packed_experts_kv_and_tied_head(
    monkeypatch,
):
    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: None,
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: None,
    )
    monkeypatch.setattr(
        "pipelinerl.vllm1.pipelinerl.vllm_quantization."
        "invalidate_fp32_cache",
        lambda: None,
    )

    result = WorkerExtension.receive_weight_update(
        _worker(),
        _request(collect_evidence=True).model_dump_json(),
    )

    assert result["rank"] == 1
    receipts = {
        receipt["source_name"]: receipt
        for receipt in result["receipts"]
    }
    assert set(receipts) == {
        info.name
        for info in _request(
            collect_evidence=True
        ).parameters_info
    }
    assert receipts[
        "model.embed_tokens.weight"
    ]["categories"] == [
        "embedding",
        "output_head",
    ]
    expert = receipts[
        "model.layers.0.experts.gate_up_proj"
    ]
    assert expert["categories"] == ["expert"]
    assert len(expert["loaded_names"]) == 2
    assert receipts[
        "model.layers.0.router.proj.weight"
    ]["categories"] == ["router"]
    assert receipts[
        "model.layers.0.self_attn.k_proj.weight"
    ]["loaded_names"] == [
        "model.layers.0.self_attn.k_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
    ]
    for receipt in receipts.values():
        fingerprint = receipt["received_fingerprint"]
        assert fingerprint["numel"] == 2
        assert fingerprint["finite_count"] == 2


def test_default_off_accepts_multi_name_load_without_fingerprinting(
    monkeypatch,
):
    class _NoInspectionModel(_Model):
        def named_parameters(self):
            raise AssertionError(
                "default-off path inspected the model"
            )

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: None,
    )
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda: None,
    )
    monkeypatch.setattr(
        "pipelinerl.vllm1.tensor_fingerprint",
        lambda tensor: (_ for _ in ()).throw(
            AssertionError(
                "default-off path fingerprinted weights"
            )
        ),
    )
    monkeypatch.setattr(
        "pipelinerl.vllm1.pipelinerl.vllm_quantization."
        "invalidate_fp32_cache",
        lambda: None,
    )

    result = WorkerExtension.receive_weight_update(
        _worker(_NoInspectionModel()),
        _request(collect_evidence=False).model_dump_json(),
    )

    assert result is None


def test_evidence_rejects_vision_model(monkeypatch):
    class _VisionModel(_Model):
        def named_parameters(self):
            return [
                (
                    "model.vision_tower.encoder.weight",
                    torch.nn.Parameter(torch.ones(2)),
                )
            ]

    monkeypatch.setattr(
        torch.cuda,
        "synchronize",
        lambda device: None,
    )

    with pytest.raises(ValueError, match="vision parameters"):
        WorkerExtension.receive_weight_update(
            _worker(_VisionModel()),
            _request(collect_evidence=True).model_dump_json(),
        )


def test_completion_logprob_helpers_use_exact_fixed_tokens():
    logits = torch.zeros((1, 4, 6))
    logits[0, 1, 4] = 2.0
    logits[0, 2, 5] = 3.0

    trainer = completion_logprobs_from_logits(
        logits,
        prompt_length=2,
        completion_token_ids=[4, 5],
    )
    expected = [
        torch.log_softmax(logits[0, 1], dim=-1)[4].item(),
        torch.log_softmax(logits[0, 2], dim=-1)[5].item(),
    ]
    assert trainer == pytest.approx(expected)

    payload = {
        "choices": [
            {
                "prompt_logprobs": [
                    None,
                    None,
                    {"4": {"logprob": trainer[0]}},
                    {"5": {"logprob": trainer[1]}},
                ]
            }
        ]
    }
    assert parse_vllm_completion_logprobs(
        payload,
        [4, 5],
    ) == pytest.approx(trainer)


def test_trainer_records_fixed_token_parity_before_and_after(
    tmp_path,
    monkeypatch,
):
    class _ParityModel:
        def __init__(self):
            self.training = True

        def eval(self):
            self.training = False

        def train(self):
            self.training = True

        def __call__(self, *, input_ids):
            assert input_ids.tolist() == [[1, 2, 3, 4]]
            return SimpleNamespace(
                logits=torch.zeros((1, 4, 8))
            )

    expected_logprob = -math.log(8)
    posts = []

    class _Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "choices": [
                    {
                        "prompt_logprobs": [
                            None,
                            None,
                            {"3": {"logprob": expected_logprob}},
                            {"4": {"logprob": expected_logprob}},
                        ]
                    }
                ]
            }

    def post(url, *, json):
        posts.append((url, json))
        return _Response()

    accelerator = SimpleNamespace(
        device=torch.device("cpu"),
        is_main_process=True,
    )
    monkeypatch.setattr(
        "pipelinerl.finetune_loop.get_accelerator",
        lambda: accelerator,
    )
    monkeypatch.setattr(
        "pipelinerl.finetune_loop.requests.post",
        post,
    )
    cfg = SimpleNamespace(
        output_dir=str(tmp_path),
        tau2_prerun=SimpleNamespace(
            enabled=True,
            evidence_dir=str(tmp_path),
            fixed_prompt_token_ids=[1, 2],
            fixed_completion_token_ids=[3, 4],
            policy_model=GEMMA_POLICY_IDENTITY,
        ),
    )
    manager = TrainerWeightUpdateManager(
        ["http://actor-0:8000"],
        _ParityModel(),
        None,
        None,
        cfg,
    )
    try:
        manager._record_parity(phase="initial", version=0)
        manager._record_parity(
            phase="after_optimizer",
            version=192,
        )
    finally:
        manager.shutdown()

    records = read_evidence_records(tmp_path, "parity")
    assert [record["phase"] for record in records] == [
        "before_update",
        "after_update",
    ]
    assert [record["version"] for record in records] == [
        0,
        192,
    ]
    assert all(record["passed"] for record in records)
    assert [payload["prompt"] for _, payload in posts] == [
        [1, 2, 3, 4],
        [1, 2, 3, 4],
    ]
    assert all(payload["max_tokens"] == 0 for _, payload in posts)
    assert all(payload["add_special_tokens"] is False for _, payload in posts)
    assert all(payload["model"] == GEMMA_POLICY_IDENTITY for _, payload in posts)


def test_manager_returns_actual_collective_receipts_and_advances_version():
    class _Engine:
        def __init__(self):
            self.events = []

        async def pause_generation(self, **kwargs):
            self.events.append(("pause", kwargs))

        async def resume_generation(self):
            self.events.append(("resume", None))

    workers = [
        {"rank": 0, "receipts": []},
        {"rank": 1, "receipts": []},
    ]

    class _Client:
        async def collective_rpc_async(
            self,
            method,
            args=(),
        ):
            assert method == "receive_weight_update"
            assert args
            return workers

    args = SimpleNamespace()
    engine = _Engine()
    manager = VllmWeightUpdateManager(
        args,
        engine,
        _Client(),
    )
    request = WeightUpdateRequest(
        version=192,
        parameters_info=[],
    )

    result = asyncio.run(
        manager.receive_weight_update(request)
    )

    assert result == workers
    assert manager.served_version == 192
    assert [event[0] for event in engine.events] == [
        "pause",
        "resume",
    ]
