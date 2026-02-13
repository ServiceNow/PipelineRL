#!/usr/bin/env python3
"""Helper script for running vLLM engine in a subprocess with proper CUDA isolation.

This script is run as a separate process with CUDA_VISIBLE_DEVICES set,
ensuring the engine only sees the intended GPU.
"""

import sys
import argparse
import asyncio


async def init_engine_and_process_group(
    model_name: str,
    init_method: str,
    actor_llm_idx: int,
    world_size: int,
):
    """Initialize vLLM engine and process group.

    create_engine() automatically calls init_actor_update_group() when
    disable_weight_updates=False, and calls destroy_actor_update_group()
    on context manager exit.
    """
    from pipelinerl.vllm1 import EngineManager
    import argparse as ap

    print("[vLLM Engine] Starting engine initialization")

    # Create args for engine with process group params
    args = ap.Namespace(
        model=model_name,
        tensor_parallel_size=1,
        disable_log_stats=True,
        enable_log_requests=False,
        disable_weight_updates=False,
        # Process group params - needed for automatic init_actor_update_group()
        actor_llm_idx=actor_llm_idx,
        weight_update_group_init_method=init_method,
        weight_update_group_world_size=world_size,
    )

    print(f"[vLLM Engine] Creating engine with model={model_name}")

    # create_engine automatically:
    # 1. Creates engine and manager
    # 2. Calls manager.init_actor_update_group() (rank 1)
    # 3. On exit, calls manager.destroy_actor_update_group()
    async with EngineManager.create_engine(args) as manager:
        print("[vLLM Engine] Engine and process group created successfully")

        # Keep engine alive until trainer completes its work
        print("[vLLM Engine] Process group active, waiting for trainer...")
        await asyncio.sleep(5)

    # Context manager exit automatically cleans up process group
    print("[vLLM Engine] Engine and process group cleaned up")


async def test_weight_update(
    model_name: str,
    init_method: str,
    actor_llm_idx: int,
    world_size: int,
    prompt: str,
    max_tokens: int,
    sync_dir: str,
    expect_different: bool = False,
):
    """Test weight update with generation before and after.

    This mode:
    1. Creates engine and initializes process group
    2. Generates baseline output
    3. Signals baseline_done, waits for broadcast_done
    4. Receives weight update
    5. Generates again with same prompt
    6. Prints both outputs for comparison
    """
    from pipelinerl.vllm1 import EngineManager
    from vllm import SamplingParams
    from pathlib import Path
    import argparse as ap
    # Import sync helper from same directory
    sys.path.insert(0, str(Path(__file__).parent))
    from sync_helper import SyncPoint

    print("[vLLM Engine] Starting weight update test")

    # Create sync points
    sync_path = Path(sync_dir)
    baseline_done = SyncPoint(sync_path, "baseline_done")
    ready_to_receive = SyncPoint(sync_path, "ready_to_receive")
    request_ready = SyncPoint(sync_path, "request_ready")
    receiving_started = SyncPoint(sync_path, "receiving_started")
    broadcast_done = SyncPoint(sync_path, "broadcast_done")

    # Create args for engine with process group params
    args = ap.Namespace(
        model=model_name,
        tensor_parallel_size=1,
        disable_log_stats=True,
        enable_log_requests=False,
        disable_weight_updates=False,
        actor_llm_idx=actor_llm_idx,
        weight_update_group_init_method=init_method,
        weight_update_group_world_size=world_size,
    )

    print(f"[vLLM Engine] Creating engine with model={model_name}")

    async with EngineManager.create_engine(args) as manager:
        print("[vLLM Engine] Engine and process group created successfully")

        # Step 1: Generate baseline
        sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
            seed=42,
        )

        print(f"[vLLM Engine] Generating baseline with prompt: '{prompt}'")
        async for output in manager.engine.generate(
            prompt,
            sampling_params=sampling_params,
            request_id="baseline",
        ):
            baseline_output = output

        baseline_text = baseline_output.outputs[0].text
        print(f"[vLLM Engine] Baseline output: '{baseline_text}'")

        # Step 2: Signal baseline done and ready to receive
        baseline_done.signal()
        ready_to_receive.signal()

        # Step 3: Wait for trainer to send WeightUpdateRequest
        print("[vLLM Engine] Waiting for trainer to send weight update request...")
        request_ready.wait(timeout=60)

        # Step 4: Read WeightUpdateRequest from trainer
        from sync_helper import read_weight_update_request
        request = read_weight_update_request(sync_path)
        print(f"[vLLM Engine] Received request with {len(request.parameters_info)} parameters")

        # Step 5: Signal we're about to start receiving, then call receive_weight_update
        receiving_started.signal()
        print("[vLLM Engine] Signaled receiving_started, calling receive_weight_update...")
        print("[vLLM Engine] (This will block until trainer broadcasts all weights)")
        await manager.receive_weight_update(request)
        print("[vLLM Engine] Weight update received!")

        # Step 6: Wait for trainer to signal broadcast complete
        broadcast_done.wait(timeout=60)
        print("[vLLM Engine] Trainer confirmed broadcast complete")

        # Step 7: Generate again with same prompt
        print(f"[vLLM Engine] Generating after update with prompt: '{prompt}'")
        async for output in manager.engine.generate(
            prompt,
            sampling_params=sampling_params,
            request_id="after_update",
        ):
            updated_output = output

        updated_text = updated_output.outputs[0].text
        print(f"[vLLM Engine] Updated output: '{updated_text}'")

        # Step 8: Compare outputs
        if expect_different:
            # Perturbed weights - expect different outputs
            if baseline_text != updated_text:
                print("[vLLM Engine] ✓ Outputs differ (as expected for perturbed weights)")
                print(f"[vLLM Engine]   Baseline: '{baseline_text}'")
                print(f"[vLLM Engine]   Updated:  '{updated_text}'")
            else:
                print("[vLLM Engine] ✗ Outputs are the same!")
                print(f"[vLLM Engine]   Both:     '{baseline_text}'")
                print("[vLLM Engine] ERROR: Perturbed weights should have changed the output")
                sys.exit(1)
        else:
            # Same weights - expect same outputs
            if baseline_text == updated_text:
                print("[vLLM Engine] ✓ Outputs match (as expected for same weights)")
            else:
                print("[vLLM Engine] ✗ Outputs differ!")
                print(f"[vLLM Engine]   Baseline: '{baseline_text}'")
                print(f"[vLLM Engine]   Updated:  '{updated_text}'")
                sys.exit(1)

    print("[vLLM Engine] Engine and process group cleaned up")


async def test_cross_validation(
    model_name: str,
    init_method: str,
    actor_llm_idx: int,
    world_size: int,
    prompt: str,
    max_tokens: int,
    sync_dir: str,
):
    """Cross-validation test for weight updates.

    Tests that broadcasting weights produces same results as loading from disk.
    Flow:
    1. Generate with original model → res_un_1
    2. Receive perturbed weights, generate → res_mod_1
    3. Recreate engine with perturbed model from disk, generate → res_mod_2
    4. Receive original weights, generate → res_un_2
    5. Verify: res_un_1 == res_un_2 and res_mod_1 == res_mod_2
    """
    from pipelinerl.vllm1 import EngineManager
    from vllm import SamplingParams
    from pathlib import Path
    import argparse as ap
    sys.path.insert(0, str(Path(__file__).parent))
    from sync_helper import SyncPoint, read_weight_update_request

    print("[vLLM Engine] Starting cross-validation test")

    # Create sync points
    sync_path = Path(sync_dir)
    baseline_done = SyncPoint(sync_path, "baseline_done")
    perturbed_model_saved = SyncPoint(sync_path, "perturbed_model_saved")
    ready_to_receive_perturbed = SyncPoint(sync_path, "ready_to_receive_perturbed")
    perturbed_broadcast_done = SyncPoint(sync_path, "perturbed_broadcast_done")
    mod1_done = SyncPoint(sync_path, "mod1_done")
    first_engine_destroyed = SyncPoint(sync_path, "first_engine_destroyed")
    engine_recreated = SyncPoint(sync_path, "engine_recreated")
    ready_to_receive_original = SyncPoint(sync_path, "ready_to_receive_original")
    original_broadcast_done = SyncPoint(sync_path, "original_broadcast_done")

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=42,
    )

    # Step 1: Generate with original model
    args = ap.Namespace(
        model=model_name,
        tensor_parallel_size=1,
        disable_log_stats=True,
        enable_log_requests=False,
        disable_weight_updates=False,
        actor_llm_idx=actor_llm_idx,
        weight_update_group_init_method=init_method,
        weight_update_group_world_size=world_size,
    )

    print(f"[vLLM Engine] Step 1: Creating engine with original model: {model_name}")
    async with EngineManager.create_engine(args) as manager:
        print(f"[vLLM Engine] Generating res_un_1 with prompt: '{prompt}'")
        async for output in manager.engine.generate(
            prompt,
            sampling_params=sampling_params,
            request_id="res_un_1",
        ):
            res_un_1_output = output
        res_un_1 = res_un_1_output.outputs[0].text
        print(f"[vLLM Engine] res_un_1: '{res_un_1}'")

        baseline_done.signal()

        # Wait for perturbed model to be saved
        print("[vLLM Engine] Waiting for trainer to save perturbed model...")
        perturbed_model_saved.wait(timeout=180)

        # Step 2: Receive perturbed weights and generate
        ready_to_receive_perturbed.signal()
        print("[vLLM Engine] Waiting for perturbed weight update request...")

        # Wait a moment for request file to be written
        import time
        time.sleep(0.5)

        request = read_weight_update_request(sync_path)
        print(f"[vLLM Engine] Received perturbed request with {len(request.parameters_info)} parameters")

        print("[vLLM Engine] Receiving perturbed weights...")
        await manager.receive_weight_update(request)

        perturbed_broadcast_done.wait(timeout=900)
        print("[vLLM Engine] Perturbed weights received")

        print(f"[vLLM Engine] Generating res_mod_1 with prompt: '{prompt}'")
        async for output in manager.engine.generate(
            prompt,
            sampling_params=sampling_params,
            request_id="res_mod_1",
        ):
            res_mod_1_output = output
        res_mod_1 = res_mod_1_output.outputs[0].text
        print(f"[vLLM Engine] res_mod_1: '{res_mod_1}'")

        mod1_done.signal()

    # Engine destroyed here (context manager exit)
    print("[vLLM Engine] First engine destroyed")
    first_engine_destroyed.signal()

    # Step 3: Recreate engine with perturbed model from disk
    perturbed_model_path = (sync_path / "perturbed_model_path.txt").read_text().strip()
    print(f"[vLLM Engine] Step 3: Recreating engine with perturbed model from: {perturbed_model_path}")

    args_perturbed = ap.Namespace(
        model=perturbed_model_path,
        tensor_parallel_size=1,
        disable_log_stats=True,
        enable_log_requests=False,
        disable_weight_updates=False,
        actor_llm_idx=actor_llm_idx,
        weight_update_group_init_method=init_method,
        weight_update_group_world_size=world_size,
    )

    async with EngineManager.create_engine(args_perturbed) as manager:
        # Signal immediately after engine is created
        engine_recreated.signal()
        print("[vLLM Engine] Engine recreated, signaled to trainer")

        print(f"[vLLM Engine] Generating res_mod_2 with prompt: '{prompt}'")
        async for output in manager.engine.generate(
            prompt,
            sampling_params=sampling_params,
            request_id="res_mod_2",
        ):
            res_mod_2_output = output
        res_mod_2 = res_mod_2_output.outputs[0].text
        print(f"[vLLM Engine] res_mod_2: '{res_mod_2}'")

        # Step 4: Receive original weights and generate
        ready_to_receive_original.signal()
        print("[vLLM Engine] Waiting for original weight update request...")

        time.sleep(0.5)
        request = read_weight_update_request(sync_path)
        print(f"[vLLM Engine] Received original request with {len(request.parameters_info)} parameters")

        print("[vLLM Engine] Receiving original weights...")
        await manager.receive_weight_update(request)

        original_broadcast_done.wait(timeout=900)
        print("[vLLM Engine] Original weights received")

        print(f"[vLLM Engine] Generating res_un_2 with prompt: '{prompt}'")
        async for output in manager.engine.generate(
            prompt,
            sampling_params=sampling_params,
            request_id="res_un_2",
        ):
            res_un_2_output = output
        res_un_2 = res_un_2_output.outputs[0].text
        print(f"[vLLM Engine] res_un_2: '{res_un_2}'")

    # Step 5: Verify
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"res_un_1:  '{res_un_1}'")
    print(f"res_un_2:  '{res_un_2}'")
    print(f"res_mod_1: '{res_mod_1}'")
    print(f"res_mod_2: '{res_mod_2}'")
    print("="*60)

    # Check assertions
    success = True
    if res_un_1 == res_un_2:
        print("✓ res_un_1 == res_un_2 (original weights produce same output)")
    else:
        print("✗ res_un_1 != res_un_2 (FAILED)")
        success = False

    if res_mod_1 == res_mod_2:
        print("✓ res_mod_1 == res_mod_2 (broadcast = load from disk)")
    else:
        print("✗ res_mod_1 != res_mod_2 (FAILED)")
        success = False

    if not success:
        sys.exit(1)

    print("\n✓ Cross-validation test PASSED")


async def test_back_and_forth(
    model_name: str,
    init_method: str,
    actor_llm_idx: int,
    world_size: int,
    prompt: str,
    max_tokens: int,
    sync_dir: str,
):
    """Back-and-forth test: switch between original and perturbed weights.

    Flow:
    1. Generate with original → res_or_1
    2. Receive perturbed, generate → res_mod_1
    3. Receive original, generate → res_or_2
    4. Receive perturbed again, generate → res_mod_2
    5. Verify: res_or_1 == res_or_2 and res_mod_1 == res_mod_2
    """
    from pipelinerl.vllm1 import EngineManager
    from vllm import SamplingParams
    from pathlib import Path
    import argparse as ap
    sys.path.insert(0, str(Path(__file__).parent))
    from sync_helper import SyncPoint, read_weight_update_request

    print("[vLLM Engine] Starting back-and-forth test")

    # Create sync points
    sync_path = Path(sync_dir)
    baseline_done = SyncPoint(sync_path, "baseline_done")
    ready_for_perturbed1 = SyncPoint(sync_path, "ready_for_perturbed1")
    perturbed1_done = SyncPoint(sync_path, "perturbed1_done")
    ready_for_original = SyncPoint(sync_path, "ready_for_original")
    original_done = SyncPoint(sync_path, "original_done")
    ready_for_perturbed2 = SyncPoint(sync_path, "ready_for_perturbed2")
    perturbed2_done = SyncPoint(sync_path, "perturbed2_done")

    sampling_params = SamplingParams(
        temperature=0.0,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=42,
    )

    # Create engine args
    args = ap.Namespace(
        model=model_name,
        tensor_parallel_size=1,
        disable_log_stats=True,
        enable_log_requests=False,
        disable_weight_updates=False,
        actor_llm_idx=actor_llm_idx,
        weight_update_group_init_method=init_method,
        weight_update_group_world_size=world_size,
    )

    print(f"[vLLM Engine] Creating engine with model: {model_name}")
    async with EngineManager.create_engine(args) as manager:
        # Step 1: Generate with original weights
        print(f"[vLLM Engine] Step 1: Generating res_or_1")
        async for output in manager.engine.generate(
            prompt, sampling_params=sampling_params, request_id="res_or_1"
        ):
            res_or_1 = output.outputs[0].text
        print(f"[vLLM Engine] res_or_1: '{res_or_1}'")
        baseline_done.signal()

        # Step 2: Receive perturbed weights, generate
        ready_for_perturbed1.signal()
        import time
        time.sleep(0.5)
        request = read_weight_update_request(sync_path)
        print(f"[vLLM Engine] Step 2: Receiving perturbed weights (1st time)")
        await manager.receive_weight_update(request)
        perturbed1_done.wait(timeout=900)

        print(f"[vLLM Engine] Generating res_mod_1")
        async for output in manager.engine.generate(
            prompt, sampling_params=sampling_params, request_id="res_mod_1"
        ):
            res_mod_1 = output.outputs[0].text
        print(f"[vLLM Engine] res_mod_1: '{res_mod_1}'")

        # Step 3: Receive original weights, generate
        ready_for_original.signal()
        time.sleep(0.5)
        request = read_weight_update_request(sync_path)
        print(f"[vLLM Engine] Step 3: Receiving original weights")
        await manager.receive_weight_update(request)
        original_done.wait(timeout=900)

        print(f"[vLLM Engine] Generating res_or_2")
        async for output in manager.engine.generate(
            prompt, sampling_params=sampling_params, request_id="res_or_2"
        ):
            res_or_2 = output.outputs[0].text
        print(f"[vLLM Engine] res_or_2: '{res_or_2}'")

        # Step 4: Receive perturbed weights again, generate
        ready_for_perturbed2.signal()
        time.sleep(0.5)
        request = read_weight_update_request(sync_path)
        print(f"[vLLM Engine] Step 4: Receiving perturbed weights (2nd time)")
        await manager.receive_weight_update(request)
        perturbed2_done.wait(timeout=900)

        print(f"[vLLM Engine] Generating res_mod_2")
        async for output in manager.engine.generate(
            prompt, sampling_params=sampling_params, request_id="res_mod_2"
        ):
            res_mod_2 = output.outputs[0].text
        print(f"[vLLM Engine] res_mod_2: '{res_mod_2}'")

    # Step 5: Save results for server test
    import json
    results_file = sync_path / "expected_results.json"
    expected_results = {
        "res_or_1": res_or_1,
        "res_mod_1": res_mod_1,
        "res_or_2": res_or_2,
        "res_mod_2": res_mod_2,
    }
    with open(results_file, "w") as f:
        json.dump(expected_results, f, indent=2)
    print(f"[vLLM Engine] Saved expected results to {results_file}")

    # Step 6: Verify
    print("\n" + "="*60)
    print("BACK-AND-FORTH TEST RESULTS")
    print("="*60)
    print(f"res_or_1:  '{res_or_1}'")
    print(f"res_or_2:  '{res_or_2}'")
    print(f"res_mod_1: '{res_mod_1}'")
    print(f"res_mod_2: '{res_mod_2}'")
    print("="*60)

    # Check assertions
    success = True
    if res_or_1 == res_or_2:
        print("✓ res_or_1 == res_or_2 (can switch back to original)")
    else:
        print("✗ res_or_1 != res_or_2 (FAILED)")
        success = False

    if res_mod_1 == res_mod_2:
        print("✓ res_mod_1 == res_mod_2 (perturbed weights consistent)")
    else:
        print("✗ res_mod_1 != res_mod_2 (FAILED)")
        success = False

    if not success:
        sys.exit(1)

    print("\n✓ Back-and-forth test PASSED")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM engine helper")
    parser.add_argument("command", choices=["init", "weight_update", "cross_validation", "back_and_forth"])
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--init-method", required=True)
    parser.add_argument("--actor-llm-idx", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=2)
    # For weight_update command
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--sync-dir", type=str, help="Directory for sync files")
    parser.add_argument("--expect-different", action="store_true", help="Expect outputs to be different (for perturbed weights)")

    args = parser.parse_args()

    try:
        if args.command == "init":
            asyncio.run(init_engine_and_process_group(
                args.model_name,
                args.init_method,
                args.actor_llm_idx,
                args.world_size,
            ))
        elif args.command == "weight_update":
            if not args.sync_dir:
                print("Error: --sync-dir required for weight_update command")
                sys.exit(1)
            asyncio.run(test_weight_update(
                args.model_name,
                args.init_method,
                args.actor_llm_idx,
                args.world_size,
                args.prompt,
                args.max_tokens,
                args.sync_dir,
                args.expect_different,
            ))
        elif args.command == "cross_validation":
            if not args.sync_dir:
                print("Error: --sync-dir required for cross_validation command")
                sys.exit(1)
            asyncio.run(test_cross_validation(
                args.model_name,
                args.init_method,
                args.actor_llm_idx,
                args.world_size,
                args.prompt,
                args.max_tokens,
                args.sync_dir,
            ))
        elif args.command == "back_and_forth":
            if not args.sync_dir:
                print("Error: --sync-dir required for back_and_forth command")
                sys.exit(1)
            asyncio.run(test_back_and_forth(
                args.model_name,
                args.init_method,
                args.actor_llm_idx,
                args.world_size,
                args.prompt,
                args.max_tokens,
                args.sync_dir,
            ))
    except Exception as e:
        print(f"[vLLM Engine] Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
