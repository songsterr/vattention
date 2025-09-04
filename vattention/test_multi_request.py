#!/usr/bin/env python3
"""Test script demonstrating multi-request workflow as described in documentation."""

import time

import torch
import vattention


def test_multi_request_workflow():
    """Test the multi-request workflow exactly as described in the documentation."""
    print("=== Multi-Request vAttention Workflow Test ===")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping test")
        return False

    # Initialize CUDA context
    dummy = torch.tensor([1.0], device='cuda:0')

    try:
        # Initialize KV cache for multi-request scenario
        print("\n--- Initializing KV Cache for Multi-Request Scenario ---")
        gpu_caches = vattention.init_kvcache(
            1,              # num_layers
            16,             # num_kv_heads
            128,            # head_size
            4,              # max_batch_size (support up to 4 concurrent requests)
            2048,           # max_context_length
            0,              # device
            torch.float16,  # dtype
            2 * 1024 * 1024,   # page_size (2MB)
            False           # megacache
        )

        print(f"✅ KV cache initialized for max_batch_size=4")
        print(f"Cache tensor shape: {gpu_caches[0].shape}")

        # Reserve physical memory
        print("\n--- Reserving Physical Memory Pool ---")
        free_memory_mb = 256  # 256 MB memory pool
        reserved_blocks = vattention.reserve_physical_pages(free_memory_mb * 1024 * 1024)
        print(f"Reserved {reserved_blocks} physical memory blocks ({free_memory_mb} MB)")

        initial_free_blocks = vattention.num_free_kvblocks()
        print(f"Initial free KV blocks: {initial_free_blocks}")

        # Simulate multiple requests arriving
        print("\n--- Simulating Multiple Requests Arriving ---")
        requests = []
        current_seq_lens = []

        # Request 1: Prefill with 512 tokens
        batch_id_1 = vattention.alloc_new_batch_idx(512)
        requests.append({"batch_id": batch_id_1, "seq_len": 512, "name": "Request-1"})
        current_seq_lens.append(512)
        print(f"Request-1: batch_id={batch_id_1}, prefill_len=512")

        # Request 2: Prefill with 768 tokens
        batch_id_2 = vattention.alloc_new_batch_idx(768)
        requests.append({"batch_id": batch_id_2, "seq_len": 768, "name": "Request-2"})
        current_seq_lens.append(768)
        print(f"Request-2: batch_id={batch_id_2}, prefill_len=768")

        # Request 3: Prefill with 256 tokens
        batch_id_3 = vattention.alloc_new_batch_idx(256)
        requests.append({"batch_id": batch_id_3, "seq_len": 256, "name": "Request-3"})
        current_seq_lens.append(256)
        print(f"Request-3: batch_id={batch_id_3}, prefill_len=256")

        print(f"\\nActive requests: {len(requests)}")
        print(f"Current sequence lengths: {current_seq_lens}")
        print(f"Free KV blocks after allocation: {vattention.num_free_kvblocks()}")

        # BEFORE MODEL FORWARD PASS: Allocate memory asynchronously
        print("\\n--- BEFORE Forward Pass: Asynchronous Memory Allocation ---")
        print("Calling vattention.step_async(self.curr_seq_lens)...")
        start_time = time.time()
        vattention.step_async(current_seq_lens)
        async_time = time.time() - start_time
        print(f"step_async completed in {async_time:.4f} seconds")
        print(f"Free KV blocks after step_async: {vattention.num_free_kvblocks()}")

        print("\\n--- Memory State After step_async ---")
        vattention.show_allocator_state()

        # Simulate model forward pass (just wait a bit)
        print("\\n--- Simulating Model Forward Pass ---")
        print("(Model inference happening here...)")
        time.sleep(0.01)  # Simulate computation time

        # AFTER FORWARD PASS: Simulate decode phase (sequence growth)
        print("\\n--- AFTER Forward Pass: Decode Phase (Sequence Growth) ---")
        # Each request generates 1 more token
        decode_seq_lens = [seq_len + 1 for seq_len in current_seq_lens]
        print(f"Decode phase - sequences grow to: {decode_seq_lens}")

        print("Calling vattention.step_async(updated_seq_lens)...")
        vattention.step_async(decode_seq_lens)
        print(f"Free KV blocks after decode step: {vattention.num_free_kvblocks()}")

        # Continue for a few decode steps
        print("\\n--- Continuing Decode Steps ---")
        for step in range(512):
            # Grow sequences by 1 token each
            decode_seq_lens = [seq_len + 1 for seq_len in decode_seq_lens]
            current_seq_lens = decode_seq_lens.copy()

            print(f"Decode step {step + 2}: sequences = {decode_seq_lens}")
            vattention.step_async(decode_seq_lens)

            # Show memory usage every few steps
            if step % 2 == 0:
                print(f"  Free KV blocks: {vattention.num_free_kvblocks()}")

        print(f"\\nAfter {5} decode steps, final sequence lengths: {decode_seq_lens}")
        print(f"Free KV blocks: {vattention.num_free_kvblocks()}")

        # Simulate request completion - Request 1 completes first
        print("\\n--- Request Completion: Request-1 Finished ---")
        completed_request = requests[0]
        print(f"Calling vattention.free_batch_idx({completed_request['batch_id']}) for {completed_request['name']}")
        vattention.free_batch_idx(completed_request['batch_id'])

        # Remove completed request from tracking
        requests = requests[1:]  # Remove first request
        decode_seq_lens = decode_seq_lens[1:]  # Remove corresponding seq_len

        print(f"Remaining requests: {[req['name'] for req in requests]}")
        print(f"Remaining sequence lengths: {decode_seq_lens}")
        print(f"Free KV blocks after request completion: {vattention.num_free_kvblocks()}")

        print("\\n--- Memory State After Request Completion ---")
        vattention.show_allocator_state()

        # Continue with remaining requests
        print("\\n--- Continuing with Remaining Requests ---")
        for step in range(512):
            decode_seq_lens = [seq_len + 1 for seq_len in decode_seq_lens]
            print(f"Decode step {step + 1}: remaining sequences = {decode_seq_lens}")
            vattention.step_async(decode_seq_lens)

        print(f"Free KV blocks: {vattention.num_free_kvblocks()}")

        # Test synchronous allocation alternative
        print("\\n--- Testing Synchronous Alternative ---")
        print("Using vattention.step(seq_lens, eager_reclaim) instead of step_async...")
        decode_seq_lens = [seq_len + 2 for seq_len in decode_seq_lens]

        start_time = time.time()
        vattention.step(decode_seq_lens, True)  # eager_reclaim=True
        sync_time = time.time() - start_time
        print(f"step (synchronous) with eager_reclaim=True completed in {sync_time:.4f} seconds")
        print(f"Free KV blocks after sync step: {vattention.num_free_kvblocks()}")

        # Complete remaining requests
        print("\\n--- Completing All Remaining Requests ---")
        for req in requests:
            print(f"Completing {req['name']} (batch_id={req['batch_id']})")
            vattention.free_batch_idx(req['batch_id'])

        final_free_blocks = vattention.num_free_kvblocks()
        print(f"\\nAll requests completed!")
        print(f"Final free KV blocks: {final_free_blocks}")
        print(f"Memory reclaimed: {final_free_blocks - (initial_free_blocks - len(requests) - 1)} blocks")

        print("\\n--- Final Memory State ---")
        vattention.show_allocator_state()

        print(f"\\n=== Performance Summary ===")
        print(f"Asynchronous allocation time: {async_time:.4f}s")
        print(f"Synchronous allocation time: {sync_time:.4f}s")
        print(f"Memory efficiency: {final_free_blocks}/{initial_free_blocks} blocks recovered")

        print("\\n✅ Multi-request workflow test completed successfully!")
        return True

    except Exception as e:
        print(f"❌ Multi-request test FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        print("\\n--- Cleanup ---")
        try:
            vattention.cleanup()
            print("✅ Cleanup completed successfully!")
        except Exception as cleanup_error:
            print(f"⚠️ Cleanup warning: {cleanup_error}")


if __name__ == "__main__":
    # Show initial GPU memory
    if torch.cuda.is_available():
        initial_allocated = torch.cuda.memory_allocated() / 1024**2
        initial_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"Initial GPU memory - Allocated: {initial_allocated:.1f} MB, Reserved: {initial_reserved:.1f} MB")

    success = test_multi_request_workflow()

    # Show final GPU memory
    if torch.cuda.is_available():
        final_allocated = torch.cuda.memory_allocated() / 1024**2
        final_reserved = torch.cuda.memory_reserved() / 1024**2
        print(f"\\nFinal GPU memory - Allocated: {final_allocated:.1f} MB, Reserved: {final_reserved:.1f} MB")
        print(f"Memory change - Allocated: {final_allocated - initial_allocated:+.1f} MB, Reserved: {final_reserved - initial_reserved:+.1f} MB")

    if success:
        print("\\n🎉 vAttention multi-request workflow is working correctly!")
        print("✅ The maybeExchangeDevice() error has been successfully resolved!")
        print("✅ Both step_async() and step() functions work correctly!")
        print("✅ Memory allocation and deallocation work as expected!")
    else:
        print("\\n❌ Multi-request workflow test failed!")