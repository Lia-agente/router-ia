from __future__ import annotations

"""Overlap benchmark: synchronous vs asynchronous block acquisition.

Measures whether deferring H2D synchronization (Proposal #1) hides transfer
latency behind computation. Both paths exercise the SAME access sequence
(routed-expert streaming pattern), so any difference is attributable to the
acquire semantics, not to a different workload.

Two scenarios per mode:

  plain      pure transfer, no compute between acquire and consume
  overlapped a simulated compute step (CPU-bound delay) runs between the
             acquire dispatch and the moment the block is actually consumed,
             mimicking the time the router spends computing the current layer
             while the next layer's experts stream in.

Requires the compiled native library (CUDA). Run from repository root:

    python native_memory/overlap_benchmark.py

The acceptance criterion from PROPOSAL.md applies: the async path must show a
measurable, reproducible reduction in wall-clock time per turn under cache
miss pressure. If it does not (bottleneck is compute/dequant, not H2D), the
the proposal is discarded per project rules.
"""

import ctypes
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

DLL = ROOT / "native_memory" / "build-ninja" / "router_ia_native_memory.dll"
os.environ.setdefault("ROUTER_IA_NATIVE_MEMORY_LIB", str(DLL))

from router_ia.memory_manager import MemoryManager

SLOT_BYTES = 8 * 1024 * 1024
PAYLOAD_BYTES = 4 * 1024 * 1024
VRAM_SLOTS = 8
RAM_SLOTS = 32
BLOCKS = 64          # total logical blocks (256 experts scaled down for smoke)
WORKING_SET = 8      # blocks touched per turn (top-8 experts)
TURNS = 6            # generation steps
REPEATS = 3          # repetitions for stable timings

# Simulated per-turn compute (s) that can overlap with the next turn's H2D.
COMPUTE_DELAY = 0.002


def payload(block_id: int) -> bytes:
    return bytes(((i * 29 + block_id * 47 + 13) & 0xFF) for i in range(PAYLOAD_BYTES))


def _register_all(mm: MemoryManager, buffers: list) -> None:
    for block_id in range(BLOCKS):
        mm.register_block(block_id, buffers[block_id], PAYLOAD_BYTES)


def bench_sync(mm: MemoryManager, route: list[list[int]], overlap: bool) -> float:
    """Legacy synchronous acquire. Each acquire blocks until H2D completes."""
    t0 = time.perf_counter()
    for turn in route:
        # (compute would run AFTER acquiring is already serialized here)
        for block_id in turn:
            mm.acquire(block_id, PAYLOAD_BYTES)
        if overlap:
            time.sleep(COMPUTE_DELAY * WORKING_SET)
    return time.perf_counter() - t0


def bench_async(mm: MemoryManager, route: list[list[int]], overlap: bool) -> float:
    """Async acquire: dispatch all H2D, then wait once, compute in between."""
    t0 = time.perf_counter()
    for turn in route:
        # dispatch the whole turn's transfers without blocking
        for block_id in turn:
            mm.acquire_async(block_id, PAYLOAD_BYTES)
        if overlap:
            # compute the current layer while the next experts stream in
            time.sleep(COMPUTE_DELAY * WORKING_SET)
        # consume: wait only now, when the data is actually needed
        for block_id in turn:
            mm.wait_acquire(block_id)
    return time.perf_counter() - t0


def build_route(seed: int) -> list[list[int]]:
    """Generate a top-8 expert access pattern with cache-miss pressure.

    Turns step through distinct expert regions so that most accesses miss the
    (small) VRAM pool and force RAM->VRAM transfers, matching the real
    256-expert/4GB-VRAM regime.
    """
    import random
    rng = random.Random(seed)
    route = []
    cursor = 0
    for _ in range(TURNS):
        turn = [(cursor + rng.randint(0, 15)) % BLOCKS for _ in range(WORKING_SET)]
        cursor = (cursor + WORKING_SET * 2) % BLOCKS  # jump to force misses
        route.append(turn)
    return route


def main() -> None:
    print("=" * 68)
    print("OVERLAP BENCHMARK: SYNC vs ASYNC BLOCK ACQUIRE")
    print("=" * 68)
    print(f"VRAM slots   : {VRAM_SLOTS}")
    print(f"RAM slots    : {RAM_SLOTS}")
    print(f"Blocks       : {BLOCKS}")
    print(f"Working set  : {WORKING_SET}/turn")
    print(f"Turns        : {TURNS}")
    print(f"Payload      : {PAYLOAD_BYTES / 1024 / 1024:.1f} MiB")
    print(f"Compute delay: {COMPUTE_DELAY * 1000:.1f} ms x {WORKING_SET}/turn")

    buffers = [ctypes.create_string_buffer(payload(i)) for i in range(BLOCKS)]

    results: dict[str, dict[str, list[float]]] = {
        "plain": {"sync": [], "async": []},
        "overlapped": {"sync": [], "async": []},
    }

    for rep in range(REPEATS):
        route = build_route(seed=1234)  # same access pattern for both modes
        with MemoryManager(
            vram_slot_bytes=SLOT_BYTES, vram_slots=VRAM_SLOTS,
            ram_slot_bytes=SLOT_BYTES, ram_slots=RAM_SLOTS, streams=2,
        ) as mm:
            _register_all(mm, buffers)
            results["plain"]["sync"].append(bench_sync(mm, route, overlap=False))

        with MemoryManager(
            vram_slot_bytes=SLOT_BYTES, vram_slots=VRAM_SLOTS,
            ram_slot_bytes=SLOT_BYTES, ram_slots=RAM_SLOTS, streams=2,
        ) as mm:
            _register_all(mm, buffers)
            results["plain"]["async"].append(bench_async(mm, route, overlap=False))

        with MemoryManager(
            vram_slot_bytes=SLOT_BYTES, vram_slots=VRAM_SLOTS,
            ram_slot_bytes=SLOT_BYTES, ram_slots=RAM_SLOTS, streams=2,
        ) as mm:
            _register_all(mm, buffers)
            results["overlapped"]["sync"].append(bench_sync(mm, route, overlap=True))

        with MemoryManager(
            vram_slot_bytes=SLOT_BYTES, vram_slots=VRAM_SLOTS,
            ram_slot_bytes=SLOT_BYTES, ram_slots=RAM_SLOTS, streams=2,
        ) as mm:
            _register_all(mm, buffers)
            results["overlapped"]["async"].append(bench_async(mm, route, overlap=True))

    print("\n" + "-" * 68)
    print(f"{'scenario':<12} {'mode':<7} {'mean(s)':>10} {'speedup':>10}")
    print("-" * 68)

    for scenario in ("plain", "overlapped"):
        s = statistics.mean(results[scenario]["sync"])
        a = statistics.mean(results[scenario]["async"])
        speedup = s / a if a > 0 else float("inf")
        print(f"{scenario:<12} {'sync':<7} {s:>10.4f} {'1.00x':>10}")
        print(f"{scenario:<12} {'async':<7} {a:>10.4f} {speedup:>9.2f}x")

    print("-" * 68)

    # Acceptance verdict against PROPOSAL.md criterion
    o_s = statistics.mean(results["overlapped"]["sync"])
    o_a = statistics.mean(results["overlapped"]["async"])
    gain = (o_s - o_a) / o_s * 100 if o_s > 0 else 0.0

    print(f"\nOverlapped scenario: sync {o_s:.4f}s vs async {o_a:.4f}s "
          f"({gain:+.1f}% wall-clock)")

    if gain >= 5.0:
        print("VERDICT: >=5% measurable gain -> async path is justified.")
        print("OVERLAP BENCHMARK = ACCEPT")
    else:
        print("VERDICT: <5% gain -> H2D is not the bottleneck; "
              "per PROPOSAL.md the async path is NOT justified here.")
        print("OVERLAP BENCHMARK = REJECT (bottleneck elsewhere)")


if __name__ == "__main__":
    main()