// residency_plan.cpp — Host-side residency budget planner for the native
// Qwen3.6-35B-A3B decode runtime (Project: 1 token/s on 3.8GB VRAM + ~5GB RAM).
//
// This is the FIRST native deliverable because "1 tk/s" is fundamentally a
// data-delivery problem, not a flops problem. This planner computes whether
// the model's hot working set actually FITS in the available memory, how it
// should be split across (VRAM resident / RAM hot / SSD cold), and the
// projected warm-decode wall clock. If the budget doesn't close, no kernel
// speeds it up — the bytes physically cannot be delivered in time.
//
// Pure host C++ (no CUDA) so it builds and runs anywhere:
//   g++ -O2 -std=c++17 native_runtime/residency_plan.cpp -o /tmp/resplan && /tmp/resplan
//

#include <cstdint>
#include <cstdio>
#include <algorithm>

namespace {

// ---- Qwen3.6-35B-A3B model constants (from the project README) ----
constexpr uint64_t kHidden      = 2048;
constexpr uint64_t kVocab       = 248320;
constexpr uint32_t kLayers      = 40;
constexpr uint32_t kFullAttnLayers = 10;    // validated full-attention
constexpr uint32_t kDeltaLayers = 30;       // Gated DeltaNet / linear-attn
constexpr uint32_t kExpertsPerLayer = 256;
constexpr uint32_t kTopK        = 8;
constexpr bool    kFp8          = true;     // 1 byte/param checkpoint

// RFC numbers — we derive these from the total param count rather than guess.
constexpr uint64_t kTotalParams = 35'000'000'000ULL;  // ~35B
constexpr uint64_t kB          = 1000ULL;
constexpr uint64_t kMiB        = 1024ULL * 1024ULL;
constexpr uint64_t kGiB        = 1024ULL * kMiB;

constexpr double kBytesPerParam = kFp8 ? 1.0 : 1.0;  // FP8 checkpoint, 1 byte

// Elective constants (user-tunable for the host)
double g_vram_gib   = 3.8;   // ~3.8 GB usable VRAM
double g_ram_gib    = 5.0;   // native budget target (free RAM)
double g_pcie_gbs   = 10.0;  // PCIe Gen3 x16 effective H2D bandwidth
double g_ssd_gbs    = 2.5;   // downstream for SSD cold spill

uint64_t params_embeddings() { return kVocab * kHidden; }        // ~508M
uint64_t params_lm_head()    { return kVocab * kHidden; }        // ~508M

// Per-layer fixed (non-routed) weights: attention QKV+O projections + norms.
uint64_t params_attn_per_layer() {
    // 4 projections hidden x hidden = 4 * 2048 * 2048 = 16.8M
    return 4ull * kHidden * kHidden;
}

uint64_t params_fixed_per_layer() {
    return params_attn_per_layer();
}

// Total fixed (attention + embeddings + lm_head + shared experts + norms).
uint64_t params_fixed_total() {
    return kLayers * params_fixed_per_layer()
         + params_embeddings() + params_lm_head();
}

// The remaining params are the routed MoE experts (attention is fixed).
uint64_t params_moe_total() {
    uint64_t f = params_fixed_total();
    return kTotalParams > f ? kTotalParams - f : 0;
}

uint64_t params_shared_total() {
    // 1 shared expert per layer (approx physiological size of a routed expert)
    uint64_t per = params_moe_total() / (kLayers * kExpertsPerLayer);
    return kLayers * per;  // shared experts are the same block size
}

uint64_t bytes_of(uint64_t params) { return (uint64_t)(params * kBytesPerParam); }

uint64_t moe_expert_bytes() {
    return bytes_of(params_moe_total() / (kLayers * kExpertsPerLayer));
}

void header() {
    std::printf("===========================================================\n"
                " RESIDENCY BUDGET PLANNER — native decode for Qwen3.6-A3B\n"
                " Goal: 1 token/s @ 3.8GB VRAM + ~5GB RAM\n"
                "===========================================================\n\n");
}

uint64_t f2b(double gib) { return (uint64_t)(gib * kGiB); }

} // namespace

int main() {
    header();

    const uint64_t vram_b = f2b(g_vram_gib);
    const uint64_t ram_b  = f2b(g_ram_gib);

    // ---- Derive per-block bytes ----
    const uint64_t emb_b    = bytes_of(params_embeddings());   // ~508MB
    const uint64_t lmhead_b = bytes_of(params_lm_head());      // ~508MB
    const uint64_t attn_per = bytes_of(params_attn_per_layer()); // ~17MB
    const uint64_t moe_ex   = moe_expert_bytes();              // ~3.3MB FP8
    const uint64_t fixed_t  = bytes_of(params_fixed_total());
    const uint64_t moe_t    = bytes_of(params_moe_total());    // ~33.8GB -> never all resident

    std::printf("--- Derived model sizes (FP8) ---\n");
    std::printf(" embeddings (row: only 1 needed per step) : %4.1f MB (%4.1f KB/row)\n",
                emb_b / (double)kMiB, emb_b / (double)kVocab / 1024.0);
    std::printf(" lm_head    (needed EVERY token)          : %4.1f MB\n", lmhead_b / (double)kMiB);
    std::printf(" attention  per layer                     : %4.1f MB x%u\n",
                attn_per / (double)kMiB, kLayers);
    std::printf(" MoE routed expert per block              : %4.1f MB\n", moe_ex / (double)kMiB);
    std::printf(" full fixed total (attn+emb+lmhead, no MoE): %4.1f GB\n",
                fixed_t / (double)kGiB);
    std::printf(" full MoE total (routed experts)          : %4.1f GB  <-- the beast\n\n",
                moe_t / (double)kGiB);

    // ---- Physical feasibility check: can attention + shared + lm_head be resident? ----
    const uint64_t shared_t = bytes_of(params_shared_total());
    const uint64_t must_hot_in_vram = attn_per * kLayers + shared_t + lmhead_b;
    std::printf("--- Residency split ---\n");
    std::printf(" ALWAYS-HOT (VRAM) attention+shared+lm_head: %4.1f GB  (must fit in VRAM %.2f GB)\n",
                must_hot_in_vram / (double)kGiB, g_vram_gib);
    std::printf(" VRAM left for active-expert slots         : %4.1f GB\n",
                (vram_b > must_hot_in_vram ? (vram_b - must_hot_in_vram) / (double)kGiB : 0.0));
    std::printf(" expert slots that fit in that VRAM head   : %4.0f\n",
                vram_b > must_hot_in_vram ? double(vram_b - must_hot_in_vram) / moe_ex : 0.0);
    std::printf(" expert slots that fit in 5GB RAM (hot)    : %4.0f\n",
                ram_b / (double)moe_ex);
    std::printf(" routed-expert loads per single token      : %u (top-%u x %u layers)\n\n",
                kLayers * kTopK, kTopK, kLayers);

    // ---- Per-token data that must arrive from RAM/VRAM (decode) ----
    const uint64_t per_token_moe = kLayers * kTopK * moe_ex;
    std::printf("--- Per-token decode physics ---\n");
    std::printf(" routed-expert bytes touched per token     : %4.2f GB\n",
                per_token_moe / (double)kGiB);
    std::printf("  if 100%% served from RAM (hot) @ %.0f GB/s : %.0f ms\n",
                g_pcie_gbs, per_token_moe / g_pcie_gbs / 1e6);
    std::printf("  if 50%% cold from SSD @ %.1f GB/s          : ~%.0f ms  <-- danger zone\n",
                g_ssd_gbs,
                (0.5 * per_token_moe / g_ssd_gbs + 0.5 * per_token_moe / g_pcie_gbs) / 1e6);

    // ---- Compute side: is 1 token/s flop-bound or bandwidth-bound? ----
    // ~3B active params/token -> ~6 GFLOP; consumers fp8 do ~200-500 GFLOPS.
    const double active_gflop = 6.0;
    const double gpu_gflops   = 250.0;   // conservative fp8 on a 4GB consumer card
    std::printf(" active compute ~%.0f GFLOP/token @ %.0f GFLOPS : ~%.0f ms (not the bottleneck)\n",
                active_gflop, gpu_gflops, active_gflop / gpu_gflops * 1000.0);

    // ---- Verdict ----
    const double estimate_ms =
        must_hot_in_vram <= vram_b
        ? (per_token_moe / g_pcie_gbs / 1e6) + (active_gflop / gpu_gflops * 1000.0)
        : 1e9;

    std::printf("\n--- Verdict ---\n");
    if (must_hot_in_vram <= vram_b) {
        std::printf(" ALWAYS-HOT set FITS in VRAM (%.1f/%.1f GB).\n",
                    must_hot_in_vram / (double)kGiB, g_vram_gib);
        std::printf(" Projected warm-decode, experts served from RAM: %.0f ms/token.\n\n",
                    estimate_ms);
        std::printf(" CONCLUSION: 1 token/s is PHYSICALLY REACHABLE for the warm/decode\n");
        std::printf(" regime IF (1) attention+shared+lm_head stay resident in VRAM,\n");
        std::printf(" (2) active-expert working set is served from RAM (hot) via\n");
        std::printf(" routing-predictor prefetch, and (3) Python overhead is removed.\n");
        std::printf(" The remaining risk is pure SSD-leakage at long generations.\n");
    } else {
        std::printf(" ALWAYS-HOT set does NOT fit (%.1f/%.1f GB) -> need to demote\n",
                    must_hot_in_vram / (double)kGiB, g_vram_gib);
        std::printf(" lm_head or refit. 1 tk/s requires either a smaller lm_head trade\n");
        std::printf(" or an SSD pipeline; otherwise the budget does not close.\n");
    }
    std::printf("===========================================================\n");
    return 0;
}