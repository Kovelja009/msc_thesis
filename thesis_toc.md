# Master Thesis — Table of Contents & Writing Outline

**Title (working):** 3D Gaussian Splatting on Tenstorrent Hardware — A Proof-of-Concept Alpha-Blending Kernel

**Scope:** Forward-pass (inference) only. The most compute-intensive stage — alpha blending — is implemented as a tt-metal kernel and benchmarked against CPU (PyTorch reference) and CUDA backends.

**Target length:** ~30–34 pages of prose (~38–40 with figures).

**Repo:** https://github.com/Kovelja009/gsplat_tt

---

## 1. Uvod *(~2–3 str.)*

- **1.1 Motivacija**
  - Novel view synthesis; NeRF → 3D Gaussian Splatting.
  - The gap: GS has only been mapped to GPU (CUDA) and web (WebGL/WebGPU) backends, never to a dataflow AI accelerator. (One sentence on why Tenstorrent is an interesting, unexplored target.)
  - Optional: one short narrative hook line (not a full passage).
- **1.2 Doprinosi i struktura rada**
  - Itemized contributions:
    - tt-metal alpha-blend kernel running on Wormhole.
    - Tri-backend benchmarking harness (CPU / CUDA / TT).
    - LPT (Longest-Processing-Time) load-balancing scheme.
    - Algorithmic optimization ladder (approx exp, opacity cull, radius cap).
  - One-paragraph roadmap of the chapters.

---

## 2. 3D Gaussian Splatting *(~8–9 str. — priority chapter, keep deep)*

- **2.1 Reprezentacija scene**
  - Gaussian as mean μ, covariance Σ, opacity α, color.
  - Short paragraph on spherical harmonics for color (note: PoC uses degree 0, so no full derivation).
- **2.2 Kovarijansa i parametrizacija**
  - Why Σ is stored as scale **S** + rotation quaternion **q**.
  - Decomposition Σ = R S Sᵀ Rᵀ.
  - Quaternion → rotation matrix derivation.
  - Positive semi-definiteness guarantee.
- **2.3 Projekcija 3D → 2D**
  - World → camera → ray-space chain.
  - EWA splatting framework (one citation sentence, no deep history).
  - Jacobian **J** of the projective transform.
  - 2D covariance: Σ′ = J W Σ Wᵀ Jᵀ.
  - **[FIGURE: projection pipeline diagram]**
- **2.4 Alfa stapanje (volume rendering)**
  - Derive C = Σ cᵢ αᵢ Πⱼ<ᵢ(1 − αⱼ) from the volume rendering integral.
  - Define transmittance T and the front-to-back recurrence.
  - **Mark explicitly: this is the equation the kernel implements.**
- **2.5 Tile-based rasterizacija**
  - 16×16 tiles, per-tile depth sorting, why this structure exists (locality, bounded per-pixel work).
  - **[FIGURE: full six-stage forward pipeline, alpha-blend stage highlighted]**

---

## 3. Tenstorrent arhitektura i programski model *(~6–7 str.)*

- **3.1 Tensix jezgro**
  - Open with one paragraph: dataflow grid vs SIMT thread-massive GPUs.
  - Five baby RISC-V cores; FPU (matrix) and SFPU (32-lane vector); packers/unpackers; 1.5 MB L1 SRAM.
  - Wormhole vs Blackhole: two sentences here, not a section.
  - **[FIGURE: Tensix core diagram — you provide]**
- **3.2 Memorija i NoC**
  - DRAM ↔ L1; tiled bf16 layout (32×32 tiles); interleaved buffers; address model.
  - NoC: explicit programmer-managed data movement, multicast.
- **3.3 Programski model**
  - Reader / compute / writer three-kernel pattern.
  - Circular buffers as synchronization primitive; compile-time vs runtime args.
  - Host ↔ device: PCIe, subprocess/daemon orchestration model used in this work.
  - **[FIGURE: reader → CB → compute → CB → writer dataflow]**

---

## 4. Dizajn i implementacija *(~8–9 str. — core contribution)*

- **4.1 Postavka i opseg (PoC)**
  - One kernel on device, rest of pipeline on host; justify as feasibility study.
  - Why alpha blending specifically — quantitative argument using bonsai (1.16M Gaussians) numbers showing blend dominates.
  - Hybrid pipeline: load → project → tile-assign → sort (CPU) → alpha-blend (backend).
  - Tri-backend rationale: CPU = correctness baseline, CUDA = incumbent comparison, TT = contribution.
  - **[FIGURE: hybrid pipeline diagram]**
- **4.2 Mapiranje na Tensix**
  - Tiles → cores mapping.
  - 256 pixels per 16×16 tile → 8 SFPU vector passes (32 lanes).
  - Reader / compute / writer breakdown for the blend kernel; circular-buffer staging of per-tile Gaussian lists.
- **4.3 Balansiranje opterećenja (LPT)** *(own section — nameable contribution)*
  - Problem: naive contiguous tile split → uneven Gaussian counts → slowest core dominates.
  - Solution: Longest-Processing-Time scheduling.
  - Result: 120 ms → 80 ms; 14.5× → 21.7× over CPU.
- **4.4 Optimizacione tehnike** *(optimization ladder, with bonsai numbers)*
  - Approximate exp on SFPU (6.0 s → 5.1 s).
  - Opacity culling at α < 1/255 (→ 2.5 s).
  - Radius capping (→ 0.6–0.9 s).
  - For each: what changed vs reference algorithm + why the hardware made it beneficial.
- **4.5 Numerička preciznost**
  - bf16 on device vs fp32 reference; PSNR/SSIM validation of output. (Short.)

---

## 5. Eksperimenti i rezultati *(~5–6 str.)*

- **5.1 Eksperimentalna postavka**
  - Hardware/software table: TT device, host CPU, GPU, tt-metal version, scenes used.
- **5.2 Poređenje backenda (latency)**
  - Frame time CPU vs CUDA vs TT at fixed scene/resolution (multiple resolutions then make 3 graphs for each backend).
- **5.3 Ablacija optimizacija**
  - Show how each of the optimizations (approx exp, opacity cull, radius cap, LPT) affects the performance.
- **5.4 Analiza uskih grla**
  - Per-stage timing breakdown (blend vs CPU stages vs host↔device transfer).
  - Key finding: once blend is on-device, CPU stages + transfer become the new bottleneck → motivates future work.

---

## 6. Zaključak i budući rad *(~2 str.)*

- **6.1 Zaključak**
  - Restate the feasibility question and answer it: 21.7× over CPU from a single offloaded stage + LPT contribution.
  - Broader claim: dataflow accelerators are a viable, unexplored target for radiance-field rendering.
- **6.2 Budući rad**
  - Move projection / tiling / sort on-device for end-to-end execution.
  - On-device sorting (bitonic / radix on Tensix).
  - Higher SH degrees for view-dependent color.
  - Multi-chip scaling across the NoC fabric.
  - Backward pass / differentiable rasterization → training on Tenstorrent.
  - Production hardening (memory management for multi-million-Gaussian scenes).

---

## Page budget (prose only, before figures)

| Chapter | Pages |
|---------|-------|
| 1. Uvod | 2.5 |
| 2. 3D Gaussian Splatting | 8.5 |
| 3. Tenstorrent arhitektura | 6.5 |
| 4. Dizajn i implementacija | 8.5 |
| 5. Eksperimenti i rezultati | 5.5 |
| 6. Zaključak i budući rad | 2.0 |
| **Total** | **~33.5** |

→ ~38–40 pages with figures.

---

## Notes to self while writing

- **Protect Chapter 2** from cuts — it's the stated goal and what readers most need. If running long, trim 3.2 (cite Tenstorrent memory docs instead of re-explaining) and 4.5 first.
- **Frame CUDA as the incumbent** ("comparison against the hardware GS was designed for"), not just a sanity peer — lets you make claims about TT *relative to the incumbent*.
- **Give LPT its own section heading** so it reads as a distinct intellectual contribution in the ToC.
- The bottleneck analysis (5.5) should flow directly into future work (6.2) — they're the same argument.

## Figures checklist

- [ ] 2.3 — 3D→2D projection diagram
- [ ] 2.5 — full six-stage forward pipeline (blend highlighted)
- [ ] 3.1 — Tensix core architecture (you provide)
- [ ] 3.3 — reader/compute/writer dataflow
- [ ] 4.1 — hybrid pipeline (CPU stages + backend blend)
- [ ] 5.2 — cross-backend latency bar/table
- [ ] 5.3 — core-scaling curves (contiguous vs LPT)
- [ ] 5.4 — optimization ablation bar chart
- [ ] 5.5 — per-stage timing breakdown
