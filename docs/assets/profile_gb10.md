# Profile of the served path on the GB10

torch 2.14.0+cu130, transformers 5.17.0, NVIDIA GB10, v3.0-2B, bf16 tower, fla. Documents 80 and 1052 tokens; 4-option choice questions. HTTP: uvicorn http=h11 loop=asyncio, one keep-alive client.

**Before (easy-infer ebed5ff): wall time by stage** (ms, p50; stage rows are measured with a synchronise between steps)

| stage | 1 q, 80 tok | 1 q, 1,052 tok | 8 q, 80 tok | 8 q, 1,052 tok | 32 q, 80 tok | 32 q, 1,052 tok |
|---|---|---|---|---|---|---|
| HTTP: receive, parse, pydantic, dispatch (server, before handler) | 0.75 | 0.75 | 0.54 | 0.83 | 0.92 | 0.62 |
| HTTP: response, middleware, return (server, after answer) | 0.36 | 0.34 | 0.38 | 0.44 | 0.49 | 0.45 |
| HTTP: socket + h11/httptools + client (outside server) | 0.70 | 0.78 | 0.83 | 0.75 | 0.75 | 0.67 |
| json.loads of the body (in-process) | 0.01 | 0.01 | 0.01 | 0.02 | 0.03 | 0.04 |
| pydantic validation (in-process) | 0.01 | 0.01 | 0.02 | 0.02 | 0.04 | 0.04 |
| wire -> Question objects | 0.01 | 0.01 | 0.02 | 0.02 | 0.04 | 0.04 |
| tokenization (encode_question, all questions) | 0.21 | 0.99 | 1.31 | 7.37 | 4.80 | 29.15 |
| shared-prefix check (re-tokenizes the state) | 0.08 | 0.83 | 0.08 | 0.86 | 0.09 | 0.97 |
| collate + H2D | 0.19 | 0.28 | 0.64 | 0.66 | 2.22 | 2.29 |
|   of which option first-token lookups (unused by this readout) | 0.04 | 0.04 | 0.33 | 0.34 | 1.20 | 1.21 |
| state pass (prefix cache), when taken | – | – | 23.06 | 82.70 | 23.26 | 83.70 |
| cache replicate (reorder_cache) | – | – | 1.15 | 1.74 | 4.16 | 6.41 |
| question forward pass(es) | 25.41 | 86.69 | 42.42 | 47.77 | 160.51 | 181.88 |
| unpermute + softmax | 0.05 | 0.07 | 0.07 | 0.08 | 0.20 | 0.21 |
| GPU->CPU copies (.tolist) | 0.02 | 0.03 | 0.11 | 0.11 | 0.40 | 0.41 |
| answer building (to_answer) | 0.01 | 0.01 | 0.02 | 0.02 | 0.05 | 0.05 |
| JSON serialisation (stdlib) | 0.01 | 0.01 | 0.03 | 0.03 | 0.08 | 0.08 |
| **in-process wall (served scorer)** | 27.00 | 90.03 | 68.48 | 140.45 | 195.41 | 302.83 |
| **HTTP client total** | 30.49 | 93.85 | 71.41 | 143.90 | 199.99 | 308.01 |
| real / padded tokens in question passes | 124 / 124 | 1096 / 1096 | 344 / 360 | 344 / 360 | 1392 / 1536 | 1392 / 1536 |
| path (forward passes) | full (1) | full (1) | state once (2) | state once (2) | state once (3) | state once (3) |

**Before: GPU time by module** (GPU kernel ms in one call, torch.profiler; share of kernel time)

| module | 1 q, 80 tok | 1 q, 1,052 tok | 8 q, 80 tok | 8 q, 1,052 tok | 32 q, 80 tok | 32 q, 1,052 tok |
|---|---|---|---|---|---|---|
| Gated DeltaNet mixers (18 layers) | 7.69 (34%) | 35.38 (43%) | 22.40 (38%) | 49.73 (40%) | 75.54 (42%) | 102.57 (39%) |
| MLPs (24) | 10.37 (46%) | 28.34 (34%) | 22.84 (39%) | 40.69 (33%) | 51.83 (29%) | 69.97 (27%) |
| full-attention mixers (6) | 1.70 (7%) | 5.84 (7%) | 3.83 (6%) | 12.94 (10%) | 11.46 (6%) | 35.78 (14%) |
| RMSNorms (48) | 1.19 (5%) | 9.61 (12%) | 3.12 (5%) | 11.06 (9%) | 13.60 (8%) | 22.09 (8%) |
| final norm | 0.02 (0%) | 0.19 (0%) | 0.06 (0%) | 0.22 (0%) | 0.27 (0%) | 0.42 (0%) |
| embedding | 0.01 (0%) | 0.01 (0%) | 0.01 (0%) | 0.01 (0%) | 0.04 (0%) | 0.04 (0%) |
| rotary | 0.02 (0%) | 0.03 (0%) | 0.04 (0%) | 0.05 (0%) | 0.08 (0%) | 0.09 (0%) |
| tower, other (masks, casts, cache cat) | 0.17 (1%) | 2.09 (3%) | 0.60 (1%) | 2.53 (2%) | 3.08 (2%) | 5.04 (2%) |
| scorer (option_xattn + MLP combine, fp32) | 1.31 (6%) | 1.31 (2%) | 4.90 (8%) | 4.61 (4%) | 18.23 (10%) | 18.41 (7%) |
| calibration head | 0.10 (0%) | 0.07 (0%) | 0.09 (0%) | 0.09 (0%) | 0.18 (0%) | 0.18 (0%) |
| outside the model (collate H2D, unpermute, softmax, copies) | 0.10 (0%) | 0.07 (0%) | 1.12 (2%) | 1.70 (1%) | 4.52 (3%) | 6.53 (2%) |
| **kernel time, total** | 22.68 | 82.935 | 59.005 | 123.63 | 178.807 | 261.103 |
| GPU span of the call | 29.54 | 86.917 | 69.596 | 131.507 | 194.671 | 275.172 |
| GPU idle inside the span (host-bound gaps) | 6.861 | 3.982 | 10.592 | 7.876 | 15.864 | 14.069 |
| wall of the profiled call (profiler on) | 32.722 | 90.952 | 71.67 | 140.569 | 200.588 | 306.975 |
| kernels launched | 1879 | 1837 | 3768 | 3702 | 5682 | 5604 |

**After A (serve-speed-2 default): wall time by stage** (ms, p50; stage rows are measured with a synchronise between steps)

| stage | 1 q, 80 tok | 1 q, 1,052 tok | 8 q, 80 tok | 8 q, 1,052 tok | 32 q, 80 tok | 32 q, 1,052 tok |
|---|---|---|---|---|---|---|
| HTTP: receive, parse, pydantic, dispatch (server, before handler) | 0.50 | 0.55 | 0.57 | 0.57 | 0.65 | 0.68 |
| HTTP: response, middleware, return (server, after answer) | 0.14 | 0.14 | 0.15 | 0.15 | 0.19 | 0.20 |
| HTTP: socket + h11/httptools + client (outside server) | 0.69 | 0.69 | 0.61 | 0.74 | 0.64 | 0.64 |
| json.loads of the body (in-process) | 0.01 | 0.02 | 0.02 | 0.02 | 0.03 | 0.04 |
| pydantic validation (in-process) | 0.02 | 0.02 | 0.03 | 0.03 | 0.04 | 0.04 |
| wire -> Question objects | 0.01 | 0.01 | 0.02 | 0.02 | 0.05 | 0.04 |
| tokenization (encode_question, all questions) | 0.46 | 1.27 | 0.95 | 1.44 | 1.04 | 2.75 |
| shared-prefix check (re-tokenizes the state) | 0.01 | 0.01 | 0.01 | 0.02 | 0.01 | 0.06 |
| collate + H2D | 0.27 | 0.37 | 0.35 | 0.36 | 1.00 | 1.00 |
| state pass (prefix cache), when taken | – | – | 23.64 | 83.74 | 23.86 | 83.95 |
| cache replicate (reorder_cache) | – | – | 1.19 | 1.78 | 4.19 | 6.42 |
| question forward pass(es) | 26.22 | 87.39 | 43.24 | 48.21 | 163.43 | 184.14 |
| unpermute + softmax | 0.07 | 0.09 | 0.09 | 0.10 | 0.16 | 0.17 |
| GPU->CPU copies (.tolist) | 0.03 | 0.04 | 0.06 | 0.06 | 0.19 | 0.20 |
| answer building (to_answer) | 0.01 | 0.01 | 0.02 | 0.02 | 0.05 | 0.05 |
| JSON serialisation (stdlib) | 0.02 | 0.02 | 0.04 | 0.04 | 0.08 | 0.08 |
| JSON serialisation (orjson) | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 | 0.01 |
| **in-process wall (served scorer)** | 27.67 | 90.63 | 70.15 | 135.35 | 194.06 | 277.32 |
| **HTTP client total** | 30.51 | 93.91 | 72.10 | 138.96 | 196.57 | 281.33 |
| real / padded tokens in question passes | 124 / 124 | 1096 / 1096 | 344 / 360 | 344 / 360 | 1392 / 1536 | 1392 / 1536 |
| path (forward passes) | full (1) | full (1) | state once (2) | state once (2) | state once (3) | state once (3) |

**After A: GPU time by module** (GPU kernel ms in one call, torch.profiler; share of kernel time)

| module | 1 q, 80 tok | 1 q, 1,052 tok | 8 q, 80 tok | 8 q, 1,052 tok | 32 q, 80 tok | 32 q, 1,052 tok |
|---|---|---|---|---|---|---|
| Gated DeltaNet mixers (18 layers) | 7.97 (34%) | 35.77 (43%) | 22.64 (38%) | 50.58 (40%) | 75.65 (42%) | 104.33 (39%) |
| MLPs (24) | 10.60 (46%) | 29.19 (35%) | 22.51 (38%) | 40.86 (33%) | 52.41 (29%) | 71.01 (27%) |
| full-attention mixers (6) | 1.51 (6%) | 5.90 (7%) | 3.79 (6%) | 13.55 (11%) | 11.60 (6%) | 36.07 (14%) |
| RMSNorms (48) | 1.12 (5%) | 9.39 (11%) | 2.96 (5%) | 11.05 (9%) | 13.66 (8%) | 21.77 (8%) |
| final norm | 0.05 (0%) | 0.19 (0%) | 0.06 (0%) | 0.23 (0%) | 0.27 (0%) | 0.43 (0%) |
| embedding | 0.01 (0%) | 0.01 (0%) | 0.01 (0%) | 0.01 (0%) | 0.04 (0%) | 0.03 (0%) |
| rotary | 0.02 (0%) | 0.06 (0%) | 0.05 (0%) | 0.05 (0%) | 0.08 (0%) | 0.11 (0%) |
| tower, other (masks, casts, cache cat) | 0.17 (1%) | 2.16 (3%) | 0.59 (1%) | 2.54 (2%) | 3.08 (2%) | 5.05 (2%) |
| scorer (option_xattn + MLP combine, fp32) | 1.62 (7%) | 1.30 (2%) | 5.02 (9%) | 4.55 (4%) | 18.61 (10%) | 19.23 (7%) |
| calibration head | 0.07 (0%) | 0.07 (0%) | 0.15 (0%) | 0.09 (0%) | 0.18 (0%) | 0.19 (0%) |
| outside the model (collate H2D, unpermute, softmax, copies) | 0.06 (0%) | 0.08 (0%) | 1.12 (2%) | 1.71 (1%) | 4.27 (2%) | 6.87 (3%) |
| **kernel time, total** | 23.193 | 84.118 | 58.892 | 125.23 | 179.841 | 265.097 |
| GPU span of the call | 29.939 | 88.505 | 72.623 | 133.78 | 196.783 | 278.801 |
| GPU idle inside the span (host-bound gaps) | 6.746 | 4.387 | 13.731 | 8.55 | 16.941 | 13.703 |
| wall of the profiled call (profiler on) | 32.642 | 90.805 | 74.386 | 136.36 | 199.536 | 282.421 |
| kernels launched | 1879 | 1837 | 3761 | 3695 | 5652 | 5574 |

