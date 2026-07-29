# Bug report: per-leaf densities are corrupted on the CUDA + oblivious fit path

**Component:** `gbrl` (repo `/auto/swgwork1/bfuhrer/projects/gbrl_project/nvlabs/gbrl`, version 1.2.0, editable install)
**Affects:** `GBTLearner.predict_densities()`, `GBTLearner.get_tree()['densities']`, and any consumer of
`edata->multi_objective_data->densities`
**Trigger:** `device='cuda'` **and** `grow_policy='oblivious'` **and** `n_objs > 1`
**Does not affect:** CPU (either grow policy), or CUDA + greedy
**Status:** reproduced deterministically; root cause identified; fix proposed below

---

## 1. Symptom

Per-leaf density vectors are documented — and relied upon by `predict_densities` — to sum to 1:

> *"Since each leaf's density vector sums to 1, every output row sums to 1 and can be read as a
> distribution over objectives / label classes."*
> — `gbrl/learners/gbt_learner.py:671` docstring

In practice, on the CUDA + oblivious path many leaves carry exactly `[1.0, 1.0]` (sum = 2, i.e. `n_objs`),
so `predict_densities` returns rows that do not sum to 1 and cannot be read as a distribution.

Observed on the Split-RL MiniGrid-Corner checkpoint
`saved_models/minigrid/minigrid/MiniGrid-Corner-v0/split_rl/fully_obs_seed_0_1000000_steps.zip`
(`grow_policy=oblivious`, `device=cuda`, `n_objs=2`, `max_depth=4`, 39,200 trees / 627,200 leaves):

```
predict_densities(obs) at the episode start state
  ->  [[0.99868685, 0.6954145]]      row sum = 1.694     # should be 1.0
```

Per-leaf breakdown of individual trees in that checkpoint:

| tree | leaf density rows (value, count) |
|---|---|
| 0 | `[0,1]`×7, `[.5,.5]`×2, `[.667,.333]`×1, `[.943,.057]`×1, `[.992,.008]`×1, `[1,0]`×2, **`[1,1]`×2** |
| 3000 | `[0,1]`×1, `[1,0]`×6, **`[1,1]`×9** |
| 20000 | **`[1,1]`×16** (all leaves) |
| 39199 | **`[1,1]`×16** (all leaves) |

Across the ensemble (sampling every 50th of the 39,200 trees):

| statistic | value |
|---|---|
| leaves with a valid (sum-to-1) density | **15.8%** |
| trees valid in every leaf (16/16) | **0.0%** |
| trees dead in every leaf (0/16) | **67.0%** |
| trees partially valid | 33.0% |

The corruption is **scattered, not monotonic** — dead leaves appear throughout training, and valid ones
survive to the very end:

| ensemble segment | valid leaves |
|---|---|
| trees 0–7,840 | 33.3% |
| trees 7,840–15,680 | 2.4% |
| trees 15,680–23,520 | 3.8% |
| trees 23,520–31,360 | 23.4% |
| trees 31,360–39,200 | 16.3% |

That scattering is what makes the post-hoc mitigation in §5.2 viable.

## 2. Root cause

Two lines interact.

**(a)** `allocate_child_tree_node` initializes every child node's density vector to **all ones** —
`gbrl/src/cuda/cuda_fitter.cu:1943`:

```cpp
ones_kernel<<<n_blocks, THREADS_PER_BLOCK, 0, stream>>>(host_child.densities, n_objs);
```

This is deliberate for the *no-labels* case: `node->densities[k]` multiplies the per-objective split gain
(`cuda_fitter.cu:190`, `:730`, `:961`) and the leaf value (`:1670`), so all-ones means "every objective
contributes fully" when no `obj_labels` were supplied.

**(b)** `calc_node_densities_kernel` — which overwrites that init with the real label distribution —
returns early for empty nodes without ever writing, `gbrl/src/cuda/cuda_fitter.cu:2306`:

```cpp
__global__ void calc_node_densities_kernel(
    TreeNodeGPU* __restrict__ node,
    const TreeNodeGPU* __restrict__ parent_node,   // <-- declared, NEVER USED in the body
    const float* __restrict__ obj_labels,
    const int n_objs){

    int obj_idx = blockIdx.x;
    extern __shared__ float s_label_count[];
    if (node->n_samples == 0)
        return;                                    // <-- all-ones init survives to the leaf
    ...
    if (threadIdx.x == 0)
        node->densities[obj_idx] = s_label_count[threadIdx.x] / static_cast<float>(node->n_samples);
}
```

So **every empty leaf ships `[1, 1, …]`**. `copy_node_to_data` (`cuda_fitter.cu:2081`) then copies that
straight into the ensemble:

```cpp
for (int i = 0; i < n_objs; ++i)
    densities[leaf_idx * n_objs + i] = node->densities[i];
```

and it is serialized to disk from there (`ensemble_io.cpp:755`). Serialization itself is correct — the
garbage is already in the ensemble before saving.

The unused `parent_node` parameter is a strong signal that inheriting the parent's density for empty
nodes was the intended behaviour and was simply never implemented.

### Why only oblivious?

Oblivious trees force a complete `2^max_depth` leaf set regardless of how the data partitions, so empty
leaves are routine. Greedy stops splitting when a node runs out of samples, so it rarely produces them.

### Why only CUDA?

The CPU path is structurally immune. `update_ensemble_per_tree` (`gbrl/src/cpp/fitter.cpp:612`)
initializes to a valid one-hot rather than all-ones:

```cpp
edata->multi_objective_data->densities[metadata->n_leaves * metadata->n_objs + k] = (k == 0) ? 1.0f : 0.0f;
```

and `calc_leaf_value` guards its write with `if (count > 0)` (`fitter.cpp:683`), so an empty leaf keeps
that valid `[1, 0, …]`.

### How the damage is distributed

The share of dead leaves tracks how concentrated the rollout data is: the more the policy's state
distribution collapses onto a few distinct observations, the more of the forced `2^depth` leaves receive
no samples. That share moves up and down over a run rather than trending in one direction (see the
quintile table in §1), so **no contiguous range of trees is safe** and no `start_idx`/`stop_idx` window
can be used to dodge the problem.

## 3. Evidence

### 3.1 Minimal reproduction across all four fit paths

Synthetic data, `n_objs=2`, `max_depth=4`, 5 boosting steps, `obj_labels` supplied every step
(`/tmp/density_matrix_test.py`):

| device | grow_policy | leaves | rowsum min | rowsum max | frac rowsum == 1 | verdict |
|---|---|---|---|---|---|---|
| cpu | greedy | 75 | 1.000 | 1.000 | 100.0% | OK |
| cpu | oblivious | 80 | 1.000 | 1.000 | 100.0% | OK |
| cuda | greedy | 80 | 1.000 | 1.000 | 100.0% | OK |
| **cuda** | **oblivious** | **80** | **1.000** | **2.000** | **93.8%** | **BROKEN** |

### 3.2 Breakage scales with empty-leaf count

Same setup, CUDA + oblivious, varying depth. More leaves per tree over a fixed 400-sample dataset means
more empty leaves, and the valid fraction falls exactly as predicted:

| max_depth | leaves/tree | frac rowsum == 1 |
|---|---|---|
| 2 | 4 | 100.0% |
| 3 | 8 | 100.0% |
| 4 | 16 | 93.8% |
| 6 | 64 | 67.2% |
| 8 | 256 | 40.2% |

The corrupted value is always exactly `[1, 1]` — matching the `ones_kernel` init, not random memory.

## 4. Impact

**Training dynamics are NOT affected.** `node->densities` is consumed only in split-gain accumulation and
leaf-value computation, both of which are weighted by sample counts. An empty node contributes zero gain
and produces no leaf value, so a wrong density on an empty node changes nothing about the fitted trees.
**Policies already trained on this path remain valid.**

**Analysis and introspection ARE affected.** `predict_densities` averages the leaf density along each
sample's path across every tree. Empty-at-fit-time leaves are still *reachable at inference time* by
different samples, so the corruption propagates directly into the output. Any Split-RL diagnostic reading
densities — objective routing, leaf purity, label-attribution maps — is invalid on a CUDA + oblivious
checkpoint.

**Partially recoverable post-hoc.** The dead leaves' densities were never computed and cannot be
reconstructed — but they are *detectable*, and existing checkpoints can be read usefully without
retraining. See §5.2.

## 5. Fixes

### 5.1 The fit-time fix

Use the already-passed `parent_node` to give empty nodes their parent's density — the semantically
correct estimate for a leaf with no data — with a one-hot fallback at the root.
`gbrl/src/cuda/cuda_fitter.cu:2305`:

```cpp
    if (node->n_samples == 0) {
        if (threadIdx.x == 0)
            node->densities[obj_idx] = (parent_node != nullptr)
                                     ? parent_node->densities[obj_idx]
                                     : (obj_idx == 0 ? 1.0f : 0.0f);
        return;
    }
```

This is deliberately surgical:

- It only runs when `obj_labels != nullptr` (the kernel is launched under that guard,
  `cuda_fitter.cu:1985`), so the intentional all-ones "no labels → all objectives contribute" behaviour
  is untouched.
- It leaves the `ones_kernel` init in place, so gain weighting for the unlabeled case is unchanged.
- It restores the sum-to-1 invariant that `predict_densities` and `get_tree()['densities']` document.

### 5.2 Post-hoc mitigation — read existing checkpoints without retraining

A dead leaf is **unambiguously identifiable**: `calc_node_densities_kernel` increments exactly one
counter per sample, so a genuinely computed row always sums to 1. A row of all ones (sum = `n_objs`)
can only come from the `ones_kernel` init. There are no false positives.

That makes a masked estimator possible. Instead of averaging the leaf density over *all* trees, average
over only the trees whose leaf is valid:

```python
acc = np.zeros((n_samples, n_objs)); cnt = np.zeros(n_samples)
for t in range(n_trees):
    d = learner.predict_densities(obs, start_idx=t, stop_idx=t + 1)   # this tree's leaf row
    valid = np.abs(d.sum(1) - 1.0) < 1e-3
    acc[valid] += d[valid]; cnt += valid
density = acc / cnt[:, None]
```

Semantically this says *trees that had no data in this region do not vote* — which is the right thing
to do even after the fitter is fixed, since an empty leaf carries no information either way.

Measured on the Corner checkpoint over all 468 (cell, direction) states, probing every 4th tree:

| estimator | spread | mean on lava | mean off lava | separation |
|---|---|---|---|---|
| unmasked (current `predict_densities`) | 0.039 | 0.426 | 0.414 | +0.013 |
| **masked** | **0.170** | **0.058** | **0.010** | **+0.047** |

Every state retained at least 2,546 valid votes (30.3% of probed trees — higher than the 15.8% global
valid-leaf rate, because real observations land preferentially on leaves that had data at fit time). The
masked absolute values are also sane: off-lava ≈ 0.01 means "routed almost entirely to the reward
objective", whereas the unmasked 0.41 was pure artifact.

**Caveat:** this recovers the signal that *was* recorded; it cannot invent the 84% of leaves that were
never computed. Treat it as a way to read existing checkpoints, not as a substitute for the fit-time fix.

Implemented as `--density_estimator masked` (the default) in `scripts/exp/corner_density_maps.py`.

### 5.3 Secondary issue worth checking while you are in there

The shared-memory reduction at `cuda_fitter.cu:2318` assumes `blockDim.x` is a power of two:

```cpp
for(int offset = blockDim.x / 2; offset > 0; offset >>= 1) {
    if(threadIdx.x < offset) s_label_count[threadIdx.x] += s_label_count[threadIdx.x + offset];
    __syncthreads();
}
```

If `get_tpb_dimensions` can return a non-power-of-two `threads_per_block`, the tail elements are silently
dropped and densities under-count (sum < 1). This did not show up in the reproduction above, but it is
worth confirming `get_tpb_dimensions` always rounds to a power of two.

## 6. Second, independent bug: all-zero labels are silently discarded

**Status:** found after bug #1 was fixed and the model retrained (300k-step checkpoint, Jul 26 19:41).
Bug #1 is confirmed fixed — the four-path matrix now passes 100% — but the retrained checkpoint is still
only **55.8%** clean, and the failure has a different shape: whole trees are now dead all-or-nothing
(55.8% of trees fully valid, 44.2% fully dead, none partial), rather than scattered leaves.

### Cause

`gbrl/learners/gbt_learner.py:169`:

```python
if obj_labels is not None and (obj_labels == 0).all():
    obj_labels = None
```

This conflates *"every sample belongs to objective 0"* with *"no label information"*. They are not the
same statement. Once the labels are nulled, `calc_node_densities_kernel` is never launched (it is guarded
by `if (dataset->obj_labels->data != nullptr)`, `cuda_fitter.cu:1985`), the **root** keeps its `ones_kernel`
init, and — because bug #1's fix makes empty children inherit their parent — every leaf in the tree
inherits `[1,1]`. Hence the all-or-nothing per-tree pattern.

### Minimal reproduction

CUDA + oblivious, `n_objs=2`, 512 samples, 5 boosting steps:

| labels in minibatch | frac densities summing to 1 | sample rows |
|---|---|---|
| mixed (~50/50) | 100% | `[0,1]`, `[.5,.5]`, `[.74,.26]` |
| **all label 0** | **0%** | `[1,1]` |
| all label 1 | 100% | `[0,1]` |
| 3/512 label 1 | 100% | `[.98,.02]` |
| `obj_labels=None` | 0% | `[1,1]` |

All-zero is byte-for-byte identical to passing nothing; all-*one* is fine. That asymmetry isolates the
zero-check as the cause.

### Why this specifically ruins Split-RL safety runs

A successful safety agent stops entering unsafe states, so `safety_label_rate → 0` and **most late-training
minibatches are all-zero**. The better the agent gets, the more of its ensemble loses its density record.
That matches the observed progression exactly: trees 0–~2,384 fully valid, then increasingly sporadic.

### It is not only a logging bug — CPU and CUDA train differently

Leaf values are `Σ_k mean_grad_k · density_k · λ_k` (`cuda_fitter.cu:1670`, `fitter.cpp:700`). With the
labels dropped, the two backends disagree about what the densities are:

| device | labels | densities | resulting leaf value |
|---|---|---|---|
| cpu | none / all-zero | `[1, 0]` | **0.0303** |
| cuda | none / all-zero | `[1, 1]` | **−0.866** |

CPU's `calc_leaf_value` defaults `lbl = 0` when `obj_labels == nullptr` (`fitter.cpp:664`), producing a
correct one-hot. CUDA leaves the all-ones init in place. So on CUDA, every all-safe minibatch adds the
**cost gradient into the policy at full weight** instead of applying the reward gradient alone. This
divergence predates the bug #1 fix and is unaffected by it.

Assuming Split-RL's intent is "reward-only-labelled states get the reward gradient", the CPU behaviour is
the correct one — worth confirming against your intended semantics before changing it.

### On the `[1,1]` convention itself

`[1,1]` for an all-zero-label minibatch is a **defensible convention**, not obviously a bug: with no
objective-1 samples there is no distribution to record, and the density-weighting machinery has nothing
to weight. Two consequences follow from that reading, and only the second is clearly a defect.

**Downstream readers must decode it, not average it.** `[1,1]` row-normalises to `0.5`, which any
consumer will read as *"mixed"* — the exact opposite of the truth, *"purely objective 0"*. Because
all-ones cannot arise from a real label count, the intended value is recoverable exactly: substitute the
one-hot `[1, 0, …]`. With bug #1 fixed this is unambiguous, since all-ones can now only mean "the
minibatch was all label 0" — confirmed empirically on the 300k checkpoint, where 55.8% of trees are fully
valid, 44.2% fully all-ones, and **0% partially damaged**. That substitution is implemented as
`--density_estimator zerofill` in `scripts/exp/corner_density_maps.py` and recovers 100% of the ensemble
(vs. `masked`, which discards 44% of it). The two agree on ranking — AUC 0.945 vs 0.952, ratio 8.6x vs
8.8x — and differ only by the expected 0.55 dilution factor.

**The CPU/CUDA divergence is a real defect regardless.** Whatever the convention should be, the two
backends should not disagree about it, and they do (table above): identical inputs give leaf values
0.0303 on CPU and −0.866 on CUDA. Worth deciding which is intended and making both match. Caveat on the
magnitude: that measurement used independent random gradients for the two objectives, which likely
exaggerates the gap versus real Split-RL gradients — the *existence* of the divergence is the solid part,
not its size.

### 6.1 Concrete patch: align CUDA's no-label init with CPU

CPU already does the right thing (`fitter.cpp:739-742`):

```cpp
if (obj_labels == nullptr || n_objs <= 1) {
    // No labels or single objective: all density on obj 0
    for (int k = 0; k < n_objs; ++k)
        node->densities[k] = (k == 0) ? 1.0f : 0.0f;
}
```

CUDA initialises to all-ones instead, at **two** sites — the root (`cuda_fitter.cu:1851`, inside
`allocate_root_tree_node`) and every child (`cuda_fitter.cu:1943`, inside `allocate_child_tree_node`):

```cpp
ones_kernel<<<n_blocks, THREADS_PER_BLOCK, 0, stream>>>(tempNode.densities, metadata->n_objs);   // :1851
ones_kernel<<<n_blocks, THREADS_PER_BLOCK, 0, stream>>>(host_child.densities, n_objs);           // :1943
```

**Step 1** — add a one-hot kernel next to `ones_kernel` in `gbrl/src/cuda/cuda_preprocess.cu:47`:

```cpp
__global__ void one_hot_obj0_kernel(float *arr, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < size) arr[i] = (i == 0) ? 1.0f : 0.0f;
}
```

and declare it in `gbrl/src/cuda/cuda_preprocess.h` beside the `ones_kernel` declaration (`:224`):

```cpp
__global__ void one_hot_obj0_kernel(float *arr, int size);
```

**Step 2** — swap both call sites to use it:

```cpp
one_hot_obj0_kernel<<<n_blocks, THREADS_PER_BLOCK, 0, stream>>>(tempNode.densities, metadata->n_objs);
one_hot_obj0_kernel<<<n_blocks, THREADS_PER_BLOCK, 0, stream>>>(host_child.densities, n_objs);
```

Both backends then agree: absent labels ⇒ density `[1, 0, …]` ⇒ only objective 0 shapes the gain and the
leaf value. `[1,1]` disappears from checkpoints entirely, so `--density_estimator raw` becomes correct and
the `zerofill`/`masked` decoders are no longer needed.

Keeping the `gbt_learner.py:169` short-circuit is then harmless for correctness — nulled all-zero labels
and real all-zero labels both yield `[1, 0]`. Removing it as well is still slightly preferable, since it
lets the density record distinguish "all samples were reward-labelled" from "caller supplied no labels".

**Caveat:** this changes CUDA training results for any Split-RL run with all-safe minibatches — which,
for a working safety agent, is most of them. Expect different (and, on this reading, more correct)
policies after the change; it is not a no-op refactor.

## 7. How to verify the fix

Rebuild the CUDA extension, then re-run the four-path matrix — all four rows must show 100%:

```bash
python3.10 /tmp/density_matrix_test.py     # expects frac rowsum == 1 -> 100.0% on every row
```

Then re-run the depth sweep; depths 2 through 8 must all report 100%.

Note the build caveat: two extension binaries coexist in the gbrl package —
`gbrl_cpp.cpython-310-*.so` (has `predict_densities`) and `gbrl_cpp.cpython-312-*.so` (does **not**).
Everything here was run under `python3.10`; rebuild both if 3.12 is used anywhere.

After rebuilding, the Corner checkpoint must be **retrained** — `scripts/exp/corner_density_maps.py`
validates the sum-to-1 invariant on load and will warn loudly if it is still violated.
