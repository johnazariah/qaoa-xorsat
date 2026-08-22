# AMD GPU support and verified evidence

QaoaXorsat v0.7 provides an optional AMDGPU/ROCm backend for the infinite-tree
evaluator. AMDGPU, CUDA, and Metal are weak dependencies: installing a provider
activates its package extension without making GPU packages mandatory for CPU
users. Explicit selection fails rather than silently falling back:

```julia
using AMDGPU, QaoaXorsat

backend = gpu_backend(:amdgpu) # :amd, :rocm, and :hip are aliases
println((backend.kind, backend.device, backend.complex_type))
evaluator = make_gpu_evaluator(backend)
value, gamma_gradient, beta_gradient =
    evaluator(TreeParams(3, 4, 3),
              QAOAAngles([0.27, -0.19, 0.11], [0.36, 0.24, -0.13]);
              clause_sign=1)
```

## Verified Windows configuration

The live AMD validation used Julia 1.12.5, HIP SDK 7.2.3 installed at
`C:\Program Files\AMD\ROCm\7.2`, and a Radeon 860M. The unified Julia resolver
selected AMDGPU 2.7.3, CUDA 6.3.0, Metal 1.10.1, GPUCompiler 2.2.2, and LLVM
9.13.0.

In PowerShell, install the HIP SDK first, then reproduce the package and strict
device checks from the repository root:

```powershell
$env:HIP_PATH = 'C:\Program Files\AMD\ROCm\7.2'
$env:PATH = "$env:HIP_PATH\bin;$env:PATH"

julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
julia --project=. -e 'using Pkg; Pkg.add(name="AMDGPU", version="2.7.3")'
julia --project=. -e 'using AMDGPU, QaoaXorsat; b=gpu_backend(:amdgpu); @assert b.complex_type === ComplexF64; @assert validate_gpu_backend(b); println((b.device, b.label))'
julia --project=. test/test_gpu_amdgpu.jl
```

The retained verification record is:

- CPU public-API tests: 8/8.
- Live AMD tests: 18/18.
- Strict device smoke: Radeon 860M, including a real
  KernelAbstractions kernel and ComplexF64 round trip.
- PFQE live integration tests: 13/13.

The `gfx1152` target may produce a support warning in the current AMDGPU stack.
It did not prevent compilation or the verified kernels from running. MIOpen and
rocSPARSE are not used by this evaluator and are therefore not setup gates.

## Retained scaling evidence

These are measured full MaxCut value-and-gradient timings from the consuming
PFQE integration, not fresh benchmark runs:

| Depth | State configurations | CPU median | AMD | Speedup | Value error | Max gamma error | Max beta error |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 12 | 16,777,216 | 124.9864094 s | 42.5408467 s | 2.9380x | 1.532108e-14 | 3.585160e-12 | 9.569234e-12 |

Here `p` is QAOA depth and `N_cfg = 4^p` is the branch-state configuration
count. It is not a physical graph size. A physical graph vertex count `N` is
not applicable to the infinite-tree evaluator.

At `p=13`, `N_cfg=67,108,864`. The estimated CPU live state was 40 GiB, above
both 31.65 GiB physical memory and the 23.73 GiB safety limit, so the CPU run
was deliberately skipped. A full GPU gradient reached the shared-UMA limit
during the backward pass and did **not** complete. A lower-memory GPU forward
value did complete in 150.8598194 s with finite value
`0.4865786622328167` and 8.0 GiB process RSS. This is forward-only evidence;
it is not evidence for a `p=13` full gradient.

CUDA and Metal use the same KernelAbstractions kernels, participate in the
unified resolver, and retain their existing golden tests. No live CUDA or Metal
hardware run is claimed by this record.
