# AMD GPU support on Windows

QaoaXorsat 0.7 uses the same KernelAbstractions kernels for CUDA, AMDGPU/ROCm,
and Metal. Vendor packages are optional weak dependencies, so CPU-only users do
not install a GPU runtime.

## Radeon 860M setup

The validated host uses:

- Windows with AMD Radeon 860M (`gfx1152`)
- AMD HIP SDK 7.2.3 in `C:\Program Files\AMD\ROCm\7.2`
- Julia 1.12.5 at
  `C:\Users\johnaz\AppData\Local\Julia-1.12.5\bin\julia.exe`
- AMDGPU 2.7.3, GPUCompiler 2.2.2, LLVM 9.13.0, and
  AMDGPU_LLVM_Backend_jll 22.1.8+2

Create an environment and pin the tested package revision:

```powershell
$env:ROCM_PATH = 'C:\Program Files\AMD\ROCm\7.2'
$julia = 'C:\Users\johnaz\AppData\Local\Julia-1.12.5\bin\julia.exe'
& $julia -e '
using Pkg
Pkg.activate("amd-qaoa"; shared=true)
Pkg.add(PackageSpec(name="AMDGPU", version="2.7.3"))
Pkg.add(PackageSpec(
    url="https://github.com/johnazariah/qaoa-xorsat.git",
    rev="johnazariah-amd-gpu-backend",
))
'
```

Validate the runtime and execute a representative MaxCut value and gradient:

```powershell
& $julia --project=@amd-qaoa -e '
using QaoaXorsat
backend = gpu_backend(:amdgpu)
@assert validate_gpu_backend(backend)
evaluator = make_gpu_evaluator(backend)
params = TreeParams(2, 3, 3)
angles = QAOAAngles([0.31, -0.22, 0.14], [0.42, 0.33, -0.18])
println(backend.label, ": ", backend.device)
println(evaluator(params, angles; clause_sign=-1))
'
```

AMDGPU may warn that optional MIOpen is unavailable. QaoaXorsat uses dense
KernelAbstractions kernels and does not require MIOpen, rocSPARSE, or
`AMDGPU.versioninfo()`. LLVM may also print that `gfx1152` is not a recognized
processor; the AMDGPU 2.7.3 backend still compiled and completed all live tests
on this host. Backend validation deliberately performs a real dense kernel
launch and host round-trip.

## Backend API and failure semantics

```julia
gpu_backend(kind::Symbol=:auto; validate::Bool=true)::GPUBackend
gpu_backend_available(kind::Symbol; validate::Bool=true)::Bool
validate_gpu_backend(backend::GPUBackend)::Bool
gpu_array(backend::GPUBackend, array::AbstractArray)
make_gpu_evaluator(backend::GPUBackend=gpu_backend();
                   checkpoint_interval::Int=0)
make_statevector_evaluator(
    backend::GPUBackend,
    N::Int,
    diagonal::AbstractVector{<:Real};
    memory_fraction=0.8,
)::DeviceStatevectorEvaluator
make_statevector_evaluator(
    backend::GPUBackend,
    N::Int,
    terms::AbstractVector{PauliTerm};
    memory_fraction=0.8,
)::DeviceStatevectorEvaluator
statevector_execution_stats(
    evaluator::DeviceStatevectorEvaluator,
)::StatevectorExecutionStats
synchronize(evaluator::DeviceStatevectorEvaluator)
```

Explicit `:amdgpu` selection, including aliases `:amd`, `:rocm`, and `:hip`,
throws `GPUBackendError` if AMDGPU, HIP, the device libraries, the device, or
the validation kernel is unavailable. It never substitutes CPU. `:auto` tries
CUDA, AMDGPU, and Metal in that order, then returns the CPU backend. AMDGPU and
CUDA use `ComplexF64`; Metal uses `ComplexF32`.

The statevector evaluator accepts a real diagonal or Z/ZZ `PauliTerm`s, starts
from normalized `|+>^N`, applies exact cost phases and `sum(X)` mixer
butterflies, and returns
`(value::Float64, gamma_gradient::Vector{Float64},
beta_gradient::Vector{Float64})`. Gradient ordering is
`[gamma_1, ..., gamma_p, beta_1, ..., beta_p]`. Non-diagonal costs fail
explicitly. Calls synchronize before returning; the explicit `synchronize`
method is also available.

`statevector_execution_stats` is immutable telemetry containing backend kind,
device, complex dtype, actual KA kernel-launch count, predicted live bytes,
allocator reservation before/after, memory cap, source revision, evaluation
counts, first-evaluation and steady-state wall times, allocation/transfer and
verification wall times, synchronized-launch wall time, and pure kernel time
when available. AMD kernel time uses HIP events. Portable compile-only and
launch-only times are reported as unavailable rather than inferred; the first
evaluation is the reproducible compile/warmup measurement.

### Statevector memory admission

Admission happens before validation or device-state allocation. For
`dimension = 2^N`, ComplexF64 predicts four complex device buffers, one real
device diagonal, one complex host initial-state copy, one real host diagonal,
and 25% transient headroom: `120 * dimension` bytes. ComplexF32 predicts
`60 * dimension` bytes. Construction requires both
`reserved + predicted <= memory_fraction * total` and `predicted <= free`,
where `memory_fraction` defaults to and cannot exceed 0.8. The observed
allocator reservation is checked again after allocation and every evaluation.
Telemetry failures and allocation failures throw `GPUBackendError`; there is
no CPU fallback.

The retained regression telemetry `43,117,445,120 / 47,102,148,608` bytes is
91.54% reserved and is rejected by the 80% cap before state allocation.

The mutually compatible provider floors are AMDGPU 2.7.3, CUDA 6.3, and Metal
1.10.1. CUDA 6.0-6.2 require GPUCompiler 1.x and cannot share an environment
with AMDGPU 2.7.3, which requires GPUCompiler 2.x.

## Correctness and scaling

Run the live compatibility test in an environment containing AMDGPU:

```powershell
& $julia --project=@amd-qaoa test\test_gpu_amdgpu.jl
```

The test checks the WHT and MaxCut/XORSAT forward and adjoint paths plus the
physical statevector matrix `N={4,6,8}`, `p={1,2,3}` against CPU `ComplexF64`
references. On the Radeon 860M all 126 current assertions pass, including
source-bound kernel launches, HIP-event timing, and the `1e-10` absolute and
relative statevector tolerances. At Basso depth `p=12`,
the measured absolute errors were `2.84e-14` for the value, `4.64e-11` for the
gamma gradient, and `5.84e-11` for the beta gradient. The live tolerance is
`rtol=2e-9`, reflecting accumulated floating-point reordering rather than a
precision downgrade. Existing CUDA/Metal tests and golden expectations remain
the compatibility basis when those devices are unavailable; no live CUDA or
Metal execution is claimed.

The scaling variable is the Basso configuration count `N_cfg = 4^p`, not a
finite physical graph size. This evaluator describes the infinite
D-regular-tree limit, so the raw CSV reports `physical_problem_n` as
`not_applicable`. One `ComplexF64` state vector occupies `16*N_cfg` bytes. The
production optimizer's full CPU adjoint model is 40 such vectors, or
`640*N_cfg` bytes.

Reproduce the bounded sweep:

```powershell
& $julia --project=@amd-qaoa scripts\benchmark_gpu_scaling.jl `
  results\amd-gpu-scaling.csv 11 12
```

The benchmark records compile/warmup and steady-state times, CPU memory,
observed HIP pool reservation, speedup, and numerical errors. CPU execution is
classified as impractical above 60 seconds per value+gradient; measurements are
only attempted when projected below 180 seconds and the 40-vector estimate is
at most 60% of host RAM. GPU runs are limited to 80% of the HIP aperture.

The committed raw sweep measured crossover at `p=10`. At `p=12`
(`N_cfg=16,777,216`) its deterministic full-gradient case took 89.57 seconds
on CPU and 51.50 seconds on AMD (1.74x). A PFQE consumer acceptance case at the
same depth took 124.99 seconds on CPU and 42.54 seconds on AMD (2.938x), with
errors around `1e-11`. These CPU measurements exceed the one-minute practical
limit while AMD remains close to it.

At `p=13`, `N_cfg=67,108,864` and one state vector is 1 GiB. The established
CPU full-adjoint estimate is 40 GiB, exceeding the 31.65 GiB installed host
RAM, so the CPU gradient was not launched. The lower-memory AMD forward-only
path completed in 150.86 seconds with value `0.4865786622328167`. The full
checkpointed AMD gradient reached the shared-UMA limit during its backward
pass and did not complete. This is an explicit resource boundary, not a
correctness fallback: `gpu_backend(:amdgpu)` still surfaces the allocation
failure. The reproducible sweep enforces the 80% HIP-aperture guard rather than
manufacturing an OOM or destabilizing the host.
