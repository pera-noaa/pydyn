"""What an MI100 under CuPy (ROCm) gives pydyn: the device, hiprtc's options, contraction off,
double division and sqrt against numpy bit for bit, 32-lane groups in a 64-lane wavefront,
pydyn's own source compiled as it is."""

import sys
from pathlib import Path

import cupy as cp
import numpy as np

rt = cp.cuda.runtime
print(
    "cupy",
    cp.__version__,
    "is_hip",
    rt.is_hip,
    "runtime",
    rt.runtimeGetVersion(),
    "driver",
    rt.driverGetVersion(),
)
props = rt.getDeviceProperties(0)
print(
    "device",
    props["name"],
    props.get("gcnArchName"),
    "CUs",
    props["multiProcessorCount"],
    "warp",
    props["warpSize"],
    "mem GB",
    round(props["totalGlobalMem"] / 2**30, 1),
)
print(
    "driver.launchKernel:",
    hasattr(cp.cuda.driver, "launchKernel"),
    "| begin_capture:",
    hasattr(cp.cuda.Stream, "begin_capture"),
)

SRC = r"""
extern "C" __global__ void arith(const double *a, const double *b, const double *c, double *fma_like,
                                 double *quot, double *root, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    fma_like[i] = a[i] * b[i] + c[i];
    quot[i] = a[i] / b[i];
    root[i] = sqrt(fabs(a[i]));
}

extern "C" __global__ void lanes(const double *v, double *bc, double *up, int *anyv, int *maxi, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned ln = threadIdx.x % 32;
    double x = v[i];
    bc[i] = __shfl(x, 5, 32);                    /* lane 5 of this 32-lane group */
    up[i] = __shfl_up(x, 1, 32);
    unsigned long long bal = __ballot(x > 0.5);  /* 64 bits: both groups of the wavefront */
    unsigned half = (unsigned)(bal >> (threadIdx.x & 32));
    anyv[i] = half != 0u;
    int m = (int)(x * 1000.0);
    for (int d = 16; d > 0; d >>= 1) m = max(m, __shfl_xor(m, d, 32));
    maxi[i] = m;
}
"""


def bits(x):
    return np.asarray(x).view(np.uint64)


def compile_(opts):
    return cp.RawModule(code=SRC, options=opts)


rng = np.random.default_rng(1)
n = 1 << 20
a = rng.standard_normal(n) * np.exp(rng.uniform(-40, 40, n))
b = rng.standard_normal(n) * np.exp(rng.uniform(-40, 40, n))
c = rng.standard_normal(n) * np.exp(rng.uniform(-40, 40, n))
a[:1000] = rng.uniform(1e-310, 1e-306, 1000)  # subnormals
b[1000:2000] = rng.uniform(1e-310, 1e-306, 1000)
host = {"fma_like": a * b + c, "quot": a / b, "root": np.sqrt(np.abs(a))}
for opts in (("-ffp-contract=off", "-std=c++14"), ("-ffp-contract=fast", "-std=c++14")):
    try:
        mod = compile_(opts)
    except Exception as e:
        print(opts, "compile failed:", str(e)[:300])
        continue
    da, db, dc = cp.asarray(a), cp.asarray(b), cp.asarray(c)
    out = {k: cp.empty(n) for k in host}
    mod.get_function("arith")(
        ((n + 255) // 256,), (256,), (da, db, dc, out["fma_like"], out["quot"], out["root"], np.int64(n))
    )
    cp.cuda.Device().synchronize()
    res = {k: int((bits(cp.asnumpy(v)) != bits(host[k])).sum()) for k, v in out.items()}
    print(opts, "differ from numpy (of 2^20):", res)

mod = compile_(("-ffp-contract=off", "-std=c++14"))
m = 256
v = rng.uniform(0, 1, m)
v[64:96] = 0.1  # a group with nothing above 0.5
dv = cp.asarray(v)
bc, up, maxi = cp.empty(m), cp.empty(m), cp.empty(m, np.int32)
anyv = cp.empty(m, np.int32)
mod.get_function("lanes")((1,), (m,), (dv, bc, up, anyv, maxi, np.int64(m)))
g = v.reshape(-1, 32)
exp_bc = np.repeat(g[:, 5], 32)
exp_up = np.where(np.arange(m) % 32 == 0, v, np.roll(v, 1))
exp_any = np.repeat((g > 0.5).any(axis=1).astype(np.int32), 32)
exp_max = np.repeat((g * 1000.0).astype(np.int32).max(axis=1), 32)
print(
    "32-lane groups: broadcast",
    np.array_equal(cp.asnumpy(bc), exp_bc),
    "| shuffle up",
    np.array_equal(cp.asnumpy(up), exp_up),
    "| any (ballot half)",
    np.array_equal(cp.asnumpy(anyv), exp_any),
    "| xor max",
    np.array_equal(cp.asnumpy(maxi), exp_max),
)

# pydyn's source as it is, with HIP's options in place of NVRTC's
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from pydyn.gpu import backend as B  # noqa: E402
from pydyn.gpu import convec_gen, coop  # noqa: E402

code = (
    (B.CSRC / "all.cu").read_text()
    + convec_gen.convec_source(26, "cuda")
    + coop.source(B.kernel_signatures(), "cuda")
)
try:
    cp.RawModule(
        code=code,
        options=("-ffp-contract=off", "-std=c++14", f"-I{B.CSRC}", "-DPD_CONVEC_N=26", "-DPD_COOP_FACES=64"),
    ).compile()
    print("pydyn's all.cu + passes + coop: compiled as is")
except Exception as e:
    msg = str(e)
    errs = [line for line in msg.splitlines() if "error" in line]
    print(f"pydyn's source: {len(errs)} error lines; first ones:")
    for line in errs[:15]:
        print("   ", line[:200])
