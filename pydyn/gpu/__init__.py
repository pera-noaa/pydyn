"""pydyn.gpu: the reference's dynamic core with every region's state resident on a GPU.

TM5's transport is bound by memory traffic, not arithmetic (tm5-zoom
docs/memory_bandwidth_floor.md), so the port is organised around where the arrays live: the
state m, rm..rzm, am, bm, cm, pu, pv, sp, phlb of every region is allocated on the device once
and stays there; every operator runs where it is; the host keeps the tree, the split orders
and the sub-cycle counts.  Inputs read from files go up when read; state comes down only to be
compared, and the transfer ledger (backend.Ledger) is what shows it.

Bitwise to the NumPy reference, the same standard as the reference to the Fortran:

    csrc/          one C source compiled for each device -- by NVRTC with --fmad=false (CUDA),
                   by xcrun metal -ffp-contract=off (Metal, float32), and by the host compiler
                   with -ffp-contract=off for the emulation backends -- so each pair runs the
                   same IEEE operations in the same order; the kernels have no
                   scheduling-dependent reductions (only order-free max / or)
    backend.py     `cuda` (CuPy on its own stream, uploads through pinned memory), `emu`
                   (emu.EmuArray: host arrays that refuse what CuPy refuses, so an emu run is
                   also a residency check), `emu32` (the float32 build, the CPU's FZ set as
                   the Apple GPU flushes); the transfer ledger, the queue's marks
    metal.py       `metal`: an Apple GPU in float32, one shared-storage arena, a library built
                   offline; with convec_gen.py (convection as unrolled passes per lmax) and
                   coop.py (the line kernels a SIMD group / warp a line), as cuda
    kernels.py     pydyn.kernels on device views: x rows (reduced grid), y and z with their
                   sub-cycle searches run to a fixpoint on the device, dynam0, convec with
                   Crout or the reference-LAPACK twin, Check_CFL's mass-only kernels, the loss
                   rates and decay
    diffusion.py   Calc_Kzz
    zoom.py        zoom.host's bodies for a device ZoomState (pydyn.zoom's driver runs
                   them); Pipeline runs intervals deciding nothing on the host, in flight,
                   each checked once the device ran it and redone if wrong
    exchange.py    the exchange, halos, compress / uncompress and the reduced grid as box
                   programs (boxes.py, csrc/boxes.cuh): staged, hazard-checked, built once
    engine.py      the replay's DeviceEngine: `python3 -m pydyn.replay <set> --device emu|cuda`
    check.py       `python3 -m pydyn.gpu.check <set>`: the set replayed on the reference and
                   on the device, every array of every checkpoint fingerprinted, the two matched
    bench.py       `python3 -m pydyn.gpu.bench`: a production-sized interval's throughput

Where the reference's bits come from a library the device cannot run -- the model's LAPACK
(MKL, Accelerate) and libm (libimf, glibc, Apple's) -- the port has open twins, compiled from
the same csrc/ into the host library that `--lapack twin --libm twin` loads into the reference:
csrc/convec.cuh's DGETRF/DGETRI in reference-LAPACK order and csrc/pmath.cuh's fdlibm
exp/log/pow.  A device run equals the reference run under the twins bit for bit; the twins are
within an ulp of the libraries they stand in for (docs/gpu_port.md).
"""
