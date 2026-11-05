// pydyn.gpu's Metal runtime, called through ctypes: one shared-storage arena that both the CPU
// (as NumPy views) and the GPU (by gpuAddress) address, a .metallib of csrc/ compiled offline
// (xcrun metal -ffp-contract=off), and dispatches queued on one compute encoder until a sync.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static id<MTLDevice> g_dev;
static id<MTLCommandQueue> g_queue;
static id<MTLBuffer> g_arena;
static id<MTLCommandBuffer> g_cb;
static id<MTLComputeCommandEncoder> g_enc;
static NSMutableArray *g_keep;  // libraries and pipelines live as long as the process
static double g_gpu_seconds;
static long g_dispatches, g_buffers;
// a command buffer is committed every g_commit_every dispatches, so the GPU starts while the
// host goes on queueing (the queue runs its buffers in order); a sync commits the last and
// waits for every one in flight
static long g_commit_every = 8, g_in_buffer;
static NSMutableArray *g_flight;
static long g_committed, g_retired;  // buffers committed, and completed in order, so far
// profiling: an encoder per dispatch, GPU timestamps at its boundaries (Apple GPUs sample at
// stage boundaries only)
static id<MTLCounterSampleBuffer> g_samples;
static long g_sample_cap, g_sample_n;

static void put_err(char *err, int n, NSString *s)
{
    if (err && n > 0) {
        strncpy(err, s ? [s UTF8String] : "unknown", n - 1);
        err[n - 1] = 0;
    }
}

extern "C" int pdm_init(uint64_t bytes, void **host, uint64_t *gpu, char *name, int namelen)
{
    @autoreleasepool {
        if (!g_dev) {
            g_dev = MTLCreateSystemDefaultDevice();
            if (!g_dev) return 1;
            g_queue = [g_dev newCommandQueue];
            g_keep = [NSMutableArray new];
            g_flight = [NSMutableArray new];
            const char *e = getenv("PYDYN_METAL_COMMIT_EVERY");
            if (e) g_commit_every = atol(e);
        }
        if (!g_arena || g_arena.length < bytes) {
            g_arena = [g_dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            if (!g_arena) return 2;
        }
        *host = g_arena.contents;
        *gpu = g_arena.gpuAddress;
        put_err(name, namelen, g_dev.name);
        return 0;
    }
}

extern "C" uint64_t pdm_max_buffer(void) { return g_dev ? (uint64_t)g_dev.maxBufferLength : 0; }

extern "C" void *pdm_library(const char *path, char *err, int errlen)
{
    @autoreleasepool {
        NSError *e = nil;
        NSURL *url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
        id<MTLLibrary> lib = [g_dev newLibraryWithURL:url error:&e];
        if (!lib) {
            put_err(err, errlen, e.localizedDescription);
            return 0;
        }
        [g_keep addObject:lib];
        return (__bridge void *)lib;
    }
}

extern "C" void *pdm_pipeline(void *lib_, const char *fn, int *width, int *maxthreads, char *err, int errlen)
{
    @autoreleasepool {
        id<MTLLibrary> lib = (__bridge id<MTLLibrary>)lib_;
        id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:fn]];
        if (!f) {
            put_err(err, errlen, @"no such function");
            return 0;
        }
        NSError *e = nil;
        id<MTLComputePipelineState> p = [g_dev newComputePipelineStateWithFunction:f error:&e];
        if (!p) {
            put_err(err, errlen, e.localizedDescription);
            return 0;
        }
        [g_keep addObject:p];
        *width = (int)p.threadExecutionWidth;
        *maxthreads = (int)p.maxTotalThreadsPerThreadgroup;
        return (__bridge void *)p;
    }
}

static void ensure_encoder(void)
{
    if (!g_cb) {
        g_cb = [g_queue commandBufferWithUnretainedReferences];
        g_buffers++;
    }
    if (g_samples) {  // a fresh encoder for this dispatch, its start and end sampled
        if (g_enc) [g_enc endEncoding];
        MTLComputePassDescriptor *d = [MTLComputePassDescriptor computePassDescriptor];
        d.dispatchType = MTLDispatchTypeSerial;
        long i = g_sample_n < g_sample_cap ? g_sample_n : g_sample_cap - 1;
        d.sampleBufferAttachments[0].sampleBuffer = g_samples;
        d.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * i;
        d.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * i + 1;
        g_sample_n++;
        g_enc = [g_cb computeCommandEncoderWithDescriptor:d];
        [g_enc useResource:g_arena usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
    } else if (!g_enc) {
        g_enc = [g_cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
        [g_enc useResource:g_arena usage:(MTLResourceUsageRead | MTLResourceUsageWrite)];
    }
}

static int retire(long upto);

// commit the open buffer (wait: and every one in flight); 0 = all completed so far
static int flush(bool wait)
{
    if (g_cb) {
        [g_enc endEncoding];
        [g_cb commit];
        [g_flight addObject:g_cb];
        g_committed++;
        g_enc = nil;
        g_cb = nil;
        g_in_buffer = 0;
    }
    if (!wait) return 0;
    return retire(g_committed);
}

// wait for the buffers in flight up to the upto'th committed; 0 = completed
static int retire(long upto)
{
    int rc = 0;
    while (g_retired < upto && g_flight.count) {
        id<MTLCommandBuffer> cb = g_flight[0];
        [cb waitUntilCompleted];
        if (cb.status != MTLCommandBufferStatusCompleted) rc = 1;
        g_gpu_seconds += cb.GPUEndTime - cb.GPUStartTime;
        [g_flight removeObjectAtIndex:0];
        g_retired++;
    }
    return rc;
}

// commit what is queued without waiting: the mark that pdm_wait waits for
extern "C" long pdm_mark(void)
{
    @autoreleasepool {
        flush(false);
        return g_committed;
    }
}

// buffers committed and retired so far: what is queued now goes in buffer committed + 1
extern "C" void pdm_progress(long *committed, long *retired)
{
    *committed = g_committed;
    *retired = g_retired;
}

// wait for what was queued before the mark (not what came after); 0 = completed
extern "C" int pdm_wait(long mark, char *err, int errlen)
{
    @autoreleasepool {
        int rc = retire(mark);
        if (rc) put_err(err, errlen, @"command buffer failed");
        return rc;
    }
}

// args: the kernel's argument struct (setBytes, <= 4 kB); n threads in groups of tg
extern "C" void pdm_dispatch(void *pso, const void *args, int len, uint64_t n, int tg)
{
    @autoreleasepool {
        ensure_encoder();
        [g_enc setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso];
        [g_enc setBytes:args length:len atIndex:0];
        [g_enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(tg, 1, 1)];
        [g_enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
        g_dispatches++;
        if (++g_in_buffer >= g_commit_every && g_commit_every > 0 && !g_samples) flush(false);
    }
}

// commit what is queued and wait; 0 = completed
extern "C" int pdm_sync(char *err, int errlen)
{
    @autoreleasepool {
        id<MTLCommandBuffer> last = g_cb;
        int rc = flush(true);
        if (rc) put_err(err, errlen, last && last.error ? last.error.localizedDescription : @"command buffer failed");
        return rc;
    }
}

// start sampling (cap dispatches at most), 0 = ok
extern "C" int pdm_profile_begin(long cap)
{
    @autoreleasepool {
        id<MTLCounterSet> ts = nil;
        for (id<MTLCounterSet> c in g_dev.counterSets)
            if ([c.name isEqualToString:MTLCommonCounterSetTimestamp]) ts = c;
        if (!ts || ![g_dev supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) return 1;
        MTLCounterSampleBufferDescriptor *d = [MTLCounterSampleBufferDescriptor new];
        d.counterSet = ts;
        d.storageMode = MTLStorageModeShared;
        d.sampleCount = 2 * cap;
        NSError *e = nil;
        g_samples = [g_dev newCounterSampleBufferWithDescriptor:d error:&e];
        if (!g_samples) return 2;
        g_sample_cap = cap;
        g_sample_n = 0;
        return 0;
    }
}

// stop sampling: the dispatches' GPU durations in seconds into out (n of them), their count
extern "C" long pdm_profile_end(double *out, long n)
{
    @autoreleasepool {
        if (!g_samples) return -1;
        long m = g_sample_n < g_sample_cap ? g_sample_n : g_sample_cap;
        NSData *d = [g_samples resolveCounterRange:NSMakeRange(0, 2 * m)];
        const MTLCounterResultTimestamp *t = (const MTLCounterResultTimestamp *)d.bytes;
        // GPU ticks -> seconds, from two CPU/GPU timestamp correlations
        MTLTimestamp c0, g0, c1, g1;
        [g_dev sampleTimestamps:&c0 gpuTimestamp:&g0];
        [NSThread sleepForTimeInterval:0.02];
        [g_dev sampleTimestamps:&c1 gpuTimestamp:&g1];
        double ns_per_tick = (double)(c1 - c0) / (double)(g1 - g0);
        for (long i = 0; i < m && i < n; i++) out[i] = (double)(t[2 * i + 1].timestamp - t[2 * i].timestamp) * ns_per_tick * 1e-9;
        g_samples = nil;
        return m;
    }
}

extern "C" void pdm_stats(double *gpu_seconds, long *dispatches, long *buffers)
{
    *gpu_seconds = g_gpu_seconds;
    *dispatches = g_dispatches;
    *buffers = g_buffers;
}
