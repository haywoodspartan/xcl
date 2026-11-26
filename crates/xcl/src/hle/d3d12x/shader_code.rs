//! Compute pipelines not used for a while are released and made again when they are bound.
//!
//! NVIDIA's driver keeps the machine code of the pipelines its lists use in one heap of at most
//! 512 MB. A submission that needs more removes the device: `DXGI_ERROR_DEVICE_REMOVED`, nothing
//! in DRED and no event in the system log, as there was no GPU fault. The wave64 programs of a
//! console title are several times their size on 32-lane hardware, and a retail title's world is
//! past that in its first seconds. Code goes into the heap when a pipeline is first used and out
//! of it with the pipeline.
//!
//! A pipeline that is released can still be in a recorded list. So `pso_async::resolve` and its
//! kin give the thread a reference to it, a bind hands that to the allocator the list records on
//! (`bound`), and the allocator's `Reset` – the title's word that the GPU is past its lists –
//! gives it back (`allocator_reset`).
//!
//! `XCL_SHADER_CODE_MB`: the compiled code to keep in use (`GetCachedBlob` sizes, which are about
//! 1/9 less than what the code takes in the heap; 0: no limit). Default: 320 on NVIDIA, no limit
//! elsewhere.

use std::cell::RefCell;
use std::collections::{HashMap, HashSet, VecDeque};
use std::ffi::c_void;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::Instant;

use windows::core::Interface;
use windows::Win32::Graphics::Direct3D12::ID3D12PipelineState;

/// a thread's references to pipelines it resolved and has not bound yet, at most
const HELD_MAX: usize = 4;
/// a pipeline used this recently is not released
pub const IDLE_MS: u64 = 1000;
const NVIDIA: u32 = 0x10DE;

/// A reference to a pipeline.
#[derive(PartialEq)]
pub struct Ref(ID3D12PipelineState);

impl Ref {
    /// # Safety
    /// `pso` is a live pipeline.
    pub unsafe fn new(pso: usize) -> Option<Self> {
        ID3D12PipelineState::from_raw_borrowed(&(pso as *mut c_void)).cloned().map(Self)
    }
}

thread_local! {
    static HELD: RefCell<VecDeque<Ref>> = const { RefCell::new(VecDeque::new()) };
}

/// allocator → the pipelines its recordings use
static BOUND: Mutex<Option<HashMap<usize, HashMap<usize, Ref>>>> = Mutex::new(None);
/// pipeline that was used → bytes of its compiled code, until the driver frees it
static SIZES: Mutex<Option<HashMap<usize, usize>>> = Mutex::new(None);
/// their sum
static USED: AtomicUsize = AtomicUsize::new(0);

fn nvidia() -> Option<bool> {
    use windows::Win32::Graphics::Direct3D12::ID3D12Device;
    use windows::Win32::Graphics::Dxgi::{CreateDXGIFactory1, IDXGIAdapter, IDXGIFactory4};
    let raw = super::real::device_raw();
    unsafe {
        let dev = ID3D12Device::from_raw_borrowed(&raw)?;
        let factory: IDXGIFactory4 = CreateDXGIFactory1().ok()?;
        let adapter: IDXGIAdapter = factory.EnumAdapterByLuid(dev.GetAdapterLuid()).ok()?;
        Some(adapter.GetDesc().ok()?.VendorId == NVIDIA)
    }
}

/// The megabytes the driver reserves for this process's compiled code: what an NVIDIA profile
/// sets for the executable (`xcl nvprofile install`, `gpu::nvprofile`), 512 without one.
fn heap_mb() -> usize {
    static H: OnceLock<usize> = OnceLock::new();
    *H.get_or_init(|| {
        let exe = std::env::current_exe().ok().and_then(|p| p.file_name().map(|n| n.to_string_lossy().into_owned()));
        let set = exe.and_then(|e| crate::gpu::nvprofile::status(&e).ok().flatten()).map(|bytes| (bytes >> 20) as usize);
        if let Some(mb) = set {
            crate::xlog_info!("D3D12.X: the NVIDIA profile of this executable reserves {mb} MB for compiled shader code");
        }
        set.unwrap_or(512)
    })
}

/// The megabytes of compute code to keep in use under a driver heap of `heap_mb`: 320 under the
/// driver's own 512 MB; under a bigger heap 13/32 of it, which leaves the graphics pipelines
/// room: they are never released and only grow. Under 1 GiB a retail title's world was removed
/// at 750 s with 640 MB kept (645 MB compute + 235 MB graphics in use, about 990 MB in the heap)
/// and not in 556 s with 420 MB kept (541 + 232).
fn default_limit_mb(heap_mb: usize) -> usize {
    if heap_mb <= 512 { 320 } else { heap_mb * 13 / 32 }
}

/// How many bytes of compiled code to keep in use (0: no limit).
pub fn limit() -> usize {
    static L: OnceLock<usize> = OnceLock::new();
    *L.get_or_init(|| {
        let mb = std::env::var("XCL_SHADER_CODE_MB").ok().and_then(|v| v.trim().parse().ok());
        // (the tests give `pso_async` pipelines that are none)
        mb.unwrap_or_else(|| if !cfg!(test) && nvidia() == Some(true) { default_limit_mb(heap_mb()) } else { 0 }) << 20
    })
}

/// How many bytes of compiled code may be in use at all (`XCL_SHADER_CODE_CAP_MB`; 0: no cap).
/// Default: 384 MB where there is a `limit`, none elsewhere. At it, a pipeline not in use yet
/// keeps its stand-in (`pso_async`'s `take`): its draws and dispatches are left out until code is
/// released.
///
/// `limit` releases what no recording holds, and a retail title's first world frame holds more
/// than the driver's heap takes in its recordings alone: the device was removed at its first
/// submit with 419 MB counted here, in every run. Left-out work in one frame is the lesser loss.
pub fn cap() -> usize {
    static C: OnceLock<usize> = OnceLock::new();
    *C.get_or_init(|| {
        let limit = limit();
        cap_of(std::env::var("XCL_SHADER_CODE_CAP_MB").ok().as_deref(), limit, if limit != 0 { heap_mb() } else { 512 })
    })
}

/// The cap in bytes for the switch's value `v`, the release limit `limit` (bytes) and the
/// driver's heap of `heap_mb`: none under a heap a profile made bigger, where no draw has to be
/// left out for the code to fit.
fn cap_of(v: Option<&str>, limit: usize, heap_mb: usize) -> usize {
    match v.and_then(|v| v.trim().parse::<usize>().ok()) {
        Some(mb) => mb << 20,
        None if limit != 0 && heap_mb <= 512 => 384 << 20,
        None => 0,
    }
}

/// The code in use is at its `cap`.
pub fn over_cap() -> bool {
    over(used(), cap())
}

fn over(used: usize, cap: usize) -> bool {
    cap != 0 && used >= cap
}

pub fn now_ms() -> u64 {
    static T: OnceLock<Instant> = OnceLock::new();
    T.get_or_init(Instant::now).elapsed().as_millis() as u64
}

/// This thread is about to use pipeline `r`.
pub fn hold(r: Ref) {
    let old = HELD.with(|h| {
        let mut h = h.borrow_mut();
        if h.contains(&r) {
            return Some(r);
        }
        h.push_back(r);
        if h.len() > HELD_MAX {
            h.pop_front()
        } else {
            None
        }
    });
    // (outside the borrow: the last reference's `Release` comes back into XCL)
    drop(old);
}

/// The title is done with the pipeline `pso`: this thread's reference to it goes.
pub fn let_go(pso: usize) {
    let old = HELD.with(|h| {
        let mut h = h.borrow_mut();
        let i = h.iter().position(|r| r.0.as_raw() as usize == pso)?;
        h.remove(i)
    });
    drop(old);
}

/// `list` records with the pipelines this thread resolved: they live until its allocator's `Reset`.
pub fn bound(list: usize) {
    let held = HELD.with(|h| std::mem::take(&mut *h.borrow_mut()));
    if held.is_empty() {
        return;
    }
    let allocator = match super::alloc_pool::enabled() {
        true => super::alloc_pool::recording_backing(list).unwrap_or(0),
        false => super::list_x::allocator_of(list),
    };
    keep(allocator, held);
}

/// `bound` for a list recording on the PC allocator `allocator`.
pub fn bound_on(allocator: usize) {
    let held = HELD.with(|h| std::mem::take(&mut *h.borrow_mut()));
    if !held.is_empty() {
        keep(allocator, held);
    }
}

fn keep(allocator: usize, held: VecDeque<Ref>) {
    let mut left = Vec::new();
    {
        let mut g = BOUND.lock().unwrap();
        let m = g.get_or_insert_with(HashMap::new).entry(allocator).or_default();
        for r in held {
            left.extend(m.insert(r.0.as_raw() as usize, r));
        }
    }
    drop(left);
}

/// `allocator` was reset or released: the GPU is past its recordings.
pub fn allocator_reset(allocator: usize) {
    let gone = BOUND.lock().unwrap().as_mut().and_then(|m| m.remove(&allocator));
    drop(gone);
}

/// Pipeline `pso` is used: its code counts until it is freed (`freed`).
///
/// # Safety
/// `pso` is a live pipeline.
pub unsafe fn count(pso: usize) -> usize {
    if let Some(&n) = SIZES.lock().unwrap().as_ref().and_then(|m| m.get(&pso)) {
        return n;
    }
    // (outside the lock: the driver's call)
    let n = super::real::cached_blob_size(pso as *mut c_void);
    if SIZES.lock().unwrap().get_or_insert_with(HashMap::new).insert(pso, n).is_none() {
        USED.fetch_add(n, Ordering::AcqRel);
    }
    n
}

/// `count` for `pso` and the other pieces of its program, if it is the first of a split one.
///
/// # Safety
/// `pso` is a live pipeline.
pub unsafe fn count_all(pso: usize) -> usize {
    let tail = super::real::dispatch_chain::tail(pso);
    count(pso) + tail.iter().map(|p| count(p.as_raw() as usize)).sum::<usize>()
}

/// The last reference to pipeline `pso` goes.
pub fn freed(pso: usize) {
    if let Some(n) = SIZES.lock().unwrap().as_mut().and_then(|m| m.remove(&pso)) {
        USED.fetch_sub(n, Ordering::AcqRel);
    }
}

/// The pipelines recordings use: released, they would take their room until those are done.
pub fn in_recordings() -> HashSet<usize> {
    BOUND.lock().unwrap().as_ref().map_or(HashSet::new(), |m| m.values().flat_map(|p| p.keys().copied()).collect())
}

/// Give back what the allocators hold whose lists are all reset and done on the GPU.
pub fn let_go_done() {
    let all: Vec<usize> = BOUND.lock().unwrap().as_ref().map_or(Vec::new(), |m| m.keys().copied().collect());
    // (a chain's recordings keep its pieces too)
    let gone = super::alloc_pool::when_done(&all, |a| (BOUND.lock().unwrap().as_mut().and_then(|m| m.remove(&a)), super::real::dispatch_chain::let_go(a)));
    drop(gone);
}

pub fn used() -> usize {
    USED.load(Ordering::Acquire)
}

/// Which of `pipelines` (bytes of code, when last used) to release for `used` bytes to come down
/// to 7/8 of `limit`: the ones not used for the longest time, and none used in the last `IDLE_MS`.
pub fn choose(pipelines: &[(usize, u64)], used: usize, limit: usize, now: u64) -> HashSet<usize> {
    let mut order: Vec<usize> = (0..pipelines.len()).filter(|&i| pipelines[i].1 + IDLE_MS <= now).collect();
    order.sort_by_key(|&i| pipelines[i].1);
    let (mut left, mut out) = (used, HashSet::new());
    for i in order {
        if left <= limit / 8 * 7 {
            break;
        }
        left = left.saturating_sub(pipelines[i].0);
        out.insert(i);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The cap is 384 MB where code is released at a limit, none where it is not, and what the
    /// switch says otherwise (0: none); the code in use is over it from the cap on.
    #[test]
    fn the_cap_follows_the_limit_unless_the_switch_names_one() {
        const MB: usize = 1 << 20;
        assert_eq!(cap_of(None, 320 * MB, 512), 384 * MB);
        assert_eq!(cap_of(None, 0, 512), 0);
        assert_eq!(cap_of(Some(" 450 "), 320 * MB, 512), 450 * MB);
        assert_eq!(cap_of(Some("0"), 320 * MB, 512), 0);
        assert_eq!(cap_of(Some("x"), 0, 512), 0);
        // a profile's bigger heap: no cap unless the switch names one
        assert_eq!(cap_of(None, 416 * MB, 1024), 0);
        assert_eq!(cap_of(Some("700"), 416 * MB, 1024), 700 * MB);
        assert!(!over(383 * MB, 384 * MB));
        assert!(over(384 * MB, 384 * MB));
        assert!(!over(usize::MAX, 0), "no cap");
    }

    /// 320 MB of compute code are kept under the driver's own heap, 13/32 of a bigger one.
    #[test]
    fn the_code_kept_in_use_follows_the_drivers_heap() {
        assert_eq!([256, 512, 1024, 2048].map(default_limit_mb), [320, 320, 416, 832]);
    }

    #[test]
    fn the_longest_unused_go_first_and_only_as_many_as_it_takes() {
        // 100 each, 400 used, the limit 320 (7/8: 280): two have to go
        let p = [(100, 5_000), (100, 1_000), (100, 3_000), (100, 9_500)];
        assert_eq!(choose(&p, 400, 320, 10_000), HashSet::from([1, 2]));
        assert_eq!(choose(&p, 280, 320, 10_000), HashSet::new());
        // never one just used, also when that leaves too much
        assert_eq!(choose(&p, 400, 8, 10_000), HashSet::from([0, 1, 2]));
    }

    /// A pipeline a list was recorded with lives until the list's allocator is reset.
    #[test]
    fn a_bound_pipeline_lives_until_its_allocator_is_reset() {
        use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
        use windows::Win32::Graphics::Direct3D12::*;
        unsafe {
            let Ok(dev) = crate::gpu::agility::create_device(None, D3D_FEATURE_LEVEL_11_0) else {
                return crate::gpu::recomp::runner::skip_test("no GPU");
            };
            let mut sig = None;
            D3D12SerializeRootSignature(&D3D12_ROOT_SIGNATURE_DESC::default(), D3D_ROOT_SIGNATURE_VERSION_1, &mut sig, None).unwrap();
            let sig = sig.unwrap();
            let rs: ID3D12RootSignature = dev.CreateRootSignature(0, std::slice::from_raw_parts(sig.GetBufferPointer() as *const u8, sig.GetBufferSize())).unwrap();
            let code = super::super::real::compile("[numthreads(1, 1, 1)] void main() {}", "cs_5_0");
            let desc = D3D12_COMPUTE_PIPELINE_STATE_DESC {
                pRootSignature: std::mem::transmute_copy(&rs),
                CS: D3D12_SHADER_BYTECODE { pShaderBytecode: code.as_ptr() as _, BytecodeLength: code.len() },
                ..Default::default()
            };
            let pso: ID3D12PipelineState = dev.CreateComputePipelineState(&desc).unwrap();
            let raw = pso.as_raw() as usize;
            let refs = || {
                type F = unsafe extern "system" fn(usize) -> u32;
                let slot = |k: usize| std::mem::transmute::<usize, F>(*(*(raw as *const *const usize)).add(k));
                slot(1)(raw);
                slot(2)(raw)
            };
            let take = || HELD.with(|h| std::mem::take(&mut *h.borrow_mut()));
            assert_eq!(refs(), 1);
            let allocator = 0x7E57_A110usize;
            hold(Ref::new(raw).unwrap());
            hold(Ref::new(raw).unwrap());
            assert_eq!(refs(), 2, "held once by this thread");
            keep(allocator, take());
            hold(Ref::new(raw).unwrap());
            keep(allocator, take());
            assert_eq!(refs(), 2, "held once by the allocator");
            assert!(in_recordings().contains(&raw));
            allocator_reset(allocator);
            assert_eq!(refs(), 1);
            assert!(!in_recordings().contains(&raw));
        }
    }
}
