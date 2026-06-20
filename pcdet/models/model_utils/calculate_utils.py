import time
import torch
from typing import Optional

def _cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def fmt_bytes(num: int) -> str:
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if abs(num) < 1024.0:
            return f"{num:.2f} {unit}"
        num /= 1024.0
    return f"{num:.2f} PiB"

@torch.no_grad()
def profile_one_module(cur_module, batch_dict, *, reset_peak: bool = False):
    """
    Return:
      batch_dict_out,
      dt_ms,
      peak_inc_alloc_bytes, alloc_delta_bytes,
      peak_inc_reserved_bytes, reserved_delta_bytes
    """
    if torch.cuda.is_available() and reset_peak:
        torch.cuda.reset_peak_memory_stats()

    if torch.cuda.is_available():
        before_alloc = torch.cuda.memory_allocated()
        before_reserved = torch.cuda.memory_reserved()

    _cuda_sync()
    t0 = time.perf_counter()
    batch_dict_out = cur_module(batch_dict)
    _cuda_sync()
    dt_ms = (time.perf_counter() - t0) * 1000.0

    if torch.cuda.is_available():
        after_alloc = torch.cuda.memory_allocated()
        after_reserved = torch.cuda.memory_reserved()

        peak_alloc = torch.cuda.max_memory_allocated()
        peak_reserved = torch.cuda.max_memory_reserved()

        peak_inc_alloc = int(max(0, peak_alloc - before_alloc))
        alloc_delta = int(after_alloc - before_alloc)

        peak_inc_reserved = int(max(0, peak_reserved - before_reserved))
        reserved_delta = int(after_reserved - before_reserved)
    else:
        peak_inc_alloc = alloc_delta = 0
        peak_inc_reserved = reserved_delta = 0

    return batch_dict_out, dt_ms, peak_inc_alloc, alloc_delta, peak_inc_reserved, reserved_delta


@torch.no_grad()
def profile_modules(module_list, module_topology, batch_dict, name_fn=None):
    """
    Returns:
      batch_dict_out,
      per_mod[name] -> {
        lat_ms,
        peak_inc_alloc, alloc_delta,
        peak_inc_reserved, reserved_delta
      }
      total -> {
        lat_ms,
        peak_inc_alloc, alloc_delta,
        peak_inc_reserved, reserved_delta
      }
    """
    if name_fn is None:
        def name_fn(i, m):
            if module_topology is not None and i < len(module_topology):
                return str(module_topology[i])
            return getattr(m, "name", f"{i}_{m.__class__.__name__}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()  # IMPORTANT: only once for the whole forward
        total_before_alloc = torch.cuda.memory_allocated()
        total_before_reserved = torch.cuda.memory_reserved()

    _cuda_sync()
    total_t0 = time.perf_counter()

    per_mod = {}
    for i, cur_module in enumerate(module_list):
        name = name_fn(i, cur_module)

        # DO NOT reset peak inside per-module measurement; it would break total peak.
        batch_dict, dt_ms, peak_inc_a, delta_a, peak_inc_r, delta_r = profile_one_module(
            cur_module, batch_dict, reset_peak=False
        )

        per_mod[name] = {
            "lat_ms": dt_ms,
            "peak_inc_alloc": peak_inc_a,
            "alloc_delta": delta_a,
            "peak_inc_reserved": peak_inc_r,
            "reserved_delta": delta_r,
        }

    _cuda_sync()
    total_lat_ms = (time.perf_counter() - total_t0) * 1000.0

    if torch.cuda.is_available():
        total_after_alloc = torch.cuda.memory_allocated()
        total_after_reserved = torch.cuda.memory_reserved()

        total_peak_alloc = torch.cuda.max_memory_allocated()
        total_peak_reserved = torch.cuda.max_memory_reserved()

        total_peak_inc_alloc = int(max(0, total_peak_alloc - total_before_alloc))
        total_alloc_delta = int(total_after_alloc - total_before_alloc)

        total_peak_inc_reserved = int(max(0, total_peak_reserved - total_before_reserved))
        total_reserved_delta = int(total_after_reserved - total_before_reserved)
    else:
        total_peak_inc_alloc = total_alloc_delta = 0
        total_peak_inc_reserved = total_reserved_delta = 0

    total = {
        "lat_ms": total_lat_ms,
        "peak_inc_alloc": total_peak_inc_alloc,
        "alloc_delta": total_alloc_delta,
        "peak_inc_reserved": total_peak_inc_reserved,
        "reserved_delta": total_reserved_delta,
    }
    return batch_dict, per_mod, total


def print_total_profile(total: dict):
    if total is None:
        print("\n[Total] total=None")
        return

    print("\n[Total]")
    print(f"  latency: {total['lat_ms']:.2f} ms")
    print(f"  peak_alloc+:    {fmt_bytes(int(total['peak_inc_alloc']))}")
    print(f"  Δalloc:         {fmt_bytes(int(total['alloc_delta']))}")
    print(f"  peak_reserved+: {fmt_bytes(int(total['peak_inc_reserved']))}")
    print(f"  Δreserved:      {fmt_bytes(int(total['reserved_delta']))}")


def print_profile(per_mod: dict, topk: Optional[int] = None):
    items = list(per_mod.items())
    if topk is not None:
        items = sorted(items, key=lambda kv: kv[1].get("lat_ms", 0.0), reverse=True)[:topk]

    print("\n[Per-module]")
    for name, m in items:
        print(
            f"  {name:30s} | "
            f"{m['lat_ms']:8.2f} ms | "
            f"peakA+ {fmt_bytes(int(m['peak_inc_alloc'])):>10s} | "
            f"ΔA {fmt_bytes(int(m['alloc_delta'])):>10s} | "
            f"peakR+ {fmt_bytes(int(m['peak_inc_reserved'])):>10s} | "
            f"ΔR {fmt_bytes(int(m['reserved_delta'])):>10s}"
        )
        
def save_feature(vis_feat, batch_dict, module_name: str):
    """
    Save a single frame's feature map for a given module if VIS_FEAT config matches.
    Expected VIS_FEAT has fields: MODULE, FRAME_ID, (optional) SAVE_DIR.
    """
    if vis_feat is None:
        return

    # only act on the configured module
    if module_name != vis_feat.MODULE:
        return

    # pick feature tensor by module
    if module_name == "map_to_bev_module":
        feature = batch_dict.get("spatial_features", None)
    elif module_name == "backbone_2d":
        feature = batch_dict.get("spatial_features_2d", None)
    elif module_name == "fuser":
        feature = batch_dict.get()    
    else:
        # not supported / not needed
        return

    if feature is None:
        return

    idx = frame_ids_list.index(target_id)

    save_dir = getattr(vis_feat, "SAVE_DIR", "/scratch/sding32/weigths")
    save_path = f"{save_dir}/{target_id}_{module_name}_feature.pt"

    print(f"******save {module_name} feature map******")
    torch.save(feature[idx].cpu(), save_path)
