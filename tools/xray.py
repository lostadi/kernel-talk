"""
tools/xray.py
──────────────
Filesystem X-Ray

The crown jewel of Kernel-Talk's UX.

A user sees `/sys/class/net/wlan0/operstate` and gets back:
  "This attribute is served by net/core/net-sysfs.c, function operstate_show().
   It reads net_device.operstate, a u8 field that maps to the IF_OPER_* enum
   defined in include/uapi/linux/if.h. Currently the value is 'up' (4)."

That's the Logos/Eros synthesis in action — static source mapping fused
with live drgn state.

The X-Ray works in three stages:
  1. Pattern Matching — known /sys and /proc paths have predictable source locations.
     We maintain a hand-curated map of the most common ones.
  2. Vector Search — for unknown paths, we construct a semantic query and
     search the Mirror for likely handlers (show_*, read_*, store_* functions).
  3. Live Probe (optional) — if drgn is available, read the actual current value.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.mirror.store import KernelStore, HybridResult
from core.probe.drgn_bridge import DrgnBridge, LiveSnapshot


# ─── Result ───────────────────────────────────────────────────────────────────

@dataclass
class XRayResult:
    """The output of a filesystem X-Ray scan."""
    path: str                       # The /sys or /proc path queried
    source_files: list[str]         # Likely source files (e.g., "net/core/net-sysfs.c")
    handler_functions: list[str]    # The show/read/store functions responsible
    data_structures: list[str]      # Kernel structs involved
    description: str                # Human-readable explanation
    live_value: str | None = None   # Current value from drgn (if available)
    confidence: str = "medium"      # "high" (pattern match) | "medium" (vector) | "low"

    def pretty(self) -> str:
        lines = [
            f"X-RAY: {self.path}",
            f"Confidence: {self.confidence}",
            "",
            f"Source: {', '.join(self.source_files) or 'unknown'}",
            f"Handlers: {', '.join(self.handler_functions) or 'unknown'}",
            f"Structs: {', '.join(self.data_structures) or 'unknown'}",
        ]
        if self.live_value is not None:
            lines.append(f"Live value: {self.live_value}")
        lines.append("")
        lines.append(self.description)
        return "\n".join(lines)


# ─── Known Path Patterns ──────────────────────────────────────────────────────
# A curated map of common /sys and /proc paths to their source locations.
# This gives us fast, high-confidence answers for the most common queries.
# Format: regex_pattern → {"files": [...], "funcs": [...], "structs": [...], "desc": "..."}

KNOWN_PATHS: list[tuple[str, dict[str, Any]]] = [
    # ── Networking ──────────────────────────────────────────────────────────
    (r"/sys/class/net/\w+/operstate", {
        "files":   ["net/core/net-sysfs.c"],
        "funcs":   ["operstate_show"],
        "structs": ["net_device"],
        "desc":    "Network interface operational state. Reads net_device.operstate "
                   "which maps to IF_OPER_* enum values (0=UNKNOWN, 6=UP). "
                   "Written by carrier change events from the driver layer.",
    }),
    (r"/sys/class/net/\w+/speed", {
        "files":   ["net/core/net-sysfs.c"],
        "funcs":   ["speed_show"],
        "structs": ["net_device", "ethtool_link_ksettings"],
        "desc":    "Link speed in Mbps. Calls ethtool_get_link_ksettings() which "
                   "dispatches to the driver's .get_link_ksettings ndo operation.",
    }),
    (r"/sys/class/net/\w+/mtu", {
        "files":   ["net/core/net-sysfs.c"],
        "funcs":   ["mtu_show", "mtu_store"],
        "structs": ["net_device"],
        "desc":    "Maximum Transmission Unit. Reads/writes net_device.mtu. "
                   "Writes call dev_set_mtu() which validates against driver min/max.",
    }),
    (r"/sys/class/net/\w+/tx_queue_len", {
        "files":   ["net/core/net-sysfs.c"],
        "funcs":   ["tx_queue_len_show", "tx_queue_len_store"],
        "structs": ["net_device"],
        "desc":    "Transmit queue length. Controls how many packets can be queued "
                   "before traffic control kicks in. Maps to net_device.tx_queue_len.",
    }),

    # ── CPU / Scheduler ─────────────────────────────────────────────────────
    (r"/sys/devices/system/cpu/cpu\d+/cpufreq/scaling_cur_freq", {
        "files":   ["drivers/cpufreq/cpufreq.c"],
        "funcs":   ["show_scaling_cur_freq"],
        "structs": ["cpufreq_policy", "cpufreq_driver"],
        "desc":    "Current CPU frequency in kHz. Reads cpufreq_policy.cur. "
                   "Updated by the cpufreq driver's ->get() callback each time "
                   "the frequency governor requests a change.",
    }),
    (r"/sys/devices/system/cpu/cpu\d+/topology/core_id", {
        "files":   ["drivers/base/topology.c"],
        "funcs":   ["core_id_show"],
        "structs": ["cpu_topology"],
        "desc":    "Physical core ID of this logical CPU. Read from ACPI/CPUID tables "
                   "during boot and stored in per-cpu cpu_topology struct.",
    }),
    (r"/proc/schedstat", {
        "files":   ["kernel/sched/stats.c"],
        "funcs":   ["show_schedstat"],
        "structs": ["rq", "sched_info"],
        "desc":    "Scheduler statistics per CPU and runqueue. Shows yield/schedule "
                   "counts, wait times, and running times accumulated since boot.",
    }),

    # ── Memory ──────────────────────────────────────────────────────────────
    (r"/proc/meminfo", {
        "files":   ["fs/proc/meminfo.c"],
        "funcs":   ["meminfo_proc_show"],
        "structs": ["sysinfo", "vm_stat"],
        "desc":    "System memory statistics. Reads from zone->vm_stat arrays and "
                   "global vmstat counters. Most fields are calculated on-demand "
                   "by summing per-zone and per-cpu counters.",
    }),
    (r"/proc/\d+/maps", {
        "files":   ["fs/proc/task_mmu.c"],
        "funcs":   ["show_map", "proc_pid_maps_open"],
        "structs": ["vm_area_struct", "mm_struct"],
        "desc":    "Virtual memory areas for a process. Walks mm->mmap VMA list "
                   "(a sorted linked list of vm_area_struct). Each line represents "
                   "one contiguous virtual memory region with permissions and backing.",
    }),
    (r"/proc/\d+/status", {
        "files":   ["fs/proc/array.c"],
        "funcs":   ["proc_pid_status"],
        "structs": ["task_struct", "mm_struct", "cred"],
        "desc":    "Process status information. Reads from task_struct fields directly: "
                   "VmPeak/VmRSS from mm_struct, threads from signal->nr_threads, "
                   "UIDs from task credentials (cred struct).",
    }),
    (r"/proc/\d+/smaps", {
        "files":   ["fs/proc/task_mmu.c"],
        "funcs":   ["show_smap"],
        "structs": ["vm_area_struct", "mem_size_stats"],
        "desc":    "Detailed memory maps with RSS, PSS (proportional), and shared/private "
                   "accounting. More expensive than /proc/pid/maps — iterates each VMA "
                   "and calls smaps_account() to walk page tables.",
    }),

    # ── Processes ────────────────────────────────────────────────────────────
    (r"/proc/\d+/stat", {
        "files":   ["fs/proc/array.c"],
        "funcs":   ["do_task_stat"],
        "structs": ["task_struct", "signal_struct"],
        "desc":    "Process statistics in single-line format (used by ps, top). "
                   "Reads CPU times from task->utime/stime, state from task->__state, "
                   "and RSS from mm_struct. The format is documented in proc(5).",
    }),
    (r"/proc/\d+/cmdline", {
        "files":   ["fs/proc/base.c"],
        "funcs":   ["proc_pid_cmdline_read"],
        "structs": ["mm_struct", "task_struct"],
        "desc":    "Null-separated argv for the process. Reads from mm->arg_start "
                   "to mm->arg_end in the process's address space via access_process_vm().",
    }),

    # ── Block I/O ────────────────────────────────────────────────────────────
    (r"/sys/block/\w+/queue/scheduler", {
        "files":   ["block/elevator.c", "block/blk-sysfs.c"],
        "funcs":   ["elv_iosched_show", "elv_iosched_store"],
        "structs": ["elevator_type", "request_queue"],
        "desc":    "I/O scheduler for this block device. Shows [current] alternatives. "
                   "Writing a scheduler name calls elevator_switch() which tears down "
                   "the old elevator and initializes the new one.",
    }),
    (r"/sys/block/\w+/stat", {
        "files":   ["block/blk-sysfs.c"],
        "funcs":   ["queue_stat_show"],
        "structs": ["disk_stats", "gendisk"],
        "desc":    "Block device I/O statistics: reads/writes completed, sectors, "
                   "time spent. Accumulated in disk_stats per CPU, summed on read.",
    }),

    # ── Power Management ─────────────────────────────────────────────────────
    (r"/sys/class/power_supply/\w+/capacity", {
        "files":   ["drivers/power/supply/power_supply_sysfs.c"],
        "funcs":   ["power_supply_show_property"],
        "structs": ["power_supply", "power_supply_desc"],
        "desc":    "Battery capacity percentage. Calls driver's .get_property() "
                   "with POWER_SUPPLY_PROP_CAPACITY. Driver reads from hardware gauge IC.",
    }),
]

_COMPILED_PATTERNS = [
    (re.compile(pattern), info)
    for pattern, info in KNOWN_PATHS
]


# ─── X-Ray Engine ─────────────────────────────────────────────────────────────

class XRay:
    """
    Filesystem X-Ray: maps /sys and /proc paths to kernel source code.

    Usage:
        xray = XRay(store=my_kernel_store, probe=my_drgn_bridge)
        result = xray.scan("/sys/class/net/wlan0/operstate")
        print(result.pretty())
    """

    def __init__(
        self,
        store: KernelStore | None = None,
        probe: DrgnBridge | None = None,
    ):
        """
        store: KernelStore for vector-backed fallback lookup.
               If None, only pattern matching is used.
        probe: DrgnBridge for live value reading.
               If None, live values are not fetched.
        """
        self.store = store
        self.probe = probe

    def scan(self, path: str) -> XRayResult:
        """
        Main entry point. Try pattern matching first, fall back to vector search.
        """
        path = path.strip()

        # Stage 1: High-confidence pattern match
        result = self._pattern_match(path)
        if result:
            # Stage 3: Optionally enrich with live value
            if self.probe and self.probe.is_available():
                result.live_value = self._read_live_value(path)
            return result

        # Stage 2: Vector search (requires Mirror to be indexed)
        if self.store:
            return self._vector_scan(path)

        # No store, no pattern — return what we can
        return XRayResult(
            path=path,
            source_files=[],
            handler_functions=[],
            data_structures=[],
            description=(
                "Path not recognized in pattern database and no Mirror index is loaded.\n"
                "Run `ktalk index` to build the Mirror, then retry."
            ),
            confidence="low",
        )

    def _pattern_match(self, path: str) -> XRayResult | None:
        """Try each known pattern against the path."""
        for pattern, info in _COMPILED_PATTERNS:
            if pattern.fullmatch(path):
                return XRayResult(
                    path=path,
                    source_files=info["files"],
                    handler_functions=info["funcs"],
                    data_structures=info["structs"],
                    description=info["desc"],
                    confidence="high",
                )
        return None

    def _vector_scan(self, path: str) -> XRayResult:
        """
        Construct a semantic query from the path and search the Mirror.
        """
        parts = path.strip("/").split("/")
        attr_name = parts[-1]
        subsystem_hint = parts[1] if len(parts) > 2 else ""

        query = (
            f"sysfs attribute {attr_name.replace('_', ' ')} "
            f"show read write store handler function "
            f"{subsystem_hint} kernel"
        )

        results = self.store.vector_search(query, top_k=5)

        if not results:
            return XRayResult(
                path=path,
                source_files=[],
                handler_functions=[],
                data_structures=[],
                description="No matching kernel code found in the Mirror index.",
                confidence="low",
            )

        source_files = list(dict.fromkeys(r.node.file_path for r in results))
        handler_functions = [r.node.symbol_name for r in results if r.node.node_type == "function"]
        data_structures = [r.node.symbol_name for r in results if r.node.node_type in ("struct", "union")]

        # Build a description from the top result's docstring
        top = results[0]
        desc = top.node.docstring or f"Handler function in {top.node.file_path}"
        desc += f"\n\n[Vector search result, confidence: {results[0].score:.2f}]"

        live_value = None
        if self.probe and self.probe.is_available():
            live_value = self._read_live_value(path)

        return XRayResult(
            path=path,
            source_files=source_files,
            handler_functions=handler_functions,
            data_structures=data_structures,
            description=desc,
            live_value=live_value,
            confidence="medium",
        )

    def _read_live_value(self, path: str) -> str | None:
        """
        Attempt to read the actual current value of a /sys or /proc path.
        This is the simplest form of live probing — just read the file.
        For richer structural data, DrgnBridge.read_struct() is used.
        """
        try:
            fpath = Path(path)
            if fpath.exists() and fpath.is_file():
                content = fpath.read_text(errors="replace").strip()
                return content[:256]  # Cap at 256 chars
        except (PermissionError, OSError):
            pass
        return None

    def batch_scan(self, paths: list[str]) -> list[XRayResult]:
        """Scan multiple paths at once."""
        return [self.scan(p) for p in paths]

    def list_known_patterns(self) -> list[str]:
        """Return all path patterns in the knowledge base."""
        return [pattern for pattern, _ in KNOWN_PATHS]
