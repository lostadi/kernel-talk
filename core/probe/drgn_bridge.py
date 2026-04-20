"""
core/probe/drgn_bridge.py
──────────────────────────
The Probe — Live Kernel Memory Bridge

This is where Theory meets Reality.

drgn (github.com/osandov/drgn) is a programmable debugger from Meta that
reads live kernel memory via /proc/kcore and kernel debug symbols (DWARF).
Unlike ptrace-based debuggers, drgn does NOT stop execution — it reads
kernel memory the way a kernel module would, safely, non-intrusively.

The core insight: every object the Mirror indexed as static C code
(struct task_struct, struct mm_struct, etc.) has *live instances* in kernel
memory right now. drgn can give us those instances. We can ask:

  "What are the current values of task->state, task->pid, task->comm?"

and get ground truth — not what the code *says* might happen, but what
*is* happening on this exact machine at this exact moment.

This transforms the system from a code search engine into a genuine
Digital Twin: the Mirror provides the theory (static structure), the
Probe provides the reality (live state), and the Synthesizer merges them.

Requirements:
  - Linux only (reads /proc/kcore)
  - Root or CAP_SYS_PTRACE capability
  - Kernel compiled with CONFIG_PROC_KCORE=y (most distros enable this)
  - Debug symbols: vmlinux with DWARF info, or kernel-debuginfo package
  - pip install drgn

Safety: drgn is READ-ONLY — it cannot write to kernel memory. This makes
it safe to use on production systems (Meta uses it in prod). We never modify
kernel state.
"""

from __future__ import annotations

import os
import platform
import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Iterator


# ─── Data Model ───────────────────────────────────────────────────────────────

@dataclass
class LiveField:
    """A single field from a live kernel struct instance."""
    name: str
    value: str      # String representation (could be int, pointer address, enum name)
    c_type: str     # e.g. "unsigned int", "pid_t", "char[16]"
    offset: int     # byte offset within the struct


@dataclass
class LiveSnapshot:
    """
    A snapshot of live kernel state at a specific moment.

    This is the Eros side of the Logos/Eros synthesis — the actual runtime
    reality that the Mirror's static code is supposed to describe.

    struct_name:  e.g. "task_struct"
    instance_id:  e.g. pointer address "0xffff888100a58000"
    fields:       The actual field values
    captured_at:  Timestamp of the snapshot
    context:      Free-form notes (e.g. "PID 1234, comm='nginx'")
    """
    struct_name: str
    instance_id: str
    fields: list[LiveField]
    captured_at: str = field(default_factory=lambda: datetime.now().isoformat())
    context: str = ""

    def to_text(self) -> str:
        """Render snapshot as text for LLM synthesis context."""
        lines = [
            f"LIVE SNAPSHOT: {self.struct_name} @ {self.instance_id}",
            f"Captured: {self.captured_at}",
        ]
        if self.context:
            lines.append(f"Context: {self.context}")
        lines.append("")
        for f in self.fields:
            lines.append(f"  .{f.name} ({f.c_type}) = {f.value}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "struct_name": self.struct_name,
            "instance_id": self.instance_id,
            "captured_at": self.captured_at,
            "context": self.context,
            "fields": [
                {"name": f.name, "value": f.value, "c_type": f.c_type, "offset": f.offset}
                for f in self.fields
            ],
        }


@dataclass
class ProcessSnapshot:
    """High-level snapshot of a running process (from task_struct)."""
    pid: int
    ppid: int
    comm: str           # process name (15 chars max in kernel)
    state: str          # TASK_RUNNING, TASK_INTERRUPTIBLE, etc.
    flags: str          # hex flags
    priority: int
    cpu: int
    mm_ptr: str         # memory descriptor pointer address
    files_ptr: str      # open file table pointer

    def to_text(self) -> str:
        return (
            f"Process [{self.pid}] {self.comm}\n"
            f"  ppid={self.ppid}, state={self.state}, cpu={self.cpu}, prio={self.priority}\n"
            f"  flags={self.flags}, mm={self.mm_ptr}, files={self.files_ptr}"
        )


# ─── drgn Availability Check ──────────────────────────────────────────────────

def _check_drgn() -> tuple[bool, str]:
    """Check if drgn is available and we're on Linux with kernel access."""
    if platform.system() != "Linux":
        return False, "drgn requires Linux"
    try:
        import drgn  # noqa
    except ImportError:
        return False, "drgn not installed: pip install drgn"
    if not os.path.exists("/proc/kcore"):
        return False, "/proc/kcore not found: kernel may lack CONFIG_PROC_KCORE"
    if os.geteuid() != 0:
        # Non-root might still work with CAP_SYS_PTRACE — try anyway
        pass
    return True, "ok"


# ─── The Bridge ───────────────────────────────────────────────────────────────

class DrgnBridge:
    """
    Interface to live kernel memory via drgn.

    This class is intentionally lazy — the drgn.Program object is expensive
    to create (it reads /proc/kcore and parses debug symbols). We create it
    once on first use and cache it.

    On non-Linux or without root, most methods return empty/mock results
    with an explanation, rather than crashing. This lets the rest of the
    system work on macOS during development.
    """

    def __init__(self, vmlinux_path: str | None = None):
        """
        vmlinux_path: Path to the vmlinux ELF with debug symbols.
                      If None, drgn tries to auto-locate it.
                      On Debian/Ubuntu: /usr/lib/debug/boot/vmlinux-$(uname -r)
                      On Arch: /usr/lib/modules/$(uname -r)/build/vmlinux
        """
        self.vmlinux_path = vmlinux_path
        self._prog = None   # drgn.Program — created lazily
        self._available, self._reason = _check_drgn()

    def _get_prog(self):
        """Lazy-init the drgn Program object."""
        if self._prog is not None:
            return self._prog

        if not self._available:
            raise RuntimeError(f"drgn not available: {self._reason}")

        import drgn
        from drgn.helpers.linux import task_state_to_char

        if self.vmlinux_path:
            self._prog = drgn.Program()
            self._prog.load_debug_info([self.vmlinux_path])
            self._prog.set_core_dump("/proc/kcore")
        else:
            # Let drgn auto-detect (works on most modern distros)
            self._prog = drgn.program_from_kernel()

        return self._prog

    # ── High-Level Probes ─────────────────────────────────────────────────────

    def list_processes(self, max_tasks: int = 64) -> list[ProcessSnapshot]:
        """
        Read the live process list from kernel memory.

        In the kernel, all processes are linked via task_struct.tasks,
        a doubly-linked list anchored at `init_task`. We walk it.

        This is the "telepathy" demo: we're reading the same data structures
        that `ps` reads via /proc, but directly from kernel memory, and we
        can see fields that /proc doesn't expose.
        """
        if not self._available:
            return self._mock_process_list()

        try:
            import drgn
            from drgn.helpers.linux import for_each_task

            prog = self._get_prog()
            snapshots = []

            for task in for_each_task(prog):
                try:
                    # F-20: task.__state was added in Linux 5.14 (commit
                    # 2f064a59a11f).  Before that the field is plain task.state.
                    # Try __state first (modern kernel), fall back to state.
                    try:
                        raw_state = int(task.__state)
                    except (AttributeError, drgn.ObjectAbsentError):
                        raw_state = int(task.state)

                    snapshot = ProcessSnapshot(
                        pid=int(task.pid),
                        ppid=int(task.real_parent.pid),
                        comm=task.comm.string_().decode("utf-8", errors="replace"),
                        state=_task_state_name(raw_state),
                        flags=hex(int(task.flags)),
                        priority=int(task.prio),
                        cpu=int(task.cpu) if hasattr(task, "cpu") else -1,
                        mm_ptr=hex(int(task.mm)) if task.mm else "0x0 (kernel thread)",
                        files_ptr=hex(int(task.files)) if task.files else "0x0",
                    )
                    snapshots.append(snapshot)
                    if len(snapshots) >= max_tasks:
                        break
                except drgn.FaultError:
                    # Some tasks may be in inconsistent state — skip gracefully
                    continue

            return snapshots

        except Exception as e:
            print(f"[probe] list_processes failed: {e}")
            return []

    def read_struct(
        self,
        struct_name: str,
        address: str,
        fields: list[str] | None = None,
    ) -> LiveSnapshot | None:
        """
        Read a specific kernel struct instance by memory address.

        struct_name: e.g. "task_struct"
        address:     hex string e.g. "0xffff888100a58000"
        fields:      which fields to read (None = try common fields)

        This is the precise inverse of what the Mirror does: the Mirror
        takes source code and indexes it. The Probe takes a live memory
        address and reads what's actually there.
        """
        if not self._available:
            return None

        try:
            import drgn

            prog = self._get_prog()
            addr = int(address, 16)
            obj = prog.object(f"struct {struct_name} *", value=addr)

            live_fields = []
            target_fields = fields or _default_fields(struct_name)

            for field_name in target_fields:
                try:
                    field_obj = getattr(obj, field_name)
                    live_fields.append(LiveField(
                        name=field_name,
                        value=str(field_obj.value_()),
                        c_type=str(field_obj.type_),
                        offset=prog.type(f"struct {struct_name}").member(field_name).offset // 8,
                    ))
                except (AttributeError, drgn.ObjectAbsentError):
                    continue

            return LiveSnapshot(
                struct_name=struct_name,
                instance_id=address,
                fields=live_fields,
                context=f"Read via drgn on {platform.node()}",
            )

        except Exception as e:
            print(f"[probe] read_struct {struct_name}@{address} failed: {e}")
            return None

    def get_sysfs_attr_struct(self, sysfs_path: str) -> LiveSnapshot | None:
        """
        Given a /sys path, attempt to find and read the kernel_attr or
        device_attribute struct that backs it.

        This is the Filesystem X-Ray's live component:
        - Static: Mirror finds the C source file that writes this sysfs attr
        - Live: Probe reads the actual kobject/attribute at runtime
        """
        # TODO: implement via drgn sysfs kobject walking
        # This is complex — sysfs is backed by kobject trees and attribute
        # groups. For now, return a descriptive stub.
        print(f"[probe] sysfs attribute walking not yet implemented for: {sysfs_path}")
        return None

    def read_runqueue(self, cpu: int = 0) -> LiveSnapshot | None:
        """
        Read the scheduler's run queue for a specific CPU.
        struct rq contains the currently running task, load, nr_running, etc.

        This answers "why is CPU X at 100%?" directly from kernel memory.
        """
        if not self._available:
            return None

        try:
            import drgn
        except ImportError:
            return None

        try:
            from drgn.helpers.linux import cpu_rq
        except ImportError:
            print(
                "[probe] read_runqueue: cpu_rq helper not found in drgn.helpers.linux. "
                "Your drgn version may be too old — try: pip install --upgrade drgn"
            )
            return None

        try:
            prog = self._get_prog()
            rq = cpu_rq(prog, cpu)

            fields = []
            for fname in ["nr_running", "nr_switches", "clock", "load"]:
                try:
                    val = getattr(rq, fname)
                    fields.append(LiveField(
                        name=fname,
                        value=str(val.value_()),
                        c_type=str(val.type_),
                        offset=0,
                    ))
                except Exception:
                    continue

            # Also read the currently running task
            try:
                curr = rq.curr
                fields.append(LiveField(
                    name="curr->comm",
                    value=curr.comm.string_().decode("utf-8", errors="replace"),
                    c_type="char[16]",
                    offset=0,
                ))
                fields.append(LiveField(
                    name="curr->pid",
                    value=str(int(curr.pid)),
                    c_type="pid_t",
                    offset=0,
                ))
            except Exception:
                pass

            return LiveSnapshot(
                struct_name="rq",
                instance_id=f"cpu{cpu}",
                fields=fields,
                context=f"Run queue for CPU {cpu}",
            )

        except Exception as e:
            print(f"[probe] read_runqueue cpu={cpu} failed: {e}")
            return None

    # ── Availability ──────────────────────────────────────────────────────────

    def is_available(self) -> bool:
        return self._available

    def status(self) -> dict:
        return {
            "available": self._available,
            "reason": self._reason,
            "platform": platform.system(),
            "kernel": platform.release() if platform.system() == "Linux" else "N/A",
            "uid": os.geteuid() if platform.system() == "Linux" else -1,
        }

    # ── Mock Data (for development on macOS/non-Linux) ────────────────────────

    def _mock_process_list(self) -> list[ProcessSnapshot]:
        """Return synthetic process data for testing without a Linux kernel."""
        return [
            ProcessSnapshot(pid=1,    ppid=0, comm="systemd",  state="TASK_INTERRUPTIBLE",
                            flags="0x400000", priority=20, cpu=0,
                            mm_ptr="0xffff888100a58000", files_ptr="0xffff888100a59000"),
            ProcessSnapshot(pid=42,   ppid=1, comm="kworker",  state="TASK_RUNNING",
                            flags="0x4000100", priority=20, cpu=1,
                            mm_ptr="0x0 (kernel thread)", files_ptr="0x0"),
            ProcessSnapshot(pid=1337, ppid=1, comm="nginx",    state="TASK_INTERRUPTIBLE",
                            flags="0x400000", priority=20, cpu=2,
                            mm_ptr="0xffff888200b68000", files_ptr="0xffff888200b69000"),
        ]


# ─── Utilities ────────────────────────────────────────────────────────────────

def _task_state_name(state: int) -> str:
    """Convert numeric task state to human-readable name."""
    # Linux 5.14+ uses __state instead of state, values shifted
    state_map = {
        0x00: "TASK_RUNNING",
        0x01: "TASK_INTERRUPTIBLE",
        0x02: "TASK_UNINTERRUPTIBLE",
        0x04: "TASK_STOPPED",
        0x08: "TASK_TRACED",
        0x10: "EXIT_DEAD",
        0x20: "EXIT_ZOMBIE",
        0x40: "TASK_PARKED",
        0x80: "TASK_DEAD",
    }
    return state_map.get(state, f"UNKNOWN(0x{state:x})")


def _default_fields(struct_name: str) -> list[str]:
    """Return the most interesting fields for well-known kernel structs."""
    defaults = {
        # F-20: include both __state (≥5.14) and state (<5.14).
        # read_struct skips fields that raise AttributeError/ObjectAbsentError,
        # so having both here is safe — the wrong one is silently dropped.
        "task_struct": [
            "pid", "tgid", "comm",
            "__state",           # Linux ≥ 5.14
            "state",             # Linux < 5.14
            "flags",
            "prio", "normal_prio", "policy", "exit_code",
        ],
        "mm_struct": [
            "total_vm", "locked_vm", "pinned_vm", "data_vm",
            "exec_vm", "stack_vm", "mmap_base", "task_size",
        ],
        "file": [
            "f_count", "f_flags", "f_mode", "f_pos",
        ],
        "socket": [
            "type", "state", "flags",
        ],
        "sk_buff": [
            "len", "data_len", "pkt_type", "priority", "protocol",
        ],
        "net_device": [
            "name", "flags", "mtu", "type", "state", "operstate",
        ],
    }
    return defaults.get(struct_name, [])
