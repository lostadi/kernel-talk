# Kernel-Talk

> *"Telepathy for the kernel."*

A Digital Twin for the Linux operating system. Bridges the gap between
**static source code** (Theory / Logos) and **live runtime memory** (Reality / Eros)
to answer the question that static analysis alone can never answer:

**Not "what does this code do?" — but "why is my machine doing *this*, right now?"**

---

## The Problem

The Linux kernel has grown to over **30 million lines of C**. Users interact
with cryptic virtual filesystem paths like `/sys/class/net/wlan0/operstate` or
`/proc/meminfo`, but cannot connect them to the code generating them. Engineers
read source code but cannot see what's actually executing. The gap between
Theory and Reality is where all hard debugging lives.

Existing tools give you one side or the other:
- `grep` / `cscope` — static code search with no runtime awareness
- `strace` / `perf` — runtime data with no source-level context
- `/proc` / `/sys` — sanitized observables, not ground-truth kernel state

Kernel-Talk collapses that gap.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        KERNEL-TALK                              │
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │  THE MIRROR  │    │  THE PROBE   │    │  THE SYNTHESIS   │   │
│  │   (Theory)   │    │  (Reality)   │    │   (Understanding)│   │
│  │              │    │              │    │                  │   │
│  │ tree-sitter  │    │    drgn      │    │  Hybrid Search   │   │
│  │  AST Parser  │    │  Live Kernel │    │  (vector+graph)  │   │
│  │      +       │    │  Memory Read │    │       +          │   │
│  │  CodeBERT    │    │              │    │   LLM Synthesis  │   │
│  │  Embeddings  │    │  task_struct │    │                  │   │
│  │      +       │    │  mm_struct   │    │  Theory + Reality│   │
│  │  NetworkX    │    │  rq, sk_buff │    │  → Explanation   │   │
│  │  KnowledgeG  │    │  net_device  │    │                  │   │
│  │              │    │  ...         │    │                  │   │
│  └──────────────┘    └──────────────┘    └──────────────────┘   │
│        │                    │                      │            │
│    ChromaDB             /proc/kcore            Ollama /         │
│    GraphML              (read-only)            OpenAI /         │
│                                                Anthropic        │
└─────────────────────────────────────────────────────────────────┘
```

### The Mirror (Static)

Parses the kernel source tree using **tree-sitter** at the AST level — not naive
text chunking. Extracts semantic units: functions, structs, enums, macros. Each
becomes a `CodeNode`: the atomic unit of the system, simultaneously a vector
document and a graph node.

The knowledge graph (NetworkX MultiDiGraph) encodes:
- `CALLS` — function → function call edges
- `USES_STRUCT` — function → struct/union dependency edges
- `DEFINED_IN` — symbol → source file edges
- `INCLUDES` — file → header dependency edges

**Why graph-augmented from day one?** Vector similarity finds semantically
relevant nodes. Graph traversal expands those seeds into their architectural
neighborhood — callers, callees, referenced structs — giving the LLM the full
structural picture. You cannot retrofit graph structure onto flat embeddings.

### The Probe (Live)

Uses **drgn** (Meta's programmable kernel debugger) to safely read live kernel
memory objects via `/proc/kcore`. Read-only, non-intrusive — Meta runs it in
production. Captures ground-truth runtime state: field values, scheduler queues,
process tables, network device state.

Requires Linux with `CONFIG_PROC_KCORE=y` and root / `CAP_SYS_PTRACE`.
Degrades gracefully on macOS (returns mock data) for development.

### The Synthesis

Constructs a structured prompt fusing Theory (static code context) and Reality
(live drgn snapshots), then calls an LLM to generate a causal explanation —
not just code echoing, but genuine synthesis of mechanism and current state.

### The Filesystem X-Ray

Maps `/sys` and `/proc` paths back to their C source. Three-stage pipeline:
1. **Pattern Match** — curated high-confidence map for common paths
2. **Vector Search** — semantic query over Mirror for unknown paths
3. **Live Read** — actual current value from the filesystem

---

## Requirements

- **Python 3.10+**
- **Linux** (for live kernel probing via drgn) — all major distributions supported
- **macOS** is supported for static analysis / development (live probing unavailable)
- A Linux kernel source tree (download or distro package — see below)
- An LLM backend: [Ollama](https://ollama.ai) (local, recommended), OpenAI, or Anthropic

---

## Installation

### 1 · System Prerequisites

Install Python 3.10+, pip, git, and C build tools for your distribution.

> **Note:** The live kernel probe (`ktalk probe`) additionally requires kernel debug symbols.
> See [Debug Symbols](#debug-symbols-for-live-probing) below.

---

#### Ubuntu / Debian (apt / dpkg)

```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git build-essential
```

---

#### Arch Linux (pacman)

```bash
sudo pacman -Syu
sudo pacman -S --needed python python-pip git base-devel
```

---

#### Fedora / RHEL / AlmaLinux / Rocky Linux (dnf)

```bash
sudo dnf install -y python3 python3-pip python3-devel git gcc make
# python3-devel is required — several pip deps (tokenizers, chromadb)
# compile C/Rust extensions and need Python.h
```

---

#### openSUSE Tumbleweed / Leap (zypper)

```bash
sudo zypper refresh
sudo zypper install -y python3 python3-pip git gcc make
```

---

#### Gentoo (emerge / Portage)

```bash
sudo emerge --sync

# Set Python targets BEFORE emerging — Portage will reject packages
# for undeclared targets. Add to /etc/portage/make.conf:
echo 'PYTHON_TARGETS="python3_12"' | sudo tee -a /etc/portage/make.conf
echo 'PYTHON_SINGLE_TARGET="python3_12"' | sudo tee -a /etc/portage/make.conf
# Adjust python3_12 to whichever ≥ python3_10 your system supports.

sudo emerge -av dev-lang/python:3.12 dev-python/pip dev-vcs/git \
               sys-devel/gcc sys-devel/make dev-python/setuptools
```

Ensure Python 3.10+ is selected as the active interpreter:
```bash
eselect python list
eselect python set python3.12   # or whichever ≥ 3.10 you have
```

---

#### Alpine Linux (apk)

```bash
sudo apk update
sudo apk add python3 py3-pip git build-base
```

---

#### macOS (Homebrew) — development only

Live kernel probing (drgn) is Linux-only. All other features work on macOS.

```bash
brew install python git
```

---

### 2 · Clone and Set Up Python Environment

```bash
git clone https://github.com/lostadi/kernel-talk.git
cd kernel-talk

python3 -m venv .venv
source .venv/bin/activate      # Fish: source .venv/bin/activate.fish
                               # Windows (WSL recommended): .venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Or use the convenience script which activates the venv and exports defaults:

```bash
source activate.sh
```

---

### 3 · LLM Backend

Choose one (or more) of the following.

#### Ollama — local, private, no API key (recommended)

```bash
# Universal install script (Linux / macOS)
curl -fsSL https://ollama.ai/install.sh | sh

# Pull the default model
ollama pull deepseek-coder:6.7b

# Alternatively, a smaller model for lower-VRAM machines:
ollama pull deepseek-coder:1.3b
```

#### OpenAI

```bash
pip install openai
export OPENAI_API_KEY=sk-...
export KTALK_MODEL=openai:gpt-4o
```

#### Anthropic

```bash
pip install anthropic
export ANTHROPIC_API_KEY=sk-ant-...
export KTALK_MODEL=anthropic:claude-3-5-sonnet-20241022
```

---

### 4 · Live Kernel Probing — drgn (Linux only)

```bash
pip install drgn
```

`drgn` requires:
- Linux with `CONFIG_PROC_KCORE=y` (true on most distros by default)
- Root or `CAP_SYS_PTRACE`
- Kernel debug symbols (vmlinux with DWARF) — see next section

#### Debug Symbols for Live Probing

> Skip this section if you only want static analysis (`ktalk index` / `ktalk ask`).

##### Ubuntu

```bash
# 1. Install the keyring FIRST (it lives in the standard ubuntu repos)
sudo apt install -y ubuntu-dbgsym-keyring

# 2. Add the ddebs source — now apt trusts the repo
echo "deb http://ddebs.ubuntu.com $(lsb_release -cs) main restricted universe multiverse
deb http://ddebs.ubuntu.com $(lsb_release -cs)-updates main restricted universe multiverse" \
  | sudo tee /etc/apt/sources.list.d/ddebs.list

# 3. Update and install debug symbols for the running kernel
sudo apt update
sudo apt install -y linux-image-$(uname -r)-dbgsym
# vmlinux lives at: /usr/lib/debug/boot/vmlinux-$(uname -r)
```

##### Debian

```bash
# Debian ships debug symbols via a separate mirror
echo "deb http://debug.mirrors.debian.org/debian-debug/ $(lsb_release -cs)-debug main" \
  | sudo tee /etc/apt/sources.list.d/debian-debug.list
sudo apt update
sudo apt install -y linux-image-$(uname -r)-dbg
# vmlinux lives at: /usr/lib/debug/boot/vmlinux-$(uname -r)
```

##### Arch Linux

`linux-headers` only ships header files for module building — it does **not** contain DWARF debug info. Use one of these instead:

```bash
# Option A — debuginfod (easiest, no extra packages, fetches on demand)
export DEBUGINFOD_URLS="https://debuginfod.archlinux.org"
# drgn picks this up automatically. Add to ~/.bashrc / ~/.zshrc to persist.

# Option B — recompile your kernel with debug info
# In your kernel .config: CONFIG_DEBUG_INFO=y
# Then rebuild and install. vmlinux is at /usr/src/linux/vmlinux after build.

# Option C — AUR package with debug kernel
# yay -S linux-debug   (or paru -S linux-debug)
# vmlinux lives at: /usr/lib/modules/$(uname -r)-debug/vmlinux
```

##### Fedora / RHEL

```bash
sudo dnf debuginfo-install kernel
# vmlinux lives at: /usr/lib/debug/lib/modules/$(uname -r)/vmlinux
```

##### openSUSE

```bash
sudo zypper install kernel-default-debuginfo
```

##### Gentoo

Rebuild the kernel with `CONFIG_DEBUG_INFO=y` in your `.config`, or use:
```bash
sudo emerge sys-kernel/gentoo-kernel-bin   # ships with debug symbols
```

---

### 5 · Kernel Source Tree

You need a kernel source tree to build the Mirror index.

#### Option A — Download from kernel.org (any distro)

```bash
wget https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.6.tar.xz
tar -xf linux-6.6.tar.xz
# then: ktalk index --kernel ./linux-6.6
```

#### Option B — Distro source package

##### Ubuntu / Debian

```bash
sudo apt install linux-source
# Source extracted to /usr/src/linux-source-*.tar.bz2
sudo tar -xf /usr/src/linux-source-*.tar.bz2 -C /usr/src/
```

##### Arch Linux

```bash
# Modern Arch (devtools ≥ 1.0) — use pkgctl
sudo pacman -S devtools
pkgctl repo clone linux
# Source tree is in ./linux/

# Older Arch — use asp (if available)
# sudo pacman -S asp && asp export linux
```

##### Fedora

```bash
sudo dnf install fedpkg
fedpkg clone kernel
cd kernel && fedpkg sources
# Or install the source RPM directly:
sudo dnf download --source kernel
rpm -i kernel-*.src.rpm
# Source lands in ~/rpmbuild/SOURCES/
```

##### openSUSE

```bash
sudo zypper install kernel-source
# Source at: /usr/src/linux-$(uname -r)/
```

##### Gentoo

```bash
sudo emerge sys-kernel/gentoo-sources
# Source at: /usr/src/linux
```

---

## Usage

### Quick Start

```bash
# Activate the environment (sets defaults, creates 'ktalk' alias)
source activate.sh

# Index the scheduler subsystem (fast — good first test)
ktalk index --kernel /path/to/linux --subsystem kernel/sched

# Ask a question
ktalk ask "why does schedule() yield the CPU?"
```

---

### 1 · Build the Mirror

Index a kernel subsystem (minutes) or the full tree (hours):

```bash
# Scheduler subsystem — fast, great for testing
python cli/ktalk.py index --kernel /path/to/linux --subsystem kernel/sched

# Memory management
python cli/ktalk.py index --kernel /path/to/linux --subsystem mm

# Networking
python cli/ktalk.py index --kernel /path/to/linux --subsystem net

# Full kernel (get coffee — or several)
python cli/ktalk.py index --kernel /path/to/linux
```

---

### 2 · Ask Questions

```bash
python cli/ktalk.py ask "why does schedule() yield the CPU?"
python cli/ktalk.py ask "how does kmalloc decide which slab cache to use?"
python cli/ktalk.py ask "what happens in the kernel when a process calls fork()?"
python cli/ktalk.py ask "why would a process be in TASK_UNINTERRUPTIBLE state?"

# Stream the response token by token
python cli/ktalk.py ask --stream "walk me through the OOM killer decision path"

# Restrict search to a subsystem
python cli/ktalk.py ask --subsystem net "how does TCP handle retransmission?"
```

---

### 3 · Filesystem X-Ray

Map any `/sys` or `/proc` path to its kernel C source:

```bash
python cli/ktalk.py xray /sys/class/net/wlan0/operstate
python cli/ktalk.py xray /proc/meminfo
python cli/ktalk.py xray /sys/block/sda/queue/scheduler
python cli/ktalk.py xray /proc/1234/maps
python cli/ktalk.py xray /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
```

Works without a Mirror index for the 30+ built-in known paths (pattern-match mode).
With a Mirror index, unknown paths are resolved via vector search.

---

### 4 · Live Kernel Probe (Linux, root)

```bash
# List processes directly from kernel memory
sudo python cli/ktalk.py probe --processes

# Read scheduler run queue for CPU 0
sudo python cli/ktalk.py probe --runqueue 0

# Read a specific struct by address (get address from --processes output)
sudo python cli/ktalk.py probe --struct task_struct --addr 0xffff888100a58000
```

---

### 5 · Explore the Graph

```bash
# Call graph around schedule(), 2 hops out
python cli/ktalk.py graph schedule --hops 2

# Everything that references task_struct
python cli/ktalk.py graph task_struct

# Index statistics
python cli/ktalk.py stats
```

---

## Configuration

All settings can be overridden via environment variables (persistent in `activate.sh`)
or CLI flags (per-invocation):

| Environment Variable | CLI Flag    | Default                        | Description                                  |
|----------------------|-------------|--------------------------------|----------------------------------------------|
| `KTALK_STORAGE`      | `--storage` | `~/.kernel-talk/store`         | Storage directory for vector index + graph   |
| `KTALK_KERNEL`       | `--kernel`  | `/usr/src/linux`               | Path to Linux kernel source tree             |
| `KTALK_MODEL`        | `--model`   | `ollama:deepseek-coder:6.7b`   | LLM backend (`ollama:`, `openai:`, `anthropic:`) |

```bash
# Example overrides
export KTALK_STORAGE=/fast/ssd/kernel-talk
export KTALK_KERNEL=/usr/src/linux-6.6
export KTALK_MODEL=openai:gpt-4o          # needs OPENAI_API_KEY
```

---

## Project Structure

```
kernel-talk/
├── core/
│   ├── mirror/
│   │   ├── parser.py      # tree-sitter AST parsing → CodeNode stream
│   │   ├── graph.py       # NetworkX knowledge graph + traversal
│   │   ├── embedder.py    # CodeBERT embeddings, batched
│   │   └── store.py       # ChromaDB + graph, unified interface
│   ├── probe/
│   │   └── drgn_bridge.py # Live kernel memory via drgn
│   └── synthesis/
│       └── synthesizer.py # Prompt construction + LLM backends
├── tools/
│   └── xray.py            # Filesystem X-Ray (/sys, /proc → source)
├── cli/
│   └── ktalk.py           # Click CLI with Rich terminal output
├── activate.sh            # Convenience: venv activation + env defaults
└── requirements.txt
```

---

## Design Principles

**AST over text.** Kernel C doesn't chunk cleanly on line boundaries.
`struct task_struct` has members accessed by the scheduler, memory manager,
and signal handler — a sliding window misses all of it. tree-sitter gives us
semantic units.

**Graph-first, not graph-later.** The `CodeNode` carries both embedding text
and graph edge metadata. Adding graph structure to an existing flat index
requires reprocessing everything. Design it in from the start.

**Two retrieval modes, one result.** Vector search finds what's semantically
similar. Graph traversal expands structural context. Both are needed because
the most relevant code (vector hit) and the most explanatory code (architectural
context) are often different nodes.

**Read-only probing only.** drgn cannot write kernel memory. This is not a
limitation — it's the guarantee that makes production use safe.

**LLM-agnostic synthesis.** Ollama (local) is the default because privacy
matters in systems debugging. Any OpenAI-compatible API works as a drop-in.

---

## Roadmap

- [ ] Incremental indexing (only re-parse changed files)
- [ ] Sysfs kobject walking via drgn (full live X-Ray)
- [ ] Web UI (React + FastAPI, for the "Mandala Kernel" topology map)
- [ ] DWARF type introspection (auto-discover struct fields without hardcoding)
- [ ] eBPF probe integration (complement drgn with dynamic tracing)
- [ ] Cross-kernel-version diff (what changed between 6.1 LTS and 6.6 LTS?)

---

*Built by Lee Ostadi. First-principles kernel understanding via Graph-RAG.*
