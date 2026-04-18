#!/usr/bin/env python3
"""
cli/ktalk.py
─────────────
Kernel-Talk CLI

The user-facing interface. Built with Click for clean subcommand structure
and Rich for readable terminal output.

Commands:
  ktalk index   [--kernel /usr/src/linux] [--subsystem kernel/sched]
                Build or update the Mirror (static knowledge graph + vector index)

  ktalk ask     "why does schedule() yield the CPU?"
                Hybrid search + LLM synthesis

  ktalk xray    /sys/class/net/wlan0/operstate
                Filesystem X-Ray: map a /sys or /proc path to source code

  ktalk probe   [--struct task_struct] [--cpu 0]
                Live kernel memory probe via drgn

  ktalk stats
                Show index statistics (node counts, edge counts, etc.)

  ktalk graph   [--symbol schedule] [--hops 2]
                Show the graph neighborhood of a kernel symbol
"""

import sys
import platform
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import click
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn
    from rich import print as rprint
except ImportError:
    print("Missing dependencies. Run: pip install click rich")
    sys.exit(1)

from core.mirror.parser import KernelParser
from core.mirror.store import KernelStore
from core.probe.drgn_bridge import DrgnBridge
from core.probe.kallsyms import KallsymsBridge
from core.synthesis.synthesizer import KernelSynthesizer
from tools.xray import XRay

console = Console()

# Default paths
DEFAULT_STORAGE = Path.home() / ".kernel-talk" / "store"
DEFAULT_KERNEL  = "/usr/src/linux"
DEFAULT_MODEL   = "ollama:deepseek-coder:6.7b"


# ─── CLI Root ─────────────────────────────────────────────────────────────────

@click.group()
@click.option("--storage", default=str(DEFAULT_STORAGE), envvar="KTALK_STORAGE",
              help="Storage directory for the Mirror index.")
@click.option("--model", default=DEFAULT_MODEL, envvar="KTALK_MODEL",
              help="LLM backend: ollama:MODEL, openai:MODEL, or anthropic:MODEL")
@click.pass_context
def cli(ctx: click.Context, storage: str, model: str):
    """
    Kernel-Talk — A Digital Twin for the Linux Kernel.

    Bridges static source code (Theory) with live kernel memory (Reality)
    to explain what your system is actually doing and why.
    """
    ctx.ensure_object(dict)
    ctx.obj["storage"] = storage
    ctx.obj["model"]   = model


# ─── index ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--kernel", default=DEFAULT_KERNEL, envvar="KTALK_KERNEL",
              help="Path to Linux kernel source tree.")
@click.option("--subsystem", default="", help="Limit to a subsystem (e.g. kernel/sched, mm, net).")
@click.option("--ext", default=".c,.h", help="File extensions to parse, comma-separated.")
@click.pass_context
def index(ctx, kernel, subsystem, ext):
    """
    Build the Mirror: parse kernel source, embed, and store in the index.

    First run will take a while (hours for the full kernel, minutes for a subsystem).
    Subsequent runs are incremental — only new/changed files are re-indexed.

    Examples:
      ktalk index --kernel /usr/src/linux --subsystem kernel/sched
      ktalk index --subsystem net
    """
    storage_dir = ctx.obj["storage"]

    kernel_path = Path(kernel)
    if not kernel_path.exists():
        console.print(f"[red]Kernel source not found: {kernel}[/red]")
        console.print("Download from: https://www.kernel.org/")
        raise click.Abort()

    extensions = tuple(e.strip() for e in ext.split(","))

    console.print(Panel(
        f"[bold cyan]Kernel-Talk: Building the Mirror[/bold cyan]\n"
        f"Kernel: {kernel}\n"
        f"Subsystem: {subsystem or '(entire tree)'}\n"
        f"Storage: {storage_dir}",
        title="[bold]MIRROR[/bold]",
    ))

    parser = KernelParser(kernel_path)
    store  = KernelStore.create(storage_dir)

    # Collect nodes with progress display
    nodes = []
    file_count = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.completed} nodes"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        parse_task = progress.add_task("Parsing kernel source...", total=None)

        for node in parser.parse_directory(subsystem, extensions=extensions):
            nodes.append(node)
            if node.node_type == "file":
                file_count += 1
                progress.update(parse_task, description=f"Parsing {node.file_path}...")
            progress.advance(parse_task)

    console.print(f"[green]Parsed {len(nodes)} nodes from {file_count} files[/green]")

    if not nodes:
        console.print("[yellow]No nodes found. Check --kernel and --subsystem paths.[/yellow]")
        return

    # Index (embed + graph)
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=40),
        TextColumn("{task.completed}/{task.total}"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        embed_task = progress.add_task("Embedding and indexing...", total=len(nodes))

        def progress_cb(done: int, total: int):
            progress.update(embed_task, completed=done, total=total)

        store.embedder.progress_callback = progress_cb
        store.index_nodes(nodes, verbose=False)
        progress.update(embed_task, completed=len(nodes))

    store.save_graph()

    # Final stats
    stats = store.stats()
    table = Table(title="Mirror Index Stats", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="green")
    table.add_row("Nodes in vector index", str(stats["vector_index"]["count"]))
    table.add_row("Nodes in graph",        str(stats["graph"]["total_nodes"]))
    table.add_row("Graph edges",           str(stats["graph"]["total_edges"]))
    table.add_row("Unique symbols",        str(stats["graph"]["unique_symbols"]))
    for ntype, count in stats["graph"]["node_types"].items():
        table.add_row(f"  {ntype}s", str(count))

    console.print(table)
    console.print(f"\n[bold green]Mirror built successfully.[/bold green] Storage: {storage_dir}")


# ─── ask ──────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("question")
@click.option("--top-k",    default=8,    help="Number of vector search results.")
@click.option("--hops",     default=2,    help="Graph expansion hops.")
@click.option("--subsystem", default=None, help="Restrict search to a subsystem path.")
@click.option("--no-live",   is_flag=True, help="Skip live kernel probing.")
@click.option("--stream",    is_flag=True, help="Stream the LLM response token-by-token.")
@click.pass_context
def ask(ctx, question, top_k, hops, subsystem, no_live, stream):
    """
    Ask a natural language question about the kernel.

    The system retrieves relevant code via hybrid search (vector + graph),
    optionally enriches with live drgn probes, then synthesizes an answer.

    Examples:
      ktalk ask "why does the scheduler yield the CPU?"
      ktalk ask "how does kmalloc decide which slab to use?"
      ktalk ask "what happens when a process calls fork()?"
    """
    storage_dir = ctx.obj["storage"]
    model       = ctx.obj["model"]

    console.print(Panel(f"[bold]{question}[/bold]", title="[cyan]QUESTION[/cyan]"))

    # Load store
    try:
        store = KernelStore.load(storage_dir)
    except Exception as e:
        console.print(f"[red]Failed to load Mirror: {e}[/red]")
        console.print("Run [bold]ktalk index[/bold] first.")
        raise click.Abort()

    # Retrieve
    with console.status("Searching the Mirror..."):
        results = store.hybrid_search(
            question,
            top_k=top_k,
            hops=hops,
            subsystem_filter=subsystem,
        )

    # Show sources
    if results.primary:
        table = Table(title="Retrieved Sources", show_header=True, header_style="bold blue")
        table.add_column("Score", width=6, style="green")
        table.add_column("Type",  width=10)
        table.add_column("Symbol", width=30)
        table.add_column("File")
        for r in results.primary:
            table.add_row(
                f"{r.score:.3f}",
                r.node.node_type,
                r.node.symbol_name,
                f"{r.node.file_path}:{r.node.line_start}",
            )
        console.print(table)
        console.print(f"[dim]+ {len(results.context)} graph-context nodes[/dim]\n")

    # Live probe
    live_snapshots = []
    if not no_live:
        probe = DrgnBridge()
        if probe.is_available():
            with console.status("Probing live kernel state..."):
                snap = probe.read_runqueue(cpu=0)
                if snap:
                    live_snapshots.append(snap)

    # Synthesize
    synthesizer = KernelSynthesizer(model=model)

    console.print(Panel("", title="[bold green]SYNTHESIS[/bold green]"))

    if stream:
        for token in synthesizer.synthesize_stream(question, results, live_snapshots or None):
            console.print(token, end="")
        console.print()
    else:
        with console.status(f"Synthesizing with {model}..."):
            result = synthesizer.synthesize(question, results, live_snapshots or None)
        console.print(result.answer)

    # Source citations
    if results.primary:
        console.print("\n[dim]Sources:[/dim]")
        for r in results.primary[:5]:
            console.print(f"  [dim]• {r.node.file_path}:{r.node.line_start} — {r.node.symbol_name}[/dim]")


# ─── xray ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("path")
@click.option("--no-live", is_flag=True, help="Skip reading the live file value.")
@click.pass_context
def xray(ctx, path, no_live):
    """
    X-Ray a /sys or /proc path to find the kernel source responsible for it.

    Examples:
      ktalk xray /sys/class/net/wlan0/operstate
      ktalk xray /proc/meminfo
      ktalk xray /sys/block/sda/queue/scheduler
    """
    storage_dir = ctx.obj["storage"]

    # Load store (optional — xray works with just pattern matching)
    store = None
    try:
        store = KernelStore.load(storage_dir)
    except Exception:
        console.print("[yellow]Mirror not loaded — using pattern-only mode.[/yellow]")

    probe = None if no_live else DrgnBridge()

    ray = XRay(store=store, probe=probe)

    with console.status(f"X-Raying {path}..."):
        result = ray.scan(path)

    # Display result
    confidence_color = {"high": "green", "medium": "yellow", "low": "red"}[result.confidence]

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key",   style="cyan bold", width=18)
    table.add_column("Value", style="white")

    table.add_row("Path",       result.path)
    table.add_row("Confidence", f"[{confidence_color}]{result.confidence}[/{confidence_color}]")
    table.add_row("Source Files",  "\n".join(result.source_files) or "[dim]unknown[/dim]")
    table.add_row("Handlers",   "\n".join(result.handler_functions) or "[dim]unknown[/dim]")
    table.add_row("Structs",    "\n".join(result.data_structures) or "[dim]unknown[/dim]")
    if result.live_value is not None:
        table.add_row("Live Value", f"[bold green]{result.live_value}[/bold green]")

    console.print(Panel(table, title=f"[bold]FILESYSTEM X-RAY[/bold]: {path}"))
    console.print()
    console.print(result.description)


# ─── probe ────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--processes", is_flag=True, help="List live kernel processes.")
@click.option("--runqueue",  default=-1, type=int, help="Read run queue for CPU N.")
@click.option("--struct",    default=None, help="Struct name to read (requires --addr).")
@click.option("--addr",      default=None, help="Memory address in hex (e.g. 0xffff...).")
@click.pass_context
def probe(ctx, processes, runqueue, struct, addr):
    """
    Probe live kernel memory via drgn.

    Requires Linux with /proc/kcore and usually root.

    Examples:
      ktalk probe --processes
      ktalk probe --runqueue 0
      ktalk probe --struct task_struct --addr 0xffff888100a58000
    """
    bridge = DrgnBridge()
    status = bridge.status()

    if not status["available"]:
        console.print(Panel(
            f"[red]drgn not available:[/red] {status['reason']}\n\n"
            f"Platform: {status['platform']}\n"
            f"Kernel: {status['kernel']}\n"
            f"UID: {status['uid']}\n\n"
            "On Linux, run as root and ensure:\n"
            "  • pip install drgn\n"
            "  • /proc/kcore exists (CONFIG_PROC_KCORE=y)\n"
            "  • vmlinux debug symbols are available",
            title="[bold red]PROBE[/bold red]",
        ))
        return

    if processes:
        with console.status("Reading live process list from kernel memory..."):
            procs = bridge.list_processes(max_tasks=32)

        table = Table(title="Live Processes (from kernel memory)", header_style="bold cyan")
        table.add_column("PID",   width=7,  style="green")
        table.add_column("PPID",  width=7)
        table.add_column("COMM",  width=16, style="bold")
        table.add_column("STATE", width=22)
        table.add_column("CPU",   width=4)
        table.add_column("PRIO",  width=6)

        for p in procs:
            table.add_row(str(p.pid), str(p.ppid), p.comm, p.state,
                          str(p.cpu), str(p.priority))
        console.print(table)

    elif runqueue >= 0:
        with console.status(f"Reading run queue for CPU {runqueue}..."):
            snap = bridge.read_runqueue(cpu=runqueue)
        if snap:
            console.print(Panel(snap.to_text(), title=f"[bold]RUN QUEUE CPU {runqueue}[/bold]"))
        else:
            console.print("[red]Failed to read run queue.[/red]")

    elif struct and addr:
        with console.status(f"Reading {struct} @ {addr}..."):
            snap = bridge.read_struct(struct, addr)
        if snap:
            console.print(Panel(snap.to_text(), title=f"[bold]{struct} @ {addr}[/bold]"))
        else:
            console.print(f"[red]Failed to read {struct} @ {addr}.[/red]")

    else:
        console.print("[yellow]Specify --processes, --runqueue N, or --struct + --addr[/yellow]")
        console.print("Run [bold]ktalk probe --help[/bold] for examples.")


# ─── stats ────────────────────────────────────────────────────────────────────

@cli.command()
@click.pass_context
def stats(ctx):
    """Show Mirror index statistics."""
    storage_dir = ctx.obj["storage"]

    try:
        store = KernelStore.load(storage_dir)
    except Exception as e:
        console.print(f"[red]Failed to load Mirror: {e}[/red]")
        raise click.Abort()

    s = store.stats()

    table = Table(title="Kernel-Talk Mirror Statistics", header_style="bold cyan")
    table.add_column("Metric", style="cyan")
    table.add_column("Value",  style="green")

    table.add_row("Vector index size",  str(s["vector_index"]["count"]))
    table.add_row("Graph nodes",        str(s["graph"]["total_nodes"]))
    table.add_row("Graph edges",        str(s["graph"]["total_edges"]))
    table.add_row("Unique symbols",     str(s["graph"]["unique_symbols"]))
    table.add_row("Storage directory",  s["storage_dir"])
    table.add_row("Embedder model",     s["embedder_model"])
    table.add_row("", "")
    for ntype, count in s["graph"].get("node_types", {}).items():
        table.add_row(f"  {ntype}s", str(count))

    # Most connected nodes
    top_nodes = store.graph.most_connected(top_k=10)
    if top_nodes:
        table.add_row("", "")
        table.add_row("[bold]Top connected symbols[/bold]", "")
        for sym, deg in top_nodes:
            table.add_row(f"  {sym}", f"{deg} edges")

    console.print(table)


# ─── graph ────────────────────────────────────────────────────────────────────

@cli.command(name="graph")
@click.argument("symbol")
@click.option("--hops", default=1, help="Number of hops to expand.")
@click.pass_context
def graph_cmd(ctx, symbol, hops):
    """
    Show the knowledge graph neighborhood of a kernel symbol.

    Example:
      ktalk graph schedule --hops 2
      ktalk graph task_struct --hops 1
    """
    storage_dir = ctx.obj["storage"]

    try:
        store = KernelStore.load(storage_dir)
    except Exception as e:
        console.print(f"[red]Failed to load Mirror: {e}[/red]")
        raise click.Abort()

    nodes = store.graph.find_by_symbol(symbol)
    if not nodes:
        console.print(f"[yellow]Symbol '{symbol}' not found in graph.[/yellow]")
        return

    seed_ids = [n.id for n in nodes]
    ctx_result = store.graph.neighborhood(seed_ids, hops=hops)

    console.print(Panel(
        f"Symbol: [bold]{symbol}[/bold]\n"
        f"Definitions: {len(nodes)}\n"
        f"Neighborhood ({hops} hops): {len(ctx_result.all_ids)} nodes",
        title="[bold cyan]GRAPH NEIGHBORHOOD[/bold cyan]",
    ))

    # Show the seed nodes
    for node in nodes:
        console.print(f"\n[bold cyan]{node.node_type}[/bold cyan]: {node.symbol_name}")
        console.print(f"  File: {node.file_path}:{node.line_start}")
        if node.calls:
            console.print(f"  Calls: {', '.join(node.calls[:10])}")
        if node.uses_structs:
            console.print(f"  Uses:  {', '.join(node.uses_structs[:10])}")

    # Show callers
    callers = store.graph.callers_of(symbol)
    if callers:
        console.print(f"\n[bold]Called by ({len(callers)} functions):[/bold]")
        for c in callers[:10]:
            console.print(f"  [green]←[/green] {c.symbol_name} [{c.file_path}:{c.line_start}]")

    # Show callees
    callees = store.graph.callees_of(symbol)
    if callees:
        console.print(f"\n[bold]Calls ({len(callees)} functions):[/bold]")
        for c in callees[:10]:
            console.print(f"  [blue]→[/blue] {c.symbol_name} [{c.file_path}:{c.line_start}]")


# ─── addr2line ────────────────────────────────────────────────────────────────

@cli.command(name="addr2line")
@click.argument("addresses", nargs=-1, required=True)
@click.option("--vmlinux", default=None, envvar="KTALK_VMLINUX",
              help="Path to vmlinux with DWARF debug info.")
@click.option("--no-cache", is_flag=True, help="Re-parse DWARF (ignore cache).")
@click.pass_context
def addr2line_cmd(ctx, addresses, vmlinux, no_cache):
    """
    Map virtual kernel address(es) to C source lines.

    The full Digital Twin reverse traversal:
      address → kallsyms (symbol name) → DWARF (function range)
              → line number program (exact source line)
              → Mirror CodeNode (static analysis context)

    Examples:
      ktalk addr2line 0xffffffff811abc04
      ktalk addr2line ffffffff811abc04 ffffffff81200018
      ktalk addr2line --vmlinux /boot/vmlinux 0xffffffff811abc04
    """
    storage_dir = ctx.obj["storage"]

    # Locate vmlinux
    vmlinux_candidates = [
        vmlinux,
        f"/boot/vmlinux-{platform.release()}",
        "/boot/vmlinux",
        f"/usr/lib/debug/boot/vmlinux-{platform.release()}",
        f"/usr/lib/debug/lib/modules/{platform.release()}/vmlinux",
        f"/usr/lib/modules/{platform.release()}/build/vmlinux",
    ]
    vmlinux_path = None
    for candidate in vmlinux_candidates:
        if candidate and Path(candidate).exists():
            vmlinux_path = candidate
            break

    if not vmlinux_path:
        console.print("[red]vmlinux not found.[/red]")
        console.print("Specify with --vmlinux or set KTALK_VMLINUX.")
        console.print("See README § Debug Symbols for installation instructions.")
        raise click.Abort()

    # Load DWARF
    from core.dwarf.bridge import DwarfBridge
    dwarf = DwarfBridge(vmlinux_path, cache_dir=str(Path(storage_dir) / "dwarf"))

    with console.status(f"Loading DWARF from {vmlinux_path} ..."):
        dwarf.load(verbose=False, use_cache=not no_cache)

    # Load kallsyms (best-effort — requires root)
    kallsyms = KallsymsBridge()
    try:
        kallsyms.load(verbose=False)
    except Exception:
        kallsyms = None

    # Load store (optional — for CodeNode cross-reference)
    store = None
    try:
        store = KernelStore.load(storage_dir)
    except Exception:
        pass

    xray = XRay(store=store, dwarf=dwarf, kallsyms=kallsyms)

    for addr_str in addresses:
        result = xray.addr2line(addr_str)

        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_column("Key",   style="cyan bold", width=16)
        table.add_column("Value", style="white")

        table.add_row("Address", result.address_hex)

        if result.line_source_file and result.line_source_line:
            stmt = " [stmt]" if result.line_is_stmt else ""
            table.add_row("Source",   f"[bold green]{result.line_source_file}:{result.line_source_line}[/bold green]{stmt}")

        if result.function_name:
            fn_offset = result.address - int(result.function_range.split(" – ")[0], 16) \
                        if result.function_range else result.kallsym_offset
            table.add_row("Function", f"{result.function_name}()+0x{fn_offset:x}")
            if result.function_range:
                table.add_row("Range",    result.function_range)
            if result.dwarf_source_file:
                table.add_row("Decl",     f"{result.dwarf_source_file}:{result.dwarf_source_line}")

        if result.kallsym_name and result.kallsym_name != result.function_name:
            table.add_row("Symbol",   f"{result.kallsym_name}+0x{result.kallsym_offset:x}")

        if result.code_node_id:
            table.add_row("Mirror",   f"[dim]{result.code_node_id}[/dim]")

        console.print(Panel(table, title=f"[bold]addr2line[/bold]: {addr_str}"))


# ─── twin ─────────────────────────────────────────────────────────────────────

@cli.command()
@click.option("--vmlinux", default=None, envvar="KTALK_VMLINUX",
              help="Path to vmlinux with DWARF debug info.")
@click.option("--no-cache", is_flag=True, help="Re-parse DWARF (ignore cache).")
@click.pass_context
def twin(ctx, vmlinux, no_cache):
    """
    Build the full Digital Twin: link all four layers.

    Runs after `ktalk index` to connect the Mirror (Layer 1) to compiled
    binary addresses via DWARF (Layer 2), then to live /proc/kallsyms
    addresses (Layer 3), and annotates structs with memory layout.

    After this, `ktalk graph <symbol>` shows live virtual addresses,
    and `ktalk addr2line` can decode any stack trace or oops address.

    Requires: vmlinux with DWARF + (optionally) root for kallsyms addresses.
    """
    storage_dir = ctx.obj["storage"]

    # Load store
    try:
        store = KernelStore.load(storage_dir)
    except Exception as e:
        console.print(f"[red]Failed to load Mirror: {e}[/red]")
        console.print("Run [bold]ktalk index[/bold] first.")
        raise click.Abort()

    # Locate vmlinux
    import platform as _platform
    vmlinux_candidates = [
        vmlinux,
        f"/boot/vmlinux-{_platform.release()}",
        "/boot/vmlinux",
        f"/usr/lib/debug/boot/vmlinux-{_platform.release()}",
        f"/usr/lib/debug/lib/modules/{_platform.release()}/vmlinux",
        f"/usr/lib/modules/{_platform.release()}/build/vmlinux",
    ]
    vmlinux_path = None
    for candidate in vmlinux_candidates:
        if candidate and Path(candidate).exists():
            vmlinux_path = candidate
            break

    if not vmlinux_path:
        console.print("[red]vmlinux not found. Cannot build Layer 2 (DWARF).[/red]")
        console.print("Install debug symbols — see README § Debug Symbols.")
        raise click.Abort()

    console.print(Panel(
        f"[bold cyan]Kernel-Talk: Building the Digital Twin[/bold cyan]\n"
        f"vmlinux:  {vmlinux_path}\n"
        f"Storage:  {storage_dir}\n\n"
        f"[dim]Layer 1: Mirror (source graph)  — already loaded\n"
        f"Layer 2: DWARF (binary addresses) — loading...\n"
        f"Layer 3: kallsyms (live addrs)    — loading...\n"
        f"Layer 4: /proc/kcore (memory)     — on-demand via drgn[/dim]",
        title="[bold]DIGITAL TWIN[/bold]",
    ))

    from core.dwarf.bridge import DwarfBridge
    import pathlib

    dwarf = DwarfBridge(
        vmlinux_path,
        cache_dir=str(pathlib.Path(storage_dir) / "dwarf")
    )

    with Progress(SpinnerColumn(), TextColumn("{task.description}"),
                  TimeElapsedColumn(), console=console) as progress:
        t = progress.add_task("Parsing DWARF (may take 1–2 min on first run)...", total=None)
        dwarf.load(verbose=False, use_cache=not no_cache)
        progress.update(t, description=f"DWARF loaded: {dwarf.stats()['functions']} functions, "
                                        f"{dwarf.stats()['struct_types']} structs")

    # Layer 2 linking
    with console.status("Linking source → binary (SOURCE_TO_BINARY edges)..."):
        r2 = store.graph.link_dwarf(dwarf, verbose=False)
    console.print(f"  [green]✓[/green] Layer 1→2: {r2['SOURCE_TO_BINARY']} SOURCE_TO_BINARY edges")

    # Layer 2 struct layout
    with console.status("Linking struct field offsets (FIELD_TO_OFFSET edges)..."):
        rl = store.graph.link_struct_layouts(dwarf, verbose=False)
    console.print(f"  [green]✓[/green] Struct layouts: {rl['structs_linked']} structs, "
                  f"{rl['FIELD_TO_OFFSET']} FIELD_TO_OFFSET edges")

    # Layer 3 (kallsyms — requires root)
    kallsyms = KallsymsBridge()
    try:
        kallsyms.load(verbose=False)
        if kallsyms.is_available():
            with console.status("Linking binary → live addresses (BINARY_TO_LIVE edges)..."):
                r3 = store.graph.link_kallsyms(kallsyms, dwarf, verbose=False)
            slide = r3.get("kaslr_slide")
            slide_str = f"0x{slide:016x}" if slide else "unknown"
            console.print(f"  [green]✓[/green] Layer 2→3: {r3['BINARY_TO_LIVE']} BINARY_TO_LIVE edges "
                          f"(KASLR slide: {slide_str})")
        else:
            console.print("  [yellow]⚠[/yellow]  Layer 3 skipped: /proc/kallsyms addresses not readable "
                          "(run as root for full Digital Twin)")
    except Exception as e:
        console.print(f"  [yellow]⚠[/yellow]  Layer 3 skipped: {e}")

    # Save the enriched graph
    with console.status("Saving enriched graph..."):
        store.save_graph()

    console.print(f"\n[bold green]Digital Twin built.[/bold green]")
    console.print("Use [bold]ktalk graph <symbol>[/bold] to see live addresses.")
    console.print("Use [bold]ktalk addr2line <hex_addr>[/bold] to decode stack traces.")


# ─── eval retrieval ───────────────────────────────────────────────────────────

@cli.command(name="eval")
@click.argument("target", type=click.Choice(["retrieval"]))
@click.option("--gold",   default="eval/retrieval_gold.jsonl",
              help="Path to gold JSONL file (query + expected symbols/files).")
@click.option("--top-k",  default=10, help="Number of results to retrieve per query.")
@click.option("--hops",   default=2,  help="Graph expansion hops.")
@click.option("--out",    default=None, help="Write per-query results to this JSON file.")
@click.pass_context
def eval_cmd(ctx, target, gold, top_k, hops, out):
    """
    Evaluate retrieval quality against a gold query set.

    Reads eval/retrieval_gold.jsonl (or --gold path), runs hybrid_search
    for each query, and reports:
      • Recall@k   — what fraction of expected symbols appeared in top-k
      • Hit@1      — did the top result match an expected symbol/file
      • MRR@k      — mean reciprocal rank

    Examples:
      ktalk eval retrieval
      ktalk eval retrieval --top-k 20 --out results.json
    """
    import json, time
    storage_dir = ctx.obj["storage"]

    gold_path = Path(gold)
    if not gold_path.exists():
        console.print(f"[red]Gold file not found: {gold}[/red]")
        raise click.Abort()

    # Load store
    try:
        store = KernelStore.load(storage_dir)
    except Exception as e:
        console.print(f"[red]Failed to load Mirror: {e}[/red]")
        console.print("Run [bold]ktalk index[/bold] first.")
        raise click.Abort()

    # Load gold pairs
    gold_pairs = []
    with open(gold_path) as f:
        for line in f:
            line = line.strip()
            if line:
                gold_pairs.append(json.loads(line))

    console.print(Panel(
        f"Evaluating retrieval on [bold]{len(gold_pairs)}[/bold] queries\n"
        f"top-k={top_k}  hops={hops}  gold={gold_path}",
        title="[bold cyan]EVAL: RETRIEVAL[/bold cyan]",
    ))

    hits1 = 0
    recall_sum = 0.0
    mrr_sum = 0.0
    per_query = []

    with Progress(
        SpinnerColumn(), TextColumn("{task.description}"),
        BarColumn(), TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("Evaluating...", total=len(gold_pairs))

        for item in gold_pairs:
            query    = item["query"]
            expected_syms  = set(item.get("expected_symbols", []))
            expected_files = set(item.get("expected_files",   []))

            t0 = time.perf_counter()
            results = store.hybrid_search(query, top_k=top_k, hops=hops)
            elapsed = time.perf_counter() - t0

            # Collect returned symbols and file paths
            returned_symbols = [r.node.symbol_name for r in results.primary]
            returned_files   = [r.node.file_path   for r in results.primary]
            all_returned_syms = set(returned_symbols)

            # Metrics
            matched_syms = expected_syms & all_returned_syms
            file_matched = any(
                any(rf.startswith(ef) or ef in rf for ef in expected_files)
                for rf in returned_files
            )

            recall = len(matched_syms) / max(1, len(expected_syms))
            recall_sum += recall

            hit1 = (
                returned_symbols[0] in expected_syms if returned_symbols else False
            ) or file_matched
            if hit1:
                hits1 += 1

            # MRR: rank of first expected symbol in primary results
            rr = 0.0
            for rank, sym in enumerate(returned_symbols, start=1):
                if sym in expected_syms:
                    rr = 1.0 / rank
                    break
            mrr_sum += rr

            per_query.append({
                "query":         query,
                "recall":        recall,
                "hit1":          hit1,
                "rr":            rr,
                "elapsed_ms":    round(elapsed * 1000, 1),
                "matched_syms":  sorted(matched_syms),
                "expected_syms": sorted(expected_syms),
                "returned_syms": returned_symbols[:top_k],
            })

            progress.advance(task)

    n = len(gold_pairs)
    recall_at_k = recall_sum / n
    hit_at_1    = hits1 / n
    mrr         = mrr_sum / n

    table = Table(title="Retrieval Evaluation Results", header_style="bold cyan")
    table.add_column("Metric",  style="cyan", width=24)
    table.add_column("Value",   style="bold green", width=12)
    table.add_column("Meaning", style="dim")
    table.add_row(f"Recall@{top_k}",  f"{recall_at_k:.3f}",
                  "Fraction of expected symbols found in top-k")
    table.add_row("Hit@1",       f"{hit_at_1:.3f}",
                  "Top result matched an expected symbol or file")
    table.add_row(f"MRR@{top_k}",     f"{mrr:.3f}",
                  "Mean reciprocal rank of first expected symbol")
    table.add_row("Queries",     str(n), "")
    avg_ms = sum(q["elapsed_ms"] for q in per_query) / n
    table.add_row("Avg latency", f"{avg_ms:.1f} ms", "Per-query hybrid_search time")

    console.print(table)

    # Worst queries (lowest recall)
    bottom = sorted(per_query, key=lambda x: x["recall"])[:5]
    console.print("\n[bold]Lowest recall queries:[/bold]")
    for item in bottom:
        console.print(
            f"  recall={item['recall']:.2f}  [dim]{item['query'][:70]}[/dim]\n"
            f"    expected: {item['expected_syms'][:4]}\n"
            f"    returned: {item['returned_syms'][:4]}"
        )

    if out:
        import json as _json
        out_path = Path(out)
        with open(out_path, "w") as f:
            _json.dump({
                "summary": {
                    f"recall@{top_k}": recall_at_k,
                    "hit@1": hit_at_1,
                    f"mrr@{top_k}": mrr,
                    "queries": n,
                },
                "per_query": per_query,
            }, f, indent=2)
        console.print(f"\n[dim]Detailed results written to {out_path}[/dim]")


# ─── bench index ──────────────────────────────────────────────────────────────

@cli.command(name="bench")
@click.argument("target", type=click.Choice(["index"]))
@click.option("--kernel",    default=DEFAULT_KERNEL, envvar="KTALK_KERNEL")
@click.option("--subsystem", default="kernel/sched",
              help="Subsystem to benchmark (default: kernel/sched — fast but representative).")
@click.option("--runs", default=3, help="Number of timed runs.")
@click.pass_context
def bench_cmd(ctx, target, kernel, subsystem, runs):
    """
    Benchmark the index pipeline (parse → embed → resolve_edges).

    Measures wall-clock time for each stage across --runs trials
    and reports mean ± stdev.  Use this to verify that Phase 1
    performance fixes (F-5/F-8 resolve_edges speedup) actually
    improved throughput.

    Example:
      ktalk bench index --subsystem kernel/sched
      ktalk bench index --subsystem mm
    """
    import time, statistics, json as _json, tempfile
    from core.mirror.parser import KernelParser
    from core.mirror.graph  import KernelGraph

    kernel_path = Path(kernel)
    if not kernel_path.exists():
        console.print(f"[red]Kernel source not found: {kernel}[/red]")
        raise click.Abort()

    console.print(Panel(
        f"Benchmarking index pipeline\n"
        f"kernel: {kernel}  subsystem: {subsystem}  runs: {runs}",
        title="[bold cyan]BENCH: INDEX[/bold cyan]",
    ))

    parse_times  : list[float] = []
    resolve_times: list[float] = []
    node_counts  : list[int]   = []

    for run in range(1, runs + 1):
        console.print(f"  Run {run}/{runs} ...", end="")

        parser = KernelParser(kernel_path)

        # ── Parse ──
        t0 = time.perf_counter()
        nodes = list(parser.parse_directory(subsystem))
        parse_t = time.perf_counter() - t0

        # ── resolve_edges (graph only, no embedding) ──
        g = KernelGraph()
        g.add_nodes(nodes)

        t0 = time.perf_counter()
        counts = g.resolve_edges()
        resolve_t = time.perf_counter() - t0

        parse_times.append(parse_t)
        resolve_times.append(resolve_t)
        node_counts.append(len(nodes))

        total_edges = sum(counts.values())
        console.print(
            f"  {len(nodes)} nodes  {total_edges} edges  "
            f"parse={parse_t:.2f}s  resolve={resolve_t:.3f}s"
        )

    def _fmt(vals: list[float], unit: str = "s") -> str:
        mean = statistics.mean(vals)
        std  = statistics.stdev(vals) if len(vals) > 1 else 0.0
        return f"{mean:.3f}{unit} ± {std:.3f}{unit}"

    table = Table(title="Benchmark Results", header_style="bold cyan")
    table.add_column("Stage",  style="cyan",  width=24)
    table.add_column("Mean ± Stdev", style="bold green", width=20)
    table.add_column("Notes", style="dim")

    table.add_row("Parse (tree-sitter)",  _fmt(parse_times),
                  f"{statistics.mean(node_counts):.0f} nodes avg")
    table.add_row("resolve_edges (graph)", _fmt(resolve_times),
                  "O(N) after F-5 fix")
    table.add_row(
        "Total (parse + resolve)",
        _fmt([p + r for p, r in zip(parse_times, resolve_times)]),
        f"{subsystem} subsystem",
    )

    console.print(table)


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli(obj={})
