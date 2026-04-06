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


# ─── Entry Point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    cli(obj={})
