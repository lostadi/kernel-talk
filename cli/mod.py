"""
cli/mod.py — ktalk mod <subcommand>

The Kernel Modification CLI.  Uses the vocabulary from the spec:
  propose / review / accept / discard / pending / preview / apply / history / revert

No Git verbs. Each word names what the user is doing, not Git's internals.
"""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

import click
import yaml

from core.mod.diff_parser import parse_diff_file, parse_unified_diff
from core.mod.models import Proposal, Snapshot
from core.mod.preview import PreviewContext
from core.mod.store import ModStore, file_sha256


# ── Helper: get ModStore from click context ────────────────────────────────────
def _get_mod_store(ctx: click.Context) -> ModStore:
    storage = ctx.obj.get("storage", str(Path.home() / ".kernel-talk/store"))
    mod_dir = str(Path(storage).parent / "mod")
    return ModStore(mod_dir)


# ── mod group ──────────────────────────────────────────────────────────────────
@click.group()
@click.pass_context
def mod(ctx: click.Context) -> None:
    """Kernel source modification commands.

    Propose → Review → Accept → Preview → Apply a patch to the kernel tree.
    """
    ctx.ensure_object(dict)


# ── propose ────────────────────────────────────────────────────────────────────
@mod.command()
@click.option("--from", "from_path", required=True,
              help="Path to a .diff / .patch file (unified diff format).")
@click.option("--description", default="", help="Human-readable description of this proposal.")
@click.option("--agent", default="user", help="Agent or author name.")
@click.option("--kernel", default=None, help="Kernel source root (for stripping paths).")
@click.pass_context
def propose(ctx: click.Context, from_path: str, description: str, agent: str, kernel: str | None) -> None:
    """Parse a unified diff and create a new Proposal."""
    store = _get_mod_store(ctx)

    diff_path = Path(from_path)
    if not diff_path.exists():
        click.echo(f"Error: diff file not found: {from_path}", err=True)
        sys.exit(1)

    hunks = parse_diff_file(diff_path, kernel_root=kernel)
    if not hunks:
        click.echo("Error: no hunks found in diff file.", err=True)
        sys.exit(1)

    if not description:
        description = f"Patch from {diff_path.name}"

    # Attach base snapshot ID if one exists
    base_snapshot_id = store.latest_snapshot_id()

    proposal = Proposal.create(
        description=description,
        hunks=hunks,
        author=agent,
        base_snapshot_id=base_snapshot_id,
    )
    store.save_proposal(proposal)

    click.echo(f"Proposal created: {proposal.id}")
    click.echo(f"  Files: {len(proposal.files_changed)}")
    click.echo(f"  Hunks: {len(proposal.hunks)}")
    click.echo(f"  State: {proposal.state}")
    click.echo("")
    click.echo("  Next: ktalk mod review " + proposal.id)


# ── review ─────────────────────────────────────────────────────────────────────
@mod.command()
@click.argument("proposal_id", required=False)
@click.option("--all", "show_all", is_flag=True, help="Review all proposals.")
@click.pass_context
def review(ctx: click.Context, proposal_id: str | None, show_all: bool) -> None:
    """Display a Proposal with colorized diff and Mirror context."""
    store = _get_mod_store(ctx)

    proposals = []
    if show_all:
        proposals = store.list_proposals()
    elif proposal_id:
        try:
            proposals = [store.load_proposal(proposal_id)]
        except FileNotFoundError:
            click.echo(f"Proposal {proposal_id} not found.", err=True)
            sys.exit(1)
    else:
        proposals = store.list_proposals(state="proposed")

    if not proposals:
        click.echo("No proposals to review.")
        return

    for p in proposals:
        _render_proposal(p)


def _render_proposal(p: Proposal) -> None:
    """Rich-formatted display of a Proposal."""
    state_color = {
        "proposed":  "yellow",
        "accepted":  "green",
        "discarded": "red",
        "applied":   "blue",
    }.get(p.state, "white")

    click.echo("─" * 70)
    click.echo(f"Proposal: {p.id}")
    click.echo(f"  Author:  {p.author}  |  Created: {p.created_at}")
    click.echo(f"  State:   " + click.style(p.state, fg=state_color, bold=True))
    click.echo(f"  Desc:    {p.description}")
    click.echo(f"  Files:   {len(p.files_changed)}")
    click.echo("")

    for hunk in p.hunks:
        click.echo(click.style(f"  --- {hunk.file_path}", fg="cyan"))
        click.echo(click.style(f"  @@ -{hunk.start_line} +{hunk.start_line} @@", fg="cyan"))
        for _, text in hunk.removed_lines:
            click.echo(click.style(f"  -{text}", fg="red"))
        for _, text in hunk.added_lines:
            click.echo(click.style(f"  +{text}", fg="green"))
        if hunk.affected_nodes:
            click.echo(f"  Mirror nodes: {', '.join(hunk.affected_nodes[:5])}")
        click.echo("")


# ── accept ─────────────────────────────────────────────────────────────────────
@mod.command()
@click.argument("proposal_id")
@click.pass_context
def accept(ctx: click.Context, proposal_id: str) -> None:
    """Accept a proposal and push it onto the pending stack."""
    store = _get_mod_store(ctx)
    try:
        proposal = store.update_proposal_state(proposal_id, "accepted")
    except FileNotFoundError:
        click.echo(f"Proposal {proposal_id} not found.", err=True)
        sys.exit(1)

    store.push_pending(proposal_id)
    click.echo(f"Accepted: {proposal_id}")
    click.echo(f"  '{proposal.description}'")
    click.echo("  Pushed to pending stack.")
    click.echo("  Next: ktalk mod preview  (or ktalk mod apply)")


# ── discard ────────────────────────────────────────────────────────────────────
@mod.command()
@click.argument("proposal_id")
@click.pass_context
def discard(ctx: click.Context, proposal_id: str) -> None:
    """Discard a proposal (preserved on disk; never deleted)."""
    store = _get_mod_store(ctx)
    try:
        store.update_proposal_state(proposal_id, "discarded")
    except FileNotFoundError:
        click.echo(f"Proposal {proposal_id} not found.", err=True)
        sys.exit(1)
    click.echo(f"Discarded: {proposal_id}")
    click.echo("  The proposal is preserved on disk and can be reconsidered later.")


# ── pending ────────────────────────────────────────────────────────────────────
@mod.command()
@click.pass_context
def pending(ctx: click.Context) -> None:
    """List the accepted-not-yet-applied proposal stack."""
    store = _get_mod_store(ctx)
    proposals = store.pending_proposals()

    if not proposals:
        click.echo("No pending proposals.")
        return

    click.echo(f"Pending stack ({len(proposals)} proposals):")
    for i, p in enumerate(proposals, 1):
        click.echo(f"  {i:2d}. [{p.id[:8]}] {p.description}  ({len(p.hunks)} hunks)")


# ── preview ────────────────────────────────────────────────────────────────────
@mod.command()
@click.option("--kernel", default=None, help="Kernel source root.")
@click.pass_context
def preview(ctx: click.Context, kernel: str | None) -> None:
    """Materialize pending proposals into a shadow tree and report changes."""
    store = _get_mod_store(ctx)
    proposals = store.pending_proposals()

    if not proposals:
        click.echo("No pending proposals to preview.")
        return

    kernel_root = kernel or ctx.obj.get("kernel", "/usr/src/linux")

    click.echo(f"Previewing {len(proposals)} proposal(s) against {kernel_root}...")

    with PreviewContext(kernel_root, proposals) as (shadow_root, result):
        click.echo(f"  Shadow tree: {shadow_root}")
        click.echo(f"  Files changed: {len(result.files_changed)}")
        for f in result.files_changed:
            click.echo(f"    ~ {f}")
        if result.ambiguities:
            click.echo(click.style(f"  ⚠ Ambiguities: {len(result.ambiguities)}", fg="yellow"))

    click.echo("Shadow tree removed (Preview never writes to real tree).")


# ── apply ──────────────────────────────────────────────────────────────────────
@mod.command()
@click.option("--kernel", default=None, help="Kernel source root to apply patches to.")
@click.option("--dry-run", is_flag=True, help="Show what would be applied without writing.")
@click.pass_context
def apply(ctx: click.Context, kernel: str | None, dry_run: bool) -> None:
    """Apply pending proposals to the real kernel source tree.

    Takes a Snapshot before writing any file.
    After Apply, use your normal build toolchain (make bzImage, etc.).
    """
    store = _get_mod_store(ctx)
    proposals = store.pending_proposals()

    if not proposals:
        click.echo("Nothing to apply.")
        return

    kernel_root = Path(kernel or ctx.obj.get("kernel", "/usr/src/linux"))

    if not kernel_root.exists():
        click.echo(f"Error: kernel root not found: {kernel_root}", err=True)
        sys.exit(1)

    click.echo(f"Applying {len(proposals)} proposal(s) to {kernel_root}...")

    if dry_run:
        click.echo(click.style("  DRY RUN — no files will be written.", fg="yellow"))

    # Take a snapshot before writing
    changed_files: dict[str, bytes] = {}
    manifest: list[tuple[str, str]] = []
    for p in proposals:
        for hunk in p.hunks:
            src = kernel_root / hunk.file_path
            if src.exists():
                content = src.read_bytes()
                changed_files[hunk.file_path] = content
                manifest.append((hunk.file_path, hashlib.sha256(content).hexdigest()))

    if not dry_run:
        snap = Snapshot.create(
            description=f"Before applying {len(proposals)} proposal(s)",
            parent_snapshot_id=store.latest_snapshot_id(),
            applied_proposals=[],
            manifest=manifest,
        )
        store.save_snapshot(snap, changed_files)
        click.echo(f"  Snapshot taken: {snap.id}")

    # Apply hunks
    from core.mod.preview import _apply_hunk
    applied_ids = []
    for p in proposals:
        for hunk in p.hunks:
            target = kernel_root / hunk.file_path
            if not dry_run:
                target.parent.mkdir(parents=True, exist_ok=True)
                _apply_hunk(target, hunk)
                click.echo(f"  ✓ {hunk.file_path}")
            else:
                click.echo(f"  ~ {hunk.file_path}  (dry-run)")
        applied_ids.append(p.id)

    if not dry_run:
        # Mark proposals as applied
        for pid in applied_ids:
            store.update_proposal_state(pid, "applied")
        store.clear_pending()
        click.echo("")
        click.echo(click.style("Apply complete.", fg="green", bold=True))
        click.echo("  Now rebuild your kernel: make -j$(nproc) bzImage")
        click.echo("  A reboot is required for changes that aren't loadable modules.")


# ── history ────────────────────────────────────────────────────────────────────
@mod.command()
@click.pass_context
def history(ctx: click.Context) -> None:
    """List all snapshots (taken before each Apply)."""
    store = _get_mod_store(ctx)
    snapshots = store.list_snapshots()

    if not snapshots:
        click.echo("No snapshots found.")
        return

    click.echo(f"Snapshot history ({len(snapshots)} entries):")
    for s in snapshots:
        click.echo(f"  [{s.id[:8]}] {s.timestamp}  {s.description}")
        if s.applied_proposals:
            click.echo(f"           Proposals: {', '.join(s.applied_proposals)}")


# ── revert ─────────────────────────────────────────────────────────────────────
@mod.command()
@click.argument("snapshot_id")
@click.option("--kernel", default=None, help="Kernel source root.")
@click.option("--dry-run", is_flag=True, help="Show what would be reverted without writing.")
@click.pass_context
def revert(ctx: click.Context, snapshot_id: str, kernel: str | None, dry_run: bool) -> None:
    """Revert the kernel source tree to the state at a given Snapshot."""
    store = _get_mod_store(ctx)
    kernel_root = Path(kernel or ctx.obj.get("kernel", "/usr/src/linux"))

    try:
        snap = store.load_snapshot(snapshot_id)
    except FileNotFoundError:
        click.echo(f"Snapshot {snapshot_id} not found.", err=True)
        sys.exit(1)

    click.echo(f"Reverting to snapshot: {snap.id[:8]}  ({snap.description})")
    if dry_run:
        click.echo(click.style("  DRY RUN", fg="yellow"))

    for rel_path, expected_sha in snap.manifest:
        content = store.snapshot_file(snap.id, rel_path)
        if content is None:
            click.echo(f"  ⚠ {rel_path}: not found in snapshot chain")
            continue
        target = kernel_root / rel_path
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            click.echo(f"  ✓ {rel_path}")
        else:
            click.echo(f"  ~ {rel_path}  (dry-run)")

    if not dry_run:
        click.echo(click.style("Revert complete.", fg="green", bold=True))
