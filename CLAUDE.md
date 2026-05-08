# CLAUDE.md — PyPIVTools-stereo-coc (worktree pointer)

This is a **git worktree** of `MTT69/python-PIVTOOLs` on branch `feature/stereo-ensemble-coc`. Verify with `git worktree list`.

> **Do not consult this file for content.** All project context — principles, gotchas, wiki, auto-update contract — lives at the fullstack root:
>
> - [`../CLAUDE.md`](../CLAUDE.md) — always-loaded principles
> - [`../wiki/index.md`](../wiki/index.md) — wiki map (read first)
> - [`../wiki/log.md`](../wiki/log.md) — recent sessions
>
> The previous 1697-line CLAUDE.md that lived here has been retired. Permanent archive: [`../backup/CLAUDE-PyPIVTools-stereo-coc-2026-04-29.md`](../backup/CLAUDE-PyPIVTools-stereo-coc-2026-04-29.md).

## Worktree-specific notes

- This worktree carries the in-progress stereo-CoC work. Mainline-stable code lives in `../PyPIVTools/` (worktree on `main`).
- Active threads: see the `wiki/sessions/2026-04-{26..29}-*.md` notes via `wiki/log.md`.
- When making a change in this worktree, ask: should it also land on `main` (via `../PyPIVTools/`)? The two will diverge until the feature branch merges.

## Local conveniences (this worktree only)

- Working data: `/Users/morgan/Downloads/stereo_ensemble/processed`
- Debug figures: `figures/debug/YYYY-MM-DD-<slug>/` (per CLAUDE.md figure rule).
- Manual diagnostic tools live under `manual_tools/` and are documented inline + via session notes (e.g. `coc_kspace_vs_gaussian.py`, `coc_vs_ab_autocorr_inspector.py`).
