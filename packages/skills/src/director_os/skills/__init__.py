"""DirectorOS skills — niche playbooks loaded as dynamic system instructions.

Playbooks live as markdown files next to this module; M2's Director Agent loads
them at runtime based on the project's `niche_skill`.
"""

from __future__ import annotations

from importlib import resources


def load_skill(niche: str) -> str:
    """Return the markdown playbook for a niche (e.g. `bodycam_documentary`)."""
    res = resources.files("director_os").joinpath(f"skills/{niche}.md")
    if not res.is_file():
        raise FileNotFoundError(f"no skill playbook for niche '{niche}'")
    return res.read_text(encoding="utf-8")
