"""Which documents are about a release, and which are about the project.

RSI-Jev ships versioned releases, so a document is one of two things. A document
*about a release* -- the README's shop window and each version record -- carries a
marker naming it, because one that quietly describes an older release is worse
than no document: it reads as current. A document about the *project* -- the wire
contract, the code guide, how to contribute -- outlives every release and carries
no marker, because stamping it means editing files each cut to change nothing.

Both directions are checked, so the distinction cannot rot in either.

    python -m pytest tests/test_docs_version.py -q
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_ROOT = Path(__file__).resolve().parent.parent
# About one release, so marked. Only version records are: the README is the shop
# window and the rest describe the project, both of which outlive a version.
RELEASE_DOCS = []
PROJECT_DOCS = ["BENCHMARKS.md", "CONTRIBUTING.md", "README.md", "EXPLORE.md",
                "rsijev/README.md", "serve/README.md"]
DOCS = RELEASE_DOCS + PROJECT_DOCS
MARKER = re.compile(r"\*Describes release (v\d+\.\d+) · updated (\d{4}-\d{2}-\d{2})\*")


def released_versions() -> list[str]:
    return sorted(p.stem for p in (ROOT / "versions").glob("v*.md"))


@pytest.mark.parametrize("rel", RELEASE_DOCS + ["versions/<latest>"])
def test_a_release_document_says_which_release(rel):
    if rel == "versions/<latest>":
        rel = f"versions/{released_versions()[-1]}.md"
    m = MARKER.search((ROOT / rel).read_text())
    assert m, (f"{rel} describes a release, so it needs a marker just under the title:\n"
               f"    *Describes release <version> · updated <YYYY-MM-DD>*")


@pytest.mark.parametrize("rel", PROJECT_DOCS)
def test_a_project_document_does_not_claim_one(rel):
    """Otherwise every release means editing these to change nothing."""
    assert not MARKER.search((ROOT / rel).read_text()), (
        f"{rel} is about the project, not a release. Drop the marker; if it really "
        f"only holds for one release, move that part into versions/.")


def test_every_root_document_is_classified():
    """A new root .md must be declared one or the other, not silently neither."""
    unfiled = {p.name for p in ROOT.glob("*.md")} - set(RELEASE_DOCS) - set(PROJECT_DOCS)
    assert not unfiled, (f"add {sorted(unfiled)} to RELEASE_DOCS or PROJECT_DOCS. "
                         f"A document is about a release or about the project.")


def test_docs_describe_the_current_release():
    """A release document must name the newest release, not a superseded one."""
    latest = released_versions()[-1]
    stale = {rel: MARKER.search((ROOT / rel).read_text()).group(1)
             for rel in RELEASE_DOCS
             if MARKER.search((ROOT / rel).read_text()).group(1) != latest}
    assert not stale, (f"these describe an older release than {latest}: {stale}. "
                       f"Update them, or say in the file why they lag.")


def test_versions_holds_nothing_but_release_records():
    """Records accumulate here; everything else about a release does not.

    `main` carries one version of the recipe, the scripts and the guides, and older
    trees live on their own branch -- but every release's card stays, because the
    chain of what was tried and killed is the only way to see whether the loop
    improves. So this directory grows, and must grow with nothing but records in it:
    a draft or a stray comparison table here would be indistinguishable from
    evidence.
    """
    stray = sorted(p.name for p in (ROOT / "versions").iterdir()
                   if not re.fullmatch(r"v\d+\.\d+\.md", p.name))
    assert not stray, (f"versions/ holds only v<major>.<minor>.md release records; found "
                       f"{stray}. Anything else belongs in the root docs or in scripts/.")


# A chain of records is only readable if the records are the same shape. These are
# the sections a reader compares across releases; a card may add more.
COMPARABLE = ["the task", "built and trained", "checkpoints", "results",
              "speed", "limitations", "how it got here"]


@pytest.mark.parametrize("version", released_versions())
def test_every_release_card_is_comparable_to_the_others(version):
    headings = " | ".join(re.findall(r"^## .+$", (ROOT / f"versions/{version}.md").read_text(),
                                     re.M)).lower()
    missing = [s for s in COMPARABLE if s not in headings]
    assert not missing, (f"versions/{version}.md is missing {missing}. Every release card "
                         f"carries these sections so two releases can be read side by side.")


def test_a_version_card_describes_itself():
    """versions/v1.0.md claims v1.0, not some other release."""
    wrong = {}
    for p in sorted((ROOT / "versions").glob("v*.md")):
        m = MARKER.search(p.read_text())
        if m and m.group(1) != p.stem:
            wrong[p.name] = m.group(1)
    assert not wrong, f"version cards whose marker disagrees with their filename: {wrong}"


# ---------------------------------------------------------------------------
# The README quotes the version card. Nothing else may restate its numbers.
# ---------------------------------------------------------------------------

FIGURE = re.compile(r"^0\.\d{3,4}$")
QUOTING = [d for d in DOCS if "/" not in d]


# Columns that hold a fit statistic rather than an accuracy. A goodness-of-fit or a
# standard deviation is a property of a measurement taken here; it is not a score
# the release record has to back.
NOT_A_SCORE = ("r2", "r²", "sd", "brier", "error")


def cells(line: str) -> list[str]:
    return [x.strip().strip("*") for x in line.strip().strip("|").split("|")]


def quoted_figures(rel: str) -> list[str]:
    """Every accuracy-shaped figure in a markdown table, by column.

    Column-aware on purpose: an R2 of 0.994 looks exactly like a top-1 of 0.994
    to a regex, and only one of the two is a claim about the model.
    """
    found, block = [], []
    for line in (ROOT / rel).read_text().splitlines() + [""]:
        if line.startswith("|"):
            block.append(line)
            continue
        if len(block) >= 3:
            header = [h.lower() for h in cells(block[0])]
            for row in block[2:]:
                for i, c in enumerate(cells(row)):
                    head = header[i] if i < len(header) else ""
                    if FIGURE.match(c) and not any(k in head for k in NOT_A_SCORE):
                        found.append(c)
        block = []
    return found


def test_the_front_pages_only_quote_figures_the_card_backs():
    """A number outside the release record must be findable inside it.

    The root documents show the headline; versions/<latest>.md is where it is
    earned. Change one without the other and this fails, naming the figure --
    which is the failure mode that made the docs drift in the first place.
    """
    card = (ROOT / f"versions/{released_versions()[-1]}.md").read_text()
    seen = {rel: quoted_figures(rel) for rel in QUOTING}
    assert any(seen.values()), f"no accuracy figures found in any of {QUOTING}; did a table move?"
    missing = {rel: [f for f in figs if f not in card]
               for rel, figs in seen.items() if any(f not in card for f in figs)}
    assert not missing, (f"these quote figures the release record does not: {missing}. "
                         f"Either the card is stale or a document invented a number.")


def test_the_superseded_confidence_formula_appears_only_as_history():
    """`1 - H(p)/ln K` was wrong. It may be recorded, never documented as current."""
    offenders = {rel for rel in DOCS if "H(p)" in (ROOT / rel).read_text()}
    assert not offenders, (f"{offenders} still present the entropy confidence statistic. "
                           "The published one is the peak statistic (K*p_max - 1)/(K - 1).")
    assert "(K * p_max - 1) / (K - 1)" in (ROOT / f"versions/{released_versions()[-1]}.md").read_text()


def test_relative_links_between_docs_resolve():
    """A moved section or renamed file breaks a link silently. Not any more."""
    link = re.compile(r"\[[^\]]+\]\((?!https?:|mailto:)([^)]+)\)")
    broken = []
    for rel in DOCS + [f"versions/{v}.md" for v in released_versions()]:
        src = ROOT / rel
        text = src.read_text()
        anchors = {re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-")
                   for h in re.findall(r"^#+ (.+)$", text, re.M)}
        for target in link.findall(text):
            path, _, frag = target.partition("#")
            if not path:                                    # same-file anchor
                if frag not in anchors:
                    broken.append(f"{rel} -> #{frag}")
                continue
            dest = (src.parent / path).resolve()
            if not dest.exists():
                broken.append(f"{rel} -> {path}")
            elif frag and dest.suffix == ".md":
                dest_anchors = {re.sub(r"[^a-z0-9 -]", "", h.lower()).replace(" ", "-")
                                for h in re.findall(r"^#+ (.+)$", dest.read_text(), re.M)}
                if frag not in dest_anchors:
                    broken.append(f"{rel} -> {path}#{frag}")
    assert not broken, f"broken relative links: {broken}"


def test_no_doc_embeds_an_svg_that_needs_animation_to_be_visible():
    """A figure whose marks start invisible must not ship as an SVG.

    Our figures reveal themselves with SMIL: the dots are `opacity="0"` until an
    `<animate>` brings them in. Any renderer that strips SMIL -- and GitHub's
    handling of user-content SVG is not something we can rely on -- would show an
    empty chart rather than a still one. GIF has no such failure mode, so the docs
    embed GIFs and the SVGs stay as the source they are generated from.
    """
    img = re.compile(r'<img\s+src="([^"]+)"')
    bad = []
    for rel in DOCS + [f"versions/{v}.md" for v in released_versions()]:
        src = ROOT / rel
        for target in img.findall(src.read_text()):
            if target.startswith("http") or not target.endswith(".svg"):
                continue
            f = (src.parent / target).resolve()
            if f.exists() and 'opacity="0"' in (s := f.read_text()) and "<animate" in s:
                bad.append(f"{rel} -> {target}")
    assert not bad, (f"these embed an SVG that is blank without SMIL: {bad}. "
                     f"Embed the generated .gif instead and keep the .svg as source.")
