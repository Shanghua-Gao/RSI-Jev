"""The demo: one page, one server, and the data it opens on.

These run with no weights, no GPU and no network. They cover the three things that
have actually broken here: a shipped example that the wire contract rejects, a page
that drifts from the endpoint it calls, and the sys.path order that makes
`from serve.X import ...` resolve to scripts/serve.py instead of the package.

    python -m pytest tests/test_demo.py -q
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT))          # root first: scripts/serve.py shadows serve/

from serve.wire import parse_questions                     # noqa: E402

PAGE = (ROOT / "serve" / "ui.html").read_text()
SERVER = (ROOT / "scripts" / "demo_web.py").read_text()
EXAMPLES = json.loads((ROOT / "serve" / "examples.json").read_text())


def test_every_example_parses_against_the_contract():
    """A prefilled example that the contract rejects breaks the demo on open. This
    caught `score` criteria shipped as a mapping when the contract wants an ordered
    array, worst level first."""
    assert EXAMPLES, "no examples shipped"
    for e in EXAMPLES:
        qs = parse_questions(e["questions"])              # raises on an invalid spec
        assert {q.key for q in qs} == set(e["gold"]), e["id"]
        for q in qs:
            assert e["gold"][q.key] in q.options, \
                f"{e['id']}/{q.key}: gold {e['gold'][q.key]!r} not in {q.options}"


def test_score_criteria_are_ordered_arrays():
    seen = 0
    for e in EXAMPLES:
        for key, q in e["questions"].items():
            if q["type"] == "score":
                assert isinstance(q["criteria"], list), \
                    f"{e['id']}/{key}: score criteria must be an array, worst level first"
                seen += 1
    assert seen, "no score questions among the examples; one mode is unexercised"


def test_all_three_modes_are_offered():
    modes = {q["type"] for e in EXAMPLES for q in e["questions"].values()}
    assert modes == {"noul", "choice", "score"}, modes


def test_documents_are_pretty_printed():
    """These states are JSON. Shipped as one long line, the page reads as a wall of
    text, which is exactly what it looked like."""
    for e in EXAMPLES:
        json.loads(e["state"])
        assert "\n" in e["state"], f"{e['id']} is not pretty-printed"


def test_the_page_and_the_server_agree_on_the_wire():
    """Both routes the page calls must exist, and every field it reads must be sent."""
    for route in ("demo/config", "demo/compare"):
        assert f'"{route}"' in PAGE, f"the page no longer calls {route}"
        assert f'"/{route}"' in SERVER, f"the server no longer serves /{route}"
    for field in ("questions", "answers", "picks", "gold", "ms", "tokens"):
        assert f'd["{field}"]' in PAGE or f'd.{field}' in PAGE, \
            f"the page stopped reading {field}"
        assert f'"{field}"' in SERVER, f"the server stopped sending {field}"


def test_the_document_is_a_form_not_a_json_box():
    """Every state in this dataset is a JSON object. Shown as a JSON string, a reader
    parses braces instead of reading the case, so it is rendered as labelled fields
    and edited in place."""
    assert "function renderDoc" in PAGE and "function fieldRow" in PAGE
    assert "scalarInput" in PAGE, "field values must be real inputs"
    assert '"+ Add field"' in PAGE, "a field must be addable"
    assert 'singular(key).toLowerCase()' in PAGE, \
        "a list item is added by its own name: + Add order, not + Add item"
    assert 'id="fmtText"' in PAGE and 'id="fmtJson"' in PAGE, \
        "the input format must be switchable; prose documents have no fields"


def test_the_questions_are_a_builder_not_a_json_box():
    """One card per question with its type, what to ask, and the options or rubric
    levels: the same shape the API takes, one control per field."""
    assert "function renderQs" in PAGE and "function wireQuestions" in PAGE
    assert 'document.createElement("select")' in PAGE, \
        "the type picker is a labelled dropdown, as in the reference playground"
    for control in ('"+ Add option"', '"+ Add level"', "+ Add question"):
        assert control in PAGE, f"missing control: {control}"
    for field in ('labelled("ID"', 'labelled("Type"', 'labelled("Instructions"'):
        assert field in PAGE, f"every question field needs a visible label: {field}"
    assert '"Option name"' in PAGE and '"Description (optional)"' in PAGE
    assert "worst first" in PAGE, "score levels are ordered; say so"


def test_the_builder_keeps_noul_wording():
    """A noul question can carry wording for each outcome and the model reads it:
    dropping the example's wording moved one answer from 0.339 to 0.282. The builder
    must round-trip it, not quietly discard it."""
    assert "q.yes" in PAGE and "q.no" in PAGE
    assert 'o.criteria = {true: q.yes.trim() || "Yes", false: q.no.trim() || "No"}' in PAGE
    for e in EXAMPLES:                      # and the examples must exercise that path
        for q in e["questions"].values():
            if q["type"] == "noul" and q.get("criteria"):
                return
    pytest.fail("no noul example carries criteria, so the round-trip is untested")


def test_the_reference_is_matched_structurally_not_textually():
    """The page rebuilds the document from its fields, and Python and JavaScript do not
    serialise identically -- a float of 32.0 comes back as "32" -- so a string compare
    would drop the reference column the moment an example passed through the form."""
    assert "_same_document" in SERVER
    assert "json.loads(a) == json.loads(b)" in SERVER


def test_no_reference_is_invented_for_an_edited_document():
    assert "_same_document(e[\"state\"], state_text)" in SERVER
    assert "No teacher" in PAGE


def test_the_demo_runs_on_apple_silicon_and_cpu_too():
    """DGX Spark is the target, but the device choice must not be CUDA-only."""
    assert "mps" in SERVER and '"cuda", "mps", "cpu"' in SERVER
    # bf16 is a CUDA-only decision; MPS and CPU stay fp32.
    assert '"bf16" if device == "cuda" else "fp32"' in SERVER


def test_scripts_put_the_root_before_their_own_directory():
    """scripts/serve.py shadows the serve/ package, so a script that inserts its own
    directory first gets "serve is not a package" the moment it imports serve.infer."""
    for script in sorted((ROOT / "scripts").glob("*.py")):
        lines = [i for i, l in enumerate(script.read_text().splitlines())
                 if "sys.path.insert" in l]
        if len(lines) < 2:
            continue
        text = script.read_text().splitlines()
        root_at = next((i for i in lines if re.search(r"parent\.parent|str\(ROOT\)\)", text[i])), None)
        own_at = next((i for i in lines if root_at is not None and i != root_at), None)
        if root_at is not None and own_at is not None:
            assert root_at > own_at, (
                f"{script.name} inserts the root before its own directory, so "
                f"scripts/ ends up first on sys.path and shadows serve/")


def test_the_page_loads_no_external_resources():
    """It has to work on a desk machine with no internet, so nothing may be fetched:
    no CDN script, no web font, no remote stylesheet or image. Plain links out are
    fine -- they are only followed if someone clicks."""
    for tag in re.findall(r"<(script|link|img|source)\b[^>]*>", PAGE):
        assert "http" not in tag, f"external resource: {tag}"
    assert "@import" not in PAGE and "fonts.googleapis" not in PAGE
    tree = ast.parse(SERVER)
    assert ast.get_docstring(tree), "the demo must explain itself at the top"


def test_the_page_says_what_it_does_before_anything_else():
    """A visitor could not tell what the page was for. It now opens with the job in one
    sentence, before any control."""
    opening = PAGE.split("<section>")[0]
    assert "typed questions" in opening and "probabilities" in opening, \
        "the opening line must say what you give it and what comes back"
    assert '<span class="n">' not in PAGE, \
        "numbered step markers claim a sequence the section headings already carry"
    assert 'id="run"' in PAGE, "an explicit Run button, so the action is visible"


def test_the_readout_draws_the_distribution_against_a_scale():
    """The subject's characteristic object is a probability distribution, so the page
    shows its shape: one track per option, one trace per selected version inside it, and
    quartile rules behind them so a threshold is read rather than estimated. A table of
    percentages hides exactly the thing a version comparison is about."""
    assert 'class="track"' in PAGE and 'class="bar"' in PAGE
    assert "repeating-linear-gradient" in PAGE, "the track needs its quartile rules"
    assert "function distribution" in PAGE, "every option must get a bar, not just the pick"
    assert '"--v" + ci' in PAGE or "var(--v${ci})" in PAGE, \
        "each version needs its own trace colour"
    # A legend for those colours, or the traces are unreadable.
    assert 'class="scale"' in PAGE


def test_both_themes_are_complete():
    """One palette is the base on bare :root and the other is a full counterpart, reached
    both by the reader's system setting and by an explicit choice. A colour defined only
    inside a media or [data-theme] block is the classic one-theme-unreadable bug, so the
    counterpart's tokens must be a subset of the base's."""
    css = PAGE.split("</style>")[0]
    base = set(re.findall(r"--([a-z0-9-]+):", css.split(":root{")[1].split("}")[0]))
    assert {"bg", "ink", "panel", "rule", "noul", "choice", "score", "hit", "miss"} <= base
    # Whichever theme is the base, the other one must exist as a complete counterpart.
    other = next(t for t in ("light", "dark") if f':root[data-theme="{t}"]{{' in css)
    for block in (f"@media (prefers-color-scheme:{other})", f':root[data-theme="{other}"]'):
        assert block in css, f"missing theme path: {block}"
    counter = set(re.findall(r"--([a-z0-9-]+):",
                             css.split(f':root[data-theme="{other}"]{{')[1].split("}")[0]))
    assert counter <= base, f"defined only in the {other} override: {sorted(counter - base)}"
    assert "background:var(--bg)" in css, "body needs an explicit token background"
    assert css.count("color-scheme:") >= 3, "each palette sets color-scheme"


def test_the_palette_is_ours():
    """An earlier pass lifted the reference playground's own tokens -- its pink accent, its
    green-cast black, its dot texture, its pixel display face. The layout is still theirs
    and credited; none of the colour or surface treatment is.

    Black is kept, because a black instrument face is nobody's property, but it is a warm
    neutral rather than their green cast, and the accent is amber rather than their pink.
    Five roles carry five hues, so none does two jobs."""
    css = PAGE.split("</style>")[0]
    for borrowed in ("#efb8ca", "#e89ab5", "#080908", "#101211", "#151817", "#2a302c",
                     "Pixelify", "var(--dot)", "--pixel:"):
        assert borrowed not in PAGE, f"still carrying the reference's own {borrowed}"
    modes = {re.search(rf"--{m}:(#[0-9a-f]{{6}})", css).group(1)
             for m in ("noul", "choice", "score")}
    assert "Okabe" not in PAGE, "do not claim a palette this page no longer uses"
    semantic = {re.search(rf"--{m}:(#[0-9a-f]{{6}})", css).group(1) for m in ("hit", "miss")}
    assert len(modes) == 3, f"the three modes must be distinguishable: {modes}"
    assert not (modes & semantic), f"a hue cannot mean both a mode and correctness: {modes & semantic}"
    css = PAGE.split("</style>")[0]
    # Near-black is fine and is not anyone's property. What was borrowed was the accent,
    # so that is what must differ: theirs is a pink (#efb8ca) on a green-cast black.
    ground = re.search(r"--bg:(#[0-9a-f]{6})", css).group(1)
    r, g, b = (int(ground[i:i+2], 16) for i in (1, 3, 5))
    assert max(r, g, b) - min(r, g, b) <= 4, \
        f"a near-black ground should be neutral, not tinted like theirs: {ground}"
    accent = re.search(r"--accent:(#[0-9a-f]{6})", css).group(1)
    assert accent.lower() not in ("#efb8ca", "#e89ab5"), "that accent is theirs"


def test_the_layout_is_two_columns_with_a_sticky_result():
    """Theirs is left-right, not one column: inputs on the left at
    minmax(0,1.05fr), run and results on the right at minmax(0,1fr) with a 24px gap,
    collapsing to one column at 900px, and the right column sticks so the answer stays
    in view while you edit. An earlier pass built a single column because a prose summary
    of their page said so; their stylesheet says otherwise."""
    css = PAGE.split("</style>")[0]
    assert "grid-template-columns:minmax(0,1.05fr) minmax(0,1fr)" in css, \
        "the two columns and their ratio"
    assert "gap:24px" in css
    assert re.search(r"@media \(max-width:900px\)\{\.grid\{grid-template-columns:1fr\}",
                     css.replace("\n", "").replace("  ", "")), \
        "the grid must collapse at their breakpoint"
    # Theirs sticks at 88px to clear their site navigation. This page has no nav bar, so
    # the same offset would leave a dead gap; it sticks near the top instead.
    assert re.search(r"position:sticky;top:\d+px", css), "the result column must stick"
    html = PAGE.split("<script>")[0]
    left = html.index('class="col"')
    right = html.index('class="col sticky"')
    assert left < html.index(">State<") < html.index(">Questions<") < right, \
        "state and questions belong to the left column"
    assert right < html.index(">Run decision<") < html.index(">API request<"), \
        "run, results and the request belong to the sticky right column"


def test_the_sections_run_in_the_order_you_work():
    """The reference playground numbers its sections, and here the content really is a
    sequence: fill in the state, define the questions, run, read the result. An earlier
    pass removed the numbers on the theory that numbering always implies a false
    sequence; this one does not."""
    html = PAGE.split("<script>")[0]
    order = [" ".join(re.sub(r"<[^>]+>", " ", m).split())
             for m in re.findall(r"<h[23][^>]*>(.*?)</h[23]>", html, re.S)]
    assert order == ["01 State", "02 Questions", "03 Run decision", "Results",
                     "04 API request"], order


def test_every_question_type_says_what_to_fill_in():
    """The first copy of this layout took the reference's section names and dropped all
    of its helper text, which is the half that teaches. Each type states what to fill
    in, the real limit, and what comes back -- with this project's numbers, not the
    reference's."""
    for key, must in (("noul", "probability of yes"),
                      ("choice", "options"),
                      ("score", "ordered low to high")):
        assert key in PAGE.split("const HINT")[1].split("};")[0], f"no hint for {key}"
        assert must in PAGE, f"the {key} hint does not say {must!r}"
    assert "2 to ${MAX_ANSWERS}" in PAGE, "limits must come from the contract, not be typed in"
    assert "All questions share this context" in PAGE, \
        "the state needs the line that explains why asking ten is cheap"


def test_each_question_card_is_numbered_and_removable():
    assert 'who.textContent = `Question ${qi + 1}`' in PAGE
    assert 'className = "qnum"' in PAGE
    assert PAGE.count('textContent = "Remove"') >= 2, "cards and rows both need Remove"


def test_the_noul_outcomes_are_labelled_not_just_hinted():
    """Placeholders vanish the moment you type. The reference labels these fields, and
    their wording reaches the model, so it must be visible."""
    assert '["yes", "Yes means"], ["no", "No means"]' in PAGE


def test_the_reference_playground_is_credited():
    assert "jevai.org/playground" in PAGE, "credit the layout we followed"


def test_editing_does_not_spend_the_gpu():
    """A run is one forward pass per selected version -- about 290 ms for two -- and the
    device serialises them. Re-running on each keystroke queued a pass per character, and
    aborting the fetch did not stop the server computing it, so a twelve-character edit
    cost about 3.5 s of GPU for one answer anyone wanted. Editing now marks the answer
    stale and Run spends the GPU, which is what the reference playground does."""
    assert "function schedule(){ dirty = true; paint(); }" in PAGE, \
        "editing must only mark the result stale"
    assert "setTimeout(go" not in PAGE, "no timer may fire a run"
    assert "if (running) return;" in PAGE, "one pass at a time"
    assert 'id="staleNote"' in PAGE and "#out.stale" in PAGE, \
        "a stale answer must say so rather than look current"
    assert 'e.key === "Enter"' in PAGE, "keyboard run, as the reference offers"


def test_the_run_button_reports_its_own_state():
    """It was possible to press Run while a run was in flight and see nothing change."""
    paint = PAGE[PAGE.index("function paint()"):]
    paint = paint[:paint.index("\n}")]
    assert "btn.disabled = running" in paint and '"Running"' in paint


def test_the_format_switch_round_trips():
    """Switching to Text left the state an object, and switching back parsed the literal
    "null", refused, and claimed the state was plain text -- so JSON could never be
    reached again. Text now stringifies, and only text is ever parsed."""
    fn = PAGE[PAGE.index("function setFormat"):]
    fn = fn[:fn.index("\n}")]
    assert 'if (!free && typeof DOC === "string")' in fn, \
        "only a text state needs parsing"
    assert 'if (free && typeof DOC !== "string") DOC = JSON.stringify' in fn, \
        "switching to Text must stringify, or the trip back has nothing to parse"
    assert "not a JSON object" in PAGE, "the refusal must say what is actually wrong"
    assert "isJsonObject" in PAGE, "JSON is unavailable for prose, and must look it"


def test_a_new_question_arrives_usable():
    """wireQuestions drops a question with no ID, so a new card with an empty ID did
    nothing at all -- and the counter, which only counted named questions, did not move
    either. Adding one looks exactly like a button that does not work."""
    add = PAGE[PAGE.index('$("#addq").addEventListener'):]
    add = add[:add.index("});")]
    assert "`question${n}`" in add, "a new question needs an ID that already works"
    assert "QS.some(q => q.key.trim() === key)" in add, "and it must be unique"
    assert f"if (QS.length >= MAX_QUESTIONS) return;" in add, "respect the contract limit"
    cnt = PAGE[PAGE.index("function count()"):]
    cnt = cnt[:cnt.index("\n}")]
    assert "${QS.length} of ${MAX_QUESTIONS}" in cnt, \
        "the counter must count what you added, not only what you named"
    assert 'id="unnamed"' in PAGE, "an unnamed question must say it will not be asked"


def test_a_list_of_objects_is_not_flattened_into_a_text_box():
    """The customer-service state holds a message thread: a list of objects. Every list
    item went through a text input, so each message rendered as "[object Object]" and the
    first keystroke replaced the object with that string. Objects recurse now."""
    fn = PAGE[PAGE.index("if (Array.isArray(value)){"):]
    fn = fn[:fn.index("into.append(fieldRow(key, box, true));")]
    assert 'if (item && typeof item === "object")' in fn, \
        "an object in a list must not go through scalarInput"
    assert "renderValue(k, item[k]" in fn, "recurse into each field of the object"
    assert "Object.fromEntries(Object.keys(last)" in fn, \
        "adding to a list of objects should give the shape already in it"
    data = json.loads((ROOT / "serve" / "examples.json").read_text())
    nested = [e["title"] for e in data
              if any(isinstance(v, list) and v and isinstance(v[0], dict)
                     for v in json.loads(e["state"]).values())]
    assert nested, "no example exercises a list of objects, so this would go untested"


def test_booleans_are_a_choice_and_empty_numbers_stay_numbers():
    """A boolean in a text box coerced silently -- "tru" became false with nothing to
    show it. And clearing a numeric field stored "" , turning a number in the document
    into a string."""
    assert 'if (typeof value === "boolean"){' in PAGE and "createElement(\"select\")" in PAGE
    assert 'v = v.trim() === "" ? null' in PAGE, "an emptied number is null, not \"\""
    data = json.loads((ROOT / "serve" / "examples.json").read_text())
    kinds = set()
    def walk(v):
        if isinstance(v, dict): [walk(x) for x in v.values()]
        elif isinstance(v, list): [walk(x) for x in v]
        else: kinds.add(type(v).__name__)
    for e in data: walk(json.loads(e["state"]))
    assert "bool" in kinds, "no example has a boolean, so the menu would go untested"


def test_the_page_opens_on_a_situation_a_stranger_recognises():
    """It opened on an internal agent trace -- autonomy: checkpointed, an
    irreversible_actions count -- which tells a first-time visitor nothing. The reference
    playground opens on a customer asking for a refund."""
    data = json.loads((ROOT / "serve" / "examples.json").read_text())
    assert data[0]["title"].startswith("Customer service"), \
        f"the page opens on {data[0]['title']!r}"


def test_field_labels_are_written_for_people():
    """The state's JSON keys were used directly as labels, so the panel read
    lifetime_value_usd / 243 / prior_tickets_90d / 0 -- names written for a program, and
    numbers with no units. The label is derived from the key, the unit or time window is
    pulled out of the suffix, and the raw key stays beside it so nothing is hidden and
    the Text view still matches."""
    assert "function humanKey" in PAGE
    assert 'className = "rawkey"' in PAGE, "the real key must stay visible"
    assert "const UNIT = {usd:" in PAGE and "const SPAN = {d:" in PAGE, \
        "units and time windows come out of the suffix"
    assert "w.slice(0, -1).toUpperCase()" in PAGE, "a plural acronym reads IDs, not IDS"
    assert "function singular" in PAGE or "const singular" in PAGE, \
        "one item of a list is named in the singular"


def test_an_added_field_can_be_named():
    """+ Add field invented a name -- field, field2 -- and every label renders as text, so
    there was no way to ever change it. A new field arrives with its name in an input."""
    assert "const FRESH = new Set()" in PAGE
    assert "function newFieldRow" in PAGE and 'placeholder = "field name"' in PAGE
    assert "function rekey" in PAGE, "renaming must keep the field's place in the object"
    for reset in ("function blank()", "function load(id)"):
        block = PAGE[PAGE.index(reset):][:200]
        assert "FRESH.clear()" in block, f"{reset} must drop pending new fields"


def test_the_state_opens_as_one_readable_block():
    """The field tree was the opening view, which turned the first example into 27
    labelled boxes, buried the conversation the model reads three levels down, and pushed
    the Questions panel off the screen. Text opens; fields are there when you want to
    change one value."""
    load = PAGE[PAGE.index("function load(id)"):]
    load = load[:load.index("renderDoc()")]
    assert "DOC = e.state; FREE = true;" in load, "an example opens as text"
    assert "JSON.parse(e.state)" not in load, "do not open on the field tree"


def test_each_panel_says_what_it_is_for():
    """A reader could not tell which half was the context and which was the question."""
    assert PAGE.count('class="what"') >= 3, "state, questions and results each need a line"
    for phrase in ("Everything the decision depends on",
                   "What you want decided about it",
                   "One row per option"):
        assert phrase in PAGE, f"missing: {phrase}"


def test_speed_is_read_at_the_top_of_the_answer():
    """Speed is the point of the architecture and was a line of small print under the
    result. It now leads the answer: total, per decision, and decisions per second for
    each selected version."""
    html = PAGE.split("<script>")[0]
    assert html.index('id="speed"') < html.index('id="out"'), \
        "the speed strip belongs above the answers, not under them"
    for part in ("ms / decision", "decisions/s", "tokens written"):
        assert part in PAGE, f"missing from the speed strip: {part}"


def test_the_measurement_takes_medians_and_refuses_a_degenerate_split():
    """Timing one question and then N splits the cost into a fixed read and a marginal
    per decision. With one sample per point a five-question run came back faster than a
    one-question run and the split printed "0.0 ms per decision" -- a clamp hiding a bad
    measurement. Each point is now the median of three after a discarded warm-up, and a
    split that is still degenerate says so."""
    fn = PAGE[PAGE.index("async function measure()"):]
    fn = fn[:fn.index("/* One request")]
    assert "const median" in PAGE, "points must be medians, not single samples"
    assert "if (!i) continue;" in fn, "discard a warm-up before sampling"
    assert "marg > 0.5 && fixed > 0" in fn, "a degenerate split must not be printed"
    assert "too close to split cleanly" in fn, "and it must say why"
    assert "Math.max(0, fixed)" not in fn, "clamping a bad fit to zero hides it"
