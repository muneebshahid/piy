"""Tests for the default file edit tool."""

from pathlib import Path

import pytest
from pydantic import ValidationError

import tile.tools.edit as edit
from tile.tools.edit import EditDetails
from tests.support.files import read_text, write_text
from tests.support.tool_results import details_of, tool_text


def test_edit_schema_exposes_edit_controls() -> None:
    """Expose path plus edits inputs and require both."""

    properties = edit.tool.input_schema["properties"]

    assert edit.tool.name == "edit"
    assert isinstance(properties, dict)
    assert set(properties) == {"path", "edits"}
    assert edit.tool.input_schema["required"] == ["path", "edits"]


def test_edit_input_requires_at_least_one_replacement() -> None:
    """Reject empty edit collections at the model boundary."""

    with pytest.raises(ValidationError, match="at least 1 item"):
        edit.EditInput(path="sample.txt", edits=[])


def test_edit_schema_describes_each_replacement() -> None:
    """Expose old_text and new_text for every edit item."""

    schema = edit.tool.input_schema
    properties = schema["properties"]
    assert isinstance(properties, dict)

    edits_schema = properties["edits"]
    assert isinstance(edits_schema, dict)

    item_schema = edits_schema["items"]
    assert isinstance(item_schema, dict)

    reference = item_schema["$ref"]
    assert isinstance(reference, str)
    definition_name = reference.rsplit("/", maxsplit=1)[-1]
    definitions = schema["$defs"]
    assert isinstance(definitions, dict)
    item_schema = definitions[definition_name]
    assert isinstance(item_schema, dict)

    item_properties = item_schema["properties"]
    assert isinstance(item_properties, dict)

    assert set(item_properties) == {"old_text", "new_text"}
    assert item_schema["required"] == ["old_text", "new_text"]


@pytest.mark.parametrize(
    ("requested_template", "expected_name"),
    [
        pytest.param("sample.txt", "sample.txt", id="relative"),
        pytest.param("@sample.txt", "sample.txt", id="at-prefixed"),
        pytest.param(
            "{cwd}/my\u00a0file.txt",
            "my file.txt",
            id="unicode-space-absolute",
        ),
    ],
)
def test_edit_resolves_requested_paths_against_cwd(
    tmp_path: Path,
    requested_template: str,
    expected_name: str,
) -> None:
    """Resolve relative, at-prefixed, and Unicode-space paths against the cwd."""

    requested_path = requested_template.format(cwd=tmp_path)

    assert edit._resolve_path(requested_path, tmp_path) == tmp_path / expected_name


def test_edit_expands_home_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Expand home-directory markers in edit paths."""

    monkeypatch.setenv("HOME", str(tmp_path))

    assert edit._resolve_path("~/sample.txt", Path.cwd()) == tmp_path / "sample.txt"


async def test_edit_replaces_text_in_file(tmp_path: Path) -> None:
    """Replace a unique exact text block in a file."""

    file_path = tmp_path / "sample.txt"
    write_text(file_path, "Hello, world!")

    result = tool_text(
        await edit.fn(
            _edit_input(
                path=str(file_path),
                edits=[{"old_text": "world", "new_text": "testing"}],
            ),
            cwd=Path.cwd(),
        )
    )

    assert file_path.read_text(encoding="utf-8") == "Hello, testing!"
    assert result == f"Successfully replaced 1 block(s) in {file_path}."


async def test_edit_replaces_multiple_disjoint_blocks(tmp_path: Path) -> None:
    """Apply multiple replacements against the original file content."""

    file_path = tmp_path / "sample.txt"
    write_text(file_path, "alpha\nbeta\ngamma\n")

    await edit.fn(
        _edit_input(
            path=str(file_path),
            edits=[
                {"old_text": "alpha\n", "new_text": "ALPHA\n"},
                {"old_text": "gamma\n", "new_text": "GAMMA\n"},
            ],
        ),
        cwd=Path.cwd(),
    )

    assert file_path.read_text(encoding="utf-8") == "ALPHA\nbeta\nGAMMA\n"


async def test_edit_matches_against_original_content(tmp_path: Path) -> None:
    """Match later edits against original content instead of earlier replacements."""

    file_path = tmp_path / "sample.txt"
    write_text(file_path, "foo\nbar\nbaz\n")

    await edit.fn(
        _edit_input(
            path=str(file_path),
            edits=[
                {"old_text": "foo\n", "new_text": "foo bar\n"},
                {"old_text": "bar\n", "new_text": "BAR\n"},
            ],
        ),
        cwd=Path.cwd(),
    )

    assert file_path.read_text(encoding="utf-8") == "foo bar\nBAR\nbaz\n"


async def test_edit_returns_unified_diff_details(tmp_path: Path) -> None:
    """Return a standard unified diff in edit result details."""

    file_path = tmp_path / "sample.txt"
    write_text(file_path, "alpha\nbeta\ngamma\n")

    tool_result = await edit.fn(
        _edit_input(
            path=str(file_path),
            edits=[{"old_text": "beta\n", "new_text": "BETA\n"}],
        ),
        cwd=Path.cwd(),
    )

    details = details_of(tool_result, EditDetails)
    assert details.type == "edit"
    assert details.diff == (
        f"--- a/{file_path}\n"
        f"+++ b/{file_path}\n"
        "@@ -1,3 +1,3 @@\n"
        " alpha\n"
        "-beta\n"
        "+BETA\n"
        " gamma\n"
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        ("first\r\nsecond\r\nthird\r\n", "first\r\nREPLACED\r\nthird\r\n"),
        ("\ufefffirst\nsecond\nthird\n", "\ufefffirst\nREPLACED\nthird\n"),
    ],
    ids=["crlf-line-endings", "utf8-bom"],
)
async def test_edit_preserves_file_encoding_metadata(
    tmp_path: Path,
    content: str,
    expected: str,
) -> None:
    """Match LF old_text while preserving CRLF endings and a UTF-8 BOM."""

    file_path = tmp_path / "sample.txt"
    write_text(file_path, content)

    await edit.fn(
        _edit_input(
            path=str(file_path),
            edits=[{"old_text": "second\n", "new_text": "REPLACED\n"}],
        ),
        cwd=Path.cwd(),
    )

    assert read_text(file_path) == expected


@pytest.mark.parametrize(
    ("content", "edits", "error_match"),
    [
        (
            "Hello, world!",
            [{"old_text": "missing", "new_text": "testing"}],
            "Could not find the exact text",
        ),
        (
            "foo foo foo",
            [{"old_text": "foo", "new_text": "bar"}],
            "Found 3 occurrences",
        ),
        (
            "hello\nworld\n",
            [{"old_text": "", "new_text": "replacement"}],
            "old_text must not be empty",
        ),
        (
            "one\ntwo\nthree\n",
            [
                {"old_text": "one\ntwo\n", "new_text": "ONE\nTWO\n"},
                {"old_text": "two\nthree\n", "new_text": "TWO\nTHREE\n"},
            ],
            "overlap",
        ),
        (
            "alpha\nbeta\ngamma\n",
            [
                {"old_text": "alpha\n", "new_text": "ALPHA\n"},
                {"old_text": "missing\n", "new_text": "MISSING\n"},
            ],
            "Could not find",
        ),
        (
            "abc\ncde\nghi",
            [{"old_text": "ghi\n", "new_text": "xyz\n"}],
            "Could not find",
        ),
        (
            "log(‘x’);\nother\nlog(‘x’);\n",
            [{"old_text": "log('x');", "new_text": "log('y');"}],
            "Found 2 occurrences",
        ),
    ],
    ids=[
        "old-text-missing",
        "duplicate-occurrences",
        "empty-old-text",
        "overlapping-edits",
        "one-invalid-edit-among-valid",
        "terminated-old-on-unterminated-file",
        "fuzzy-ambiguous-window",
    ],
)
async def test_edit_rejects_invalid_edits_and_leaves_file_unchanged(
    tmp_path: Path,
    content: str,
    edits: list[dict[str, str]],
    error_match: str,
) -> None:
    """Reject invalid edit requests without modifying the target file."""

    file_path = tmp_path / "sample.txt"
    write_text(file_path, content)

    with pytest.raises(RuntimeError, match=error_match):
        await edit.fn(
            _edit_input(path=str(file_path), edits=edits),
            cwd=Path.cwd(),
        )

    assert read_text(file_path) == content


@pytest.mark.parametrize(
    ("content", "old_text", "new_text", "expected"),
    [
        (
            "line one   \nline two  \nline three\n",
            "line one\nline two\n",
            "replaced\n",
            "replaced\nline three\n",
        ),
        (
            "console.log(‘hello’);\n",
            "console.log('hello');",
            "console.log('world');",
            "console.log('world');\n",
        ),
        (
            "hello\u00a0world\nrange: 1\u20135\n",
            "hello world\nrange: 1-5\n",
            "hello universe\nrange: 10-50\n",
            "hello universe\nrange: 10-50\n",
        ),
        (
            "ＡＢＣ１２３\ncafe\u0301\n",
            "ABC123\ncafé\n",
            "XYZ789\ncoffee\n",
            "XYZ789\ncoffee\n",
        ),
    ],
    ids=[
        "trailing-whitespace",
        "smart-quotes",
        "unicode-dashes-and-spaces",
        "nfkc-compatibility",
    ],
)
async def test_edit_fuzzy_matches_normalized_line_variants(
    tmp_path: Path,
    content: str,
    old_text: str,
    new_text: str,
    expected: str,
) -> None:
    """Fall back to fuzzy whole-line matching across normalization variants."""

    file_path = tmp_path / "sample.txt"
    write_text(file_path, content)

    await edit.fn(
        _edit_input(
            path=str(file_path),
            edits=[{"old_text": old_text, "new_text": new_text}],
        ),
        cwd=Path.cwd(),
    )

    assert file_path.read_text(encoding="utf-8") == expected


@pytest.mark.parametrize(
    ("content", "old_text", "new_text", "expected"),
    [
        (
            "console.log(‘hello’);",
            "console.log('hello');",
            "console.log('world');\n",
            "console.log('world');\n",
        ),
        (
            "alpha\nlog(‘x’);\nomega\n",
            "log('x');\n",
            "",
            "alpha\nomega\n",
        ),
        (
            "line1\nlog(‘x’);\n",
            "log('x');",
            "",
            "line1\n\n",
        ),
        (
            "line1\nlog(‘x’);\nline3\n",
            "log('x');",
            "a();\nb();",
            "line1\na();\nb();\nline3\n",
        ),
        (
            "alpha\nlog(‘x’);",
            "log('x');",
            "",
            "alpha\n",
        ),
    ],
    ids=[
        "eof-adds-trailing-newline",
        "terminated-line-deletion",
        "unterminated-deletion-keeps-empty-line",
        "unterminated-replacement-keeps-terminator",
        "last-line-deletion-keeps-prior-terminator",
    ],
)
async def test_edit_fuzzy_maps_terminators_like_exact_replacement(
    tmp_path: Path,
    content: str,
    old_text: str,
    new_text: str,
    expected: str,
) -> None:
    """Honor old and new text terminators exactly across window edges."""

    file_path = tmp_path / "sample.txt"
    write_text(file_path, content)

    await edit.fn(
        _edit_input(
            path=str(file_path),
            edits=[{"old_text": old_text, "new_text": new_text}],
        ),
        cwd=Path.cwd(),
    )

    assert read_text(file_path) == expected


async def test_edit_fuzzy_preserves_untouched_regions_and_diffs_original(
    tmp_path: Path,
) -> None:
    """Leave lines outside the window byte-identical and diff original content."""

    file_path = tmp_path / "sample.txt"
    write_text(file_path, "alpha ‘one’  \nconsole.log(‘hello’);\nomega “end”\t\n")

    tool_result = await edit.fn(
        _edit_input(
            path=str(file_path),
            edits=[
                {
                    "old_text": "console.log('hello');",
                    "new_text": "console.log('world');",
                }
            ],
        ),
        cwd=Path.cwd(),
    )

    assert file_path.read_text(encoding="utf-8") == (
        "alpha ‘one’  \nconsole.log('world');\nomega “end”\t\n"
    )
    details = details_of(tool_result, EditDetails)
    assert "-console.log(‘hello’);\n" in details.diff
    assert "+console.log('world');\n" in details.diff
    assert "-alpha" not in details.diff
    assert "+alpha" not in details.diff


async def test_edit_fuzzy_skips_unterminated_final_line_for_terminated_old(
    tmp_path: Path,
) -> None:
    """Match only the terminated occurrence when the final line lacks a newline."""

    file_path = tmp_path / "sample.txt"
    write_text(file_path, "log(‘x’);\nabc\nlog(‘x’);")

    await edit.fn(
        _edit_input(
            path=str(file_path),
            edits=[{"old_text": "log('x');\n", "new_text": "log('y');\n"}],
        ),
        cwd=Path.cwd(),
    )

    assert read_text(file_path) == "log('y');\nabc\nlog(‘x’);"


async def test_edit_fuzzy_matches_multiple_edits(tmp_path: Path) -> None:
    """Match every edit in fuzzy line space once the fallback is needed."""

    file_path = tmp_path / "sample.txt"
    write_text(file_path, "console.log(‘hello’);\nhello\u00a0world\n")

    await edit.fn(
        _edit_input(
            path=str(file_path),
            edits=[
                {
                    "old_text": "console.log('hello');\n",
                    "new_text": "console.log('world');\n",
                },
                {"old_text": "hello world\n", "new_text": "hello universe\n"},
            ],
        ),
        cwd=Path.cwd(),
    )

    assert file_path.read_text(encoding="utf-8") == (
        "console.log('world');\nhello universe\n"
    )


@pytest.mark.parametrize(
    ("content", "old_text", "new_text"),
    [
        ("line1\nline2\n", "line2", ""),
        ("line1\nline2\n", "line2\n", ""),
        ("line1\nline2\nline3\n", "line2", "x\ny"),
        ("line1\nline2\nline3\n", "line2", "x\n"),
        ("line1\nline2\nline3\n", "line2\n", "x\n"),
        ("line1\nline2\nline3\n", "line2\n", "x"),
        ("line1\nline2", "line2", "x\n"),
        ("line1\nline2", "line2", ""),
        ("line1\nline2", "line2", "x"),
        ("only", "only", "changed\n"),
    ],
)
def test_fuzzy_replacement_equals_exact_replacement(
    content: str,
    old_text: str,
    new_text: str,
) -> None:
    """Produce byte-identical results from both paths whenever both match."""

    replacements = [edit.EditReplacement(old_text=old_text, new_text=new_text)]

    exact = edit._apply_exact_replacements(content, replacements, "sample.txt")
    fuzzy = edit._apply_fuzzy_replacements(content, replacements, "sample.txt")

    assert fuzzy.new_content == exact.new_content


def _edit_input(*, path: str, edits: list[dict[str, str]]) -> edit.EditInput:
    """Validate JSON-shaped replacements for direct tool-function tests."""

    return edit.EditInput.model_validate({"path": path, "edits": edits})
