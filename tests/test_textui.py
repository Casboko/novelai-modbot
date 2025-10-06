from app.textui import ConsoleBlockBuilder, display_width, pad_display


def test_display_width_basic_ascii() -> None:
    assert display_width("hello") == 5


def test_display_width_cjk() -> None:
    assert display_width("評価") == 4


def test_display_width_ambiguous_default_narrow() -> None:
    assert display_width("・") == 2


def test_pad_display_aligns_columns() -> None:
    left = pad_display("レーティング", 12)
    right = pad_display("0.97", 6)
    assert display_width(left) == 12
    assert display_width(left + right) == 18


def test_console_block_builder_table_alignment() -> None:
    builder = ConsoleBlockBuilder()
    builder.add_table([
        ["g", "0.10", "[##--------]"],
        ["ｅ", "0.90", "[#########-]"],
    ], column_widths=[2, 6, 12])
    lines = builder.render().splitlines()
    width_numeric_0 = display_width(lines[0][: lines[0].index("0")])
    width_numeric_1 = display_width(lines[1][: lines[1].index("0")])
    assert width_numeric_0 == width_numeric_1

    width_bar_0 = display_width(lines[0][: lines[0].index("[")])
    width_bar_1 = display_width(lines[1][: lines[1].index("[")])
    assert width_bar_0 == width_bar_1
