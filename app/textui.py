from __future__ import annotations

import unicodedata
from typing import Iterable, List, Sequence


def display_width(text: str, *, ambiguous_is_wide: bool = False) -> int:
    """Return the display width (number of columns) for *text*.

    The calculation follows Unicode East Asian Width rules (UAX #11). Characters
    marked as Fullwidth (F) or Wide (W) count as 2 columns, Narrow (Na) and
    Halfwidth (H) as 1 column, and combining characters as 0. Ambiguous (A)
    characters default to 1 column but may be treated as 2 if
    ``ambiguous_is_wide`` is true.
    """

    total = 0
    for char in text:
        if not char:
            continue
        if unicodedata.combining(char):
            continue
        width_type = unicodedata.east_asian_width(char)
        if width_type in {"F", "W"}:
            total += 2
        elif width_type == "A":
            total += 2 if ambiguous_is_wide else 1
        else:
            total += 1
    return total


def pad_display(
    text: str,
    width: int,
    *,
    fill: str = " ",
    ambiguous_is_wide: bool = False,
) -> str:
    """Pad *text* on the right so that its display width reaches *width*.

    The padding honours East Asian width via :func:`display_width`. When the
    text already exceeds *width* it is returned unchanged.
    """

    current = display_width(text, ambiguous_is_wide=ambiguous_is_wide)
    if current >= width:
        return text
    if not fill:
        fill = " "
    pad_chars: List[str] = []
    fill_width = display_width(fill, ambiguous_is_wide=ambiguous_is_wide)
    if fill_width <= 0:
        fill = " "
        fill_width = 1
    remaining = width - current
    while remaining > 0:
        if remaining >= fill_width:
            pad_chars.append(fill)
            remaining -= fill_width
        else:
            pad_chars.append(" " * remaining)
            remaining = 0
    return text + "".join(pad_chars)


class ConsoleBlockBuilder:
    """Utility for assembling fixed-width console style blocks."""

    def __init__(self, *, ambiguous_is_wide: bool = False, separator: str = "  ") -> None:
        self._lines: List[str] = []
        self.ambiguous_is_wide = ambiguous_is_wide
        self.separator = separator

    def add_line(self, text: str = "") -> None:
        self._lines.append(text)

    def add_table(
        self,
        rows: Sequence[Sequence[str]],
        column_widths: Sequence[int],
    ) -> None:
        sep = self.separator
        for row in rows:
            padded_cells = []
            for idx, cell in enumerate(row):
                try:
                    width = column_widths[idx]
                except IndexError:
                    width = column_widths[-1]
                padded_cells.append(
                    pad_display(cell, width, ambiguous_is_wide=self.ambiguous_is_wide)
                )
            self._lines.append(sep.join(padded_cells))

    def extend(self, lines: Iterable[str]) -> None:
        for line in lines:
            self.add_line(str(line))

    def render(self) -> str:
        return "\n".join(self._lines)

