"""OpenProtein sequence specifications/expressions."""

import re
from dataclasses import dataclass
from typing import Literal

AMINO_ACIDS = "ARNDCQEGHILKMFPSTWYV"
EXTRA_TOKENS = "XOUBZ-?"


@dataclass
class SequenceSegment:
    """
    A segment of a sequence expression.

    Attributes:
        type: Type of segment - 'fixed' (known residues), 'design' (to be designed),
              or 'range' (variable length design)
        content: For 'fixed': amino acid sequence
                 For 'design': number of residues to design
                 For 'range': (min_length, max_length) tuple
    """

    type: Literal["fixed", "design", "range"]
    content: str | int | tuple[int, int]

    def __repr__(self) -> str:
        if self.type == "fixed":
            return f"Fixed({self.content})"
        elif self.type == "design":
            return f"Design({self.content})"
        else:  # range
            return f"Range{self.content}"


class SequenceExpr:
    """
    Parsed representation of a sequence expression/specification.

    Examples:
        "15..20" -> Range(15, 20)
        "AAAA" -> Fixed("AAAA")
        "6" -> Design(6)
        "15..20AAAA6C3..5" -> Range(15,20), Fixed("AAAA"), Design(6), Fixed("C"), Range(3,5)
        "3..5C6C3" -> Range(3,5), Fixed("C"), Design(6), Fixed("C"), Design(3)
    """

    def __init__(self, expr: str):
        self.expr = expr
        self.segments = self._parse(expr)

    def _parse(self, expr: str) -> list[SequenceSegment]:
        """Parse sequence expression into segments."""
        segments = []
        pos = 0

        while pos < len(expr):
            # Try to match range: digits..digits
            range_match = re.match(r"(\d+)\.\.(\d+)", expr[pos:])
            if range_match:
                min_len = int(range_match.group(1))
                max_len = int(range_match.group(2))
                if min_len == max_len == 0:
                    raise ValueError("Invalid range of 0..0")
                if min_len > max_len:
                    raise ValueError(f"Invalid range {min_len}..{max_len}: min > max")
                segments.append(SequenceSegment("range", (min_len, max_len)))
                pos += range_match.end()
                continue

            # Try to match design count: single digits
            design_match = re.match(r"(\d+)", expr[pos:])
            if design_match:
                count = int(design_match.group(1))
                if count == 0:
                    raise ValueError("Invalid count of 0")
                segments.append(SequenceSegment("design", count))
                pos += design_match.end()
                continue

            # Try to match fixed sequence: uppercase letters
            fixed_match = re.match(r"([A-Z]+)", expr[pos:])
            if fixed_match:
                sequence = fixed_match.group(1)
                segments.append(SequenceSegment("fixed", sequence))
                pos += fixed_match.end()
                continue

            # Invalid character
            raise ValueError(f"Invalid character at position {pos} in expr: {expr}")

        return segments

    def to_protein_sequence(self) -> bytes:
        """
        Convert to protein sequence string.

        Returns:
            Byte string representing the sequence
        """
        parts: list[str] = []
        for segment in self.segments:
            if segment.type == "fixed":
                part = segment.content
                assert isinstance(part, str)
                parts.append(part)
            elif segment.type == "design":
                num_design = segment.content
                assert isinstance(num_design, int)
                part = "X" * num_design
                parts.append(part)
            elif segment.type == "range":
                assert isinstance(segment.content, tuple)
                min_len, max_len = segment.content
                # Use min required + optional
                parts.append("X" * min_len + "?" * (max_len - min_len))
            else:
                raise TypeError(f"Unexpected segment type {segment.type}")

        return "".join(parts).encode()

    def min_length(self) -> int:
        """Calculate minimum possible sequence length."""
        total = 0
        for segment in self.segments:
            if segment.type == "fixed":
                part = segment.content
                assert isinstance(part, str)
                total += len(part)
            elif segment.type == "design":
                num_design = segment.content
                assert isinstance(num_design, int)
                total += num_design
            elif segment.type == "range":
                assert isinstance(segment.content, tuple)
                min_len, _ = segment.content
                total += min_len
            else:
                raise TypeError(f"Unexpected segment type {segment.type}")
        return total

    def max_length(self) -> int:
        """Calculate maximum possible sequence length."""
        total = 0
        for segment in self.segments:
            if segment.type == "fixed":
                part = segment.content
                assert isinstance(part, str)
                total += len(part)
            elif segment.type == "design":
                num_design = segment.content
                assert isinstance(num_design, int)
                total += num_design
            elif segment.type == "range":
                assert isinstance(segment.content, tuple)
                _, max_len = segment.content
                total += max_len
            else:
                raise TypeError(f"Unexpected segment type {segment.type}")
        return total

    def has_variable_length(self) -> bool:
        """Check if expression includes variable-length regions."""
        return any(seg.type == "range" for seg in self.segments)

    def __repr__(self) -> str:
        return f"SequenceExpr({self.expr!r}, segments={self.segments})"

    def __str__(self) -> str:
        return self.expr

    @classmethod
    def parse(cls, expr: str) -> "SequenceExpr":
        """
        Parse a sequence expression.

        Args:
            expr: Sequence expression string

        Returns:
            Parsed SequenceExpression object

        Examples:
            >>> SequenceExpr.parse("15..20")
            SequenceExpr('15..20', segments=[Range(15, 20)])

            >>> SequenceExpr.parse("AAAA6C")
            SequenceExpr('AAAA6C', segments=[Fixed('AAAA'), Design(6), Fixed('C')])

            >>> SequenceExpr.parse("3..5C6C3")
            SequenceExpr('3..5C6C3', segments=[Range(3, 5), Fixed('C'), Design(6), Fixed('C'), Design(3)])
        """
        return SequenceExpr(expr)
