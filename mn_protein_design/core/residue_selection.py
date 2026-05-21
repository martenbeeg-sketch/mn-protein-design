from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResidueSelection:
    chain_id: str
    start: int
    end: int

    def contains(self, chain_id: str, residue_number: int) -> bool:
        return self.chain_id == chain_id and self.start <= residue_number <= self.end

    def to_json(self) -> dict:
        return {"chain_id": self.chain_id, "start": self.start, "end": self.end}


def parse_selection(text: str) -> list[ResidueSelection]:
    selections: list[ResidueSelection] = []
    for raw in text.replace(";", ",").split(","):
        token = raw.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Residue range '{token}' must look like A:10-30 or A:42")
        chain, span = token.split(":", 1)
        chain = chain.strip()
        if not chain:
            raise ValueError(f"Residue range '{token}' is missing a chain ID")
        if "-" in span:
            start_text, end_text = span.split("-", 1)
            start, end = int(start_text), int(end_text)
        else:
            start = end = int(span)
        if end < start:
            raise ValueError(f"Residue range '{token}' has end before start")
        selections.append(ResidueSelection(chain, start, end))
    return selections
