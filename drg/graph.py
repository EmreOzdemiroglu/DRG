# drg/graph.py
from typing import List, Dict, Any, Tuple
import json

class KG:
    def __init__(self):
        self.nodes: Dict[str, Dict[str, Any]] = {}
        self.edges: List[Tuple[str, str, str]] = []  # (src, rel, dst)

    @classmethod
    def from_typed(cls, entities_typed: List[Tuple[str, str]], triples: List[Tuple[str, str, str]]):
        kg = cls()
        for name, etype in entities_typed:
            kg.nodes.setdefault(name, {"type": etype})
        for s, r, o in triples:
            kg.nodes.setdefault(s, {"type": None})
            kg.nodes.setdefault(o, {"type": None})
            kg.edges.append((s, r, o))
        return kg

    @classmethod
    def from_triples(cls, triples: List[Tuple[str, str, str]]):
        kg = cls()
        for s, r, o in triples:
            kg.nodes.setdefault(s, {"type": None})
            kg.nodes.setdefault(o, {"type": None})
            kg.edges.append((s, r, o))
        return kg

    def to_json(self, indent: int = 2) -> str:
        data = {
            "nodes": [{"id": n, **attr} for n, attr in self.nodes.items()],
            "edges": [{"source": s, "type": r, "target": o} for s, r, o in self.edges],
        }
        return json.dumps(data, indent=indent)