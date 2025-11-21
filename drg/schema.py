from dataclasses import dataclass
from typing import List, Tuple

@dataclass(frozen=True)
class Entity:
    name: str

@dataclass(frozen=True)
class Relation:
    name: str
    src: str
    dst: str

class DRGSchema:
    def __init__(self, entities: List[Entity], relations: List[Relation]):
        self.entities = entities
        self.relations = relations
        self._validate()

    def _validate(self):
        entity_names = {e.name for e in self.entities}
        for r in self.relations:
            if r.src not in entity_names or r.dst not in entity_names:
                raise ValueError(f"Relation {r.name} refers to unknown entity: {r.src}->{r.dst}")

    def relation_types(self) -> List[Tuple[str, str, str]]:
        return [(r.name, r.src, r.dst) for r in self.relations]
