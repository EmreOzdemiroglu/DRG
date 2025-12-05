"""
Declarative schema definitions for DRG - Signature-like structure.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple


@dataclass(frozen=True)
class Entity:
    """Legacy Entity class for backward compatibility."""
    name: str


@dataclass(frozen=True)
class Relation:
    """Relation definition between entity types."""
    name: str
    src: str  # Source entity type name
    dst: str  # Destination entity type name


@dataclass
class EntityType:
    """Entity type definition with metadata."""
    name: str
    description: str
    examples: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate entity type."""
        if not self.name:
            raise ValueError("EntityType name cannot be empty")
        if not self.description:
            raise ValueError("EntityType description cannot be empty")


@dataclass
class EntityGroup:
    """Group of related entity types."""
    name: str
    description: str
    entity_types: List[EntityType]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate entity group."""
        if not self.name:
            raise ValueError("EntityGroup name cannot be empty")
        if not self.entity_types:
            raise ValueError("EntityGroup must contain at least one EntityType")
    
    def get_entity_type_names(self) -> List[str]:
        """Get list of entity type names in this group."""
        return [et.name for et in self.entity_types]


@dataclass
class PropertyGroup:
    """Group of properties that can be shared across entity types."""
    name: str
    description: str
    properties: Dict[str, Any]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate property group."""
        if not self.name:
            raise ValueError("PropertyGroup name cannot be empty")
        if not self.properties:
            raise ValueError("PropertyGroup must contain at least one property")


@dataclass
class RelationGroup:
    """Group of related relations with semantic meaning."""
    name: str
    description: str
    relations: List[Relation]
    examples: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate relation group."""
        if not self.name:
            raise ValueError("RelationGroup name cannot be empty")
        if not self.relations:
            raise ValueError("RelationGroup must contain at least one Relation")
    
    def get_relation_triples(self) -> List[Tuple[str, str, str]]:
        """Get list of (relation_name, src, dst) tuples."""
        return [(r.name, r.src, r.dst) for r in self.relations]


class DRGSchema:
    """Legacy schema class for backward compatibility."""
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


class EnhancedDRGSchema:
    """Enhanced declarative schema with grouping capabilities."""
    
    def __init__(
        self,
        entity_types: List[EntityType],
        relation_groups: List[RelationGroup],
        entity_groups: Optional[List[EntityGroup]] = None,
        property_groups: Optional[List[PropertyGroup]] = None,
        auto_discovery: bool = False
    ):
        self.entity_types = entity_types
        self.relation_groups = relation_groups
        self.entity_groups = entity_groups or []
        self.property_groups = property_groups or []
        self.auto_discovery = auto_discovery
        
        self._validate()
        self._build_indexes()
    
    def _validate(self):
        """Validate schema consistency."""
        # Check entity type names are unique
        entity_names = {et.name for et in self.entity_types}
        if len(entity_names) != len(self.entity_types):
            raise ValueError("EntityType names must be unique")
        
        # Check relation groups reference valid entity types
        all_relation_triples = []
        for rg in self.relation_groups:
            for rel in rg.relations:
                if rel.src not in entity_names:
                    raise ValueError(
                        f"Relation {rel.name} in group '{rg.name}' references unknown entity type: {rel.src}"
                    )
                if rel.dst not in entity_names:
                    raise ValueError(
                        f"Relation {rel.name} in group '{rg.name}' references unknown entity type: {rel.dst}"
                    )
                all_relation_triples.append((rel.name, rel.src, rel.dst))
        
        # Check entity groups reference valid entity types
        entity_type_map = {et.name: et for et in self.entity_types}
        for eg in self.entity_groups:
            for et in eg.entity_types:
                if et.name not in entity_type_map:
                    raise ValueError(
                        f"EntityGroup '{eg.name}' references unknown EntityType: {et.name}"
                    )
    
    def _build_indexes(self):
        """Build internal indexes for fast lookup."""
        # Entity type name -> EntityType
        self._entity_type_map = {et.name: et for et in self.entity_types}
        
        # Relation name -> List of (src, dst) pairs
        self._relation_map: Dict[str, List[Tuple[str, str]]] = {}
        for rg in self.relation_groups:
            for rel in rg.relations:
                if rel.name not in self._relation_map:
                    self._relation_map[rel.name] = []
                self._relation_map[rel.name].append((rel.src, rel.dst))
        
        # Entity type -> List of relations it can participate in
        self._entity_relations: Dict[str, List[Tuple[str, str, str]]] = {}
        for rg in self.relation_groups:
            for rel in rg.relations:
                if rel.src not in self._entity_relations:
                    self._entity_relations[rel.src] = []
                if rel.dst not in self._entity_relations:
                    self._entity_relations[rel.dst] = []
                self._entity_relations[rel.src].append((rel.name, rel.src, rel.dst))
                self._entity_relations[rel.dst].append((rel.name, rel.src, rel.dst))
    
    def get_entity_type(self, name: str) -> Optional[EntityType]:
        """Get entity type by name."""
        return self._entity_type_map.get(name)
    
    def get_all_relations(self) -> List[Relation]:
        """Get all relations from all relation groups."""
        relations = []
        for rg in self.relation_groups:
            relations.extend(rg.relations)
        return relations
    
    def get_relations_for_entity_type(self, entity_type_name: str) -> List[Tuple[str, str, str]]:
        """Get all relations that involve a given entity type."""
        return self._entity_relations.get(entity_type_name, [])
    
    def is_valid_relation(self, relation_name: str, src_type: str, dst_type: str) -> bool:
        """Check if a relation is valid for given entity types."""
        if relation_name not in self._relation_map:
            return False
        return (src_type, dst_type) in self._relation_map[relation_name]
    
    def to_legacy_schema(self) -> DRGSchema:
        """Convert to legacy DRGSchema for backward compatibility."""
        entities = [Entity(et.name) for et in self.entity_types]
        relations = self.get_all_relations()
        return DRGSchema(entities=entities, relations=relations)
    
    def get_schema_summary(self) -> Dict[str, Any]:
        """Get a summary of the schema for display/debugging."""
        return {
            "entity_types": [
                {
                    "name": et.name,
                    "description": et.description,
                    "examples": et.examples,
                    "properties": et.properties
                }
                for et in self.entity_types
            ],
            "relation_groups": [
                {
                    "name": rg.name,
                    "description": rg.description,
                    "relations": [(r.name, r.src, r.dst) for r in rg.relations],
                    "examples": rg.examples
                }
                for rg in self.relation_groups
            ],
            "entity_groups": [
                {
                    "name": eg.name,
                    "description": eg.description,
                    "entity_types": eg.get_entity_type_names(),
                    "examples": eg.examples
                }
                for eg in self.entity_groups
            ],
            "property_groups": [
                {
                    "name": pg.name,
                    "description": pg.description,
                    "properties": pg.properties,
                    "examples": pg.examples
                }
                for pg in self.property_groups
            ],
            "auto_discovery": self.auto_discovery
        }
