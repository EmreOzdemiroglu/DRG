"""
Declarative schema definitions for DRG - Signature-like structure.
"""
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple, Union
import json
from pathlib import Path


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
    description: str = ""  # Bağlantı sebebi (relationship type açıklaması)
    detail: str = ""  # Bağlantı detayı (tek cümleyle neden bağlantılı olduğu)


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


def load_schema_from_json(schema_path: Union[str, Path]) -> Union[DRGSchema, EnhancedDRGSchema]:
    """Load schema from JSON file (supports both Enhanced and legacy formats).
    
    Args:
        schema_path: Path to JSON schema file
    
    Returns:
        DRGSchema or EnhancedDRGSchema instance
    
    Raises:
        FileNotFoundError: If schema file doesn't exist
        ValueError: If schema JSON is invalid
    """
    path = Path(schema_path)
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    
    try:
        with open(path, "r", encoding="utf-8") as f:
            schema_data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in schema file: {e}") from e
    
    # Enhanced schema formatını kontrol et
    if "entity_types" in schema_data:
        entity_types = [
            EntityType(
                name=et["name"],
                description=et.get("description", ""),
                examples=et.get("examples", []),
                properties=et.get("properties", {})
            )
            for et in schema_data.get("entity_types", [])
        ]
        
        relation_groups = []
        for rg_data in schema_data.get("relation_groups", []):
            relations = [
                Relation(
                    name=r["name"],
                    src=r.get("source", r.get("src", "")),
                    dst=r.get("target", r.get("dst", "")),
                    description=r.get("description", ""),
                    detail=r.get("detail", "")
                )
                for r in rg_data.get("relations", [])
            ]
            relation_groups.append(RelationGroup(
                name=rg_data["name"],
                description=rg_data.get("description", ""),
                relations=relations,
                examples=rg_data.get("examples", [])
            ))
        
        return EnhancedDRGSchema(
            entity_types=entity_types,
            relation_groups=relation_groups,
            auto_discovery=schema_data.get("auto_discovery", False)
        )
    else:
        # Legacy format
        entities = [Entity(e["name"]) for e in schema_data.get("entities", [])]
        relations = [
            Relation(
                name=r["name"],
                src=r.get("source", r.get("src", "")),
                dst=r.get("target", r.get("dst", "")),
                description=r.get("description", ""),
                detail=r.get("detail", "")
            )
            for r in schema_data.get("relations", [])
        ]
        
        return DRGSchema(entities=entities, relations=relations)
