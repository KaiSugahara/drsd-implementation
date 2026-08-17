from typing import Any, Optional

from pydantic import BaseModel


class News(BaseModel):
    news_id: str
    title: str


class NewsPair(BaseModel):
    query_news: News
    target_news: News
    relationship: Optional[str] = None


class Relationship(BaseModel):
    name: str
    definition: str


class RelationshipSchema:
    def __init__(self, relationships: dict[str, str]) -> None:
        self.relationships: dict[str, Relationship] = {
            name: Relationship(name=name, definition=definition) for name, definition in relationships.items()
        }
        self.removed: dict[str, Relationship] = {}
        self.rejected: dict[str, Relationship] = {}

    def add(self, relationship: str, definition: str) -> None:
        rel = Relationship(name=relationship, definition=definition)
        self.relationships[rel.name] = rel

    def reject(self, relationship: str) -> None:
        assert relationship in self.relationships
        self.rejected[relationship] = self.relationships.pop(relationship)

    def remove(self, relationship: str) -> None:
        assert relationship in self.relationships
        self.removed[relationship] = self.relationships.pop(relationship)

    @property
    def accepted_names(self) -> list[str]:
        return list(self.relationships.keys())

    @property
    def accepted_relationships(self) -> list[Relationship]:
        return list(self.relationships.values())

    @property
    def rejected_relationships(self) -> list[Relationship]:
        return list(self.rejected.values())

    @property
    def removed_relationships(self) -> list[Relationship]:
        return list(self.removed.values())

    @property
    def active_label_map(self) -> dict[str, int]:
        return {"Unknown": 0, **{label: idx + 1 for idx, label in enumerate(self.accepted_names)}}

    @property
    def hypothesis_memory(self) -> dict[str, Any]:
        hypothesis_memory = (
            [
                {
                    "name": relationship.name,
                    "definition": relationship.definition,
                    "status": "Accepted",
                }
                for relationship in self.accepted_relationships
            ]
            + [
                {
                    "name": relationship.name,
                    "definition": relationship.definition,
                    "status": "Rejected",
                }
                for relationship in self.rejected_relationships
            ]
            + [
                {
                    "name": relationship.name,
                    "definition": relationship.definition,
                    "status": "Removed",
                }
                for relationship in self.removed_relationships
            ]
        )
        return {
            "accepted": {name: relationship.definition for name, relationship in self.relationships.items()},
            "rejected": {name: relationship.definition for name, relationship in self.rejected.items()},
            "removed": {name: relationship.definition for name, relationship in self.removed.items()},
            "hypothesis_memory": hypothesis_memory,
        }
