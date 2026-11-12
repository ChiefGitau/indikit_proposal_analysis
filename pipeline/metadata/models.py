"""Pydantic records of the metadata module (paper Section 3.2.1 Entities, 3.2.2 Extensions).

One ``IndicatorMetadata`` per unique IndiKit ID. The paper's entity
<Aspect, KPI, Topic, Quantity> maps to ``aspect`` (sector / sub-sector),
``kpi_wording_en`` + ``name``, ``topic`` (disaggregation) and ``quantity``;
the extensions <Knowledge> and <SearchTerm> are ``knowledge`` and ``search_terms``.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from pipeline import config


class Quantity(str, Enum):
    """Measurement type derived from the wording (paper: absolute value / intensity / key action)."""

    number = "number"
    percentage = "percentage"
    score = "score"
    ratio = "ratio"
    yes_no = "yes_no"
    other = "other"


FieldSource = Literal["export", "site", "seed", "derived", "llm", "override"]


class Aspect(BaseModel):
    """One sector assignment of an indicator (the export repeats a row per sub-sector)."""

    model_config = ConfigDict(frozen=True)

    section: str = ""
    sector: str = ""
    sub_sector: str = ""


class IndicatorMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid")

    indikit_id: str
    name: str
    aspect: list[Aspect] = Field(default_factory=list)
    kpi_wording_en: str = ""
    kpi_wording_fr: Optional[str] = None
    levels: list[str] = Field(default_factory=list)
    topic: Optional[str] = None
    quantity: Quantity = Quantity.other
    purpose: Optional[str] = None
    how_to_collect: Optional[str] = None
    important_comments: Optional[str] = None
    guidance: Optional[str] = None
    url: Optional[str] = None
    on_site: bool = False
    in_export: bool = False
    field_sources: dict[str, FieldSource] = Field(default_factory=dict)
    knowledge: Optional[str] = None
    search_terms: list[str] = Field(default_factory=list)

    @field_validator("indikit_id", "name", mode="before")
    @classmethod
    def _required_text(cls, v, info):
        s = str(v).strip() if v is not None else ""
        if not s:
            raise ValueError(f"{info.field_name} must be a non-empty string")
        return s

    @field_validator("levels")
    @classmethod
    def _known_levels(cls, v: list[str]) -> list[str]:
        unknown = [x for x in v if x not in config.LEVELS]
        if unknown:
            raise ValueError(f"unknown level(s) {unknown}; allowed: {config.LEVELS}")
        # canonical order, no duplicates
        return [lvl for lvl in config.LEVELS if lvl in v]

    @property
    def primary_sector(self) -> str:
        return self.aspect[0].sector if self.aspect else ""

    @property
    def primary_sub_sector(self) -> str:
        return self.aspect[0].sub_sector if self.aspect else ""

    @property
    def needs_fetch(self) -> bool:
        """True for seeded records (neither in the export nor in the site data)."""
        return not self.on_site and not self.in_export
