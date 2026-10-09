from __future__ import annotations

from collections import Counter
from typing import Iterable
from urllib.parse import urlparse

from .schema import Target


DOMAIN_FAMILIES = (
    "literature_publishing",
    "film",
    "television",
    "radio_podcasts",
    "music",
    "video_games",
    "software_open_source",
    "internet_web_history",
    "visual_art",
    "photography",
    "architecture",
    "theatre_performing_arts",
    "museums_collections",
    "archives_manuscripts",
    "local_history",
    "archaeology",
    "biology_taxonomy",
    "medicine_health_history",
    "physics_astronomy",
    "chemistry",
    "mathematics",
    "earth_science",
    "engineering",
    "aerospace",
    "transport",
    "industry_manufacturing",
    "education_academic_history",
    "law_justice_records",
    "government_public_institutions",
    "business_company_history",
    "sports",
    "geography_places",
    "heritage_venues",
    "awards_ceremonies",
    "language_dictionaries",
    "food_culture",
)
DOMAIN_SET = set(DOMAIN_FAMILIES)


def normalize_domain(value: str) -> str:
    key = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    return key if key in DOMAIN_SET else "unknown"


def source_domains(urls: Iterable[str]) -> list[str]:
    domains: list[str] = []
    for url in urls:
        hostname = (urlparse(str(url)).hostname or "").lower()
        if hostname.startswith("www."):
            hostname = hostname[4:]
        if hostname and hostname not in domains:
            domains.append(hostname)
    return domains


def target_domain(target: Target) -> str:
    explicit = normalize_domain(target.domain_family)
    if explicit != "unknown":
        return explicit
    text = " ".join(
        [target.name, target.entity_type, target.answer_field, target.description, *target.source_urls]
    ).lower()
    rules = (
        ("sports", (" match", "football", "soccer", "cricket", "baseball", "basketball", "olympic")),
        ("video_games", ("video game", "game map", "zx spectrum", "commodore 64", "amiga game")),
        ("software_open_source", ("software", "open source", "unix", "github", "utility")),
        ("television", ("television", " tv ", "episode", "sitcom")),
        ("radio_podcasts", ("radio", "podcast", "broadcast")),
        ("film", (" film", "cinema", "movie")),
        ("music", ("music", "album", "recording", "catalog number", "composer", "orchestra")),
        ("literature_publishing", ("magazine", "publication", "book", "story", "novel", "isfdb")),
        ("visual_art", ("artwork", "painting", "engraving", "print catalog")),
        ("museums_collections", ("museum", "collection object", "accession")),
        ("heritage_venues", ("theatre", "cinema", "venue", "heritage")),
        ("biology_taxonomy", ("species", "taxonomy", "holotype", "zoology", "botany")),
        ("aerospace", ("space mission", "spacecraft", "apollo", "aviation")),
        ("education_academic_history", ("university", "academic", "course catalog")),
        ("archives_manuscripts", ("archive", "manuscript", "microfilm")),
        ("geography_places", ("place", "geography", "river", "island")),
    )
    for domain, terms in rules:
        if any(term in text for term in terms):
            return domain
    return "unknown"


def choose_domain_slots(counts: Counter[str], requested: int) -> list[str]:
    """Choose least-represented domains, reserving one model-choice slot."""
    if requested <= 0:
        return []
    ranked = sorted(DOMAIN_FAMILIES, key=lambda domain: (counts.get(domain, 0), DOMAIN_FAMILIES.index(domain)))
    directed = requested - 1 if requested >= 3 else requested
    slots = ranked[:directed]
    if directed < requested:
        slots.append("model_choice")
    return slots

