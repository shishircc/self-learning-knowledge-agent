import json
import re
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .auth import Identity


_COCO_IMG_REF_RE = re.compile(r"coco-img:(img_[0-9a-f]+)")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class PacketContent:
    gist: str = ""
    summary: str = ""
    full: str = ""


@dataclass
class PacketImage:
    id: str
    alt: str | None
    mime: str
    data_b64: str
    dimensions: list  # [w, h]
    source_url: str | None = None
    added_at: str = ""

    @classmethod
    def new(
        cls,
        alt: str | None,
        mime: str,
        data_b64: str,
        dimensions,
        source_url: str | None = None,
    ) -> "PacketImage":
        if dimensions is None:
            dims_list = [0, 0]
        else:
            dims_list = [int(dimensions[0]), int(dimensions[1])]
        return cls(
            id=new_id("img"),
            alt=alt,
            mime=mime,
            data_b64=data_b64,
            dimensions=dims_list,
            source_url=source_url,
            added_at=now_iso(),
        )

    @property
    def data_uri(self) -> str:
        return f"data:{self.mime};base64,{self.data_b64}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PacketImage":
        return cls(
            id=d["id"],
            alt=d.get("alt"),
            mime=d["mime"],
            data_b64=d["data_b64"],
            dimensions=list(d.get("dimensions") or [0, 0]),
            source_url=d.get("source_url"),
            added_at=d.get("added_at", ""),
        )


@dataclass
class PacketSource:
    """Provenance record for one write into a packet.

    `effective_authoritativeness = max(role_authoritativeness, domain_authoritativeness or 0)`
    is computed at construction time so the trust scalar is recorded as it
    was at write time (config changes later don't retroactively shift it).
    """

    type: str  # "url" | "conversation"
    url: str | None = None
    domain_authoritativeness: float | None = None
    speaker_name: str | None = None
    speaker_email: str | None = None
    speaker_role: str | None = None
    role_authoritativeness: float = 0.0
    effective_authoritativeness: float = 0.0
    recorded_at: str = ""

    @classmethod
    def from_url(
        cls,
        url: str,
        domain_authoritativeness: float | None,
        writer: "Identity",
    ) -> "PacketSource":
        role_auth = float(writer.role_authoritativeness or 0.0)
        d_auth = float(domain_authoritativeness or 0.0)
        return cls(
            type="url",
            url=url,
            domain_authoritativeness=d_auth,
            speaker_name=writer.name,
            speaker_email=writer.email,
            speaker_role=writer.role,
            role_authoritativeness=role_auth,
            effective_authoritativeness=max(role_auth, d_auth),
            recorded_at=now_iso(),
        )

    @classmethod
    def from_conversation(cls, writer: "Identity") -> "PacketSource":
        role_auth = float(writer.role_authoritativeness or 0.0)
        return cls(
            type="conversation",
            url=None,
            domain_authoritativeness=None,
            speaker_name=writer.name,
            speaker_email=writer.email,
            speaker_role=writer.role,
            role_authoritativeness=role_auth,
            effective_authoritativeness=role_auth,
            recorded_at=now_iso(),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PacketSource":
        return cls(
            type=d.get("type", "conversation"),
            url=d.get("url"),
            domain_authoritativeness=d.get("domain_authoritativeness"),
            speaker_name=d.get("speaker_name"),
            speaker_email=d.get("speaker_email"),
            speaker_role=d.get("speaker_role"),
            role_authoritativeness=float(d.get("role_authoritativeness", 0.0) or 0.0),
            effective_authoritativeness=float(
                d.get("effective_authoritativeness", 0.0) or 0.0
            ),
            recorded_at=d.get("recorded_at", ""),
        )


@dataclass
class TopicFacet:
    text: str
    vector: list  # serializable

    def vec_np(self) -> np.ndarray:
        return np.asarray(self.vector, dtype=np.float32)

    def to_dict(self) -> dict:
        return {"text": self.text, "vector": self.vector}

    @classmethod
    def from_dict(cls, d: dict):
        return cls(text=d["text"], vector=d["vector"])


@dataclass
class Packet:
    id: str
    topics: list  # list[TopicFacet]
    entities: list  # list[str], lowercased
    content: PacketContent
    strength_events: list = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    source_session_ids: list = field(default_factory=list)
    sources: list = field(default_factory=list)  # list[PacketSource]
    authoritativeness: float = 0.0
    images: list = field(default_factory=list)  # list[PacketImage]

    @classmethod
    def new(
        cls,
        topics: list,  # list of (text, vector) tuples
        entities: list,
        gist: str = "",
        summary: str = "",
        full: str = "",
        sources: list | None = None,
        images: list | None = None,
    ):
        ts = now_iso()
        facets = [
            TopicFacet(text=t, vector=(v.tolist() if hasattr(v, "tolist") else list(v)))
            for t, v in topics
        ]
        pkt = cls(
            id=new_id("pkt"),
            topics=facets,
            entities=[e.lower().strip() for e in entities if e and e.strip()],
            content=PacketContent(gist=gist, summary=summary, full=full),
            strength_events=[],
            created_at=ts,
            updated_at=ts,
            source_session_ids=[],
            sources=[],
            authoritativeness=0.0,
            images=[],
        )
        for src in sources or []:
            pkt.add_source(src)
        for img in images or []:
            pkt.add_image(img)
        return pkt

    def topic_texts(self) -> list[str]:
        return [t.text for t in self.topics]

    def topic_vectors_np(self) -> list[np.ndarray]:
        return [t.vec_np() for t in self.topics]

    def combined_topic_text(self) -> str:
        return " | ".join(self.topic_texts())

    def add_facet_if_new(self, text: str, vector, dedup_threshold: float) -> bool:
        """Add a new facet if it's not too similar to an existing one. Returns True if added."""
        new_vec = np.asarray(vector, dtype=np.float32)
        for f in self.topics:
            existing = f.vec_np()
            sim = float(np.dot(new_vec, existing))
            if sim >= dedup_threshold:
                return False
        self.topics.append(TopicFacet(text=text, vector=(vector.tolist() if hasattr(vector, "tolist") else list(vector))))
        return True

    def merge_entities(self, new_entities: list[str]):
        existing = set(self.entities)
        for e in new_entities:
            if not e or not e.strip():
                continue
            e_norm = e.lower().strip()
            if e_norm not in existing:
                self.entities.append(e_norm)
                existing.add(e_norm)

    def record_event(self, event_type: str, weight: float):
        self.strength_events.append({
            "event_type": event_type,
            "timestamp": now_iso(),
            "weight": weight,
        })

    def add_source(self, source: PacketSource) -> bool:
        """Append `source` to `sources`; recompute authoritativeness (monotone-up).

        Returns True on append. URL-source dedupe is the caller's responsibility
        (use `has_source_url(url)` first); this method always appends so the
        audit trail of write events stays intact.
        """
        if not isinstance(source, PacketSource):
            return False
        self.sources.append(source)
        eff = float(source.effective_authoritativeness or 0.0)
        if eff > self.authoritativeness:
            self.authoritativeness = eff
        return True

    def has_source_url(self, url: str) -> bool:
        if not url:
            return False
        u = url.strip().lower()
        for s in self.sources:
            if s.type == "url" and s.url and s.url.strip().lower() == u:
                return True
        return False

    def source_urls(self) -> list[str]:
        """Derived: distinct URLs across `sources`, in first-seen order."""
        out: list[str] = []
        seen: set[str] = set()
        for s in self.sources:
            if s.type != "url" or not s.url:
                continue
            key = s.url.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s.url)
        return out

    def add_image(self, image: PacketImage) -> bool:
        """Append an image. No-op if an image with the same id already exists."""
        if not isinstance(image, PacketImage):
            return False
        if any(existing.id == image.id for existing in self.images):
            return False
        self.images.append(image)
        return True

    def image_by_id(self, image_id: str) -> PacketImage | None:
        for img in self.images:
            if img.id == image_id:
                return img
        return None

    def referenced_image_ids(self) -> set[str]:
        """All image ids appearing as `coco-img:img_<id>` inside content.full."""
        return set(_COCO_IMG_REF_RE.findall(self.content.full or ""))

    def unreferenced_image_ids(self) -> set[str]:
        refs = self.referenced_image_ids()
        return {img.id for img in self.images if img.id not in refs}

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topics": [t.to_dict() for t in self.topics],
            "entities": self.entities,
            "content": asdict(self.content),
            "strength_events": self.strength_events,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_session_ids": self.source_session_ids,
            "sources": [s.to_dict() for s in self.sources],
            "authoritativeness": self.authoritativeness,
            "images": [img.to_dict() for img in self.images],
        }

    @classmethod
    def from_dict(cls, d: dict):
        # Backwards-compat: legacy packets have `source_urls: list[str]` and no
        # `sources` / `authoritativeness`. Synthesize neutral-trust PacketSource
        # entries so URL lookups still work; trust starts at 0.0.
        raw_sources = d.get("sources")
        if raw_sources is None:
            sources = []
            for u in d.get("source_urls") or []:
                if not isinstance(u, str) or not u.strip():
                    continue
                sources.append(PacketSource(
                    type="url",
                    url=u.strip(),
                    domain_authoritativeness=0.0,
                    speaker_name=None,
                    speaker_email=None,
                    speaker_role=None,
                    role_authoritativeness=0.0,
                    effective_authoritativeness=0.0,
                    recorded_at="",
                ))
        else:
            sources = [PacketSource.from_dict(s) for s in raw_sources]

        authoritativeness = float(d.get("authoritativeness", 0.0) or 0.0)
        if sources and authoritativeness == 0.0:
            # Recompute aggregate from sources if absent in the JSON.
            authoritativeness = max(
                (float(s.effective_authoritativeness or 0.0) for s in sources),
                default=0.0,
            )

        return cls(
            id=d["id"],
            topics=[TopicFacet.from_dict(t) for t in d["topics"]],
            entities=d.get("entities", []),
            content=PacketContent(**d.get("content", {})),
            strength_events=d.get("strength_events", []),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            source_session_ids=d.get("source_session_ids", []),
            sources=sources,
            authoritativeness=authoritativeness,
            images=[PacketImage.from_dict(x) for x in d.get("images", [])],
        )


@dataclass
class ScratchpadEntry:
    id: str
    topic: str
    topic_vector: list
    raw_excerpts: list
    mention_count: int = 1
    created_at: str = ""
    last_seen_at: str = ""
    last_seen_session_n: int = 0
    sessions_seen: list = field(default_factory=list)

    @classmethod
    def new(cls, topic: str, topic_vector, excerpt: str, session_id: str, session_n: int):
        tv = topic_vector.tolist() if hasattr(topic_vector, "tolist") else list(topic_vector)
        ts = now_iso()
        return cls(
            id=new_id("scratch"),
            topic=topic,
            topic_vector=tv,
            raw_excerpts=[excerpt],
            mention_count=1,
            created_at=ts,
            last_seen_at=ts,
            last_seen_session_n=session_n,
            sessions_seen=[session_id],
        )

    def topic_vec_np(self) -> np.ndarray:
        return np.asarray(self.topic_vector, dtype=np.float32)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict):
        return cls(**d)


class PacketStore:
    def __init__(self, data_dir: str):
        self.dir = Path(data_dir) / "packets"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.packets: dict[str, Packet] = {}
        self._load_all()

    def _load_all(self):
        for f in self.dir.glob("*.json"):
            with open(f) as fp:
                p = Packet.from_dict(json.load(fp))
                self.packets[p.id] = p

    def save(self, packet: Packet):
        packet.updated_at = now_iso()
        with open(self.dir / f"{packet.id}.json", "w") as f:
            json.dump(packet.to_dict(), f, indent=2)

    def add(self, packet: Packet):
        self.packets[packet.id] = packet
        self.save(packet)

    def get(self, pid: str) -> Packet | None:
        return self.packets.get(pid)

    def all(self) -> list[Packet]:
        return list(self.packets.values())

    def find_by_source_url(self, url: str) -> Packet | None:
        if not url:
            return None
        for p in self.packets.values():
            if p.has_source_url(url):
                return p
        return None

    def find_image(self, image_id: str) -> tuple[Packet, PacketImage] | None:
        """Locate an image by id across all packets — used by downstream UIs.

        Linear scan; fine at personal scale.
        """
        if not image_id:
            return None
        for p in self.packets.values():
            img = p.image_by_id(image_id)
            if img is not None:
                return p, img
        return None


class Scratchpad:
    def __init__(self, data_dir: str):
        self.dir = Path(data_dir) / "scratchpad"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.entries: dict[str, ScratchpadEntry] = {}
        self._load_all()

    def _load_all(self):
        for f in self.dir.glob("*.json"):
            with open(f) as fp:
                e = ScratchpadEntry.from_dict(json.load(fp))
                self.entries[e.id] = e

    def save(self, entry: ScratchpadEntry):
        with open(self.dir / f"{entry.id}.json", "w") as f:
            json.dump(entry.to_dict(), f, indent=2)

    def add(self, entry: ScratchpadEntry):
        self.entries[entry.id] = entry
        self.save(entry)

    def remove(self, entry_id: str):
        if entry_id in self.entries:
            del self.entries[entry_id]
            p = self.dir / f"{entry_id}.json"
            if p.exists():
                p.unlink()

    def all(self) -> list[ScratchpadEntry]:
        return list(self.entries.values())

    def prune_old(self, current_session_n: int, max_sessions_inactive: int):
        for e in list(self.entries.values()):
            if (current_session_n - e.last_seen_session_n) > max_sessions_inactive:
                self.remove(e.id)


class SessionCounter:
    def __init__(self, data_dir: str):
        self.path = Path(data_dir) / "session_counter.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.n = 0
        if self.path.exists():
            with open(self.path) as f:
                self.n = json.load(f).get("n", 0)

    def increment(self) -> int:
        self.n += 1
        with open(self.path, "w") as f:
            json.dump({"n": self.n}, f)
        return self.n
