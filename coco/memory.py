import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np


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

    @classmethod
    def new(
        cls,
        topics: list,  # list of (text, vector) tuples
        entities: list,
        gist: str = "",
        summary: str = "",
        full: str = "",
    ):
        ts = now_iso()
        facets = [
            TopicFacet(text=t, vector=(v.tolist() if hasattr(v, "tolist") else list(v)))
            for t, v in topics
        ]
        return cls(
            id=new_id("pkt"),
            topics=facets,
            entities=[e.lower().strip() for e in entities if e and e.strip()],
            content=PacketContent(gist=gist, summary=summary, full=full),
            strength_events=[],
            created_at=ts,
            updated_at=ts,
            source_session_ids=[],
        )

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
        }

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            id=d["id"],
            topics=[TopicFacet.from_dict(t) for t in d["topics"]],
            entities=d.get("entities", []),
            content=PacketContent(**d.get("content", {})),
            strength_events=d.get("strength_events", []),
            created_at=d.get("created_at", ""),
            updated_at=d.get("updated_at", ""),
            source_session_ids=d.get("source_session_ids", []),
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
