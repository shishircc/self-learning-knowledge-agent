import uuid
import numpy as np


class Session:
    def __init__(self, session_id: str | None = None):
        self.id = session_id or "ses_" + uuid.uuid4().hex[:12]
        self.topics: list[dict] = []  # {topic_text, topic_vector, first_seen_turn, last_seen_turn}
        self.current_topic_idx: int | None = None
        self.loaded_packets: dict[str, dict] = {}  # packet_id -> {packet, slice}
        self.turns: list[dict] = []  # {role: "user"|"coco", content: str}

    def add_turn(self, role: str, content: str):
        self.turns.append({"role": role, "content": content})

    def recent_turns(self, n_exchanges: int) -> list[dict]:
        # n_exchanges of (user, coco) pairs
        return self.turns[-(n_exchanges * 2):]

    def loaded_packets_list(self) -> list[dict]:
        return list(self.loaded_packets.values())

    def add_topic(self, topic_text: str, topic_vector: np.ndarray):
        self.topics.append({
            "topic_text": topic_text,
            "topic_vector": topic_vector.tolist(),
            "first_seen_turn": len(self.turns),
            "last_seen_turn": len(self.turns),
        })
        self.current_topic_idx = len(self.topics) - 1

    def update_topic(self, idx: int):
        self.topics[idx]["last_seen_turn"] = len(self.turns)
        self.current_topic_idx = idx
