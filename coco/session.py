import uuid
import numpy as np

from .auth import Identity, ANONYMOUS


class Session:
    def __init__(
        self,
        session_id: str | None = None,
        user: Identity | None = None,
    ):
        self.id = session_id or "ses_" + uuid.uuid4().hex[:12]
        # `user` is acquired once at startup by `auth.acquire_identity` and is
        # immutable for the session — re-auth requires restarting Coco.
        self.user: Identity = user or ANONYMOUS
        self.topics: list[dict] = []  # {topic_text, topic_vector, first_seen_turn, last_seen_turn}
        self.current_topic_idx: int | None = None
        self.loaded_packets: dict[str, dict] = {}  # packet_id -> {packet, slice}
        self.turns: list[dict] = []  # {role: "user"|"coco", content: str}

    @property
    def admin_mode(self) -> bool:
        """True iff the session was started with the --admin CLI flag.

        Derived from `user.provider` (not stored) so the visual mode cannot
        drift from the acquired identity. Read by run_session and the UI
        helpers that paint the admin banner and per-turn badge.
        """
        return self.user.provider == "cli_admin"

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
