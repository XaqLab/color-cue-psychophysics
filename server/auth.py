"""Simple file-backed user store with hashed passwords."""

from __future__ import annotations

import json
from pathlib import Path

from werkzeug.security import check_password_hash, generate_password_hash


class UserStore:
    """JSON-backed user database.

    Passwords are stored as Werkzeug-hashed strings; plaintext is never
    persisted to disk.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._data: dict[str, dict] = {}
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                self._data = json.load(f)

    # ------------------------------------------------------------------
    # Internal helpers

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2)

    # ------------------------------------------------------------------
    # Public API

    def create(
        self,
        subject_id: str,
        password: str,
        is_admin: bool = False,
    ) -> None:
        """Create a new user account.

        Args:
            subject_id: Unique participant identifier used as username.
            password: Plaintext password (hashed before storage).
            is_admin: Grant administrator privileges.

        Raises:
            ValueError: If ``subject_id`` or ``password`` is empty, or if the
                subject already exists.
        """
        if not subject_id or not password:
            raise ValueError("subject_id and password are required.")
        if subject_id in self._data:
            raise ValueError(f"User {subject_id!r} already exists.")
        self._data[subject_id] = {
            "subject_id": subject_id,
            "password_hash": generate_password_hash(password),
            "is_admin": is_admin,
        }
        self._save()

    def verify(self, subject_id: str, password: str) -> bool:
        """Return ``True`` if ``password`` matches the stored hash."""
        user = self._data.get(subject_id)
        if user is None:
            return False
        return check_password_hash(user["password_hash"], password)

    def get(self, subject_id: str) -> dict | None:
        """Return the user record, or ``None`` if not found."""
        return self._data.get(subject_id)

    def list_users(self) -> list[dict]:
        """Return a list of ``{subject_id, is_admin}`` dicts for all users."""
        return [
            {
                "subject_id": u["subject_id"],
                "is_admin": u.get("is_admin", False),
            }
            for u in self._data.values()
        ]

    def change_password(self, subject_id: str, new_password: str) -> None:
        """Replace the stored password hash for an existing user.

        Raises:
            ValueError: If the user does not exist.
        """
        if subject_id not in self._data:
            raise ValueError(f"User {subject_id!r} not found.")
        self._data[subject_id]["password_hash"] = generate_password_hash(new_password)
        self._save()

    def delete(self, subject_id: str) -> None:
        """Remove a user account.

        Raises:
            ValueError: If the user does not exist.
        """
        if subject_id not in self._data:
            raise ValueError(f"User {subject_id!r} not found.")
        del self._data[subject_id]
        self._save()
