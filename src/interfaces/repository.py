from typing import Protocol

from models.feedback import Feedback

class Repository(Protocol):
    def AddFeedback(self, feedback: Feedback) -> None:
        """Add a feedback item to the repository."""
        ...