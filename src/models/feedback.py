from datetime import datetime

class Feedback:
    def __init__(self, timestamp: datetime, text: str, predicted_label: str, expected_label: str, liked: bool, disliked: bool):
        self.timestamp = timestamp
        self.text = text
        self.predicted_label = predicted_label
        self.expected_label = expected_label
        self.liked = liked
        self.disliked = disliked
