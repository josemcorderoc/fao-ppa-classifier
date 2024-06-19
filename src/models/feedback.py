from datetime import datetime

class Feedback:
    def __init__(self, timestamp: datetime, text: str, predicted_label: str, selected_incorrect_ppas: list[str], selected_missing_ppas: list[str], liked: bool, disliked: bool):
        self.timestamp = timestamp
        self.text = text
        self.predicted_label = predicted_label
        self.selected_incorrect_ppas = selected_incorrect_ppas
        self.selected_missing_ppas = selected_missing_ppas
        self.liked = liked
        self.disliked = disliked
