from datetime import datetime


class Feedback:
    def __init__(self, timestamp: datetime, text: str, predicted_main_ppa: str, liked_main_ppa: bool, disliked_main_ppa: bool,
                 selected_main_ppa: str, predicted_other_ppas: list[str], liked_other_ppas: bool, disliked_other_ppas: bool,
                 selected_incorrect_other_ppas: list[str], selected_missing_other_ppas: list[str]):
        self.timestamp = timestamp
        self.text = text

        self.predicted_main_ppa = predicted_main_ppa
        self.liked_main_ppa = liked_main_ppa
        self.disliked_main_ppa = disliked_main_ppa
        self.selected_main_ppa = selected_main_ppa

        self.predicted_other_ppas = predicted_other_ppas
        self.liked_other_ppas = liked_other_ppas
        self.disliked_other_ppas = disliked_other_ppas
        self.selected_incorrect_other_ppas = selected_incorrect_other_ppas
        self.selected_missing_other_ppas = selected_missing_other_ppas
