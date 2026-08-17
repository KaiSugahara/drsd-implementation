class ConsecutivePruningManager:
    def __init__(self, threshold) -> None:
        self.threshold = threshold
        self.remove_streak_by_label: dict[str, int] = {}
        self.reject_streak_by_label: dict[str, int] = {}

    def apply_round(
        self,
        removed_labels: list[str],
        rejected_labels: list[str],
    ) -> tuple[list[str], list[str], dict[str, dict[str, int]]]:

        for label in removed_labels:
            self.remove_streak_by_label[label] = self.remove_streak_by_label.get(label, 0) + 1
        for label in set(self.remove_streak_by_label) - set(removed_labels):
            self.remove_streak_by_label.pop(label, None)

        for label in rejected_labels:
            self.reject_streak_by_label[label] = self.reject_streak_by_label.get(label, 0) + 1
        for label in set(self.reject_streak_by_label) - set(rejected_labels):
            self.reject_streak_by_label.pop(label, None)

        confirmed_removed_labels = [
            label for label in removed_labels if self.remove_streak_by_label.get(label, 0) >= self.threshold
        ]
        confirmed_rejected_labels = [
            label for label in rejected_labels if self.reject_streak_by_label.get(label, 0) >= self.threshold
        ]

        for label in confirmed_removed_labels + confirmed_rejected_labels:
            self.remove_streak_by_label.pop(label, None)
            self.reject_streak_by_label.pop(label, None)

        streak_snapshot = {
            "remove_streak_by_label": dict(self.remove_streak_by_label),
            "reject_streak_by_label": dict(self.reject_streak_by_label),
        }

        return confirmed_removed_labels, confirmed_rejected_labels, streak_snapshot
