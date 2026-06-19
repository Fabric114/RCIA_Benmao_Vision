class AutoRegisterGate:
    """Gate automatic registration until detections are confident and stable."""

    def __init__(self, min_conf=0.7, stable_frames=3):
        if stable_frames < 1:
            raise ValueError("stable_frames must be >= 1")
        self.min_conf = float(min_conf)
        self.stable_frames = int(stable_frames)
        self._stable_count = 0

    @staticmethod
    def _area(det):
        x1, y1, x2, y2, *_ = det
        return max(0, x2 - x1) * max(0, y2 - y1)

    def reset(self):
        self._stable_count = 0

    def update(self, detections):
        eligible = [det for det in detections if float(det[4]) >= self.min_conf]
        if not eligible:
            self._stable_count = 0
            return None

        self._stable_count += 1
        selected = max(eligible, key=lambda det: (float(det[4]), self._area(det)))
        if self._stable_count >= self.stable_frames:
            return selected
        return None
