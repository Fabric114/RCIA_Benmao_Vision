from collections import OrderedDict
from dataclasses import dataclass


@dataclass(frozen=True)
class LatestRgbdPair:
    color_stamp_ns: int
    depth_stamp_ns: int
    color: object
    depth: object


class LatestRgbdBuffer:
    def __init__(self, depth_history_size=30, color_history_size=30):
        self._color_stamp_ns = None
        self._color = None
        self._color_frames = OrderedDict()
        self._depth_frames = OrderedDict()
        self._depth_history_size = max(1, int(depth_history_size))
        self._color_history_size = max(1, int(color_history_size))

    def clear(self):
        self._color_stamp_ns = None
        self._color = None
        self._color_frames.clear()
        self._depth_frames.clear()

    def put_color(self, stamp_ns, image):
        stamp_ns = int(stamp_ns)
        if self._color_stamp_ns is not None and stamp_ns < self._color_stamp_ns:
            return
        self._color_stamp_ns = stamp_ns
        self._color = image
        self._color_frames[stamp_ns] = image
        self._color_frames.move_to_end(stamp_ns)
        while len(self._color_frames) > self._color_history_size:
            self._color_frames.popitem(last=False)

    def put_depth(self, stamp_ns, image):
        stamp_ns = int(stamp_ns)
        self._depth_frames[stamp_ns] = image
        self._depth_frames.move_to_end(stamp_ns)
        while len(self._depth_frames) > self._depth_history_size:
            self._depth_frames.popitem(last=False)

    def get_latest_pair(self, now_ns, tolerance_ns, max_age_ns):
        if self._color_stamp_ns is None or not self._depth_frames:
            return None
        now_ns = int(now_ns)
        tolerance_ns = int(tolerance_ns)
        max_age_ns = int(max_age_ns)
        if now_ns - self._color_stamp_ns > max_age_ns:
            return None

        depth_stamp_ns = min(
            self._depth_frames.keys(),
            key=lambda stamp_ns: abs(stamp_ns - self._color_stamp_ns),
        )
        if abs(depth_stamp_ns - self._color_stamp_ns) > tolerance_ns:
            return None
        return LatestRgbdPair(
            color_stamp_ns=self._color_stamp_ns,
            depth_stamp_ns=depth_stamp_ns,
            color=self._color,
            depth=self._depth_frames[depth_stamp_ns],
        )

    def get_pair_at(self, target_stamp_ns, tolerance_ns):
        if not self._color_frames or not self._depth_frames:
            return None
        target_stamp_ns = int(target_stamp_ns)
        tolerance_ns = int(tolerance_ns)
        color_stamp_ns = min(
            self._color_frames.keys(),
            key=lambda stamp_ns: abs(stamp_ns - target_stamp_ns),
        )
        if abs(color_stamp_ns - target_stamp_ns) > tolerance_ns:
            return None
        depth_stamp_ns = min(
            self._depth_frames.keys(),
            key=lambda stamp_ns: abs(stamp_ns - color_stamp_ns),
        )
        if abs(depth_stamp_ns - color_stamp_ns) > tolerance_ns:
            return None
        return LatestRgbdPair(
            color_stamp_ns=color_stamp_ns,
            depth_stamp_ns=depth_stamp_ns,
            color=self._color_frames[color_stamp_ns],
            depth=self._depth_frames[depth_stamp_ns],
        )
