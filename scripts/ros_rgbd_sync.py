from collections import OrderedDict


def stamp_to_nanoseconds(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def find_nearest_stamp(target_ns, stamps, tolerance_ns):
    best_stamp = None
    best_delta = None
    for stamp in stamps:
        delta = abs(int(stamp) - int(target_ns))
        if best_delta is None or delta < best_delta:
            best_stamp = int(stamp)
            best_delta = delta
    if best_stamp is None or best_delta > int(tolerance_ns):
        return None
    return best_stamp


class StampedFrameBuffer:
    def __init__(self, max_size):
        self.max_size = max(1, int(max_size))
        self._frames = OrderedDict()

    def put(self, stamp_ns, frame):
        stamp_ns = int(stamp_ns)
        self._frames[stamp_ns] = frame
        self._frames.move_to_end(stamp_ns)
        while len(self._frames) > self.max_size:
            self._frames.popitem(last=False)

    def get_nearest(self, target_ns, tolerance_ns):
        stamp_ns = find_nearest_stamp(target_ns, self._frames.keys(), tolerance_ns)
        if stamp_ns is None:
            return None
        return stamp_ns, self._frames[stamp_ns]
