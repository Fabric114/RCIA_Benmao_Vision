import json
import unittest

from scripts.yolo_detection_message import detections_to_json


class YoloDetectionMessageTest(unittest.TestCase):
    def test_detections_to_json_uses_stable_schema(self):
        payload = json.loads(
            detections_to_json(
                123,
                [
                    (1, 2, 3, 4, 0.91, 0),
                    (5.5, 6.5, 7.5, 8.5, 0.42, None),
                ],
            )
        )

        self.assertEqual(payload["stamp_ns"], 123)
        self.assertEqual(
            payload["detections"][0],
            {
                "class_id": 0,
                "confidence": 0.91,
                "bbox_xyxy": [1.0, 2.0, 3.0, 4.0],
            },
        )
        self.assertEqual(payload["detections"][1]["class_id"], -1)
        self.assertEqual(payload["detections"][1]["bbox_xyxy"], [5.5, 6.5, 7.5, 8.5])

    def test_detections_to_json_accepts_mapping_detections(self):
        payload = json.loads(
            detections_to_json(
                456,
                [
                    {
                        "bbox_xyxy": [10, 20, 30, 40],
                        "confidence": 0.8,
                        "class_id": 2,
                    }
                ],
            )
        )

        self.assertEqual(payload["stamp_ns"], 456)
        self.assertEqual(payload["detections"][0]["class_id"], 2)
        self.assertEqual(payload["detections"][0]["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
