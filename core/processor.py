class HazardProcessor:
    def __init__(self, cam_height=1.5, focal_len=500, frame_w=640, frame_h=480):
        self.cam_height = cam_height
        self.f = focal_len
        self.W = frame_w
        self.H = frame_h

    def get_distance(self, bottom_y):
        y_rel = bottom_y - (self.H / 2)
        if y_rel <= 0:
            return 20.0
        distance = (self.cam_height * self.f) / y_rel
        return round(distance, 1)

    def get_position(self, x1, x2):
        center_x = (x1 + x2) / 2
        if center_x < self.W / 3:
            return "on your left"
        elif center_x > 2 * self.W / 3:
            return "on your right"
        return "ahead"

    def process(self, detections):
        alerts = []

        for d in detections:
            # SAFETY CHECK
            if not isinstance(d, dict) or "box" not in d:
                continue

            x1, y1, x2, y2 = map(int, d["box"])

            alert_text = (
                f"{d['label']} "
                f"{self.get_distance(y2)} meters "
                f"{self.get_position(x1, x2)}"
            )

            alerts.append({
                "label": d["label"],
                "box": d["box"],
                "text": alert_text
            })

        return alerts
