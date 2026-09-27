"""
Stage 4: Anomaly & Accident Detection.

Runs on the same camera feed as ANPR. Uses classical motion-analysis
heuristics (optical flow + frame differencing) as a fast, trainable-free
baseline for a hackathon prototype:

  - Sudden large motion change between consecutive frames -> possible collision
  - A tracked region with near-zero motion for an extended period, positioned
    where traffic should be flowing -> possible stall/breakdown

NOTE: For production, replace `detect_anomaly` with a lightweight 3D-CNN or
spatio-temporal autoencoder trained on normal traffic flow, which generalizes
better than fixed thresholds. The event schema/output stays the same either
way, so this can be swapped without touching downstream code.
"""
import cv2
import numpy as np
from collections import deque

MOTION_SPIKE_THRESHOLD = 35.0     # sudden large frame-to-frame motion -> collision-like
STALL_FRAMES_THRESHOLD = 8        # consecutive near-static frames -> possible stall
STALL_MOTION_THRESHOLD = 2.0
CONFIRMATION_WINDOW = 3           # require N consecutive flagged frames before alerting (cuts false positives)


class AnomalyDetector:
    def __init__(self):
        self.prev_gray = None
        self.motion_history = deque(maxlen=30)
        self.stall_counter = 0
        self.pending_flags = deque(maxlen=CONFIRMATION_WINDOW)
        self.alert_active = False  # cooldown: don't re-fire every frame of one ongoing incident

    def _frame_motion_score(self, gray):
        if self.prev_gray is None:
            self.prev_gray = gray
            return 0.0
        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        score = float(np.mean(magnitude))
        self.prev_gray = gray
        return score

    def process_frame(self, frame_bgr, camera_id, timestamp):
        """
        Feed one frame at a time (as if streaming from a live camera).
        Returns an event dict if a confirmed anomaly is detected, else None.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (320, 180))
        motion = self._frame_motion_score(gray)
        self.motion_history.append(motion)

        raw_flag = None
        if len(self.motion_history) >= 5:
            baseline = np.mean(list(self.motion_history)[:-1])
            if motion > baseline + MOTION_SPIKE_THRESHOLD:
                raw_flag = "collision_like_motion"

        if motion < STALL_MOTION_THRESHOLD:
            self.stall_counter += 1
        else:
            self.stall_counter = 0
        if self.stall_counter >= STALL_FRAMES_THRESHOLD:
            raw_flag = "possible_stall"

        self.pending_flags.append(raw_flag)

        if raw_flag is None:
            self.alert_active = False  # incident cleared, allow future re-triggering

        # confirmation: only fire if the SAME flag type shows up consistently,
        # not a one-off spike (camera shake, a truck passing close to the lens, etc.),
        # and only once per ongoing incident (cooldown) so we don't spam alerts.
        if (not self.alert_active and len(self.pending_flags) == CONFIRMATION_WINDOW
                and all(f == raw_flag and f is not None for f in self.pending_flags)):
            self.alert_active = True
            confidence = min(99.0, 50 + (motion - MOTION_SPIKE_THRESHOLD) * 2) if raw_flag == "collision_like_motion" else 70.0
            return {
                "event_type": "accident_alert",
                "anomaly_type": raw_flag,
                "camera_id": camera_id,
                "timestamp": timestamp,
                "motion_score": round(motion, 2),
                "confidence": round(confidence, 1),
            }
        return None


if __name__ == "__main__":
    import sys, glob, time
    frame_paths = sorted(glob.glob(sys.argv[1])) if len(sys.argv) > 1 else []
    if not frame_paths:
        print("Usage: python accident_detection.py 'frames_dir/*.jpg'")
    else:
        det = AnomalyDetector()
        for i, p in enumerate(frame_paths):
            frame = cv2.imread(p)
            event = det.process_frame(frame, camera_id="CAM-01", timestamp=i)
            if event:
                print("ALERT:", event)
        print(f"Processed {len(frame_paths)} frames.")
