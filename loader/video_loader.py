import numpy as np
from jina.executors.crafters import BaseCrafter
import cv2


class VideoLoader(BaseCrafter):
    def __init__(self, *args, **kwargs):
        super(VideoLoader, self).__init__(*args, **kwargs)

    def resize_frame(self, frame, desired_size):
        min_size = np.min(frame.shape[:2])
        ratio = desired_size / min_size
        frame = cv2.resize(frame, dsize=(0, 0), fx=ratio, fy=ratio, interpolation=cv2.INTER_CUBIC)
        return frame

    def center_crop(self, frame, desired_size):
        old_size = frame.shape[:2]
        top = int(np.maximum(0, (old_size[0] - desired_size) / 2))
        left = int(np.maximum(0, (old_size[1] - desired_size) / 2))
        return frame[top: top + desired_size, left: left + desired_size, :]

    def load_video(self, video, all_frames=False):
        cap = cv2.VideoCapture(video)
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps > 144 or fps is None:
            fps = 25
        frames = []
        count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if isinstance(frame, np.ndarray):
                if int(count % round(fps)) == 0 or all_frames:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frames.append(self.center_crop(self.resize_frame(frame, 256), 256))
            else:
                break
            count += 1
        cap.release()
        return np.array(frames)

    def craft(self, buffer, *args, **kwargs):
        video_path = buffer.decode()
        self.logger.info(f"loading video {video_path}")
        frames = self.load_video(video_path)
        return dict(blob=frames)