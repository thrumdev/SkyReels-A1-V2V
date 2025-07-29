import numpy as np
import insightface
from insightface.utils import face_align 
import cv2
import onnxruntime as ort
import huggingface_hub
import os

arc_path = "pretrained_models/arc/arc.onnx"

def ensure_arc_exists():
    if not os.path.exists(arc_path):
        huggingface_hub.hf_hub_download(
            repo_id="garavv/arcface-onnx",
            filename="arc.onnx",
            local_dir="pretrained_models/arc",
        )

class FaceTracker:
    """
    The face tracker is responsible for tracking faces in a video stream.

    This class keeps track of all the individual faces registered in the video stream,
    processed on a frame-by-frame basis.

    Finally, it informs the system how to clip or trim the video into segments which wholly
    focus on a single face at a time.
    """

    def __init__(self, n_frames, max_missing_frames=8, min_clip_length=49):
        self.n_frames = n_frames
        self.last_frame = None
        self.individuals = []  # List to hold individual face trackers
        self.max_missing_frames = max_missing_frames
        self.min_clip_length = min_clip_length
        self.onnxsess = ort.InferenceSession(arc_path, providers=['CUDAExecutionProvider'])
        self.face_count_at = []


    def ingest_frame(self, image, faces):
        """
        Ingests a list of face detections and updates the internal state.
        
        Args:
            faces (list): A list of face detection dictionaries from insightface.
        """

        frame = 0 if self.last_frame is None else self.last_frame + 1
        self.face_count_at.append(len(faces))
        for face in faces:
            max_similarity = 0.0
            closest_individual = None

            input_name = self.onnxsess.get_inputs()[0].name
            output_name = self.onnxsess.get_outputs()[0].name

            aimg = face_align.norm_crop(image, face['kps'])
            aimg = (aimg.astype(np.float32) - 127.5) / 128.0
            aimg = aimg[np.newaxis, ...]
            embedding = self.onnxsess.run(
                [output_name], 
                {input_name: aimg},
            )[0][0]

            embedding = embedding / np.linalg.norm(embedding)
            for ind in self.individuals:
                missing_frames = ind.missing_frames(frame)

                # After a certain number of frames, an individual is presumed to have left
                # the scene.
                if missing_frames >= self.max_missing_frames:
                    continue

                similarity = ind.cos_sim(embedding)
                if similarity > 0.2 and similarity > max_similarity:
                    max_similarity = similarity
                    closest_individual = ind
                elif len(faces) == 1 and self.face_count_at[frame - 1] == 1:
                    # If there is only one face in the frame, we assume it is the same individual
                    # as the one in the previous frame, if it was also the only face.
                    # This is a heuristic to handle cases where the face is not detected in the
                    # previous frame.
                    if ind.last_found_in['frame'] == frame - 1:
                        max_similarity = similarity
                        closest_individual = ind
            
            if closest_individual is not None:
                closest_individual.update(embedding, frame, face)
            else:
                self.individuals.append(Individual(embedding, frame, face))


        self.last_frame = frame

    def make_clips(self):
        """
        Generates clips based on the tracked individuals.
        
        Returns a list of dictionaries, each containing:
            - 'start': Start frame of the clip
            - 'end': End frame of the clip
            - 'individual': The center of the individual face being tracked
        """

        # Take the longest possible clips that each contain a single individual.
        # Overlapping clips are not allowed.
        clips = []
        for individual in self.individuals:
            start, end = individual.range()
            clip_len = end - start + 1
            if clip_len < self.min_clip_length:
                continue

            displaced = []
            displaced_len = 0
            for i, clip in enumerate(clips):
                clip_start, clip_end = clip['start'], clip['end']
                if not (end < clip_start or start > clip_end):
                    displaced.append(i)
                    displaced_len += clip_end - clip_start + 1
            
            if displaced_len >= clip_len:
                continue
            else:
                clips = [clip for i, clip in enumerate(clips) if i not in displaced]

                # normalize the frame indices to the start of the clip
                faces = {
                    f - start: face for f, face in individual.found_in.items()
                }
                clips.append({
                    'start': start,
                    'end': end,
                    'faces': faces,
                })

        return clips


class Individual:
    """
    Represents an individual face being tracked.
    """

    def __init__(self, embedding, frame, face):
        self.found_in = {}
        self.found_in[frame] = face
        self.first_found_in = frame
        self.last_found_in = {'frame': frame, 'embedding': embedding}

    def missing_frames(self, cur_frame):
        """
        Calculate the number of frames since the individual was last seen.
        """
        return cur_frame - self.last_found_in['frame']

    def cos_sim(self, embedding):
        """
        Calculate the cosine similarity from the embedding of this individual to another embedding
        """

        

        e = self.last_embedding()
        return np.dot(e, embedding) / (np.linalg.norm(e) * np.linalg.norm(embedding))
    
    def update(self, embedding, frame, face):
        """
        Update the individual's embedding and last seen frame.
        """
        self.found_in[frame] = face
        self.last_found_in = {'frame': frame, 'embedding': embedding}
    
    def last_frame(self):
        return self.last_found_in['frame']
    
    def last_embedding(self):
        return self.last_found_in['embedding']
    
    def range(self):
        """
        Returns the range of frames this individual has been found in.
        """
        return self.first_found_in, self.last_frame()
