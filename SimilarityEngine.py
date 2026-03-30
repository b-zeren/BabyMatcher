import cv2
import numpy as np

class SimilarityEngine:
    def __init__(self, detector_path, embedder_path):
        self.detector = cv2.FaceDetectorYN.create(model=detector_path, 
                                                  config="", 
                                                  score_threshold=0.60,
                                                  nms_threshold=0.3,
                                                  input_size=(320, 320))
        self.embedder = cv2.FaceRecognizerSF.create(model=embedder_path, 
                                                    config="")

    def find_faces(self, image):
        self.detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = self.detector.detect(image)

        scale = 1.0
        if faces is None or len(faces) == 0:
            h, w = image.shape[:2]
            scale = 640 / max(h, w) 
            temp_img = cv2.resize(image, (int(w * scale), int(h * scale)))
            
            self.detector.setInputSize((temp_img.shape[1], temp_img.shape[0]))
            _, faces = self.detector.detect(temp_img)
            
            if faces is not None and len(faces) > 0:
                faces[0][:14] = faces[0][:14] / scale
            else:
                return None, None, None

        face_info = faces[0]

        x, y, w, h = face_info[:4].astype(int)
        # Ensure coordinates are within image boundaries
        x, y = max(0, x), max(0, y)
        full_face = image[y:y+h, x:x+w]
        
        aligned_face = self.embedder.alignCrop(image, face_info)
        
        # lms indices: 0,1: R.Eye | 2,3: L.Eye | 4,5: Nose | 6,7: R.Mouth | 8,9: L.Mouth
        lms = face_info[4:14].reshape(5, 2).astype(int)
        
        patches = {
            "eyes": self._get_patch(image, lms[0:2], padding=int(w//5)),
            "nose": self._get_patch(image, [lms[2]], padding=int(w//6)),
            "mouth": self._get_patch(image, lms[3:5], padding=int(w//6))
        }
        
        return full_face, aligned_face, patches

    def _get_patch(self, image, points, padding):
        """Helper to crop an area around a set of landmark points."""
        pts = np.array(points)
        x_min, y_min = np.min(pts, axis=0) - padding
        x_max, y_max = np.max(pts, axis=0) + padding
        
        # Ensure within image bounds
        x_min, y_min = max(0, x_min), max(0, y_min)
        x_max, y_max = min(image.shape[1], x_max), min(image.shape[0], y_max)
        
        return image[int(y_min):int(y_max), int(x_min):int(x_max)]

    def get_face_embedding(self, face_image):
        # First resize to 112*112
        if face_image.shape[0] != 112 or face_image.shape[1] != 112:
            face_image = cv2.resize(face_image, (112, 112))
        return self.embedder.feature(face_image)

    def compare_faces(self, embedding1, embedding2):
        # Compare using cosine similarity
        distance = np.dot(embedding1, embedding2.T) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return distance[0][0]