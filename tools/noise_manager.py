import cv2
import numpy as np

class Noise_Manager:
    def __init__(self, noise_type):
        self.noise_type = noise_type

    def add_noise(self, frame, mean=0, sigma=0.5):

        noise = None

        if self.noise_type == "gauss":
            noise =  np.random.normal(mean, sigma, frame.shape).astype('uint8')
            frame = cv2.add(frame, noise)
            return frame
        
        if self.noise_type == "salt":
            result = np.copy(frame)
            
            salt = np.ceil(0.01 * frame.size)
            coords = [np.random.randint(0, i - 1, int(salt)) for i in frame.shape]
            result[tuple(coords)] = 255

            pepper = np.ceil(0.01 * frame.size)
            coords = [np.random.randint(0, i - 1, int(pepper)) for i in frame.shape]
            result[tuple(coords)] = 0
            
            return result
        
        if self.noise_type == "speckle":
            gauss = np.random.normal(mean, sigma, frame.shape).astype("float32")
            frame = cv2.add(frame.astype("float32"), frame.astype("float32") * gauss)
            return frame.astype("uint8")

        return frame

        