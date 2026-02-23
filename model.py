import os
# --- MAGIC FIX: DISABLE BROKEN GPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import pandas as pd
import tensorflow as tf
import numpy as np
from PIL import Image

class Model:
    def __init__(self):
        self.model = None
        self.IMG_SIZE = 260
        self.SIGMAS = {
            "SPEI_30d": 0.9069,
            "SPEI_1y":  0.8832,
            "SPEI_2y":  0.8217
        }

    def load(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        model_path = os.path.join(dir_path, "best_beetle_model.keras")
        self.model = tf.keras.models.load_model(model_path)

    def predict(self, inputs):
        processed_images = []
        names = []
        domains = []
        
        for record in inputs:
            try:
                if "relative_img" in record:
                    img = record["relative_img"] 
                    img_array = tf.keras.utils.img_to_array(img)
                elif "relative_img_loc" in record:
                    path = record["relative_img_loc"]
                    if os.path.exists(path):
                        img_raw = tf.io.read_file(path)
                        img_array = tf.image.decode_image(img_raw, channels=3)
                    else:
                        continue
                else:
                    continue

                img_tensor = tf.cast(img_array, dtype=tf.float32)
                img_tensor = tf.image.resize(img_tensor, [self.IMG_SIZE, self.IMG_SIZE])
                img_tensor.set_shape([self.IMG_SIZE, self.IMG_SIZE, 3])
                
                name = str(record.get("scientificName", "Unknown"))
                domain = str(record.get("domainID", "Unknown"))
                
                processed_images.append(img_tensor)
                names.append(name)
                domains.append(domain)
                
            except Exception:
                continue

        if not processed_images:
             return {
                "SPEI_30d": {"mu": 0.0, "sigma": self.SIGMAS["SPEI_30d"]},
                "SPEI_1y":  {"mu": 0.0, "sigma": self.SIGMAS["SPEI_1y"]},
                "SPEI_2y":  {"mu": 0.0, "sigma": self.SIGMAS["SPEI_2y"]},
            }

        batch_images = tf.stack(processed_images)
        batch_names = tf.convert_to_tensor(names)
        batch_domains = tf.convert_to_tensor(domains)
        
        predictions = self.model.predict([batch_images, batch_names, batch_domains], verbose=0)
        mean_prediction = np.mean(predictions, axis=0)

        return {
            "SPEI_30d": {"mu": float(mean_prediction[0]), "sigma": self.SIGMAS["SPEI_30d"]},
            "SPEI_1y":  {"mu": float(mean_prediction[1]), "sigma": self.SIGMAS["SPEI_1y"]},
            "SPEI_2y":  {"mu": float(mean_prediction[2]), "sigma": self.SIGMAS["SPEI_2y"]}
        }