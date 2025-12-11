import os
import numpy as np
from PIL import Image
from deepface import DeepFace

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

MODEL_NAME = "Facenet512"
DETECTOR_BACKEND = "opencv"

current_dir = os.path.dirname(os.path.abspath(__file__))
DATABASE_FOLDER = os.path.join(current_dir, "database_img")

MOCK_PATIENT_DB = {
    "Sarah": {"disease": "Diabetes", "current_appointment": "2025-12-31"},
    "Peter": {"disease": "Diabetes", "current_appointment": "2025-12-25"}
}

def get_face_embedding(image: Image.Image):
    try:
        img_rgb = np.array(image.convert("RGB"))
        img_bgr = img_rgb[:, :, ::-1]
        embedding_objs = DeepFace.represent(
            img_path=img_bgr,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR_BACKEND,
            anti_spoofing=True,
            enforce_detection=True
        )
        
        if len(embedding_objs) > 0:
            return embedding_objs[0]["embedding"]
        return None

    except ValueError:
        return None
    except Exception:
        return None

def load_database_from_folder(folder_path=DATABASE_FOLDER):
    db = {}
    if not os.path.exists(folder_path):
        os.makedirs(folder_path, exist_ok=True)
        return db

    files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for filename in files:
        filepath = os.path.join(folder_path, filename)
        try:
            embedding_objs = DeepFace.represent(
                img_path=filepath,
                model_name=MODEL_NAME,
                detector_backend=DETECTOR_BACKEND,
                enforce_detection=False 
            )
            
            if len(embedding_objs) > 0:
                db[filename] = embedding_objs[0]["embedding"]
                
        except Exception:
            pass
            
    return db

def find_match(target_encoding, database, threshold=0.4):
    if not database:
        return "Database Empty", 0.0, False

    target_vector = np.array(target_encoding)
    best_match_name = "None"
    lowest_distance = 100.0 
   
    for filename, db_encoding in database.items():
        db_vector = np.array(db_encoding)
        
        dot_product = np.dot(target_vector, db_vector)
        norm_a = np.linalg.norm(target_vector)
        norm_b = np.linalg.norm(db_vector)
        
        similarity = dot_product / (norm_a * norm_b)
        distance = 1 - similarity
        
        if distance < lowest_distance:
            lowest_distance = distance
            best_match_name = filename

    similarity_score = 1.0 - lowest_distance
    is_match = lowest_distance < threshold

    return best_match_name, float(similarity_score), is_match

def get_patient_info(filename):
    if filename == "No match" or filename == "Database Empty":
        return None
    name_key = os.path.splitext(filename)[0]
    return MOCK_PATIENT_DB.get(name_key)