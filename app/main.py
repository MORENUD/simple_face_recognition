import uvicorn
import base64
import io
from PIL import Image
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from . import recognition

ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    db = recognition.load_database_from_folder()
    ml_resources["known_database"] = db
    yield
    ml_resources.clear()

app = FastAPI(title="Simple Face Auth API", lifespan=lifespan)

class ImageRequest(BaseModel):
    image_base64: str

class DebugInfo(BaseModel):
    file: str
    score: float

class PredictionResponse(BaseModel):
    user_name: str
    disease: str
    appointment_day: int
    debug_info: DebugInfo

def decode_base64_image(base64_str: str) -> Image.Image:
    try:
        if "," in base64_str:
            base64_str = base64_str.split(",")[1]
        
        image_data = base64.b64decode(base64_str)
        return Image.open(io.BytesIO(image_data))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid Base64 string")

@app.post("/recognize", response_model=PredictionResponse)
async def recognize_face(payload: ImageRequest):
    known_database = ml_resources.get("known_database")
    if not known_database:
        raise HTTPException(status_code=500, detail="Database empty")

    input_image = decode_base64_image(payload.image_base64)

    try:
        target_vector = recognition.process_image_to_vector(input_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    matched_filename, score = recognition.find_match(
        target_vector, 
        known_database, 
        threshold=0.8
    )

    if matched_filename == "No match":
        return {
            "user_name": "NA",
            "disease": "NA",
            "appointment_day": 0,
            "debug_info": {
                "file": "NA",
                "score": 0.0
            }
        }

    real_name = matched_filename.split('.')[0]
    patient_info = recognition.get_patient_info(matched_filename)

    if patient_info:
        disease = patient_info["disease"]
        appt_day = patient_info["appointment_day"]
    else:
        disease = "Unknown Data"
        appt_day = 0

    return {
        "user_name": real_name,
        "disease": disease,
        "appointment_day": appt_day,
        "debug_info": {
            "file": matched_filename,
            "score": round(score, 4)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)