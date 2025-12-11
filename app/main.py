import uvicorn
import base64
import io
import os
from PIL import Image
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from recognition import (
    load_database_from_folder, 
    get_face_embedding, 
    find_match, 
    get_patient_info
)

ml_resources = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    db = load_database_from_folder()
    ml_resources["known_database"] = db
    yield
    ml_resources.clear()

app = FastAPI(title="Face Auth API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageRequest(BaseModel):
    image_base64: str

class DebugInfo(BaseModel):
    file: str
    score: float

class PredictionResponse(BaseModel):
    name: str
    disease: str
    current_appointment: str
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
    if known_database is None:
        known_database = {}

    try:
        input_image = decode_base64_image(payload.image_base64)
        target_encoding = get_face_embedding(input_image)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image processing failed: {str(e)}")

    if target_encoding is None:
        return {
            "name": "No Face Detected",
            "disease": "NA",
            "current_appointment": '2099-12-31',
            "debug_info": {"file": "NA", "score": 0.0}
        }

    closest_filename, score, is_match = find_match(target_encoding, known_database, threshold=0.40)

    if not is_match:
        return {
            "name": "Unknown",
            "disease": "NA",
            "current_appointment": '2099-12-31',
            "debug_info": {
                "file": closest_filename,
                "score": round(score, 4)
            }
        }

    real_name = os.path.splitext(closest_filename)[0]
    patient_info = get_patient_info(closest_filename)

    disease = patient_info["disease"] if patient_info else "Data Not Found"
    appt_day = patient_info["current_appointment"] if patient_info else "2099-12-31"

    return {
        "name": real_name,
        "disease": disease,
        "current_appointment": appt_day,
        "debug_info": {
            "file": closest_filename,
            "score": round(score, 4)
        }
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)