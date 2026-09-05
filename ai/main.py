from fastapi import FastAPI, UploadFile, File, Form

app = FastAPI()


@app.get("/")
def root():
    return {"message": "AI service is running"}


@app.post("/predict")
async def predict_endpoint(
    clinical_image: UploadFile = File(...),
    dermoscopic_image: UploadFile = File(...),
    age: int = Form(...),
    sex: str = Form(...),
    skin_tone: int = Form(...),
    site: str = Form(...)
):
    return {
        "message": "Prediction endpoint is ready"
    }