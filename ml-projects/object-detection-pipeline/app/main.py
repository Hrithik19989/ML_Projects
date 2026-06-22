from app.model import predict_objects
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Automated Vision Monitoring Pipeline",
    version="1.0.0",
    description="Production API for Defect Detection or Traffic Monitoring",
)


@app.get("/health")
def health_check():
    """Simple endpoint to verify the service status."""
    return {"status": "healthy"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Accepts an image file and returns object detection coordinates."""
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG and PNG are supported.",
        )

    try:
        # Read raw image bytes
        image_bytes = await file.read()
        # Run vision pipeline
        predictions = predict_objects(image_bytes)

        return JSONResponse(
            content={"filename": file.filename, "detections": predictions}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
