import os
import uuid
import logging
import shutil
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends
from fastapi.responses import FileResponse

from api_models import (
    GenerationRequest, TrainingRequest, JobResponse, JobStatus
)

from generator import PipelineConfig, TeacherModel, DatasetGenerator
from trainer import TrainingConfig, start_training_process

# --- Globals for State Management ---
JOBS: Dict[str, Dict[str, Any]] = {}
MODEL_REGISTRY: Dict[str, Any] = {}

# --- FastAPI Lifespan Manager ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manages loading the teacher model on startup."""
    logging.info("Server startup: Loading the Teacher Model...")
    # The teacher model is loaded once and shared across all generation requests.
    model_config = PipelineConfig(pdf_path="", output_path="")
    teacher = TeacherModel(model_config)
    teacher.load()
    MODEL_REGISTRY["teacher"] = teacher
    logging.info("Teacher Model loaded and ready.")
    
    yield
    
    logging.info("Server shutdown: Clearing model from memory.")
    MODEL_REGISTRY.clear()
    
# --- FastAPI App ---
app = FastAPI(
    title="Distillation MLOps Pipeline",
    description="An API to orchestrate dataset generation and model training.",
    lifespan=lifespan,
)

# --- Background Task Functions ---
def run_generation_task(job_id: str, pdf_path: str, config_params: dict):
    """Background task for data generation."""
    job_status = JOBS[job_id]
    output_path = f"outputs/datasets/{job_id}_dataset.jsonl"
    
    config = PipelineConfig(
        pdf_path=pdf_path,
        output_path=output_path,
        **config_params
    )
    
    teacher = MODEL_REGISTRY["teacher"]
    generator = DatasetGenerator(config, teacher)
    generator.run(job_status=job_status)
    JOBS[job_id]["output_path"] = output_path

def run_training_task(job_id: str, config_params: dict):
    """Background task for model training."""
    job_status = JOBS[job_id]
    config = TrainingConfig(**config_params)
    start_training_process(config, job_status)

# --- API Endpoints ---
@app.get("/health", summary="Health Check")
def health_check():
    model_loaded = "teacher" in MODEL_REGISTRY and MODEL_REGISTRY["teacher"].model is not None
    return {"status": "ok", "teacher_model_loaded": model_loaded}

@app.post("/jobs/generate", response_model=JobResponse, status_code=202)
async def start_generation_job(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="The PDF document to process."),
    config: GenerationRequest = Depends(),
):
    job_id = f"gen-{uuid.uuid4()}"
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("outputs/datasets", exist_ok=True)
    
    pdf_path = f"uploads/{job_id}_{file.filename}"
    with open(pdf_path, "wb") as buffer:
        buffer.write(await file.read())
        
    JOBS[job_id] = {"status": "PENDING", "job_type": "generation"}
    background_tasks.add_task(run_generation_task, job_id, pdf_path, config.model_dump())
    
    return JobResponse(job_id=job_id, status="PENDING", message="Generation job started.")

@app.post("/jobs/train", response_model=JobResponse, status_code=202)
def start_training_job(
    background_tasks: BackgroundTasks,
    config: TrainingRequest,
):
    job_id = f"train-{uuid.uuid4()}"
    
    # Validate that the dataset file exists
    if not os.path.exists(config.dataset_path):
        raise HTTPException(status_code=400, detail=f"Dataset file not found at: {config.dataset_path}")
    
    JOBS[job_id] = {"status": "PENDING", "job_type": "training"}
    background_tasks.add_task(run_training_task, job_id, config.model_dump())
    
    return JobResponse(job_id=job_id, status="PENDING", message="Training job started.")

@app.get("/jobs/{job_id}/status", response_model=JobStatus)
def get_job_status(job_id: str):
    """Check the status of any job (generation or training)."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return JobStatus(**job)

@app.get("/jobs/{job_id}/download", response_class=FileResponse)
def download_job_result(job_id: str):
    """Download the output of a completed job."""
    job = JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "COMPLETED":
        raise HTTPException(status_code=400, detail=f"Job not complete. Status: {job.get('status')}")
    
    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output artifact not found.")

    if job.get("job_type") == "generation":
        return FileResponse(path=output_path, media_type='application/jsonl', filename=os.path.basename(output_path))
    elif job.get("job_type") == "training":
        # Zip the model directory for download
        dir_name = os.path.basename(output_path)
        zip_path = f"outputs/models/{dir_name}.zip"
        os.makedirs("outputs/models", exist_ok=True)
        shutil.make_archive(zip_path.replace(".zip", ""), 'zip', output_path)
        return FileResponse(path=zip_path, media_type='application/zip', filename=f"{dir_name}.zip")
    else:
        raise HTTPException(status_code=400, detail="Unknown job type for download.")