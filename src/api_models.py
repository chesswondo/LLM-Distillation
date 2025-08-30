from pydantic import BaseModel, Field
from typing import Optional

class GenerationRequest(BaseModel):
    """Configuration for a new dataset generation job."""
    chunk_size: int = Field(1024, description="Chunk size in tokens.")
    chunk_overlap: int = Field(128, description="Token overlap between chunks.")
    questions_per_chunk: int = Field(1, description="Number of questions to generate per chunk.")
    store_logits: bool = Field(True, description="Whether to store teacher logits for distillation.")
    logits_top_k: int = Field(50, description="Value of 'k' for top-k logit storage.")

class TrainingRequest(BaseModel):
    """Configuration for a new model training job."""
    dataset_path: str = Field(..., description="Path to the .jsonl dataset file generated previously.")
    output_dir: str = Field(..., description="Directory to save the trained model artifacts.")
    student_model_name: str = Field("microsoft/Phi-3-mini-4k-instruct", description="HF model ID for the student.")
    
    # Key training hyperparameters
    num_train_epochs: int = Field(3, gt=0, description="Number of training epochs.")
    learning_rate: float = Field(2e-4, gt=0, description="Learning rate.")
    per_device_train_batch_size: int = Field(1, gt=0, description="Batch size per device.")
    gradient_accumulation_steps: int = Field(8, gt=0, description="Gradient accumulation steps.")
    loss_alpha: float = Field(0.5, ge=0.0, le=1.0, description="Weight for Cross-Entropy loss in distillation.")

    # LoRA configuration
    use_lora: bool = Field(True, description="Enable LoRA for training.")
    lora_r: int = Field(16, gt=0, description="LoRA attention dimension (rank).")
    lora_alpha: int = Field(32, gt=0, description="LoRA scaling factor.")
    lora_dropout: float = Field(0.05, ge=0.0, le=1.0, description="LoRA dropout probability.")
    
class JobResponse(BaseModel):
    """Standard response after starting a job."""
    job_id: str
    status: str
    message: str

class JobStatus(BaseModel):
    """Detailed status of a job."""
    status: str
    progress: Optional[dict] = None
    latest_metrics: Optional[dict] = None
    message: Optional[str] = None
    error: Optional[str] = None
    output_path: Optional[str] = None