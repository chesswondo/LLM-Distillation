# LLM Distillation Pipeline

This test project provides a complete, end-to-end pipeline for creating specialized, lightweight language models using knowledge distillation. It features a FastAPI-based job server, containerized with Docker.

### Features
- PDF Data Processing: Ingests raw text from PDF documents to use as a knowledge base.

- Synthetic Dataset Generation: Uses a powerful "teacher" model (e.g., Llama-3.1-8B-Instruct) to generate high-quality Question/Answer pairs from the source text.

- Knowledge Distillation: Implements a custom Hugging Face trainer to distill the teacher's knowledge into a smaller, faster "student" model (e.g., Phi-3-Mini-4k-Instruct).

- Parameter-Efficient Fine-Tuning (PEFT): Utilizes LoRA and 4-bit quantization to make training feasible on consumer-grade GPUs.

- Asynchronous Job API: Uses FastAPI to manage long-running data generation and training jobs in the background.

- Dockerized Environment.

## Getting Started
#### 1. Clone the Repository
```bash
git clone https://github.com/chesswondo/LLM-Distillation
cd LLM-Distillation
```

#### 2. Hugging Face Authentication
To download gated models like Llama 3.1, the application inside the container needs access to your Hugging Face account.\
Get your token from the Hugging Face website: huggingface.co/settings/tokens.

#### 3. Build the Docker Image
Open a terminal in the project's root directory and run the build command. This will take a significant amount of time on the first run as it downloads the base CUDA image and installs all Python dependencies.

```bash
docker build -t distillation-pipeline .
```

#### 4. Run the Docker Container
Execute the following command to start the service. This command enables GPU access and uses volumes to persist your data and Hugging Face cache.

```bash
docker run -d -p 8080:8000 --gpus all -e HUGGINGFACE_HUB_TOKEN=your_hf_token -v path/to/user/.cache/huggingface:/root/.cache/huggingface -v ./outputs:/app/outputs -v ./uploads:/app/uploads distillation-pipeline
```

## API Workflow and Usage
Once the container is running, you can access the interactive API documentation at http://localhost:8080/docs.