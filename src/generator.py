import json
import logging
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

import torch
from pypdf import PdfReader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    PreTrainedTokenizer,
    PreTrainedModel,
)

# Configure logging for clear and informative output
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

@dataclass
class PipelineConfig:
    """Configuration for the data generation pipeline."""
    # File paths
    pdf_path: str
    output_path: str

    # Model configuration
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"
    load_in_4bit: bool = True
    use_bf16: bool = True

    # Text chunking parameters
    chunk_size: int = 1024
    chunk_overlap: int = 128

    # Generation parameters
    questions_per_chunk: int = 1
    max_new_tokens_question: int = 128
    max_new_tokens_answer: int = 512
    temperature_q: float = 0.7
    temperature_a: float = 0.8
    top_p: float = 0.9
    top_k: int = 50
    do_sample: bool = True
    seed: int = 42

    # Distillation settings
    store_logits: bool = True
    logits_top_k: int = 50  # Number of top logits to store if store_logits is True

    # Internal metadata
    meta: Dict[str, Any] = field(default_factory=dict, init=False)

    def __post_init__(self):
        """Populate metadata after initialization."""
        self.meta = {
            "model": self.model_name,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "questions_per_chunk": self.questions_per_chunk,
            "max_new_tokens_question": self.max_new_tokens_question,
            "max_new_tokens_answer": self.max_new_tokens_answer,
            "temperature_q": self.temperature_q,
            "temperature_a": self.temperature_a,
            "top_p": self.top_p,
            "top_k": self.top_k,
        }


class TeacherModel:
    """Encapsulates the teacher LLM, its tokenizer, and generation logic."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

    def load(self):
        """Loads the model and tokenizer based on the configuration."""
        logging.info(f"Loading tokenizer: {self.config.model_name}")
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name, use_fast=True, token=token)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        logging.info(f"Loading model: {self.config.model_name}")
        torch_dtype = torch.float16
        if self.config.use_bf16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
            logging.info("Using bfloat16 compute dtype.")

        quant_config = None
        if self.config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
            logging.info("Enabled 4-bit quantization.")

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            device_map="auto",
            torch_dtype=torch_dtype,
            quantization_config=quant_config,
            token=token
        )
        self.model.eval()
        logging.info("Model loaded successfully.")

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Applies the chat template to a list of messages."""
        if not self.tokenizer:
            raise RuntimeError("Tokenizer is not loaded.")
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def generate(self, prompt: str, max_new_tokens: int, temperature: float, seed: int) -> str:
        """Generates text from a given prompt."""
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer is not loaded.")

        torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.config.do_sample,
                temperature=temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        # Decode generated text, skipping the prompt
        prompt_len = inputs["input_ids"].shape[1]
        gen_ids = output_ids[0, prompt_len:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()

    def _collect_top_k_logits(self, prompt: str, completion: str) -> Dict[str, Any]:
        """
        Computes and returns the top-k logits for the completion tokens.
        This is much more memory-efficient than storing all logits.
        """
        if not self.model or not self.tokenizer:
            raise RuntimeError("Model or tokenizer is not loaded.")

        full_text = prompt + completion
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)
        prompt_len = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model(**inputs, output_logits=True)

        # Get logits for completion tokens (shift by 1 for next-token prediction)
        completion_logits = outputs.logits[0, prompt_len - 1 : -1, :]
        completion_token_ids = inputs["input_ids"][0, prompt_len:]

        # Collect top-k logits and their indices for each token in the completion
        top_k_logits_list = []
        for i in range(completion_logits.shape[0]):
            top_k = torch.topk(completion_logits[i], k=self.config.logits_top_k)
            top_k_logits_list.append({
                "token_id": completion_token_ids[i].item(),
                "top_logits": top_k.values.cpu().tolist(),
                "top_indices": top_k.indices.cpu().tolist(),
            })

        return {
            "logits_top_k": self.config.logits_top_k,
            "tokens": top_k_logits_list
        }


class DatasetGenerator:
    """Orchestrates the process of generating a Q&A dataset from a PDF."""

    def __init__(self, config: PipelineConfig, teacher: TeacherModel):
        self.config = config
        self.teacher = teacher

    @staticmethod
    def _load_pdf_text(pdf_path: str) -> str:
        """Extracts text from all pages of a PDF file."""
        logging.info(f"Reading PDF from: {pdf_path}")
        try:
            reader = PdfReader(pdf_path)
            pages = []
            for i, page in enumerate(reader.pages):
                try:
                    pages.append(page.extract_text() or "")
                except Exception as e:
                    logging.warning(f"Could not extract text from page {i}: {e}")
                    pages.append("")
            return "\n\n".join(pages)
        except Exception as e:
            logging.error(f"Failed to read PDF file {pdf_path}: {e}")
            raise

    @staticmethod
    def _is_useful_chunk(text: str, min_len: int = 200, min_alpha_ratio: float = 0.5) -> bool:
        """Heuristically checks if a text chunk is meaningful."""
        text = text.strip()
        if len(text) < min_len:
            return False
        alpha_ratio = sum(c.isalpha() for c in text) / max(1, len(text))
        if alpha_ratio < min_alpha_ratio:
            return False
        junk_markers = ["table of contents", "bibliography", "references", "index", "copyright"]
        if any(marker in text.lower() for marker in junk_markers):
            return False
        return True

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """Splits text into overlapping chunks based on token count."""
        if not self.teacher.tokenizer:
            raise RuntimeError("Tokenizer is not loaded.")
        
        logging.info("Tokenizing and chunking text...")
        tokens = self.teacher.tokenizer.encode(text, add_special_tokens=False)
        
        chunks = []
        start_idx = 0
        chunk_index = 0
        while start_idx < len(tokens):
            end_idx = min(start_idx + self.config.chunk_size, len(tokens))
            window_ids = tokens[start_idx:end_idx]
            context = self.teacher.tokenizer.decode(window_ids, skip_special_tokens=True)
            
            if self._is_useful_chunk(context):
                chunks.append({
                    "chunk_index": chunk_index,
                    "offset_tokens": start_idx,
                    "context": context,
                })
            
            if end_idx == len(tokens):
                break # Reached the end
            
            start_idx += self.config.chunk_size - self.config.chunk_overlap
            chunk_index += 1
            
        logging.info(f"Created {len(chunks)} useful chunks.")
        return chunks

    def _generate_qna_for_chunk(self, chunk: Dict[str, Any], seed: int) -> List[Dict[str, Any]]:
        """Generates one or more question-answer pairs for a single text chunk."""
        context = chunk["context"]
        records = []

        q_sys_prompt = (
            "You are a curious but knowledgeable student. Read the provided context and generate a single, non-trivial, precise but also not very long question that truly tests deep understanding and can be answered based on the context. "
            "If it is possible, avoid trivia, definitions, or questions answerable by a single sentence taken verbatim. Output only the question, no preface. If you can't generate a valuable question, return exactly: NO_QUESTION"
        )
        a_sys_prompt = (
            "You are a world-class expert and teacher. Provide a clear, comprehensive, accurate answer to the user's question using only the provided context and general knowledge. "
            "If there is an error in the question, or the context does not provide an answer completely, point it out in your response. But if the question is not related to the context at all and you can't provide any valuable answer, return exactly: NO_ANSWER"
        )

        for i in range(self.config.questions_per_chunk):
            # Generate Question
            q_messages = [
                {"role": "system", "content": q_sys_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nGenerate ONE deep question."},
            ]
            q_prompt = self.teacher._apply_chat_template(q_messages)
            question = self.teacher.generate(
                q_prompt,
                self.config.max_new_tokens_question,
                self.config.temperature_q,
                seed + i,
            )

            if "NO_QUESTION" in question:
                continue

            # Generate Answer
            a_messages = [
                {"role": "system", "content": a_sys_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"},
            ]
            a_prompt = self.teacher._apply_chat_template(a_messages)
            answer = self.teacher.generate(
                a_prompt,
                self.config.max_new_tokens_answer,
                self.config.temperature_a,
                seed + i + 1000, # Use a different seed for the answer
            )

            if "NO_ANSWER" in answer:
                continue
            
            record = {
                "chunk_index": chunk["chunk_index"],
                "offset_tokens": chunk["offset_tokens"],
                "question": question,
                "answer": answer,
                "context": context,
            }
            
            # Collect logits
            if self.config.store_logits:
                try:
                    record["teacher_logits"] = self.teacher._collect_top_k_logits(a_prompt, answer)
                except Exception as e:
                    logging.error(f"Failed to collect logits for chunk {chunk['chunk_index']}: {e}")
                    record["teacher_logits_error"] = str(e)
            
            records.append(record)
        return records

    def run(self, job_status: Dict[str, Any]):
        """Executes the full data generation pipeline and reports progress."""
        try:
            job_status["status"] = "LOADING_MODEL"
            self.teacher.load()
            job_status["status"] = "PROCESSING"
            full_text = self._load_pdf_text(self.config.pdf_path)
            chunks = self._chunk_text(full_text)
            
            output_dir = os.path.dirname(self.config.output_path)
            os.makedirs(output_dir, exist_ok=True)
            
            uid = 0
            total_chunks = len(chunks)
            job_status["progress"] = {"current": 0, "total": total_chunks}

            with open(self.config.output_path, "w", encoding="utf-8") as f_out:
                for i, chunk in enumerate(tqdm(chunks, desc="Processing Chunks")):
                    chunk_seed = self.config.seed + chunk["chunk_index"]
                    qna_records = self._generate_qna_for_chunk(chunk, chunk_seed)
                    for record in qna_records:
                        record["id"] = f"gen_{uid}"
                        record["meta"] = self.config.meta
                        f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        uid += 1
                    job_status["progress"]["current"] = i + 1

            job_status["status"] = "COMPLETED"
            job_status["message"] = f"Successfully generated {uid} Q&A pairs."
        except Exception as e:
            logging.error(f"Generation job failed: {e}", exc_info=True)
            job_status["status"] = "FAILED"
            job_status["error"] = str(e)
