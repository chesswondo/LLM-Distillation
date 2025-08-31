import logging
import os
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from datasets import load_dataset
import evaluate
import numpy as np
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerState,
    TrainerControl,
    DataCollatorForSeq2Seq,
)
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

@dataclass
class TrainingConfig:
    """Configuration for the student model training pipeline."""
    # Model and tokenizer
    student_model_name: str = "microsoft/Phi-3-mini-4k-instruct"
    
    # Dataset path
    dataset_path: str = "./data/qa_dataset.jsonl"
    
    # Fine-tuning and quantization
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    load_in_4bit: bool = True
    
    # Training hyperparameters
    output_dir: str = "./student_model_distilled"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    logging_steps: int = 5
    eval_steps: int = 10
    save_steps: int = 100
    fp16: bool = True # Use fp16 for training, bf16 is also an option if supported
    
    # Distillation loss configuration
    loss_alpha: float = 0.5 # 0.0 = only KD, 1.0 = only CE

class ProgressCallback(TrainerCallback):
    """A custom callback to report progress to our job status dictionary."""
    def __init__(self, job_status: Dict[str, Any]):
        self.job_status = job_status
        self.job_status["status"] = "RUNNING"

    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Called at the end of each training step."""
        self.job_status["progress"] = {
            "epoch": round(state.epoch, 2),
            "step": state.global_step,
            "total_steps": state.max_steps,
            "percentage": round((state.global_step / state.max_steps) * 100, 2) if state.max_steps > 0 else 0,
        }

    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, logs: Dict[str, float], **kwargs):
        """Called when logs are saved."""
        self.job_status["latest_metrics"] = logs


class DistillationTrainer(Trainer):
    """
    Custom Trainer that implements a combined loss function for knowledge distillation.
    The total loss is a weighted average of Cross-Entropy and KL-Divergence.
    """
    def __init__(self, *args, loss_alpha, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_alpha = loss_alpha
        logging.info(f"DistillationTrainer initialized with loss_alpha = {self.loss_alpha}")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Extract the teacher's top-k logits and indices from the inputs
        teacher_top_logits = inputs.pop("teacher_top_logits", None)
        teacher_top_indices = inputs.pop("teacher_top_indices", None)
        
        # Get student's logits
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # Calculate Hard Loss (Cross-Entropy)
        labels = inputs.get("labels")
        loss_ce = outputs.loss # The default loss from the model is Cross-Entropy
        
        # Calculate Soft Loss (KL-Divergence)
        loss_kd = 0.0
        if teacher_top_logits is not None and teacher_top_indices is not None and self.loss_alpha < 1.0:
            # We only calculate KD loss on the answer tokens (where labels are not -100)
            active_loss_mask = labels.view(-1) != -100
            
            # Select student logits and labels for active loss positions
            active_student_logits = student_logits.view(-1, student_logits.size(-1))[active_loss_mask]
            active_teacher_logits = teacher_top_logits.view(-1, teacher_top_logits.size(-1))[active_loss_mask]
            active_teacher_indices = teacher_top_indices.view(-1, teacher_top_indices.size(-1))[active_loss_mask]

            # Gather the student's logits corresponding to the teacher's top-k indices
            student_logits_for_kd = torch.gather(active_student_logits, 1, active_teacher_indices)
            
            # Convert logits to log-probabilities (for student) and probabilities (for teacher)
            log_probs_student = F.log_softmax(student_logits_for_kd, dim=-1)
            probs_teacher = F.softmax(active_teacher_logits, dim=-1)

            # Calculate KL-Divergence loss
            loss_kd = F.kl_div(log_probs_student, probs_teacher, reduction='batchmean', log_target=False)

        # Combine the losses
        total_loss = self.loss_alpha * loss_ce + (1.0 - self.loss_alpha) * loss_kd

        if self.state.is_world_process_zero and self.is_in_train:
            self.log({
                "train/loss_total": total_loss.item(),
                "train/loss_ce": loss_ce.item(),
                "train/loss_kd": loss_kd.item() if isinstance(loss_kd, torch.Tensor) else loss_kd,
                "train/lr": self.optimizer.param_groups[0]['lr']
            })
        
        return (total_loss, outputs) if return_outputs else total_loss


def start_training_process(config: TrainingConfig, job_status: Dict[str, Any]):
    """The main training logic, refactored from the original main function."""
    try:
        # Load Model and Tokenizer
        logging.info(f"Loading student model: {config.student_model_name}")
        token = os.getenv("HUGGINGFACE_HUB_TOKEN")
        
        quant_config = None
        if config.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 # Use bfloat16 for computation
            )

        model = AutoModelForCausalLM.from_pretrained(
            config.student_model_name,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
            token=token
        )
        
        tokenizer = AutoTokenizer.from_pretrained(
            config.student_model_name, 
            trust_remote_code=True,
            token=token
        )
        tokenizer.pad_token = tokenizer.eos_token
        model.config.use_cache = False # Important for gradient checkpointing
        
        # Configure LoRA (PEFT)
        if config.use_lora:
            logging.info("Configuring LoRA for efficient fine-tuning.")
            if config.load_in_4bit:
                model = prepare_model_for_kbit_training(model)
            
            peft_config = LoraConfig(
                r=config.lora_r,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=config.lora_target_modules,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, peft_config)
            model.print_trainable_parameters()

        # Load and Preprocess Dataset
        logging.info(f"Loading dataset from: {config.dataset_path}")
        full_dataset = load_dataset('json', data_files=config.dataset_path, split='train')
        split_dataset = full_dataset.train_test_split(test_size=0.1, seed=42)
        train_dataset = split_dataset['train']
        eval_dataset = split_dataset['test']

        rouge_metric = evaluate.load("rouge", cache_dir="C:/Users/chess/tmp/hf_metrics_cache", num_proc=1)
        bertscore_metric = evaluate.load("bertscore", cache_dir="C:/Users/chess/tmp/hf_metrics_cache", num_proc=1)

        def compute_metrics(eval_preds):
            preds, labels = eval_preds

            # If preds are logits: [batch, seq_len, vocab_size] -> take argmax
            if preds.ndim == 3:
                preds = np.argmax(preds, axis=-1)

            # Replace -100 with pad_token_id for decoding
            labels = np.where(labels == -100, tokenizer.pad_token_id, labels)

            # Decode
            decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
            decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

            # ROUGE
            rouge_results = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)

            # BERTScore
            bert_results = bertscore_metric.compute(
                predictions=decoded_preds,
                references=decoded_labels,
                lang="en"
            )

            results = {
                **rouge_results,
                "bertscore_precision": np.mean(bert_results["precision"]),
                "bertscore_recall": np.mean(bert_results["recall"]),
                "bertscore_f1": np.mean(bert_results["f1"]),
            }

            return {k: round(v, 4) for k, v in results.items()}


        def preprocess_function(examples):
            # Build prompts
            prompts = []
            for i in range(len(examples['question'])):
                messages = [
                    {"role": "system", "content": "You are a helpful expert. Answer the question based only on the provided context."},
                    {"role": "user", "content": f"Context:\n{examples['context'][i]}\n\nQuestion: {examples['question'][i]}"},
                ]
                formatted_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                prompts.append(formatted_prompt + examples['answer'][i] + tokenizer.eos_token)

            # Tokenize without padding -> variable length
            model_inputs = tokenizer(
                prompts,
                padding=False,
                truncation=True,
                max_length=4096,
            )

            # Copy labels
            labels = [ids.copy() for ids in model_inputs["input_ids"]]

            # Compute prompt lengths -> where assistant starts
            prompt_lengths = [
                len(tokenizer.encode(p.split('<|assistant|>')[0] + '<|assistant|>'))
                for p in prompts
            ]

            # Mask prompt part with -100
            for label, length in zip(labels, prompt_lengths):
                for i in range(min(length, len(label))):  # safeguard
                    label[i] = -100

            model_inputs["labels"] = labels

            # Return as np.object arrays so HF doesn't try to make a dense tensor here
            return {
                "input_ids": np.array(model_inputs["input_ids"], dtype=object),
                "attention_mask": np.array(model_inputs["attention_mask"], dtype=object),
                "labels": np.array(labels, dtype=object),
            }

        tokenized_train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        tokenized_eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        print(tokenized_train_dataset[0])
        tokenized_train_dataset = tokenized_train_dataset.with_format("torch")
        tokenized_eval_dataset = tokenized_eval_dataset.with_format("torch")

        logging.info(f"Dataset processed. Number of train examples: {len(tokenized_train_dataset)}. Number of eval examples: {len(tokenized_eval_dataset)}")

        # Configure Trainer
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            lr_scheduler_type=config.lr_scheduler_type,
            warmup_ratio=config.warmup_ratio,
            weight_decay=config.weight_decay,
            logging_strategy="steps",
            logging_steps=config.logging_steps,
            eval_strategy="steps",
            eval_steps=config.eval_steps,
            save_strategy="steps",
            save_steps=config.save_steps,
            fp16=config.fp16,
            report_to="wandb",
            gradient_checkpointing=True,
            load_best_model_at_end=True
        )

        # The data collator pads sequences to the max length in each batch
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding="longest",
            return_tensors="pt"
        )

        trainer = DistillationTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_dataset,
            eval_dataset=tokenized_eval_dataset,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            loss_alpha=config.loss_alpha,
            callbacks=[ProgressCallback(job_status=job_status)]
        )

        # Train
        logging.info("Starting training...")
        trainer.train()
        
        # Save Final Model
        final_path = os.path.join(config.output_dir, "final_checkpoint")
        logging.info(f"Saving final model to {final_path}")
        trainer.save_model(final_path)
        tokenizer.save_pretrained(final_path)
        logging.info("Training complete!")

        job_status["status"] = "COMPLETED"
        job_status["message"] = f"Training complete. Model saved to {final_path}"
        job_status["output_path"] = final_path
    
    except Exception as e:
        logging.error(f"Training job failed: {e}", exc_info=True)
        job_status["status"] = "FAILED"
        job_status["error"] = str(e)
