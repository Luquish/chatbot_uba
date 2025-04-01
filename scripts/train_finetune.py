import os
import logging
from pathlib import Path
from typing import Dict, List
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType
)
from datasets import Dataset
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(
        self,
        model_name: str = os.getenv('BASE_MODEL_NAME', 'mistralai/Mistral-7B-v0.1'),
        output_dir: str = os.getenv('MODEL_PATH', 'models/finetuned_model'),
        finetuning_dir: str = 'data/finetuning',
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Inicializa el entrenador del modelo.
        
        Args:
            model_name (str): Nombre del modelo base (ej: "mistralai/Mistral-7B-v0.1")
            output_dir (str): Directorio para guardar el modelo fine-tuneado
            finetuning_dir (str): Directorio con datos de entrenamiento
            device (str): Dispositivo para entrenamiento
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.finetuning_dir = Path(finetuning_dir)
        self.device = device
        
        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar tokenizer y modelo
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Configuración de bits
        use_8bit = os.getenv('USE_8BIT', 'True').lower() == 'true'
        use_4bit = os.getenv('USE_4BIT', 'False').lower() == 'true'
        
        load_kwargs = {
            "torch_dtype": torch.float16,
            "device_map": "auto"
        }
        
        if use_8bit:
            load_kwargs["load_in_8bit"] = True
        elif use_4bit:
            load_kwargs["load_in_4bit"] = True
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            **load_kwargs
        )
        
        # Configurar LoRA
        self.setup_lora()
        
    def setup_lora(self):
        """Configura y aplica LoRA al modelo."""
        # Obtener parámetros de LoRA desde variables de entorno
        lora_r = int(os.getenv('LORA_R', '16'))
        lora_alpha = int(os.getenv('LORA_ALPHA', '32'))
        lora_dropout = float(os.getenv('LORA_DROPOUT', '0.05'))
        target_modules_str = os.getenv('TARGET_MODULES', 'q_proj,k_proj,v_proj,o_proj')
        target_modules = target_modules_str.split(',')
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, lora_config)
        
    def load_training_data(self) -> Dataset:
        """
        Carga y prepara los datos de entrenamiento con detección automática de columnas.
        
        Returns:
            Dataset: Dataset de Hugging Face
        """
        # Cargar datos de entrenamiento
        data_path = self.finetuning_dir / "training_data.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"No se encontró el archivo {data_path}")
            
        df = pd.read_csv(data_path)
        
        # Detectar automáticamente columnas relevantes
        columns = df.columns.tolist()
        
        # Buscar columnas de preguntas/consultas
        question_columns = [col for col in columns if col.lower() in 
                        ['pregunta', 'question', 'consulta', 'input', 'prompt', 'query']]
        
        # Buscar columnas de respuestas
        answer_columns = [col for col in columns if col.lower() in 
                     ['respuesta', 'answer', 'contestacion', 'output', 'response']]
        
        # Si no encuentra columnas específicas, usar la primera para preguntas y la segunda para respuestas
        if not question_columns and len(columns) >= 1:
            question_columns = [columns[0]]
            logger.warning(f"No se detectaron columnas de preguntas. Usando '{columns[0]}'")
        
        if not answer_columns and len(columns) >= 2:
            answer_columns = [columns[1]]
            logger.warning(f"No se detectaron columnas de respuestas. Usando '{columns[1]}'")
        
        if not question_columns or not answer_columns:
            raise ValueError("No se pudieron detectar columnas válidas en el CSV")
        
        # Usar la primera columna detectada de cada tipo
        question_col = question_columns[0]
        answer_col = answer_columns[0]
        
        logger.info(f"Usando columna '{question_col}' para preguntas y '{answer_col}' para respuestas")
        
        # Crear textos formateados para entrenamiento
        df['text'] = "Pregunta: " + df[question_col] + "\nRespuesta: " + df[answer_col]
        
        # Tokenizar textos
        max_length = int(os.getenv('MAX_LENGTH', '512'))
        
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                padding="max_length",
                truncation=True,
                max_length=max_length
            )
            
        # Crear dataset
        dataset = Dataset.from_pandas(df)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
        
    def train(self):
        """
        Entrena el modelo usando LoRA con parámetros de las variables de entorno.
        """
        # Cargar datos
        train_dataset = self.load_training_data()
        
        # Obtener parámetros de entrenamiento desde variables de entorno
        num_train_epochs = int(os.getenv('NUM_EPOCHS', '3'))
        batch_size = int(os.getenv('BATCH_SIZE', '4'))
        gradient_accumulation_steps = int(os.getenv('GRADIENT_ACCUMULATION_STEPS', '4'))
        learning_rate = float(os.getenv('LEARNING_RATE', '2e-4'))
        fp16 = os.getenv('FP16', 'True').lower() == 'true'
        
        # Configurar argumentos de entrenamiento
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            fp16=fp16,
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="no",
            save_total_limit=2,
            remove_unused_columns=False,
            push_to_hub=False
        )
        
        # Inicializar trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False
            )
        )
        
        # Entrenar modelo
        logger.info("Iniciando entrenamiento...")
        trainer.train()
        
        # Guardar modelo
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        logger.info(f"Modelo guardado en {self.output_dir}")

def main():
    """Función principal para ejecutar el fine-tuning."""
    # Inicializar y entrenar
    trainer = ModelTrainer()
    trainer.train()

if __name__ == "__main__":
    main() 