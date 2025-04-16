import os
import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any
import pandas as pd
from tqdm import tqdm
import openai
from dotenv import load_dotenv

# Cargar variables de entorno
load_dotenv()

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OpenAIFineTuner:
    def __init__(
        self,
        model_name: str = os.getenv('PRIMARY_MODEL', 'gpt-4o-mini'),
        output_dir: str = os.getenv('MODEL_PATH', 'models/finetuned_model'),
        finetuning_dir: str = 'data/finetuning',
        api_key: str = os.getenv('OPENAI_API_KEY'),
        timeout: int = int(os.getenv('API_TIMEOUT', '30'))
    ):
        """
        Inicializa el fine-tuner para OpenAI.
        
        Args:
            model_name (str): Nombre del modelo base (ej: "gpt-4o-mini")
            output_dir (str): Directorio para guardar información del modelo fine-tuneado
            finetuning_dir (str): Directorio con datos de entrenamiento
            api_key (str): API key de OpenAI
            timeout (int): Tiempo máximo de espera para llamadas a la API
        """
        self.model_name = model_name
        self.output_dir = Path(output_dir)
        self.finetuning_dir = Path(finetuning_dir)
        self.timeout = timeout
        
        # Cargar parámetros de generación para testing
        self.temperature = float(os.getenv('TEMPERATURE', '0.7'))
        self.top_p = float(os.getenv('TOP_P', '0.9'))
        self.max_output_tokens = int(os.getenv('MAX_OUTPUT_TOKENS', '300'))
        
        # Verificar que existe la API key
        if not api_key:
            raise ValueError("Se requiere OPENAI_API_KEY para usar el fine-tuning de OpenAI")
        
        # Configurar cliente de OpenAI
        openai.api_key = api_key
        
        # Crear directorio de salida
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Crear directorio para datos si no existe
        self.finetuning_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivo para guardar el jsonl
        self.jsonl_file = self.finetuning_dir / "training_data.jsonl"
        
        logger.info(f"Inicializado fine-tuner para modelo {self.model_name} (timeout: {self.timeout}s)")
        
    def prepare_training_data(self, system_prompt: str = None) -> str:
        """
        Prepara los datos de entrenamiento en formato JSONL requerido por OpenAI.
        
        Args:
            system_prompt (str, optional): Prompt de sistema a usar en cada ejemplo
            
        Returns:
            str: Ruta al archivo JSONL generado
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
        
        # Si no hay prompt de sistema, usar uno predeterminado
        if not system_prompt:
            system_prompt = "Eres DrCecim, asistente virtual especializado de la Facultad de Medicina UBA. Tu tarea es proporcionar respuestas breves, precisas y útiles."
        
        # Crear archivo JSONL
        with open(self.jsonl_file, 'w', encoding='utf-8') as f:
            for _, row in tqdm(df.iterrows(), total=len(df), desc="Preparando datos"):
                # Crear formato de conversación para OpenAI
                conversation = {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": str(row[question_col])},
                        {"role": "assistant", "content": str(row[answer_col])}
                    ]
                }
                f.write(json.dumps(conversation, ensure_ascii=False) + '\n')
        
        logger.info(f"Datos de entrenamiento convertidos a formato JSONL: {self.jsonl_file}")
        return str(self.jsonl_file)
    
    def upload_file(self, file_path: str) -> str:
        """
        Sube un archivo a OpenAI para su uso en fine-tuning.
        
        Args:
            file_path (str): Ruta al archivo JSONL
            
        Returns:
            str: ID del archivo subido
        """
        try:
            with open(file_path, 'rb') as f:
                response = openai.files.create(
                    file=f,
                    purpose="fine-tune",
                    timeout=self.timeout
                )
            file_id = response.id
            logger.info(f"Archivo subido exitosamente. ID: {file_id}")
            return file_id
        except Exception as e:
            logger.error(f"Error al subir archivo: {str(e)}")
            raise
    
    def create_fine_tuning_job(self, file_id: str, validation_file_id: Optional[str] = None) -> str:
        """
        Crea un trabajo de fine-tuning en OpenAI.
        
        Args:
            file_id (str): ID del archivo de entrenamiento
            validation_file_id (str, optional): ID del archivo de validación
            
        Returns:
            str: ID del trabajo de fine-tuning
        """
        try:
            # Parámetros básicos para el fine-tuning
            job_params = {
                "model": self.model_name,
                "training_file": file_id,
                "hyperparameters": {
                    "n_epochs": int(os.getenv('NUM_EPOCHS', '3'))
                },
                "timeout": self.timeout
            }
            
            # Agregar archivo de validación si se proporciona
            if validation_file_id:
                job_params["validation_file"] = validation_file_id
            
            # Crear el trabajo de fine-tuning
            response = openai.fine_tuning.jobs.create(**job_params)
            
            job_id = response.id
            logger.info(f"Trabajo de fine-tuning creado. ID: {job_id}")
            
            # Guardar información del trabajo junto con los parámetros
            job_info = {
                "job_id": job_id,
                "model": self.model_name,
                "training_file": file_id,
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "hyperparameters": {
                    "n_epochs": int(os.getenv('NUM_EPOCHS', '3'))
                },
                "generation_params": {
                    "temperature": self.temperature,
                    "top_p": self.top_p,
                    "max_tokens": self.max_output_tokens
                }
            }
            
            job_info_path = self.output_dir / "job_info.json"
            with open(job_info_path, 'w') as f:
                json.dump(job_info, f, indent=2)
            
            return job_id
        except Exception as e:
            logger.error(f"Error al crear trabajo de fine-tuning: {str(e)}")
            raise
    
    def monitor_fine_tuning_job(self, job_id: str, poll_interval: int = 60) -> Dict[str, Any]:
        """
        Monitorea el progreso de un trabajo de fine-tuning.
        
        Args:
            job_id (str): ID del trabajo de fine-tuning
            poll_interval (int): Intervalo en segundos para consultar el estado
            
        Returns:
            Dict[str, Any]: Información final del trabajo
        """
        logger.info(f"Iniciando monitoreo del trabajo {job_id}")
        
        while True:
            try:
                job_info = openai.fine_tuning.jobs.retrieve(job_id, timeout=self.timeout)
                status = job_info.status
                
                logger.info(f"Estado del trabajo: {status}")
                
                if status == "succeeded":
                    logger.info("Fine-tuning completado exitosamente!")
                    
                    # Guardar información final del trabajo
                    final_info = {
                        "job_id": job_id,
                        "model": job_info.fine_tuned_model,
                        "training_file": job_info.training_file,
                        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "status": "completed",
                        "recommended_params": {
                            "temperature": self.temperature,
                            "top_p": self.top_p,
                            "max_tokens": self.max_output_tokens
                        }
                    }
                    
                    final_info_path = self.output_dir / "final_model_info.json"
                    with open(final_info_path, 'w') as f:
                        json.dump(final_info, f, indent=2)
                    
                    # Crear un archivo de configuración para el modelo
                    config_path = self.output_dir / f"{job_info.fine_tuned_model.replace(':', '_')}_config.json"
                    with open(config_path, 'w') as f:
                        config = {
                            "model_id": job_info.fine_tuned_model,
                            "base_model": self.model_name,
                            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                            "params": {
                                "temperature": self.temperature,
                                "top_p": self.top_p,
                                "max_tokens": self.max_output_tokens
                            }
                        }
                        json.dump(config, f, indent=2)
                    
                    return final_info
                
                elif status == "failed":
                    logger.error(f"El trabajo de fine-tuning falló: {job_info.error}")
                    raise RuntimeError(f"Fine-tuning falló: {job_info.error}")
                
                else:  # status in ["pending", "running"]
                    # Si hay métricas disponibles, mostrarlas
                    if hasattr(job_info, 'result_files') and job_info.result_files:
                        result_file_id = job_info.result_files[0]
                        content = openai.files.content(result_file_id, timeout=self.timeout)
                        logger.info(f"Métricas actuales: {content}")
                    
                    time.sleep(poll_interval)
            
            except Exception as e:
                logger.error(f"Error al monitorear el trabajo: {str(e)}")
                time.sleep(poll_interval)
    
    def test_fine_tuned_model(self, model_id: str, test_queries: List[str]) -> Dict[str, Any]:
        """
        Prueba el modelo fine-tuned con algunas consultas.
        
        Args:
            model_id (str): ID del modelo fine-tuned
            test_queries (List[str]): Lista de consultas para probar
            
        Returns:
            Dict[str, Any]: Resultados de las pruebas
        """
        logger.info(f"Probando modelo fine-tuned {model_id} con {len(test_queries)} consultas")
        
        results = []
        
        for query in test_queries:
            try:
                response = openai.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": "Eres DrCecim, asistente virtual especializado de la Facultad de Medicina UBA."},
                        {"role": "user", "content": query}
                    ],
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_output_tokens,
                    timeout=self.timeout
                )
                
                answer = response.choices[0].message.content.strip()
                results.append({"query": query, "response": answer})
                logger.info(f"Consulta: {query}")
                logger.info(f"Respuesta: {answer}")
                
            except Exception as e:
                logger.error(f"Error al probar el modelo con la consulta '{query}': {str(e)}")
                results.append({"query": query, "error": str(e)})
        
        # Guardar resultados
        test_results_path = self.output_dir / f"{model_id.replace(':', '_')}_test_results.json"
        with open(test_results_path, 'w') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
            
        logger.info(f"Resultados de prueba guardados en {test_results_path}")
        return results
    
    def run_fine_tuning(self, system_prompt: str = None, test_queries: List[str] = None) -> Dict[str, Any]:
        """
        Ejecuta el proceso completo de fine-tuning.
        
        Args:
            system_prompt (str, optional): Prompt de sistema para los ejemplos
            test_queries (List[str], optional): Consultas para probar el modelo
        
        Returns:
            Dict[str, Any]: Información del modelo fine-tuned
        """
        try:
            # 1. Preparar datos
            jsonl_path = self.prepare_training_data(system_prompt)
            
            # 2. Subir archivo
            file_id = self.upload_file(jsonl_path)
            
            # 3. Crear trabajo de fine-tuning
            job_id = self.create_fine_tuning_job(file_id)
            
            # 4. Monitorear progreso
            result = self.monitor_fine_tuning_job(job_id)
            
            # 5. Probar el modelo si hay consultas de prueba
            if test_queries and result and "model" in result:
                self.test_fine_tuned_model(result["model"], test_queries)
            
            logger.info(f"Proceso de fine-tuning completado. Modelo fine-tuned: {result['model']}")
            
            # Guardar el ID del modelo como variable de entorno (opcionalmente)
            logger.info("Para usar este modelo actualiza la variable FINE_TUNED_MODEL en tu archivo .env")
            logger.info(f"FINE_TUNED_MODEL={result['model']}")
            
            return result
        
        except Exception as e:
            logger.error(f"Error en el proceso de fine-tuning: {str(e)}")
            raise

def main():
    """Función principal para ejecutar el fine-tuning."""
    # System prompt personalizado para el chatbot UBA
    system_prompt = """
    Eres DrCecim, asistente virtual especializado de la Facultad de Medicina UBA. 
    Tu tarea es proporcionar respuestas breves, precisas y útiles.
    Debes seguir estas reglas:
    1. Sé muy conciso y directo.
    2. Usa la información de los documentos oficiales primero.
    3. Si hay documentos específicos, cita naturalmente su origen ("Según el reglamento...").
    4. No agregues información no presente en las fuentes.
    5. Usa formato de viñetas cuando sea útil para mayor claridad.
    6. Si la información está incompleta, sugiere dónde obtener más datos (alumnos@fmed.uba.ar).
    7. No hagas preguntas adicionales.
    """
    
    # Consultas de prueba para el modelo fine-tuned
    test_queries = [
        "¿Cuáles son los requisitos para mantener la regularidad?",
        "Necesito información sobre el régimen disciplinario",
        "¿Cómo puedo presentar una queja formal?",
        "¿Cuántas materias puedo cursar por cuatrimestre?",
        "¿Qué hago si pierdo la regularidad?"
    ]
    
    try:
        # Verificar que existe la API key
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            logger.error("Se requiere OPENAI_API_KEY para usar el fine-tuning de OpenAI")
            return
        
        # Verificar directorio de datos
        data_dir = Path('data/finetuning')
        if not data_dir.exists():
            logger.info(f"Creando directorio para datos de fine-tuning: {data_dir}")
            data_dir.mkdir(parents=True, exist_ok=True)
        
        # Verificar existencia del archivo de entrenamiento
        train_file = data_dir / "training_data.csv"
        if not train_file.exists():
            logger.error(f"No se encontró el archivo de entrenamiento: {train_file}")
            logger.error("Por favor, crea un archivo CSV con columnas 'pregunta' y 'respuesta'")
            return
        
        # Inicializar y ejecutar fine-tuning
        finetuner = OpenAIFineTuner()
        result = finetuner.run_fine_tuning(system_prompt, test_queries)
        
        logger.info(f"Modelo fine-tuned creado: {result['model']}")
        logger.info("Para usar este modelo, agrega la siguiente línea a tu archivo .env:")
        logger.info(f"FINE_TUNED_MODEL={result['model']}")
        
    except Exception as e:
        logger.error(f"Error en proceso de fine-tuning: {str(e)}")

if __name__ == "__main__":
    main() 