#!/usr/bin/env python3
"""
Test de modelos OpenAI (embeddings y generación).
Valida que los modelos funcionen correctamente.
"""

import numpy as np
from base_test import BaseTest


class TestOpenAI(BaseTest):
    """Test de modelos OpenAI."""
    
    def get_test_description(self) -> str:
        return "Test completo de modelos OpenAI (embeddings y generación)"
    
    def get_test_category(self) -> str:
        return "openai"
    
    def _run_test_logic(self) -> bool:
        """Validar modelos OpenAI."""
        print("🧠 Probando modelos OpenAI completos...")
        
        try:
            from models.openai_model import OpenAIModel, OpenAIEmbedding
            
            # Test de modelo de embeddings
            try:
                embedding_model = OpenAIEmbedding(
                    model_name=self.config.openai.embedding_model,
                    api_key=self.config.openai.openai_api_key,
                    timeout=self.config.openai.api_timeout
                )
                self.log_success("Modelo de embeddings inicializado")
            except Exception as e:
                self.log_error(f"Error inicializando modelo de embeddings: {str(e)}")
                return False
            
            # Test de modelo de generación
            try:
                generation_model = OpenAIModel(
                    model_name=self.config.openai.primary_model,
                    api_key=self.config.openai.openai_api_key,
                    timeout=self.config.openai.api_timeout,
                    max_output_tokens=self.config.openai.max_output_tokens
                )
                self.log_success("Modelo de generación inicializado")
            except Exception as e:
                self.log_error(f"Error inicializando modelo de generación: {str(e)}")
                return False
            
            # Test de configuración de modelos
            self.log_success("Modelos OpenAI configurados correctamente")
            self.log_info(f"Modelo de embeddings: {self.config.openai.embedding_model}")
            self.log_info(f"Modelo de generación: {self.config.openai.primary_model}")
            self.log_info(f"Timeout configurado: {self.config.openai.api_timeout}s")
            
            # Test de generación de texto
            try:
                test_prompt = "Explica brevemente qué es la Universidad de Buenos Aires"
                
                response = generation_model.generate(
                    prompt=test_prompt,
                    temperature=self.config.openai.temperature
                )
                
                if response and len(response) > 10:
                    self.log_success(f"Respuesta generada: {response[:100]}...")
                else:
                    self.log_error("Respuesta generada vacía o muy corta")
                    return False
                    
            except Exception as e:
                self.log_error(f"Error generando texto: {str(e)}")
                return False
            
            return True
            
        except Exception as e:
            self.log_error(f"Error en test de OpenAI: {str(e)}")
            return False 