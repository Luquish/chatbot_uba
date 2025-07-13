"""
Clase base abstracta para los modelos de lenguaje.
"""
from typing import Any, Dict


class BaseModel:
    """
    Clase base abstracta para modelos de lenguaje.
    Define la interfaz común para todos los modelos.
    """
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        Genera texto a partir de un prompt.
        
        Args:
            prompt (str): Texto de entrada
            kwargs: Argumentos adicionales para la generación
            
        Returns:
            str: Texto generado
        """
        raise NotImplementedError("Este método debe ser implementado por las subclases") 