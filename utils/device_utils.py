"""
Utilidades para configuración de dispositivos (GPU/CPU).
"""
import os
import logging
import torch

from config.settings import DEVICE_PREF

logger = logging.getLogger(__name__)


def get_device():
    """
    Configura el dispositivo según preferencias y disponibilidad.
    
    Returns:
        str: Dispositivo seleccionado ("cuda", "mps" o "cpu")
    """
    device_pref = DEVICE_PREF
    
    if device_pref == 'auto':
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_pref  # Usa específicamente cuda, cpu o mps si se especifica
    
    logger.info(f"Usando dispositivo: {device}")
    return device 