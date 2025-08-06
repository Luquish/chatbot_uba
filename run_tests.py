#!/usr/bin/env python3
"""
Script de conveniencia para ejecutar tests desde el directorio raíz.
Redirige a la suite de tests modular.
"""

import sys
import os
from pathlib import Path

# Añadir el directorio actual al path
sys.path.insert(0, str(Path(__file__).parent))

# Importar y ejecutar el runner de tests
from tests.run_tests import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main()) 