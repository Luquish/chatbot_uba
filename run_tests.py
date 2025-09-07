#!/usr/bin/env python3
"""
Script de conveniencia para ejecutar tests desde el directorio raíz.
Redirige a la suite de tests modular.
"""

import sys
import os
from pathlib import Path

# Añadir el directorio actual y la carpeta de tests al path
ROOT = Path(__file__).parent
TESTS_DIR = ROOT / 'tests'
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(TESTS_DIR))

# Importar y ejecutar el runner de tests
from tests.run_tests import main
import asyncio

if __name__ == "__main__":
    asyncio.run(main()) 