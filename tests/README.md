# Tests del Chatbot UBA

Suite completa de tests modulares para validación pre-producción del Chatbot UBA.

## 📁 Estructura

```
tests/
├── __init__.py                 # Paquete Python
├── base_test.py               # Clase base para todos los tests
├── test_config.py             # Test de configuración
├── test_database.py           # Test de PostgreSQL
├── test_openai.py             # Test de modelos OpenAI
├── test_rag_system.py         # Test del sistema RAG
├── test_chatbot_interaction.py # Test de interacciones
├── test_telegram.py           # Test de Telegram
├── test_vector_services.py    # Test de servicios vectoriales
├── test_google_services.py    # Test de Google APIs
├── test_http_endpoints.py     # Test de endpoints HTTP
├── test_simulation.py         # Test de simulación real
├── run_tests.py               # Script principal
├── run_single_test.py         # Script para tests individuales
└── README.md                  # Este archivo
```

## 🚀 Uso

### Ejecutar todos los tests
```bash
# Desde el directorio raíz
python run_tests.py

# O desde la carpeta tests
cd tests
python run_tests.py
```

### Ejecutar un test específico
```bash
# Desde la carpeta tests
python run_single_test.py config
python run_single_test.py database
python run_single_test.py openai
python run_single_test.py rag
python run_single_test.py interaction
python run_single_test.py telegram
python run_single_test.py vector
python run_single_test.py google
python run_single_test.py http
python run_single_test.py simulation
```

## 📋 Tests Disponibles

| Test | Descripción | Categoría |
|------|-------------|-----------|
| `config` | Validación de configuración completa | Configuración |
| `database` | Conexión y funcionalidad de PostgreSQL | Base de datos |
| `openai` | Modelos OpenAI (embeddings y generación) | IA |
| `rag` | Sistema RAG completo | IA |
| `interaction` | Interacciones usuario-backend | Funcionalidad |
| `telegram` | Integración de Telegram | Comunicación |
| `vector` | Servicios vectoriales avanzados | IA |
| `google` | Servicios de Google (Sheets y Calendar) | APIs |
| `http` | Endpoints HTTP críticos | Servidor |
| `simulation` | Simulación de interacción real | End-to-end |

## 🏗️ Arquitectura

### BaseTest
Clase base que proporciona:
- Configuración común
- Métodos de logging
- Manejo de errores
- Estructura de resultados

### AsyncBaseTest
Clase base para tests asíncronos que extiende BaseTest.

### TestRunner
Ejecutor principal que:
- Ejecuta todos los tests
- Genera reporte de resultados
- Calcula puntuación de readiness
- Ejecuta simulación si está listo

## 📊 Resultados

Cada test retorna un diccionario con:
```python
{
    "name": "NombreDelTest",
    "passed": True/False,
    "error_message": "Mensaje de error si falla",
    "details": {}  # Detalles adicionales
}
```

## 🎯 Puntuación de Readiness

El sistema calcula una puntuación basada en:
- **Configuración**: 100% si todas las variables están configuradas
- **Tests funcionales**: Porcentaje de tests que pasan
- **Puntuación general**: Promedio de configuración y funcionales

### Niveles
- **≥90%**: Sistema listo para producción
- **70-89%**: Sistema parcialmente listo
- **<70%**: Sistema requiere correcciones

## 🔧 Desarrollo

### Agregar un nuevo test
1. Crear archivo `test_nuevo.py`
2. Heredar de `BaseTest` o `AsyncBaseTest`
3. Implementar `_run_test_logic()`
4. Agregar a `TEST_MAP` en `run_single_test.py`
5. Agregar a la lista en `run_tests.py`

### Ejemplo de test
```python
from tests.base_test import BaseTest

class TestNuevo(BaseTest):
    def get_test_description(self) -> str:
        return "Descripción del test"
    
    def get_test_category(self) -> str:
        return "categoria"
    
    def _run_test_logic(self) -> bool:
        # Lógica del test
        return True
```

## 🐛 Debugging

Para debuggear un test específico:
```bash
python -m pdb tests/run_single_test.py config
```

Para ver logs detallados:
```bash
export LOG_LEVEL=DEBUG
python tests/run_single_test.py config
``` 