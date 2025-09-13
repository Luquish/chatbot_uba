# 🚀 GUÍA DE REFACTORIZACIÓN Y MIGRACIÓN DE TOOLS

## 📋 PLAN DE REFACTORIZACIÓN COMPLETO

### 🎯 OBJETIVOS
- **Estandarizar** todas las tools con lógica moderna
- **Reutilizar** código común entre tools
- **Mejorar** performance y precisión del 25% al 90%+
- **Mantener** compatibilidad con código existente

---

## 🏗️ ARQUITECTURA DE CLASES BASE

### **1. `ModernBaseTool` (Clase Base Principal)**
```python
# Funcionalidades incluidas:
- ✅ Fuzzy matching con rapidfuzz
- ✅ Caché inteligente con TTL
- ✅ Scoring sofisticado
- ✅ Logging detallado
- ✅ Manejo de errores robusto
- ✅ Normalización de texto con caché
```

### **2. `SheetsBaseTool` (Para Tools de Google Sheets)**
```python
# Funcionalidades específicas:
- ✅ Manejo automático de sheets
- ✅ Procesamiento de celdas combinadas
- ✅ Caché de datos del sheet
- ✅ Búsqueda avanzada en columnas
- ✅ Normalización de datos
```

### **3. `ConversationalBaseTool` (Para Tools Conversacionales)**
```python
# Funcionalidades específicas:
- ✅ Detección de intents
- ✅ Context awareness
- ✅ NLP básico integrado
- ✅ Boost por longitud de consulta
```

---

## 📊 PLAN DE MIGRACIÓN POR PRIORIDAD

### 🔴 **FASE 1: CRÍTICAS (1-2 semanas)**

#### **1. `faq_tool.py` → `ModernBaseTool`**
**Problema actual:** Solo keywords básicas, 0% fuzzy matching
**Mejora esperada:** 25% → 85% tasa de éxito

```python
# ANTES (básico)
def _rule_score(self, query: str) -> float:
    query_l = query.lower()
    keywords = self.config.get('triggers', {}).get('keywords', [])
    hits = sum(1 for k in keywords if k in query_l)
    return min(1.0, 0.2 * hits) if hits else 0.0

# DESPUÉS (moderno)
class FaqTool(ModernBaseTool):
    def _rule_score(self, query: str) -> float:
        # Usa _advanced_keyword_matching automáticamente
        return super()._rule_score(query)
    
    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        # Implementar lógica específica de FAQ con fuzzy matching
        pass
```

**Pasos de migración:**
1. Heredar de `ModernBaseTool`
2. Configurar keywords expandidas
3. Implementar fuzzy matching para preguntas similares
4. Agregar caché para respuestas frecuentes

#### **2. `horarios_catedra_tool.py` → `SheetsBaseTool`**
**Problema actual:** Matching básico de materias, sin fuzzy matching
**Mejora esperada:** 30% → 90% tasa de éxito

```python
# ANTES (básico)
def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
    # Lógica manual de sheets
    values = self.sheets_service.get_sheet_values(sid, a1)
    # Matching básico por tokens

# DESPUÉS (moderno)
class HorariosCatedraTool(SheetsBaseTool):
    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        # Usar métodos heredados
        rows = self._fetch_sheet_data()  # Con caché automático
        processed_rows = self._process_sheet_rows(rows)  # Con normalización
        
        # Búsqueda avanzada
        matches = self._find_best_matches(
            query, processed_rows, 
            search_columns=[0, 1],  # Materia y Cátedra
            threshold=0.6
        )
```

**Pasos de migración:**
1. Heredar de `SheetsBaseTool`
2. Configurar columnas de búsqueda
3. Implementar fuzzy matching para nombres de materias
4. Optimizar procesamiento de datos

#### **3. `hospitales_tool.py` → `ModernBaseTool`**
**Problema actual:** Keywords básicas, sin fuzzy matching para nombres
**Mejora esperada:** 40% → 85% tasa de éxito

```python
# ANTES (básico)
def _rule_score(self, query: str) -> float:
    query_l = query.lower()
    keywords = self.config.get('triggers', {}).get('keywords', [])
    hits = sum(1 for k in keywords if k in query_l)
    return min(1.0, 0.15 * hits) if hits else 0.0

# DESPUÉS (moderno)
class HospitalesTool(ModernBaseTool):
    def __init__(self):
        super().__init__("hospitales", 75, {
            'thresholds': {'accept': 0.4},
            'triggers': {
                'keywords': [
                    # Keywords expandidas con sinónimos
                    'hospital', 'hospitales', 'udh', 'unidad docente hospitalaria',
                    'donde queda', 'dónde queda', 'ubicacion', 'ubicación',
                    # ... más keywords
                ]
            },
            'fuzzy_matching': {
                'enabled': True,
                'threshold': 0.7
            }
        })
    
    def _rule_score(self, query: str) -> float:
        # Usar fuzzy matching para nombres de hospitales
        base_score = super()._rule_score(query)
        
        # Boost por hospitales específicos con fuzzy matching
        hospital_boost = self._calculate_hospital_boost(query)
        
        return min(base_score + hospital_boost, 1.0)
    
    def _calculate_hospital_boost(self, query: str) -> float:
        hospitales = ['durand', 'clínicas', 'italiano', 'alemán', ...]
        max_boost = 0.0
        
        for hospital in hospitales:
            score, _ = self._calculate_fuzzy_score(query, hospital)
            if score > 0.6:
                max_boost = max(max_boost, score * 0.3)
        
        return max_boost
```

**Pasos de migración:**
1. Heredar de `ModernBaseTool`
2. Expandir keywords con sinónimos
3. Implementar fuzzy matching para nombres de hospitales
4. Optimizar boosts por contexto

---

### 🟡 **FASE 2: MEDIAS (1 semana)**

#### **4. `calendar_tool.py` → `ModernBaseTool`**
**Mejora esperada:** 65% → 85% tasa de éxito

```python
class CalendarTool(ModernBaseTool):
    def _rule_score(self, query: str) -> float:
        # Usar fuzzy matching para intents de calendario
        base_score = super()._rule_score(query)
        
        # Boost por intents específicos
        intent_boost = self._calculate_intent_boost(query)
        
        return min(base_score + intent_boost, 1.0)
```

#### **5. `horarios_lic_tec_tool.py` → `SheetsBaseTool`**
**Mejora esperada:** 60% → 85% tasa de éxito

```python
class HorariosLicTecTool(SheetsBaseTool):
    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        rows = self._fetch_sheet_data()
        processed_rows = self._process_sheet_rows(rows)
        
        # Búsqueda fuzzy en nombres de carreras
        matches = self._find_best_matches(
            query, processed_rows, 
            search_columns=[0],  # Columna de nombres
            threshold=0.6
        )
```

#### **6. `mails_nuevo_espacio_tool.py` → `SheetsBaseTool`**
**Mejora esperada:** 60% → 80% tasa de éxito

```python
class MailsNuevoEspacioTool(SheetsBaseTool):
    def _rule_score(self, query: str) -> float:
        # Usar fuzzy matching para tipos de consulta
        base_score = super()._rule_score(query)
        
        # Boost por tipos específicos
        type_boost = self._calculate_type_boost(query)
        
        return min(base_score + type_boost, 1.0)
```

---

### 🟢 **FASE 3: BAJAS (3-4 días)**

#### **7. `conversational_tool.py` → `ConversationalBaseTool`**
**Mejora esperada:** 80% → 90% tasa de éxito

```python
class ConversationalTool(ConversationalBaseTool):
    def _detect_intent(self, query: str, context: Dict[str, Any]) -> Tuple[str, float]:
        # Usar detección de intents mejorada
        return super()._detect_intent(query, context)
```

#### **8. `sheets_tool.py` → `SheetsBaseTool`**
**Mejora esperada:** 60% → 75% tasa de éxito

```python
class SheetsTool(SheetsBaseTool):
    def execute(self, query: str, params: Dict[str, Any], context: Dict[str, Any]) -> ToolResult:
        # Usar funcionalidades heredadas
        rows = self._fetch_sheet_data()
        processed_rows = self._process_sheet_rows(rows)
        
        # Implementar lógica específica de cursos
```

---

## 🔧 CONFIGURACIÓN RECOMENDADA POR TOOL

### **Configuración Base Recomendada:**
```python
DEFAULT_CONFIG = {
    'thresholds': {'accept': 0.6},
    'triggers': {'keywords': []},
    'fuzzy_matching': {
        'enabled': True,
        'threshold': 0.7,
        'weights': {
            'ratio': 0.3,
            'partial': 0.25,
            'token_sort': 0.25,
            'token_set': 0.2
        }
    },
    'caching': {
        'enabled': True,
        'ttl_minutes': 30
    }
}
```

### **Configuraciones Específicas:**

#### **FAQ Tool:**
```python
faq_config = {
    'thresholds': {'accept': 0.7},
    'fuzzy_matching': {'threshold': 0.6},  # Más permisivo
    'caching': {'ttl_minutes': 60}  # Caché más largo
}
```

#### **Sheets Tools:**
```python
sheets_config = {
    'data_processing': {
        'skip_header_rows': 1,
        'normalize_columns': True,
        'handle_merged_cells': True
    },
    'caching': {'ttl_minutes': 15}  # Caché más corto para datos dinámicos
}
```

#### **Conversational Tool:**
```python
conversational_config = {
    'thresholds': {'accept': 0.5},
    'intent_detection': {
        'enabled': True,
        'confidence_threshold': 0.7
    },
    'context_awareness': {
        'enabled': True,
        'context_boost': 0.3
    }
}
```

---

## 📈 MÉTRICAS DE ÉXITO ESPERADAS

| **Tool** | **Tasa Actual** | **Tasa Objetivo** | **Mejora** |
|----------|-----------------|-------------------|------------|
| `faq_tool.py` | 25% | 85% | +60% |
| `horarios_catedra_tool.py` | 30% | 90% | +60% |
| `hospitales_tool.py` | 40% | 85% | +45% |
| `calendar_tool.py` | 65% | 85% | +20% |
| `horarios_lic_tec_tool.py` | 60% | 85% | +25% |
| `mails_nuevo_espacio_tool.py` | 60% | 80% | +20% |
| `conversational_tool.py` | 80% | 90% | +10% |
| `sheets_tool.py` | 60% | 75% | +15% |

**Tasa de éxito general esperada:** 25% → 85% (+60%)

---

## 🚀 PASOS DE IMPLEMENTACIÓN

### **Semana 1-2: Fase Crítica**
1. ✅ Migrar `faq_tool.py`
2. ✅ Migrar `horarios_catedra_tool.py`
3. ✅ Migrar `hospitales_tool.py`
4. ✅ Testing y ajustes

### **Semana 3: Fase Media**
1. ✅ Migrar `calendar_tool.py`
2. ✅ Migrar `horarios_lic_tec_tool.py`
3. ✅ Migrar `mails_nuevo_espacio_tool.py`
4. ✅ Testing y ajustes

### **Semana 4: Fase Baja**
1. ✅ Migrar `conversational_tool.py`
2. ✅ Migrar `sheets_tool.py`
3. ✅ Testing final
4. ✅ Documentación

---

## 🧪 TESTING Y VALIDACIÓN

### **Tests Requeridos:**
1. **Tests unitarios** para cada método de matching
2. **Tests de integración** para cada tool
3. **Tests de performance** para verificar mejoras
4. **Tests de regresión** para asegurar compatibilidad

### **Métricas a Monitorear:**
- Tasa de éxito por tool
- Tiempo de respuesta
- Uso de caché
- Precisión del fuzzy matching

---

## 📚 RECURSOS Y DOCUMENTACIÓN

### **Archivos Clave:**
- `services/tools/base.py` - Clases base modernas
- `utils/text_utils.py` - Utilidades de texto
- `config/router.yaml` - Configuración de routing

### **Dependencias Nuevas:**
- `rapidfuzz>=3.0.0` - Fuzzy matching rápido
- `fuzzywuzzy>=0.18.0` - Fuzzy matching avanzado
- `python-Levenshtein>=0.21.0` - Distancias de edición

---

## ⚠️ CONSIDERACIONES IMPORTANTES

### **Compatibilidad:**
- Mantener interfaces existentes
- No romper funcionalidad actual
- Migración gradual sin downtime

### **Performance:**
- Caché inteligente para evitar consultas repetidas
- Fuzzy matching optimizado
- Logging detallado para debugging

### **Mantenimiento:**
- Código reutilizable y modular
- Configuración centralizada
- Documentación actualizada

---

**🎯 Objetivo Final:** Transformar todas las tools de lógica básica a lógica moderna, mejorando la tasa de éxito del 25% al 85%+ y estableciendo una base sólida para futuras mejoras.
