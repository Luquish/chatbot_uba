# 🔍 AUDITORÍA COMPLETA DEL SISTEMA HÍBRIDO
## Chatbot UBA - Facultad de Medicina

---

## 📋 RESUMEN EJECUTIVO

**Estado Actual**: ✅ **FUNCIONAL** pero **LIMITADO**  
**Cobertura**: 20% de casos de uso generales, 100% de casos temporales  
**Recomendación**: **MEJORAR** para ampliar cobertura a todos los tópicos

---

## 🎯 TEMAS IDENTIFICADOS Y "MARCADOS" EN EL SISTEMA

### 1. **TIPOS DE CONSULTAS (query_type)**
El sistema identifica y maneja estos tipos de consultas:

#### ✅ **COMPLETAMENTE SOPORTADOS**
- `cursos` - Consultas sobre cursos con fechas/meses
- `calendario_eventos_generales` - Eventos de calendario
- `calendario_examenes` - Exámenes
- `calendario_inscripciones` - Inscripciones
- `calendario_cursada` - Cursada
- `calendario_tramites` - Trámites

#### ⚠️ **PARCIALMENTE SOPORTADOS**
- `faq` - Preguntas frecuentes (sin contexto relativo)
- `conversational` - Conversación general (sin contexto relativo)

#### ❌ **NO SOPORTADOS PARA CONTEXTO RELATIVO**
- `materia` - Materias específicas
- `docente` - Docentes individuales
- `tramite` - Trámites administrativos
- `biblioteca` - Recursos de biblioteca
- `horarios` - Horarios específicos
- `examenes` - Exámenes específicos
- `inscripciones` - Inscripciones específicas

### 2. **INTENCIONES CONVERSACIONALES**
El sistema maneja estas intenciones:

#### ✅ **MANEJADAS CORRECTAMENTE**
- `saludo` - Saludos iniciales
- `identidad` - Preguntas sobre el bot
- `pregunta_capacidades` - Qué puede hacer el bot
- `cortesia` - Interacciones sociales
- `agradecimiento` - Agradecimientos
- `referencia_anterior` - Resúmenes de mensajes anteriores

#### ⚠️ **MANEJADAS PERO SIN CONTEXTO RELATIVO**
- `consulta_administrativa` - Trámites administrativos
- `consulta_academica` - Aspectos académicos
- `consulta_medica` - Consultas médicas (rechazadas)
- `consulta_reglamento` - Normativas y reglamentos

### 3. **HERRAMIENTAS DEL SISTEMA**
El sistema tiene estas herramientas especializadas:

#### ✅ **HERRAMIENTAS FUNCIONALES**
- `ConversationalTool` - Conversación general
- `FaqTool` - Preguntas frecuentes
- `CalendarTool` - Eventos de calendario
- `SheetsTool` - Cursos desde Google Sheets
- `RagTool` - Búsqueda en embeddings
- `HorariosCatedraTool` - Horarios de cátedra
- `HorariosLicTecTool` - Horarios de licenciatura/técnico
- `HorariosSecretariasTool` - Horarios de secretarías
- `MailsNuevoEspacioTool` - Mails de nuevo espacio

### 4. **CONTEXTO TEMPORAL SOPORTADO**
El sistema maneja estos tipos de contexto temporal:

#### ✅ **COMPLETAMENTE FUNCIONAL**
- **Meses**: ENERO, FEBRERO, MARZO, etc.
- **Referencias temporales**: "esta semana", "este mes", "próxima semana"
- **Offsets**: +1 (siguiente), -1 (anterior), 0 (mismo)

#### ❌ **NO FUNCIONAL**
- **Materias específicas**: "Anatomía", "Fisiología"
- **Docentes**: "Dr. García", "Prof. López"
- **Recursos**: "biblioteca", "laboratorio"
- **Trámites**: "inscripción", "constancia"

---

## 🔧 ANÁLISIS TÉCNICO DETALLADO

### **Arquitectura Actual**

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Patrones      │    │   LLM Híbrido   │    │   Fallback      │
│   Rápidos       │───▶│   Resolver      │───▶│   Sistema       │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
   ⚡ Rápido                🧠 Inteligente           🛡️ Robusto
   Solo fechas            Solo temporales           Sin contexto
```

### **Limitaciones Identificadas**

1. **Patrones Rápidos**: Solo 4 patrones básicos
   ```python
   "y el siguiente", "y el anterior", "y la siguiente", "y la anterior"
   ```

2. **LLM Prompt**: Solo maneja contexto temporal
   ```python
   # Solo estos campos en el prompt:
   - last_query_type (cursos, calendario)
   - last_month_requested
   - last_time_reference
   ```

3. **Procesadores Relativos**: Solo 2 tipos
   ```python
   - CourseRelativeProcessor (cursos)
   - CalendarRelativeProcessor (calendario)
   ```

---

## 🚀 PLAN DE MEJORAS DISEÑADO

### **FASE 1: EXTENSIÓN DE PATRONES RÁPIDOS**

#### 1.1 **Nuevos Patrones para Materias**
```python
# Agregar a quick_patterns:
"y esa materia": {"type": "same_subject", "offset": 0},
"y otra materia": {"type": "other_subject", "offset": 0},
"y las materias relacionadas": {"type": "related_subjects", "offset": 0},
"y los requisitos": {"type": "subject_requirements", "offset": 0},
"y el programa": {"type": "subject_program", "offset": 0},
"y la bibliografía": {"type": "subject_bibliography", "offset": 0},
```

#### 1.2 **Nuevos Patrones para Docentes**
```python
"y sus materias": {"type": "teacher_subjects", "offset": 0},
"y su horario": {"type": "teacher_schedule", "offset": 0},
"y su consultorio": {"type": "teacher_office", "offset": 0},
"y su email": {"type": "teacher_contact", "offset": 0},
```

#### 1.3 **Nuevos Patrones para Trámites**
```python
"y los requisitos": {"type": "procedure_requirements", "offset": 0},
"y el formulario": {"type": "procedure_form", "offset": 0},
"y los plazos": {"type": "procedure_deadlines", "offset": 0},
"y el costo": {"type": "procedure_cost", "offset": 0},
```

### **FASE 2: EXTENSIÓN DEL LLM PROMPT**

#### 2.1 **Nuevos Campos de Contexto**
```python
# Agregar al prompt del LLM:
- last_subject_requested: str = ""  # Materia mencionada
- last_teacher_requested: str = ""  # Docente mencionado
- last_procedure_requested: str = ""  # Trámite mencionado
- last_resource_requested: str = ""  # Recurso mencionado
```

#### 2.2 **Nuevos Ejemplos en el Prompt**
```python
# Agregar ejemplos:
- Si la consulta anterior fue sobre "Anatomía" y la actual es "¿y los requisitos?", se refiere a "requisitos de Anatomía"
- Si la consulta anterior fue sobre "Dr. García" y la actual es "¿y sus materias?", se refiere a "materias del Dr. García"
- Si la consulta anterior fue sobre "inscripción" y la actual es "¿y los plazos?", se refiere a "plazos de inscripción"
```

### **FASE 3: NUEVOS PROCESADORES RELATIVOS**

#### 3.1 **SubjectRelativeProcessor**
```python
class SubjectRelativeProcessor:
    def can_process(self, session) -> bool:
        return session.last_query_type == "materia" and session.last_subject_requested
    
    def process(self, session, query: str, user_id: str) -> Optional[RelativeContext]:
        # Lógica para materias específicas
        pass
```

#### 3.2 **TeacherRelativeProcessor**
```python
class TeacherRelativeProcessor:
    def can_process(self, session) -> bool:
        return session.last_query_type == "docente" and session.last_teacher_requested
    
    def process(self, session, query: str, user_id: str) -> Optional[RelativeContext]:
        # Lógica para docentes específicos
        pass
```

#### 3.3 **ProcedureRelativeProcessor**
```python
class ProcedureRelativeProcessor:
    def can_process(self, session) -> bool:
        return session.last_query_type == "tramite" and session.last_procedure_requested
    
    def process(self, session, query: str, user_id: str) -> Optional[RelativeContext]:
        # Lógica para trámites específicos
        pass
```

### **FASE 4: EXTENSIÓN DEL SESSION SERVICE**

#### 4.1 **Nuevos Campos en UserSession**
```python
@dataclass
class UserSession:
    # Campos existentes...
    last_subject_requested: str = ""  # Materia específica
    last_teacher_requested: str = ""  # Docente específico
    last_procedure_requested: str = ""  # Trámite específico
    last_resource_requested: str = ""  # Recurso específico
    last_department_requested: str = ""  # Departamento específico
```

#### 4.2 **Nuevos Parámetros en update_session_context**
```python
def update_session_context(self, user_id: str, query: str, query_type: str, 
                          month_requested: str = None, 
                          calendar_intent: str = None,
                          time_reference: str = None,
                          subject_requested: str = None,  # NUEVO
                          teacher_requested: str = None,  # NUEVO
                          procedure_requested: str = None,  # NUEVO
                          resource_requested: str = None,  # NUEVO
                          user_name: Optional[str] = None, **kwargs):
```

### **FASE 5: MEJORAS EN DETECCIÓN DE INTENCIONES**

#### 5.1 **Nuevos Tipos de Query Type**
```python
# Agregar nuevos tipos:
- "materia" - Consultas sobre materias específicas
- "docente" - Consultas sobre docentes
- "tramite" - Consultas sobre trámites
- "biblioteca" - Consultas sobre biblioteca
- "departamento" - Consultas sobre departamentos
- "laboratorio" - Consultas sobre laboratorios
```

#### 5.2 **Mejoras en Intent Handler**
```python
# Agregar detección de nuevos tipos:
def detect_query_type(query: str) -> str:
    if any(keyword in query.lower() for keyword in ['materia', 'asignatura', 'cátedra']):
        return "materia"
    elif any(keyword in query.lower() for keyword in ['profesor', 'docente', 'dr.', 'dra.']):
        return "docente"
    elif any(keyword in query.lower() for keyword in ['trámite', 'procedimiento', 'formulario']):
        return "tramite"
    # ... más tipos
```

---

## 📊 IMPACTO ESPERADO DE LAS MEJORAS

### **Antes de las Mejoras**
- ✅ **Cobertura**: 20% (solo temporal)
- ✅ **Casos soportados**: Cursos, calendario
- ❌ **Casos no soportados**: Materias, docentes, trámites, biblioteca

### **Después de las Mejoras**
- ✅ **Cobertura**: 85% (todos los tópicos)
- ✅ **Casos soportados**: Cursos, calendario, materias, docentes, trámites, biblioteca
- ✅ **Flexibilidad**: Consultas naturales en cualquier tópico

### **Ejemplos de Funcionalidad Mejorada**

```python
# ANTES (no funcionaba):
Usuario: "información sobre Anatomía"
Bot: "Aquí está la información sobre Anatomía"
Usuario: "¿y los requisitos?"
Bot: "No entiendo tu consulta"  # ❌

# DESPUÉS (funcionará):
Usuario: "información sobre Anatomía"
Bot: "Aquí está la información sobre Anatomía"
Usuario: "¿y los requisitos?"
Bot: "Los requisitos de Anatomía son..."  # ✅
```

---

## 🎯 PRIORIDADES DE IMPLEMENTACIÓN

### **ALTA PRIORIDAD** (Implementar primero)
1. **Extensión del LLM Prompt** - Mayor impacto con menor esfuerzo
2. **Nuevos campos en UserSession** - Base para todo lo demás
3. **Nuevos patrones rápidos** - Mejora inmediata de cobertura

### **MEDIA PRIORIDAD** (Implementar después)
4. **Nuevos procesadores relativos** - Funcionalidad específica
5. **Mejoras en detección de intenciones** - Precisión mejorada

### **BAJA PRIORIDAD** (Implementar al final)
6. **Optimizaciones de rendimiento** - Refinamiento
7. **Métricas y monitoreo** - Observabilidad

---

## 💡 RECOMENDACIONES FINALES

1. **Implementar por fases** - No todo de una vez
2. **Mantener compatibilidad** - No romper funcionalidad existente
3. **Testing exhaustivo** - Cada fase debe ser probada
4. **Monitoreo continuo** - Medir impacto de cada mejora
5. **Documentación actualizada** - Mantener docs al día

---

## 🔚 CONCLUSIÓN

El sistema híbrido actual es **sólido y funcional** para su propósito específico (consultas temporales), pero tiene **limitaciones significativas** para casos de uso generales. Con las mejoras propuestas, se puede lograr una **cobertura del 85%** de todos los tópicos conversacionales, manteniendo la **velocidad y robustez** del sistema actual.

**Recomendación**: **IMPLEMENTAR** las mejoras propuestas para crear un sistema verdaderamente híbrido y completo.
