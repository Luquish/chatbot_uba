# Carpeta Data - Almacenamiento Local

Esta carpeta contiene los datos necesarios para el funcionamiento del chatbot cuando no se puede acceder a Google Cloud Storage.

## Estructura

```
data/
├── embeddings/          # Embeddings locales (fallback para GCS)
│   └── (migrado a PostgreSQL Cloud SQL)
│   ├── metadata.csv     # Metadatos de los chunks
│   └── config.json      # Configuración de embeddings
└── README.md           # Este archivo
```

## ¿Por qué es necesaria?

El sistema tiene una **lógica de fallback** en 3 niveles:

1. **Prioridad 1**: Cargar desde GCS (servicio nuevo de drcecim_upload)
2. **Prioridad 2**: Cargar desde GCS (servicio legacy)
3. **Prioridad 3**: Cargar desde archivos locales (`data/embeddings/`)

## ¿Cuándo se usa?

- **Desarrollo local** sin GCS configurado
- **Testing** local
- **Emergencias** cuando GCS no está disponible

## Mantenimiento

Los archivos en `embeddings/` deben mantenerse sincronizados con los datos en GCS para garantizar que el fallback funcione correctamente. 