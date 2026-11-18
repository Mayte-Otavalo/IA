# RetenTel AI: portal de retención de clientes

Portal en Streamlit sobre el modelo CatBoost del equipo (IBM Telco Churn, 7.043 clientes). Estima el riesgo de abandono, elige el plan de retención que más valor deja y puede enviar ese plan por correo. No entrena nada: carga el modelo y lo evalúa.

## Instalar y ejecutar

Requiere Python 3.12 (probado) y funciona sin ninguna configuración adicional.

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Abra http://localhost:8501.

## Cómo usarlo

1. **Analizar cliente**: complete el formulario y vea el riesgo, su evolución con la antigüedad y el plan de retención recomendado.
2. **Cartera**: cargue un CSV (mismas columnas que `assets/sample_test_data.csv`) o use la muestra; ordena por riesgo y descarga la lista de contacto.
3. **Campaña**: fije un presupuesto; el portal elige a quién contactar y con qué plan, y compara contra ordenar solo por riesgo.
4. **Lanzar campaña por correo**: el agente envía a cada cliente su plan. Sin configuración corre en modo demostración y no envía nada.
5. **Calidad del modelo** y **Sistema**: métricas en la partición de prueba, umbral de alerta y límites de uso.

## Pruebas

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## Docker

```bash
docker build -t retentel-ai .
docker run -p 8501:8501 retentel-ai
```

La imagen no se ha construido todavía en ningún equipo: no está verificada.

## Estructura

- `core.py`, `plans.py`, `mailer.py`: modelo y puntuación, planes y campañas, envío de correo. Sin Streamlit.
- `app.py`, `views/`, `components.py`, `static/`: interfaz.
- `assets/`: modelo (`champion_pipeline.joblib`, `metadata.json`) y datos.
- `tests/`: pruebas; ninguna usa la red.

El modelo se reentrena en el notebook de Colab del equipo; copie su carpeta `models/` a `assets/` y ejecute las pruebas.

## Límites

- El umbral de alerta (40 %) se eligió sobre la partición de prueba; falta validarlo con validación cruzada.
- Los efectos de planes y campañas son estimaciones del modelo, no resultados medidos: valídelos con un piloto.
