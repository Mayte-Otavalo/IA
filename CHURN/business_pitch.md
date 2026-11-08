# 📡 RetenTel AI: The Prescriptive Customer Retention Platform
## Propuesta de Negocio, Justificación Económica y Guion de Negociación Comercial (Business Pitch)

**Asignatura:** Aprendizaje Automático — Tarea 7  
**Modalidad:** Competencia Comercial Presencial (Rol Vendedor vs. Comprador)  
**Proyecto:** Optimización de Machine Learning Multiobjetivo, Explainable AI y Business Pitch  
**Dataset Base:** Telco Customer Churn (IBM / Kaggle)  
**Modelos:** Regresión Logística, CatBoost Classifier, optimizado con NSGA-III  
**Técnicas Clave:** ADASYN (balanceo adaptativo), Brier Score (calibración), SHAP con correlación de variables, LIME, Análisis Contrafactual y Chatbot RAG Guardrailed  

---

## 1. Identificación del Problema y Oportunidad de Negocio
### 1.1 El Dolor de la Industria (The Pain Point)
En la industria de telecomunicaciones, la tasa de abandono (*churn rate*) promedio oscila entre el **25% y 27% anual**.
- Para un operador con **50,000 suscriptores**, un churn del 26.5% implica perder más de **13,250 clientes al año**.
- **Costo de Adquisición de Clientes (CAC):** Adquirir un nuevo cliente en telecomunicaciones cuesta entre **$300 y $350 USD** (publicidad, comisiones de venta, instalación técnica, router).
- **Costo de Retención Reactiva Tradicional:** Cuando el cliente llama al *Call Center* para solicitar la baja, la empresa ofrece apresuradamente descuentos del 30% o 50% en el abono mensual, erosionando los márgenes sin saber si el cliente realmente se quedaría o si su problema no era el precio.
- **La Fuga Silenciosa:** Más del 60% de los clientes no llaman para quejarse; simplemente dejan vencer su contrato mensual o migran de operador sin previo aviso.

### 1.2 La Oportunidad
Reducir la tasa de churn apenas entre 4 y 6 puntos porcentuales mediante **detección temprana prescriptiva** protege millones de dólares en ingresos recurrentes y evita incurrir en millones de dólares de gasto en CAC.

---

## 2. Definición del Producto: "RetenTel AI"
**RetenTel AI** es una plataforma SaaS B2B (*Software as a Service*) de **Inteligencia Predictiva y Prescriptiva de Retención**.

A diferencia de los modelos analíticos convencionales de *Business Intelligence* que se limitan a generar una lista estática de clientes en un archivo Excel una vez al mes (cuando el cliente ya tomó la decisión de irse), **RetenTel AI** proporciona:
1. **Modelos Especializados (Regresión Logística & CatBoost):** Entrenados tras balanceo adaptativo **ADASYN** para sintetizar patrones de la clase minoritaria de deserción sin sesgos.
2. **Optimización Multiobjetivo Evolutiva (NSGA-III):** Calibración simultánea en 4 dimensiones críticas de producción:
   - **Maximizar F2-Score:** Pondera el Recall al doble que la precisión ($\beta=2$), garantizando capturar al máximo número de desertores reales.
   - **Minimizar Latencia de Inferencia:** Garantiza respuestas en milisegundos para integración en Call Centers y CRMs en vivo.
   - **Minimizar Tamaño del Modelo (KB):** Optimiza memoria y despliegue ágil en microservicios cloud.
   - **Maximizar ROC-AUC:** Asegura capacidad de discriminación global entre clases.
3. **Calibración Probabilística con Brier Score:** Supervisa el error cuadrático medio de las probabilidades, garantizando que un 80% de riesgo signifique exactamente 8 de cada 10 casos en la realidad.
4. **Explainable AI Operativa (XAI con SHAP y LIME):** Identifica la causa raíz del riesgo e incluye un **Índice de Correlación** entre las variables más determinantes y su impacto marginal en la fuga.
5. **Motor Contrafactual Prescriptivo (Counterfactual What-If):** Calcula las modificaciones mínimas y accionables (contrato, soporte técnico, método de pago, tarifa) para transformar a un cliente de riesgo en cliente retenido seguro.
6. **Asistente Conversacional Guardrailed (Chatbot RAG Documental):** Permite a directores y gestores comerciales consultar métricas, ROI y perfiles con la certeza absoluta de que solo responde lo documentado en la plataforma, bloqueando cualquier consulta fuera de tema.

---

## 3. Cliente Objetivo (Ideal Customer Profile - ICP)
- **Comprador Económico Principal:** 
  - Vicepresidente / Director de *Customer Success* y Retención.
  - *Chief Commercial Officer* (CCO) / *Chief Marketing Officer* (CMO).
- **Comprador Técnico Revisor:** 
  - *Chief Technology Officer* (CTO) / *Head of Data Science*.
- **Tipo de Empresas:**
  - Operadores de telecomunicaciones (móvil, fibra óptica, televisión de pago).
  - Proveedores de Servicios de Internet (ISPs) regionales y nacionales con bases de 20,000 a 500,000 clientes.

---

## 4. Propuesta de Valor Única (Value Proposition)

> *"RetenTel AI transforma el departamento de retención de un centro de costos reactivo que quema margen en descuentos a ciegas, en un motor de protección de ingresos prescriptivo, logrando salvar más de 4,000 clientes al año y generando un ROI auditado superior al 400%."*

### Diferenciales frente a la Competencia:
| Enfoque Tradicional / Competidores | RetenTel AI |
| :--- | :--- |
| **Modelos de Caja Negra:** El agente solo ve 'Cliente en Riesgo: 85%', sin saber por qué. | **XAI Incorporado (SHAP + LIME):** El agente ve la causa raíz en lenguaje natural y la correlación directa con variables determinantes. |
| **Descuentos Indiscriminados:** Rebajas de precio a clientes cuya queja es la calidad técnica. | **Matriz Contrafactual Prescriptiva:** Simulación 'what-if' para revertir el abandono con el mínimo incentivo necesario. |
| **Modelos Descalibrados / Sesgados:** Modelos estándar ignoran la clase minoritaria o fallan en calibración. | **ADASYN + Brier Score:** Muestreo adaptativo de frontera y calibración probabilística auditada. |
| **Monobjetivo Tradicional:** Optimización aislada que satura servidores o descuida el Recall. | **NSGA-III Multiobjetivo:** Frente de Pareto optimizado en F2-Score, latencia, tamaño de modelo y ROC-AUC. |
| **Sistemas Aislados:** Consultas lentas mediante tickets manuales a Data Science. | **Chatbot Documental Guardrailed:** Interfaz conversacional segura que solo responde con información validada del proyecto. |

---

## 5. Modelo Económico y Análisis Financiero de ROI

### 5.1 Parámetros del Escenario Base (Operador de 50,000 Clientes)
- **Base Total de Suscriptores:** 50,000
- **Tasa de Abandono Anual (Churn Histórico):** 26.5% $\rightarrow$ **13,250 clientes perdidos/año**.
- **Ingreso Promedio por Usuario (ARPU):** $65.00 USD/mes $\rightarrow$ **$780.00 USD/año por cliente**.
- **Costo de Re-Adquisición (CAC):** $320.00 USD.
- **Rendimiento de RetenTel AI (CatBoost Optimizado NSGA-III con ADASYN):**
  - **F2-Score:** Optimizado para maximizar captura de fugas reales.
  - **Recall:** ~79% - 81% de los clientes que fugan son detectados preventivamente $\rightarrow$ **~10,600 clientes en riesgo identificados**.
  - **Precisión:** ~61% - 63%.
  - **Brier Score:** < 0.16 (alta confiabilidad probabilística).
  - **Efectividad de la Oferta Prescriptiva Contrafactual:** 42% de éxito de retención.
  - **Costo promedio de la acción prescriptiva:** $35.00 USD por cliente contactado.

### 5.2 Balance Financiero Anual
1. **Clientes Efectivamente Rescatados:**
   $$10,600 \times 42\% = \mathbf{4,452 \text{ clientes salvados al a\~no}}$$
2. **Ingresos Anuales Protegidos:**
   $$4,452 \times \$780.00 = \mathbf{\$3,472,560 \text{ USD}}$$
3. **Ahorro Evitado en Adquisición de Sustitutos (CAC):**
   $$4,452 \times \$320.00 = \mathbf{\$1,424,640 \text{ USD}}$$
4. **Costo Operativo de Campañas de Retención Prescriptiva:**
   $$\approx 16,500 \text{ clientes contactados} \times \$35.00 = \$577,500 \text{ USD}$$
5. **Costo de Licenciamiento SaaS de RetenTel AI:**
   $$\mathbf{\$60,000 \text{ USD / a\~no}}$$
6. **Inversión Total:**
   $$\$577,500 + \$60,000 = \$637,500 \text{ USD}$$
7. **Beneficio Neto para la Empresa:**
   $$\$3,472,560 - \$637,500 = \mathbf{+\$2,835,060 \text{ USD}}$$
8. **Retorno sobre la Inversión (ROI):**
   $$\text{ROI} = \left(\frac{\$2,835,060}{\$637,500}\right) \times 100\% = \mathbf{444.7\%}$$

> **En síntesis:** Por cada $1 invertido en RetenTel AI, la telco genera **$5.45 USD** en flujo de caja positivo neto.

---

## 6. Precios y Monetización (Pricing Strategy)
- **Plan Enterprise SaaS:**
  - Tarifa fija anual de **$60,000 USD** (incluye API de baja latencia calibrada con NSGA-III, preprocesador ADASYN, interfaz Streamlit, Chatbot RAG integrado y soporte 24/7).
  - **Success Fee Opcional:** 5% sobre el valor del primer año de clientes rescatados certificados.
- **Servicio Complementario de Despliegue e Integración CRM:**
  - Implementación inicial y conexión con CRM/Billing: **$15,000 USD** (bonificada al 100% en contratos de 3 años).

---

## 7. Guion y Estrategia de Negociación Comercial (Roleplay Presencial)

### 7.1 Argumentario Central del Vendedor
1. *"Estimado Comité Directivo: Cada mes que su empresa opera con modelos tradicionales o BI estático, más de 1,100 clientes cruzan la puerta sin que nadie sepa la razón real. Reemplazar a cada cliente cuesta $320."*
2. *"Con RetenTel AI, nuestro algoritmo NSGA-III optimiza cuatro frentes clave: no solo la capacidad de captura (F2-score y ROC-AUC), sino también tiempos de respuesta en milisegundos y mínimo consumo de recursos computacionales."*
3. *"El balanceo ADASYN garantiza que aprendemos de los desertores más complejos, y el Brier Score certifica que nuestras alertas de riesgo son matemáticamente confiables."*
4. *"Finalmente, con el motor contrafactual y las explicaciones SHAP, su equipo de ventas sabrá exactamente qué ofrecer: si cambiar el contrato, subsidiar un router o activar soporte técnico, evitando quemar margen en rebajas inútiles."*

### 7.2 Manejo de Objeciones del Comprador
- **Objeción 1:** *"Ya tenemos un equipo interno de Business Intelligence que hace reportes de churn."*  
  - *Respuesta:* *"BI genera reportes forenses: le dice quién se fue el mes pasado. RetenTel AI es predictivo y prescriptivo en tiempo real: detecta la propensión antes de la baja y entrega la solución contrafactual para retenerlo."*
- **Objeción 2:** *"¿Por qué optimizar con NSGA-III en lugar de una métrica única como Accuracy?"*  
  - *Respuesta:* *"En telecomunicaciones, Accuracy es un espejismo: predecir que nadie se va da 73% de precisión pero quiebra la empresa. Con NSGA-III equilibramos F2-Score (Recall priorizado), latencia en microservicios, tamaño compacto y ROC-AUC, garantizando viabilidad técnica y rentabilidad financiera."*
- **Objeción 3:** *"¿Cómo aseguramos que nuestros directivos usen la herramienta sin inventar datos?"*  
  - *Respuesta:* *"Incluimos un Asistente IA Conversacional con Guardrail Documental estricto: responde únicamente con datos validados de las auditorías técnicas y financieras, bloqueando cualquier consulta fuera de ámbito."*
