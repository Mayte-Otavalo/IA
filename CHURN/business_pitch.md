# 📡 RetenTel AI: The Prescriptive Customer Retention Platform
## Propuesta de Negocio, Justificación Económica y Guion de Negociación Comercial (Business Pitch)

**Asignatura:** Aprendizaje Automático — Tarea 7  
**Modalidad:** Competencia Comercial Presencial (Rol Vendedor vs. Comprador)  
**Proyecto:** Optimización de Machine Learning, Explainable AI y Business Pitch  
**Dataset Base:** Telco Customer Churn (IBM / Kaggle)  

---

## 1. Identificación del Problema y Oportunidad de Negocio
### 1.1 El Dolor de la Industria (The Pain Point)
En la industria de telecomunicaciones, la tasa de abandono (*churn rate*) promedio oscila entre el **25% y 27% anual**.
- Para un operador mediano con **50,000 suscriptores**, un churn del 26.5% implica perder más de **13,250 clientes al año**.
- **Costo de Adquisición de Clientes (CAC):** Adquirir un nuevo cliente en telecomunicaciones cuesta entre **$300 y $350 USD** (publicidad, comisiones de venta, instalación técnica, router).
- **Costo de Retención Reactiva Tradicional:** Cuando el cliente llama al *Call Center* para solicitar la baja, la empresa ofrece apresuradamente descuentos del 30% o 50% en el abono mensual, erosionando los márgenes sin saber si el cliente realmente se quedaría o si su problema no era el precio.
- **La Fuga Silenciosa:** Más del 60% de los clientes no llaman para quejarse; simplemente dejan vencer su contrato mensual o migran de operador sin previo aviso.

### 1.2 La Oportunidad
Reducir la tasa de churn apenas entre 4 y 6 puntos porcentuales mediante **detección temprana prescriptiva** protege millones de dólares en ingresos recurrentes y evita incurrir en millones de dólares de gasto en CAC.

---

## 2. Definición del Producto: "RetenTel AI"
**RetenTel AI** es una plataforma SaaS B2B (*Software as a Service*) de **Inteligencia Predictiva y Prescriptiva de Retención**.

A diferencia de los modelos analíticos convencionales de *Business Intelligence* que se limitan a generar una lista estática de clientes en un archivo Excel una vez al mes (cuando el cliente ya tomó la decisión de irse), **RetenTel AI** proporciona:
1. **Detección Temprana en Streaming:** Motor de scoring en tiempo real (calibrado en el Frente de Pareto con latencias de inferencia de milisegundos) que evalúa el riesgo de fuga tras cada interacción, ticket de soporte o facturación.
2. **Explainable AI Operativa (XAI con SHAP y LIME):** No es una caja negra. Acompaña cada alerta de riesgo con la causa raíz exacta (*¿Por qué este cliente quiere marcharse?*).
3. **Motor Prescriptivo de Ofertas (Actionable Insights):** Asigna automáticamente la intervención comercial óptima que ataca la causa raíz identificada por SHAP con el menor costo posible para el operador.

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
| **Modelos de Caja Negra:** El agente de call center solo ve 'Cliente en Riesgo: 85%', sin saber por qué. | **XAI Incorporado:** El agente ve la causa raíz en lenguaje natural: 'Riesgo inducido por Contrato Mensual y falta de Soporte Técnico'. |
| **Descuentos Indiscriminados:** Se otorgan rebajas de precio a clientes cuya insatisfacción es la calidad de fibra óptica. | **Matriz Prescriptiva:** Se otorga la acción exacta (ej. soporte VIP prioritario por $15 en vez de descuento de $100). |
| **Modelos Descalibrados / Falsos Negativos:** Modelos optimizados solo por Accuracy ignoran al 50% de los clientes que fugan. | **Optimización Multiobjetivo (NSGA-II):** Modelo calibrado para maximizar Recall (78.6%) y mantener latencia ultra-baja. |

---

## 5. Modelo Económico y Análisis Financiero de ROI

### 5.1 Parámetros del Escenario Base (Operador de 50,000 Clientes)
- **Base Total de Suscriptores:** 50,000
- **Tasa de Abandono Anual (Churn Histórico):** 26.5% $\rightarrow$ **13,250 clientes perdidos/año**.
- **Ingreso Promedio por Usuario (ARPU):** $65.00 USD/mes $\rightarrow$ **$780.00 USD/año por cliente**.
- **Costo de Re-Adquisición (CAC):** $320.00 USD.
- **Rendimiento de RetenTel AI (Modelo Campeón):**
  - **Recall:** 78.6% de los clientes que fugan son detectados preventivamente $\rightarrow$ **10,414 clientes en riesgo identificados**.
  - **Precisión:** 63.0% (permite focalizar la campaña sin saturar con falsos positivos).
  - **Efectividad de la Oferta Prescriptiva:** 40% de éxito de retención (gracias a atacar la causa raíz con XAI vs 15% de ofertas ciegas).
  - **Costo promedio de la acción prescriptiva:** $35.00 USD por cliente contactado.

### 5.2 Balance Financiero Anual
1. **Clientes Efectivamente Rescatados:**
   $$10,414 \times 40\% = \mathbf{4,165 \text{ clientes salvados al a\~no}}$$
2. **Ingresos Anuales Protegidos:**
   $$4,165 \times \$780.00 = \mathbf{\$3,248,700 \text{ USD}}$$
3. **Ahorro Evitado en Adquisición de Sustitutos (CAC):**
   $$4,165 \times \$320.00 = \mathbf{\$1,332,800 \text{ USD}}$$
4. **Costo Operativo de Campañas de Retención:**
   $$\approx 15,300 \text{ clientes contactados} \times \$35.00 = \$535,500 \text{ USD}$$
5. **Costo de Licenciamiento SaaS de RetenTel AI:**
   $$\mathbf{\$60,000 \text{ USD / a\~no}}$$
6. **Inversión Total:**
   $$\$535,500 + \$60,000 = \$595,500 \text{ USD}$$
7. **Beneficio Neto para la Empresa:**
   $$\$3,248,700 - \$595,500 = \mathbf{+\$2,653,200 \text{ USD}}$$
8. **Retorno sobre la Inversión (ROI):**
   $$\text{ROI} = \left(\frac{\$2,653,200}{\$595,500}\right) \times 100\% = \mathbf{445.5\%}$$

> **En síntesis:** Por cada $1 invertido en la plataforma RetenTel AI y sus campañas prescriptivas, la empresa recupera **$5.45 USD** netos.

---

## 6. Precios y Monetización (Pricing Strategy)
- **Plan Enterprise SaaS:**
  - Tarifa fija anual de **$60,000 USD** (incluye despliegue, conexión con CRM/Billing API, soporte 24/7 y actualizaciones de modelo).
  - **Success Fee Compartido (Opcional):** 5% sobre el valor del primer año de los clientes efectivamente retenidos.
- **Servicio Complementario de Consultoría de Integración:**
  - Implementación inicial y calibración de pipelines de datos: **$15,000 USD** (pago único, condonable en contratos multianuales de 3 años).

---

## 7. Limitaciones, Riesgos y Mitigación
1. **Riesgo de Fatiga Comercial (Over-contacting):**
   - *Riesgo:* Molestar a clientes con demasiadas ofertas de retención.
   - *Mitigación:* Filtro de umbral dinámico calibrado en la aplicación Streamlit para contactar únicamente a clientes con probabilidad superior al 60% o en riesgo crítico de LTV.
2. **Confusión de Correlación vs. Causalidad:**
   - *Riesgo:* Asumir que la fibra óptica *causa* el churn.
   - *Mitigación:* La plataforma incorpora protocolos XAI que aíslan los factores intervenibles (soporte técnico, plazo contractual, método de débito) de los atributos inherentes.
3. **Deriva del Modelo (Data Drift):**
   - *Riesgo:* Cambios en las ofertas de la competencia vuelven obsoletos los pesos del modelo.
   - *Mitigación:* Reentrenamiento periódico programado y monitoreo continuo de métricas en validación cruzada.

---

## 8. Guion Estratégico para la Negociación Comercial (Roleplay de Clase)

### 8.1 Discurso del Vendedor (Pitch de 2 Minutos)
> *"Buenos días. Si revisamos los estados financieros de su compañía, hoy existe una fuga invisible que les cuesta más de $3 millones de dólares al año: 26 de cada 100 suscriptores cancelan su servicio. Cuando esos clientes se van, ustedes gastan $320 tratando de reemplazarlos con nuevos suscriptores.*  
>  
> *¿Cómo intentan resolverlo hoy? Sus comerciales llaman a ciegas ofreciendo descuentos de tarifa a personas que querían soporte técnico o que simplemente no sabían cómo domiciliar su pago automático. Eso destruye su margen operativo.*  
>  
> *Les presento **RetenTel AI**. Una plataforma inteligente desarrollada con Random Forest optimizado mediante TPE Bayesiano y calibrado con algoritmos multiobjetivo NSGA-II. Nuestro sistema detecta el 78.6% de los clientes en riesgo con una latencia de inferencia de 12 milisegundos. Pero lo más importante: gracias a nuestra tecnología de IA Explicable (SHAP y LIME), no solo decimos 'quién se va', sino 'por qué' y 'con qué oferta salvarlo'.*  
>  
> *Con una inversión anual de $60,000, su empresa salvará más de 4,000 clientes al año, generando más de $2.6 millones de dólares en beneficio neto auditado: un retorno de inversión del 445%. Permítanme mostrarles en nuestra aplicación Streamlit en vivo cómo un cliente de alto riesgo puede ser diagnosticado y rescatado en menos de 10 segundos."*

### 8.2 Matriz de Manejo de Objeciones (Comprador vs. Vendedor)

| Posible Pregunta / Objeción del Comprador | Respuesta Contundente del Vendedor |
| :--- | :--- |
| **"Ya tenemos un equipo interno de Data Science y Business Intelligence. ¿Por qué contratarlos a ustedes?"** | *"Su equipo interno es excelente, pero la mayoría de los modelos internos se quedan en un Jupyter Notebook o en reportes mensuales batch. RetenTel AI es una plataforma prescriptiva en tiempo real con Frente de Pareto optimizado para responder en milisegundos directamente en la pantalla de su agente de call center, con explicabilidad SHAP para no técnicos y una matriz de ROI automatizada lista para operar hoy mismo, ahorrándoles 9 meses de desarrollo."* |
| **"¿Por qué elegiste Random Forest y no una Red Neuronal Profunda o CatBoost?"** | *"En nuestro benchmark técnico riguroso comparamos Logistic Regression, Random Forest y Gradient Boosting. Random Forest con ponderación balanceada y optimización TPE Bayesiana logró el mejor equilibrio de Recall (78.6%) y F1-score (0.6388). Además, Random Forest es nativamente compatible con TreeSHAP, lo que nos permite calcular explicaciones matemáticas exactas basadas en la Teoría de Juegos de Shapley sin aproximaciones heurísticas inestables."* |
| **"60,000 dólares al año suena costoso para un software."** | *"Con un ARPU de $65 al mes, cada cliente salvado representa $780 anuales para su compañía. Esto significa que con solo salvar **77 clientes en todo el año** (menos de 7 clientes al mes), la licencia anual de $60,000 está completamente pagada. Todo cliente adicional a partir del cliente 78 es ganancia neta pura para su empresa. Estamos proyectando salvar más de 4,000 clientes."* |
| **"¿Qué pasa si el modelo tiene muchos falsos positivos y gastamos dinero en clientes que no se iban a ir?"** | *"Justamente por eso implementamos la Optimización Multiobjetivo NSGA-II en la Parte 3 de nuestro desarrollo. En el Frente de Pareto encontramos el balance óptimo: filtramos a los clientes de bajo riesgo y priorizamos únicamente aquellos con alta probabilidad y alto valor de vida (LTV), garantizando una precisión del 63% y evitando el gasto innecesario de presupuesto comercial."* |
