import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.patches as patches

pdf_path = r"C:\Users\mayte.otavalo\.gemini\antigravity-cli\brain\14026be1-0d45-43a3-8b1d-2a006dce2260\Guia_Maestra_Defensa_RetenTel_AI.pdf"

def setup_page(title, subtitle, page_num, total_pages=5):
    fig = plt.figure(figsize=(8.27, 11.69), dpi=150) # A4 format
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis('off')
    
    # Header Banner
    rect = patches.Rectangle((0, 0.91), 1, 0.09, facecolor='#0F172A', edgecolor='none')
    ax.add_patch(rect)
    accent_bar = patches.Rectangle((0, 0.905), 1, 0.005, facecolor='#2563EB', edgecolor='none')
    ax.add_patch(accent_bar)
    
    ax.text(0.05, 0.965, "RETENTEL AI | GUÍA MAESTRA DE DEFENSA TÉCNICA Y DE NEGOCIO", 
            fontsize=11, fontweight='bold', color='#60A5FA', family='sans-serif')
    ax.text(0.05, 0.93, title.upper(), 
            fontsize=15, fontweight='bold', color='#FFFFFF', family='sans-serif')
    ax.text(0.05, 0.915, subtitle, 
            fontsize=8.5, color='#94A3B8', family='sans-serif', style='italic')
    
    # Footer
    footer_bar = patches.Rectangle((0, 0.035), 1, 0.002, facecolor='#E2E8F0', edgecolor='none')
    ax.add_patch(footer_bar)
    ax.text(0.05, 0.018, "Concurso Beca / Pasantía Tecnológica — Jurado: Academia & Empresa Pública", 
            fontsize=8, color='#64748B', family='sans-serif')
    ax.text(0.95, 0.018, f"Página {page_num} de {total_pages}", 
            fontsize=8, color='#64748B', family='sans-serif', ha='right')
            
    return fig, ax

def add_card(ax, x, y, w, h, bg_color='#F8FAFC', border_color='#CBD5E1'):
    card = patches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.015",
                                  facecolor=bg_color, edgecolor=border_color, linewidth=1)
    ax.add_patch(card)

with PdfPages(pdf_path) as pdf:
    # ==========================================
    # PÁGINA 1: PREGUNTAS ESTRATÉGICAS Y DE NEGOCIO
    # ==========================================
    fig, ax = setup_page("1. Fundamentación Económica y Estrategia de Validación", 
                         "Costo de Fidelización vs Recaptura (5x) y Validación Cruzada Estratificada 5-Fold", 1)
    
    # Tarjeta 1: Por qué cuesta 5x más fidelizar que recapturar
    add_card(ax, 0.05, 0.51, 0.90, 0.37, bg_color='#FFFFFF', border_color='#93C5FD')
    ax.text(0.07, 0.85, "1. ¿Por qué cuesta hasta 5 veces más recapturar un usuario que fidelizarlo?", 
            fontsize=11.5, fontweight='bold', color='#1E3A8A')
    
    p1_text = (
        "En la economía digital y los servicios públicos existe una asimetría estructural entre el Costo de\n"
        "Adquisición/Recaptura (CAC) y el Costo de Retención de Clientes (CRC):\n\n"
        "• Anatomía del Costo de Adquisición / Recaptura (CAC):\n"
        "  - Inversión en campañas de marketing intensivas y pauta publicitaria agresiva.\n"
        "  - Comisiones comerciales de venta y subsidios de terminales/routers.\n"
        "  - Trámites de onboarding, scoring crediticio y validación de identidad.\n"
        "  - En el sector público: costos de reapertura burocrática de expedientes y pérdida de confianza ciudadana.\n\n"
        "• Anatomía del Costo de Retención (CRC):\n"
        "  - El usuario ya tiene infraestructura instalada, historial crediticio y hábito de uso.\n"
        "  - Retener requiere una intervención marginal proactiva: una bonificación tarifaria, soporte técnico\n"
        "    prioritario o migración contractual (en nuestro modelo: ~$35 por contacto exitoso).\n\n"
        "• Fórmula de Retorno de Inversión (ROI):\n"
        "  Un cliente leal amortiza la inversión inicial y genera margen acumulado (CLV). Al abandonar, la organización\n"
        "  debe gastar nuevamente el 100% del CAC solo para mantener la base de ingresos plana (treadmill effect)."
    )
    ax.text(0.07, 0.54, p1_text, fontsize=8.2, color='#334155', linespacing=1.35)
    
    # Tarjeta 2: Validación Cruzada Estratificada 5-Fold y 80/20
    add_card(ax, 0.05, 0.07, 0.90, 0.41, bg_color='#FFFFFF', border_color='#93C5FD')
    ax.text(0.07, 0.45, "2. ¿Qué significa: 'Sobre el 80% aplicamos Validación Cruzada Estratificada 5-Fold'?", 
            fontsize=11.5, fontweight='bold', color='#1E3A8A')
    
    p2_text = (
        "Es el estándar metodológico de oro para evitar el sobreajuste y garantizar reproducibilidad científica:\n\n"
        "1. Partición 80/20 Holdout Independiente:\n"
        "   - 80% (5,634 filas): Subconjunto exclusivo de desarrollo, ingeniería y optimización.\n"
        "   - 20% (1,409 filas): Conjunto de prueba aislado, sellado bajo 'llave' durante todo el proceso.\n"
        "     Solo se evaluó una única vez al final sobre el modelo campeón para verificar la generalización real.\n\n"
        "2. ¿Por qué 5 Pliegues (5-Fold CV)?:\n"
        "   - Dividir solo en un train/test simple es frágil: una partición afortunada o desafortunada sesga los resultados.\n"
        "   - Con 5-Fold, el 80% se divide en 5 subconjuntos disjuntos. El modelo se entrena 5 veces: en cada iteración\n"
        "     se usan 4 pliegues para entrenar (64% del total) y 1 para validar (16%). La métrica final es la media.\n\n"
        "3. ¿Por qué Estratificada (Stratified)?:\n"
        "   - El dataset tiene desbalance (26.5% cancelaciones vs 73.5% permanencia).\n"
        "   - La estratificación obliga a que cada uno de los 5 pliegues mantenga exactamente la misma proporción\n"
        "     de 26.5% de cancelaciones, evitando pliegues anómalos o subrepresentados.\n\n"
        "4. Prevención Absoluta de Fuga de Datos (Data Leakage):\n"
        "   - Los transformadores, escaladores y el balanceo jamás 'tocan' el pliegue de validación. La fuga de datos\n"
        "     invalida cualquier trabajo de investigación; RetenTel AI garantiza rigor científico absoluto."
    )
    ax.text(0.07, 0.10, p2_text, fontsize=8.2, color='#334155', linespacing=1.35)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ==========================================
    # PÁGINA 2: ADASYN Y MODELOS DE VANGUARDIA
    # ==========================================
    fig, ax = setup_page("2. Balanceo Adaptativo y Paradigmas de Modelado", 
                         "Algoritmo ADASYN, TabPFN (Foundation Model) y CatBoost (Oblivious Trees)", 2)
    
    # Tarjeta 3: ADASYN
    add_card(ax, 0.05, 0.58, 0.90, 0.30, bg_color='#FFFFFF', border_color='#93C5FD')
    ax.text(0.07, 0.85, "3. Algoritmo de Balanceo Adaptativo ADASYN (¿Por qué intra-fold?)", 
            fontsize=11.5, fontweight='bold', color='#1E3A8A')
    p3_text = (
        "ADASYN (Adaptive Synthetic Sampling Approach, He et al., IEEE 2008) supera las limitaciones de SMOTE:\n\n"
        "• Mecánica Adaptativa:\n"
        "  - SMOTE tradicional genera muestras sintéticas uniformes entre todos los vecinos minoritarios.\n"
        "  - ADASYN evalúa los K-vecinos más cercanos de cada ejemplo minoritario y calcula la proporción r_i\n"
        "    de vecinos que pertenecen a la clase mayoritaria (r_i = Delta_i / K).\n"
        "  - Concentra la síntesis de nuevas muestras en las instancias con mayor r_i, es decir, aquellas que están\n"
        "    en la frontera de decisión y son más difíciles de aprender por el algoritmo.\n\n"
        "• ¿Por qué OBLIGATORIAMENTE Intra-Fold?:\n"
        "  - Si ADASYN se aplica antes del split, las muestras sintéticas se generan usando información del futuro\n"
        "    conjunto de validación. RetenTel AI ajusta ADASYN dentro de cada pliegue de entrenamiento."
    )
    ax.text(0.07, 0.61, p3_text, fontsize=8.2, color='#334155', linespacing=1.35)

    # Tarjeta 4: TabPFN
    add_card(ax, 0.05, 0.31, 0.90, 0.25, bg_color='#FFFFFF', border_color='#93C5FD')
    ax.text(0.07, 0.53, "4. TabPFN: Foundation Model Tabular de Frontera (Nature / ICLR)", 
            fontsize=11.5, fontweight='bold', color='#1E3A8A')
    p4_text = (
        "TabPFN (Prior-data Fitted Networks for Tabular Data, Hollmann et al., 2023) es una revolución conceptual:\n\n"
        "• Arquitectura Transformer Tabular: Preentrenado sobre millones de datasets sintéticos generados mediante\n"
        "  modelos causales y procesos gaussianos bayesianos.\n"
        "• In-Context Learning Puro: No realiza backpropagation ni ajuste de gradiente al inferir. Recibe la matriz de\n"
        "  entrenamiento y los casos de test en un solo 'forward pass' de atención, computando la probabilidad posterior.\n"
        "• Rol en RetenTel AI: Sirvió como el benchmark bayesiano más riguroso de la literatura moderna para contrastar\n"
        "  nuestro pipeline frente a árboles de decisión gradient boosted."
    )
    ax.text(0.07, 0.34, p4_text, fontsize=8.2, color='#334155', linespacing=1.35)

    # Tarjeta 5: CatBoost
    add_card(ax, 0.05, 0.06, 0.90, 0.23, bg_color='#FFFFFF', border_color='#93C5FD')
    ax.text(0.07, 0.26, "5. CatBoost: Árboles Simétricos y Boosting Ordenado", 
            fontsize=11.5, fontweight='bold', color='#1E3A8A')
    p5_text = (
        "CatBoost (Yandex, Prokhorenkova et al., NeurIPS 2018) fue el motor seleccionado para el campeón:\n\n"
        "• Árboles Simétricos (Oblivious Trees): Aplica el mismo criterio de división en todo el nivel del árbol.\n"
        "  Esto actúa como un regularizador implícito y permite convertir el árbol en tablas de bits ultra-rápidas en CPU.\n"
        "• Target Statistics Ordenadas: Calcula estadísticas del target en permutaciones aleatorias, previniendo fuga.\n"
        "• Desempeño en Producción: Latencia de 12.56 ms / 1,000 predicciones y peso de 11.1 KB en disco."
    )
    ax.text(0.07, 0.09, p5_text, fontsize=8.2, color='#334155', linespacing=1.35)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ==========================================
    # PÁGINA 3: MÉTRICAS Y EVALUACIÓN PROBABILÍSTICA
    # ==========================================
    fig, ax = setup_page("3. Métricas Avanzadas y Calibración Probabilística", 
                         "ROC-AUC (Discriminación), Brier Score (Calibración) y Fundamento del F2-Score", 3)
    
    # Tarjeta 6: ROC-AUC
    add_card(ax, 0.05, 0.62, 0.90, 0.26, bg_color='#FFFFFF', border_color='#93C5FD')
    ax.text(0.07, 0.85, "6. ROC-AUC: Capacidad de Discriminación y Ordenamiento Relativo", 
            fontsize=11.5, fontweight='bold', color='#1E3A8A')
    p6_text = (
        "El Área Bajo la Curva ROC (Receiver Operating Characteristic) evalúa el modelo en todos los umbrales:\n\n"
        "• Ejes de la Curva: Tasa de Verdaderos Positivos (Recall = TP / [TP + FN]) frente a Falsos Positivos (FPR = FP / [FP + TN]).\n"
        "• Significado Probabilístico Formal: Representa la probabilidad exacta de que un cliente cancelador tomado\n"
        "  al azar reciba un score de riesgo más alto que un cliente retenido: AUC = P(f(x_churn) > f(x_no_churn)).\n"
        "• Resultado del Campeón: 0.8474 en el conjunto de prueba aislado (excelente capacidad de separación)."
    )
    ax.text(0.07, 0.65, p6_text, fontsize=8.2, color='#334155', linespacing=1.35)

    # Tarjeta 7: Brier Score
    add_card(ax, 0.05, 0.33, 0.90, 0.27, bg_color='#FFFFFF', border_color='#93C5FD')
    ax.text(0.07, 0.57, "7. Brier Score: Calibración y Fiabilidad de las Probabilidades", 
            fontsize=11.5, fontweight='bold', color='#1E3A8A')
    p7_text = (
        "El Brier Score (Glenn W. Brier, 1950) es una métrica indispensable para la empresa pública y privada:\n\n"
        "• Formulación Matemática: BS = (1 / N) * SUM_{i=1}^N (p_i - y_i)^2\n"
        "  donde p_i es la probabilidad continua estimada [0, 1] y y_i es la etiqueta binaria real {0, 1}.\n"
        "• Strictly Proper Scoring Rule: Se minimiza (BS -> 0) si y solo si las probabilidades predichas reflejan la verdadera\n"
        "  frecuencia observada condicionada P(Y=1|X).\n"
        "• ¿Por qué es vital?: Un modelo puede tener alto ROC-AUC pero emitir probabilidades pésimas (ej. 0.51 vs 0.49).\n"
        "  Para planificar presupuestos de rescate se necesitan probabilidades reales. RetenTel AI logró BS = 0.1713."
    )
    ax.text(0.07, 0.36, p7_text, fontsize=8.2, color='#334155', linespacing=1.35)

    # Tarjeta 8: F2-Score
    add_card(ax, 0.05, 0.06, 0.90, 0.25, bg_color='#FFFFFF', border_color='#93C5FD')
    ax.text(0.07, 0.28, "8. ¿Por qué optimizar el F2-Score en lugar de Accuracy o F1?", 
            fontsize=11.5, fontweight='bold', color='#1E3A8A')
    p8_text = (
        "La familia F-beta (Van Rijsbergen, 1979) pondera la importancia relativa de Precision y Recall:\n\n"
        "• Fórmula con beta = 2: F_2 = 5 * (Precision * Recall) / (4 * Precision + Recall)\n"
        "• Justificación de Negocio (Asimetría Radical de Costos):\n"
        "  - Falso Positivo (FP): Llamar a un cliente que no se iba. Costo menor de gestión (~$35).\n"
        "  - Falso Negativo (FN): No detectar a un cliente que se marcha. Pérdida del valor anual completo (~$780).\n"
        "  - El error por omisión cuesta 20 veces más. El F_2 asigna doble peso al Recall, obligando al modelo a detectar\n"
        "    a los canceladores. Resultado: Recall de 84.22% y F2-Score de 0.7353."
    )
    ax.text(0.07, 0.09, p8_text, fontsize=8.2, color='#334155', linespacing=1.35)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ==========================================
    # PÁGINA 4: XAI (SHAP) Y MOTOR CONTRAFACTUAL
    # ==========================================
    fig, ax = setup_page("4. Explicabilidad XAI y Prescripción Contrafactual", 
                         "Fundamentos Axiomáticos de SHAP y Recurso Accionable para el Operador", 4)
    
    # Tarjeta 9: SHAP
    add_card(ax, 0.05, 0.49, 0.90, 0.39, bg_color='#FFFFFF', border_color='#93C5FD')
    ax.text(0.07, 0.85, "9. ¿Qué es SHAP? (SHapley Additive exPlanations)", 
            fontsize=11.5, fontweight='bold', color='#1E3A8A')
    p9_text = (
        "SHAP (Lundberg & Lee, NeurIPS 2017) traslada la Teoría de Juegos Cooperativos (Premio Nobel Lloyd Shapley, 1953)\n"
        "a la Inteligencia Artificial Explicable (XAI):\n\n"
        "• Analogía de la Teoría de Juegos:\n"
        "  - Las variables predictivas (features) son 'jugadores' en una coalición.\n"
        "  - La predicción del modelo f(x) es el 'pago' o ganancia colectiva del juego.\n"
        "  - El valor de Shapley phi_j asigna a cada variable su contribución marginal exacta promediada en todas las coaliciones.\n\n"
        "• Los 4 Axiomas que Garantizan Rigor Inmune a Impugnaciones Legales o Académicas:\n"
        "  1. Eficiencia (Local Accuracy): f(x) = phi_0 + SUM phi_j(x). La suma de aportes explica la predicción al 100%.\n"
        "  2. Simetría: Si dos variables aportan lo mismo a cualquier coalición, reciben exactamente el mismo phi.\n"
        "  3. Jugador Nulo: Si una variable no altera el resultado en ninguna coalición, su valor SHAP es exactamente 0.\n"
        "  4. Aditividad: Permite sumar limpiamente las explicaciones entre árboles en ensambles como CatBoost.\n\n"
        "• TreeExplainer: Algoritmo optimizado en O(TLD^2) que calcula los valores exactos, revelando que el tipo de contrato\n"
        "  (mes a mes), tenure bajo y altos cargos mensuales son los mayores inductores de fuga."
    )
    ax.text(0.07, 0.52, p9_text, fontsize=8.2, color='#334155', linespacing=1.35)

    # Tarjeta 10: Prescripción Contrafactual
    add_card(ax, 0.05, 0.07, 0.90, 0.40, bg_color='#FFFFFF', border_color='#93C5FD')
    ax.text(0.07, 0.44, "10. Motor Contrafactual: De la Predicción a la Acción Operativa", 
            fontsize=11.5, fontweight='bold', color='#1E3A8A')
    p10_text = (
        "La predicción por sí sola no salva usuarios; se requiere Prescripción Algorítmica Accionable:\n\n"
        "• Búsqueda Greedy sobre Palancas Accionables:\n"
        "  - Variables Intocables (Protegidas): Género, Ciudadano Mayor, Dependientes, Antigüedad (tenure).\n"
        "    No tiene sentido decirle al operador 'rejuvenezca al cliente' o 'aumente artificialmente su antigüedad'.\n"
        "  - Palancas Modificables por la Institución: Tipo de contrato, soporte técnico prioritario, suite de seguridad\n"
        "    en línea, método de débito bancario y bonificación tarifaria promocional (-$15/mes).\n\n"
        "• Lógica Operativa del Algoritmo:\n"
        "  1. Toma a un cliente de riesgo crítico (ej. Probabilidad = 78%).\n"
        "  2. Evalúa individualmente cada palanca de servicio y acepta solo aquellas que disminuyen monótonamente el riesgo.\n"
        "  3. Se detiene tan pronto el riesgo cae por debajo del umbral de seguridad (30%).\n\n"
        "• Valor para la Empresa Pública: El funcionario no necesita interpretar matemáticas; la interfaz le dice:\n"
        "  'Paso 1: Migrar a contrato anual (-26% de riesgo). Paso 2: Bonificar soporte técnico (-24% de riesgo).\n"
        "  Resultado: Cliente rescatado al 28% de probabilidad final'."
    )
    ax.text(0.07, 0.10, p10_text, fontsize=8.2, color='#334155', linespacing=1.35)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

    # ==========================================
    # PÁGINA 5: RESUMEN DE ARQUITECTURA Y AUDITORÍA
    # ==========================================
    fig, ax = setup_page("5. Ficha Técnica Consolidada del Modelo Campeón", 
                         "Resultados Finales en Conjunto de Prueba Aislado y Guardrails de Seguridad", 5)
    
    # Tabla de Métricas
    add_card(ax, 0.05, 0.54, 0.90, 0.34, bg_color='#FFFFFF', border_color='#93C5FD')
    ax.text(0.07, 0.85, "Tabla Maestra de Desempeño: Conjunto de Prueba Aislado (20% Holdout)", 
            fontsize=11.5, fontweight='bold', color='#1E3A8A')
    
    table_data = [
        ["Métrica de Evaluación", "Valor Obtenido", "Objetivo / Interpretación Práctica"],
        ["Recall (Sensibilidad)", "84.22%", "Detecta más de 8 de cada 10 canceladores reales"],
        ["ROC-AUC", "0.8474", "Excelente capacidad de discriminación en datos nuevos"],
        ["F2-Score (beta=2)", "0.7353", "Optimización multiobjetivo con peso 2x en Recall"],
        ["F1-Score", "0.6176", "Equilibrio armónico estándar"],
        ["Precision", "48.76%", "Mitiga falsas alarmas controlando el costo de contacto"],
        ["Brier Score (Calibración)", "0.1713", "Probabilidades fiables para presupuestación"],
        ["Latencia de Inferencia", "12.56 ms / 1k", "Inferencia ultra-rápida en CPU (0.012 ms/usuario)"],
        ["Tamaño del Modelo Serializado", "11.1 KB", "Desplegable en edge, microservicios o contenedores"]
    ]
    
    col_widths = [0.32, 0.18, 0.44]
    y_pos = 0.80
    for row_idx, row in enumerate(table_data):
        is_header = (row_idx == 0)
        bg = '#1E3A8A' if is_header else ('#F1F5F9' if row_idx % 2 == 1 else '#FFFFFF')
        txt_col = '#FFFFFF' if is_header else '#0F172A'
        weight = 'bold' if is_header else ('bold' if row_idx in [1, 2, 3, 6] else 'normal')
        
        row_rect = patches.Rectangle((0.07, y_pos - 0.024), 0.86, 0.028, facecolor=bg, edgecolor='#CBD5E1', linewidth=0.5)
        ax.add_patch(row_rect)
        
        x_col = 0.08
        for col_idx, cell in enumerate(row):
            ax.text(x_col, y_pos - 0.018, cell, fontsize=8 if not is_header else 8.5, 
                    color=txt_col, fontweight=weight, family='sans-serif')
            x_col += col_widths[col_idx]
        y_pos -= 0.029

    # Tarjeta de Guardrails y Conclusión
    add_card(ax, 0.05, 0.07, 0.90, 0.44, bg_color='#FFFFFF', border_color='#93C5FD')
    ax.text(0.07, 0.48, "Arquitectura de Seguridad RAG y Guardrails para el Sector Público", 
            fontsize=11.5, fontweight='bold', color='#1E3A8A')
    
    p11_text = (
        "El componente generativo de RetenTel AI fue blindado contra riesgos de seguridad e inyección:\n\n"
        "• Búsqueda Semántica Vectorial con TF-IDF y Similitud Coseno:\n"
        "  El asistente no consulta a la nube de manera irrestricta; busca información contextual en la base de datos\n"
        "  técnica oficial del proyecto y los metadatos reales serializados (metadata.json).\n\n"
        "• Guardrail de Seguridad Algorítmico (Umbral 0.12):\n"
        "  Cualquier consulta sobre temas ajenos (cocina, deportes, política) o intentos de prompt injection sufren una\n"
        "  pérdida de similitud semántica. Si el score < 0.12, se activa un bloqueo automático que rechaza la consulta\n"
        "  y notifica que solo se responde con documentación oficial autorizada.\n\n"
        "• Transparencia Institucional y Auditoría de Respuestas:\n"
        "  Cada respuesta emitida por el asistente despliega un panel interactivo con: el Documento Oficial consultado,\n"
        "  el porcentaje exacto de similitud semántica, el motor de inferencia utilizado y el fragmento textual verificado.\n\n"
        "• CONCLUSIÓN PARA LA BECA / PASANTÍA:\n"
        "  RetenTel AI prueba que domino todo el ciclo de vida del dato: desde la prevención estricta de fuga y modelado\n"
        "  avanzado (CatBoost, TabPFN, NSGA-III), hasta la ingeniería de software seguro, interfaces ergonómicas y ROI medible.\n"
        "  Estoy preparado para generar valor técnico y comercial inmediato en la empresa tecnológica patrocinadora."
    )
    ax.text(0.07, 0.10, p11_text, fontsize=8.2, color='#334155', linespacing=1.35)

    pdf.savefig(fig, bbox_inches='tight')
    plt.close()

print(f"PDF generado exitosamente en: {pdf_path}")
print(f"Tamaño: {os.path.getsize(pdf_path)} bytes")
