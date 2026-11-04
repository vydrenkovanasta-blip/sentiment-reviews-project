import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import os
import csv
import re

# ==========================================
# ФУНКЦИЯ СОХРАНЕНИЯ ОБРАТНОЙ СВЯЗИ
# ==========================================
def save_feedback(text, correct_label, original_label):
    file_path = "data/feedback.csv"
    file_exists = os.path.isfile(file_path)
    
    with open(file_path, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["text", "label", "model_prediction"])
        writer.writerow([text, correct_label, original_label])

# ==========================================
# БЛОК 1: НАСТРОЙКА СТРАНИЦЫ И ЗАГРУЗКА МОДЕЛИ
# ==========================================
st.set_page_config(page_title="Анализ отзывов", page_icon="📊", layout="wide")

# Загрузка модели
@st.cache_resource
def load_model():
    model = joblib.load("models/nb_model.pkl")
    vectorizer = joblib.load("models/vectorizer.pkl")
    return model, vectorizer

# Загрузка словаря сленга
@st.cache_resource
def load_slang():
    slang_df = pd.read_csv("data/slang_dict.csv")
    slang_dict = {}
    for _, row in slang_df.iterrows():
        slang_dict[str(row['slang']).strip().lower()] = (
            str(row['нормализация']).strip(),
            str(row['тип']).strip()
        )
    return slang_dict

# Функция нормализации (как в model.py)
def normalize_text(text, slang_dict):
    text = str(text).lower()
    phrases = {k: v for k, v in slang_dict.items() if v[1] == 'phrase'}
    for phrase, (norm, _) in sorted(phrases.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(phrase, norm)

    polarities = {k: v for k, v in slang_dict.items() if v[1] == 'polarity'}
    for word, (norm, _) in polarities.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', norm, text)

    intensifiers = {k: v for k, v in slang_dict.items() if v[1] == 'intensifier'}
    for word, (norm, _) in intensifiers.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', 'очень', text)
    return text

try:
    model, vectorizer = load_model()
    slang_dict = load_slang()
except FileNotFoundError:
    st.error("Модель или словарь сленга не найдены. Запустите: python model.py")
    st.stop()

# ==========================================
# БЛОК 1.5: ЗАГОЛОВОК И ОПИСАНИЕ
# ==========================================
st.title("📊 Анализ тональности отзывов")
st.markdown("""
Это веб-приложение анализирует эмоциональную окраску текстовых отзывов о товарах и услугах. 
Оно автоматически распределяет отзывы на три категории: **Позитивный**, **Нейтральный** и **Негативный**, 
используя алгоритм машинного обучения (Наивный Байес) и собственный словарь современного сленга.

**Как пользоваться:**
1. Загрузите файл **CSV** или **TXT** (первая колонка — текст отзыва).
2. Или просто вставьте текст в поле справа.
3. Нажмите кнопку **«Анализировать»**.
""")
st.divider() # Добавляет красивую горизонтальную линию

# ==========================================
# БЛОК 2: ВВОД ДАННЫХ
# ==========================================
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("1. Загрузите файл")
    uploaded = st.file_uploader("CSV или TXT", type=["csv", "txt"])

with col_b:
    st.subheader("2. Или вставьте текст")
    manual_text = st.text_area("Каждый отзыв — с новой строки", height=180)

reviews = []
if uploaded is not None:
    if uploaded.name.endswith(".csv"):
        df_up = pd.read_csv(uploaded)
        reviews += df_up.iloc[:, 0].dropna().astype(str).tolist()
    else:
        content = uploaded.read().decode("utf-8", errors="ignore")
        reviews += [r.strip() for r in content.split("\n") if r.strip()]

if manual_text:
    reviews += [r.strip() for r in manual_text.split("\n") if r.strip()]

reviews = [r for r in reviews if len(r) > 2]

# ==========================================
# БЛОК 3: КЛАССИФИКАЦИЯ (с памятью сессии)
# ==========================================
if reviews:
    st.success(f"Получено отзывов: **{len(reviews)}**")

    if st.button("🔍 Анализировать", type="primary"):
        with st.spinner("Модель анализирует отзывы..."):
            # Сначала применяем сленг
            normalized_reviews = [normalize_text(r, slang_dict) for r in reviews]
            X = vectorizer.transform(normalized_reviews)
            preds = model.predict(X)
            proba = model.predict_proba(X)
            
            # Сохраняем результаты в "память" Streamlit
            st.session_state['result_df'] = pd.DataFrame({
                "Отзыв": reviews,
                "Класс": preds,
                "Уверенность": [f"{max(p):.0%}" for p in proba]
            })

# ==========================================
# БЛОК 4: ВЫВОД РЕЗУЛЬТАТОВ (если они есть в памяти)
# ==========================================
    # ==========================================
# БЛОК 4: ВЫВОД РЕЗУЛЬТАТОВ (если они есть в памяти)
# ==========================================
if 'result_df' in st.session_state:
    result_df = st.session_state['result_df']
    RU = {"positive": "Позитивный", "neutral": "Нейтральный", "negative": "Негативный"}

    # --- 1. СТАТИСТИКА ---
    st.markdown("---")
    st.subheader("📈 Статистика")
    counts = result_df["Класс"].value_counts()
    c1, c2, c3 = st.columns(3)
    c1.metric("🟢 Позитивный", counts.get("positive", 0))
    c2.metric("🟡 Нейтральный", counts.get("neutral", 0))
    c3.metric("🔴 Негативный", counts.get("negative", 0))

    # --- 2. ДИАГРАММЫ ---
    st.markdown("---")
    st.subheader("🥧 Диаграммы")
    col_pie, col_bar = st.columns(2)

    with col_pie:
        fig1, ax1 = plt.subplots(figsize=(4, 4))
        labels = [RU[k] for k in ["positive", "neutral", "negative"] if k in counts.index]
        sizes = [counts[k] for k in ["positive", "neutral", "negative"] if k in counts.index]
        colors = ["#4CAF50", "#FFC107", "#F44336"][:len(labels)]
        ax1.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
        ax1.axis("equal")
        st.pyplot(fig1)

    with col_bar:
        fig2, ax2 = plt.subplots(figsize=(5, 4))
        ax2.bar(["Позитивный", "Нейтральный", "Негативный"], 
                [counts.get("positive", 0), counts.get("neutral", 0), counts.get("negative", 0)],
                color=["#4CAF50", "#FFC107", "#F44336"])
        st.pyplot(fig2)

    # --- 3. ТРИ СТОЛБИКА С ОТЗЫВАМИ ---
    st.markdown("---")
    st.subheader("📋 Отзывы по категориям")
    col_pos, col_neu, col_neg = st.columns(3)

    def render_column(col, label_key, title, color):
        with col:
            subset = result_df[result_df["Класс"] == label_key]
            st.markdown(f"<h3 style='color:{color};'>{title} ({len(subset)})</h3>", unsafe_allow_html=True)
            if subset.empty:
                st.write("— пусто —")
            else:
                with st.container(height=500):
                    for idx, row in subset.iterrows():
                        st.markdown(
                            f"<div style='padding:8px; margin-bottom:6px; background:#f8f8f8; "
                            f"border-left:4px solid {color}; border-radius:4px; font-size:14px; "
                            f"color: #333333;'>{row['Отзыв']}<br>"
                            f"<span style='color:#666666; font-size:12px;'>уверенность: {row['Уверенность']}</span></div>",
                            unsafe_allow_html=True
                        )
                        
                        with st.expander("❌ Модель ошиблась?"):
                            correct = st.selectbox(
                                "Выберите класс:",
                                ["negative", "neutral", "positive"],
                                format_func=lambda x: {"negative": "Негативный", "neutral": "Нейтральный", "positive": "Позитивный"}[x],
                                key=f"select_{idx}"
                            )
                            if st.button("Отправить", key=f"btn_{idx}"):
                                save_feedback(row['Отзыв'], correct, row['Класс'])
                                st.success("Спасибо! Данные сохранены.")

    render_column(col_pos, "positive", "🟢 Позитивный", "#4CAF50")
    render_column(col_neu, "neutral", "🟡 Нейтральный", "#FFC107")
    render_column(col_neg, "negative", "🔴 Негативный", "#F44336")

    # --- 4. СКАЧИВАНИЕ ---
    st.markdown("---")
    csv = result_df.to_csv(index=False).encode("utf-8-sig")
    st.download_button("📥 Скачать результат (CSV)", data=csv, file_name="results.csv", mime="text/csv")

else:
    st.info("Загрузите файл или вставьте отзывы, чтобы начать.")
