import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report

# 1. Загрузка данных
print("Загружаем данные...")
try:
    df = pd.read_csv('data/Dataset.csv')
    slang_df = pd.read_csv('data/slang_dict.csv')
except FileNotFoundError:
    print("ОШИБКА: Файлы Dataset.csv или slang_dict.csv не найдены в папке проекта!")
    print("Пожалуйста, перетащи их в папку pythonProject1 в левом меню.")
    exit()

# Создаем словарь из DataFrame
slang_dict = {}
for _, row in slang_df.iterrows():
    slang_dict[str(row['slang']).strip().lower()] = (
        str(row['нормализация']).strip(),
        str(row['тип']).strip()
    )


# 2. Функция нормализации сленга
def normalize_text(text):
    text = str(text).lower()

    # Шаг 1: Заменяем фразы (phrase)
    phrases = {k: v for k, v in slang_dict.items() if v[1] == 'phrase'}
    for phrase, (norm, _) in sorted(phrases.items(), key=lambda x: len(x[0]), reverse=True):
        text = text.replace(phrase, norm)

    # Шаг 2: Заменяем обычные слова (polarity)
    polarities = {k: v for k, v in slang_dict.items() if v[1] == 'polarity'}
    for word, (norm, _) in polarities.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', norm, text)

    # Шаг 3: Усилители (intensifier) меняем на "очень"
    intensifiers = {k: v for k, v in slang_dict.items() if v[1] == 'intensifier'}
    for word, (norm, _) in intensifiers.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', 'очень', text)

    return text


print("Применяем словарь сленга...")
df['clean_text'] = df['text'].apply(normalize_text)

# 3. Подготовка к обучению
X = df['clean_text']
y = df['label']

# Делим на обучающую и тестовую выборки (80% на 20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Векторизация текста (TF-IDF)
print("Векторизуем текст...")
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 5. Обучение модели
print("Обучаем модель ComplementNB...")
model = ComplementNB(alpha=2.0)  # Используем Байес с нашей лучшей alpha
model.fit(X_train_vec, y_train)

# 6. Оценка качества
y_pred = model.predict(X_test_vec)
f1 = f1_score(y_test, y_pred, average='macro')

print("\n" + "=" * 40)
print(f"ИТОГОВЫЙ F1-MACRO: {f1:.4f}")
print("=" * 40)
print("\nДетальный отчет по классам:")
print(classification_report(y_test, y_pred))

# --- СОХРАНЕНИЕ МОДЕЛИ ДЛЯ САЙТА ---
import joblib
import os

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/nb_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")
print("\n✅ Модель и векторизатор сохранены в папку models/")