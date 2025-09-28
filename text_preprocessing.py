import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Скачиваем необходимые ресурсы
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

def preprocess_text(text):
    """
    Предобработка текста: очистка, лемматизация, удаление стоп-слов
    """
    if pd.isna(text):
        return ""
    
    # Очистка текста
    text = re.sub(r'[^a-zA-Z\s]', '', str(text), re.I|re.A)
    text = text.lower().strip()
    
    # Токенизация
    tokens = text.split()
    
    # Удаление стоп-слов и лемматизация
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def load_and_preprocess_data(file_path='fake_news_train.csv'):
    """
    Загрузка и предобработка данных
    """
    df = pd.read_csv(file_path)
    print(f"Загружено {len(df)} записей")
    
    # Предобработка текста
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    
    # Удаление пустых текстов после очистки
    df = df[df['cleaned_text'].str.len() > 0]
    
    print(f"После очистки осталось {len(df)} записей")
    return df

# text_preprocessing.py - дополнительная функция
def load_fake_news_data(filepath):
    """
    Специфичная загрузка для датасета Fake News Detection
    """
    df = pd.read_csv(filepath)
    
    # Проверяем структуру данных
    print("Структура загруженных данных:")
    print(f"Столбцы: {df.columns.tolist()}")
    print(f"Размер: {df.shape}")
    
    # Очистка данных
    df = df.dropna()  # или более сложная логика очистки
    
    # Если есть отдельно title и text, объединяем их
    if 'title' in df.columns and 'text' in df.columns:
        df['cleaned_text'] = df['title'] + ' ' + df['text']
    elif 'text' in df.columns:
        df['cleaned_text'] = df['text']
    elif 'title' in df.columns:
        df['cleaned_text'] = df['title']
    
    return df