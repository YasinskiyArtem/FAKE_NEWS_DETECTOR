import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import re
import string

class NewsAIAnalyzer:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.fake_indicators = [
            'шок', 'сенсация', 'скандал', 'разоблачение', 'замалчивают',
            'правда', 'секрет', 'срочно', 'эксклюзив', 'шокирующий'
        ]
    
    def create_demo_dataset(self):
        """Создание демонстрационного набора данных"""
        real_news = [
            "Сегодня в Москве прошла международная конференция по климату.",
            "Центральный банк опубликовал отчет по инфляции за прошлый месяц.",
            "Ученые обнаружили новую планету в ближайшей звездной системе.",
            "Правительство утвердило новую программу поддержки малого бизнеса.",
            "В городе завершился фестиваль современного искусства."
        ]
        
        fake_news = [
            "ШОК! Ученые скрывают правду о вакцинах! СРОЧНО к прочтению!",
            "Сенсационное разоблачение: правительство скрывает инопланетян!",
            "Эксклюзив: банкиры знают секрет вечной молодости!",
            "СКАНДАЛ! Медики десятилетиями скрывали простое лекарство от рака!",
            "Шокирующая правда о 5G вышках, которую от нас скрывают!"
        ]
        
        texts = real_news + fake_news
        labels = [0] * len(real_news) + [1] * len(fake_news)
        
        return pd.DataFrame({'text': texts, 'label': labels})
    
    def extract_text_features(self, text):
        """Извлечение дополнительных признаков из текста"""
        # Количество восклицательных знаков
        exclamation_count = text.count('!')
        
        # Количество заглавных букв
        upper_count = sum(1 for char in text if char.isupper())
        
        # Наличие маркеров фейков
        fake_markers = sum(1 for word in self.fake_indicators if word in text.lower())
        
        # Длина текста
        text_length = len(text)
        
        return {
            'exclamation_count': exclamation_count,
            'upper_ratio': upper_count / len(text) if text else 0,
            'fake_markers': fake_markers,
            'text_length': text_length
        }
    
    def train_model(self, df=None):
        """Обучение модели"""
        if df is None:
            df = self.create_demo_dataset()
        
        # Извлечение дополнительных признаков
        features = []
        for text in df['text']:
            feat = self.extract_text_features(text)
            features.append(feat)
        
        features_df = pd.DataFrame(features)
        
        # Векторизация текста
        self.vectorizer = TfidfVectorizer(
            max_features=1000, 
            stop_words=['и', 'в', 'на', 'с', 'по', 'для']
        )
        X_text = self.vectorizer.fit_transform(df['text'])
        
        # Объединение признаков
        X_combined = np.hstack([X_text.toarray(), features_df.values])
        y = df['label']
        
        # Разделение на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42
        )
        
        # Обучение модели
        self.model = LogisticRegression(random_state=42)
        self.model.fit(X_train, y_train)
        
        # Оценка точности
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Точность модели: {accuracy:.2f}")
        
        # Сохранение модели
        self.save_model()
        
        return accuracy
    
    def save_model(self):
        """Сохранение обученной модели"""
        if self.model and self.vectorizer:
            joblib.dump(self.model, 'models/news_model.pkl')
            joblib.dump(self.vectorizer, 'models/vectorizer.pkl')
            print("Модель сохранена")
    
    def load_model(self):
        """Загрузка обученной модели"""
        try:
            self.model = joblib.load('models/news_model.pkl')
            self.vectorizer = joblib.load('models/vectorizer.pkl')
            print("Модель загружена")
            return True
        except FileNotFoundError:
            print("Модель не найдена, требуется обучение")
            return False
    
    def analyze_text(self, text):
        """Анализ текста новости"""
        if not self.model or not self.vectorizer:
            if not self.load_model():
                self.train_model()
        
        # Извлечение признаков
        text_features = self.extract_text_features(text)
        features_df = pd.DataFrame([text_features])
        
        # Векторизация текста
        X_text = self.vectorizer.transform([text])
        
        # Объединение признаков
        X_combined = np.hstack([X_text.toarray(), features_df.values])
        
        # Предсказание
        prediction = self.model.predict(X_combined)[0]
        probability = self.model.predict_proba(X_combined)[0]
        
        # Анализ стилистических особенностей
        style_analysis = self.analyze_style(text)
        
        return {
            'prediction': 'fake' if prediction == 1 else 'real',
            'confidence': max(probability),
            'real_prob': probability[0],
            'fake_prob': probability[1],
            'style_analysis': style_analysis,
            'features': text_features
        }
    
    def analyze_style(self, text):
        """Анализ стилистических особенностей текста"""
        analysis = {
            'has_excessive_exclamations': text.count('!') > 3,
            'has_uppercase_words': any(word.isupper() and len(word) > 3 for word in text.split()),
            'contains_fake_markers': any(marker in text.lower() for marker in self.fake_indicators),
            'emotional_words_count': sum(1 for word in text.lower().split() if word in self.fake_indicators)
        }
        
        # Оценка эмоциональности текста
        emotional_score = sum([
            analysis['has_excessive_exclamations'] * 2,
            analysis['has_uppercase_words'] * 3,
            analysis['emotional_words_count'] * 1
        ])
        
        analysis['emotional_score'] = emotional_score
        analysis['is_emotional'] = emotional_score >= 3
        
        return analysis