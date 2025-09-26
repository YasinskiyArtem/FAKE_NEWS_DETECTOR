import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

class NewsAnalyzer:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=1500,
            stop_words=['и', 'в', 'на', 'с', 'по', 'для', 'о', 'от', 'к'],
            ngram_range=(1, 2)
        )
        self.model = LogisticRegression(max_iter=1000, random_state=42)
        
        self.fake_indicators = {
            'emotional': ['шок', 'сенсация', 'скандал', 'разоблачение', 'срочно', 'эксклюзив'],
            'clickbait': ['вы не поверите', 'стало известно', 'ученые в шоке', 'врачи скрывают'],
            'conspiracy': ['заговор', 'скрывают правду', 'мировой заговор', 'тайное правительство']
        }
    
    def load_dataset(self):
        """Загрузка датасета"""
        try:
            df = pd.read_csv('data/russian_news_dataset.csv')
            return df
        except:
            from data_collector import DataCollector
            collector = DataCollector()
            return collector.create_dataset()
    
    def extract_features(self, text):
        """Извлечение признаков из текста"""
        if not isinstance(text, str):
            text = ""
            
        text_lower = text.lower()
        
        features = {
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),
            'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(1, len(text)),
            'text_length': len(text),
            'word_count': len(text.split()),
            'emotional_words': sum(1 for word in self.fake_indicators['emotional'] if word in text_lower),
            'clickbait_phrases': sum(1 for phrase in self.fake_indicators['clickbait'] if phrase in text_lower),
            'conspiracy_terms': sum(1 for term in self.fake_indicators['conspiracy'] if term in text_lower),
            'has_quotes': int('«' in text or '"' in text),
            'has_hashtags': int('#' in text)
        }
        
        return features
    
    def train_model(self):
        """Обучение модели на датасете"""
        print("🔄 Обучение модели...")
        
        df = self.load_dataset()
        
        # Векторизация текста
        X_text = self.vectorizer.fit_transform(df['text'])
        
        # Извлечение признаков
        features_list = []
        for text in df['text']:
            features = self.extract_features(text)
            features_list.append(list(features.values()))
        
        features_array = np.array(features_list)
        
        # Объединение признаков
        X_combined = np.hstack([X_text.toarray(), features_array])
        y = df['is_fake']
        
        # Обучение
        X_train, X_test, y_train, y_test = train_test_split(
            X_combined, y, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Оценка
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Модель обучена! Точность: {accuracy:.3f}")
        print(classification_report(y_test, y_pred))
        
        # Сохранение
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.model, 'models/news_model.pkl')
        joblib.dump(self.vectorizer, 'models/vectorizer.pkl')
        
        return accuracy
    
    def load_trained_model(self):
        """Загрузка обученной модели"""
        try:
            self.model = joblib.load('models/news_model.pkl')
            self.vectorizer = joblib.load('models/vectorizer.pkl')
            return True
        except:
            return False
    
    def analyze_news(self, text):
        """Анализ новости"""
        if not self.load_trained_model():
            print("❌ Модель не найдена, начинаем обучение...")
            self.train_model()
            self.load_trained_model()
        
        # Извлечение признаков
        features = self.extract_features(text)
        features_array = np.array([list(features.values())])
        
        # Векторизация текста
        text_vec = self.vectorizer.transform([text])
        
        # Объединение
        X_combined = np.hstack([text_vec.toarray(), features_array])
        
        # Предсказание
        prediction = self.model.predict(X_combined)[0]
        probability = self.model.predict_proba(X_combined)[0]
        
        # Анализ стиля
        style_analysis = self.analyze_style(text, features)
        
        return {
            'prediction': 'fake' if prediction == 1 else 'real',
            'confidence': max(probability),
            'real_prob': probability[0],
            'fake_prob': probability[1],
            'features': features,
            'style_analysis': style_analysis
        }
    
    def analyze_style(self, text, features):
        """Анализ стилистических особенностей"""
        analysis = []
        
        if features['emotional_words'] > 1:
            analysis.append(f"📛 Эмоциональные слова: {features['emotional_words']}")
        
        if features['clickbait_phrases'] > 0:
            analysis.append("🎣 Обнаружены кликбейт-фразы")
        
        if features['conspiracy_terms'] > 0:
            analysis.append("🔮 Есть элементы теорий заговора")
        
        if features['exclamation_count'] > 2:
            analysis.append(f"❗️ Восклицательных знаков: {features['exclamation_count']}")
        
        if features['uppercase_ratio'] > 0.05:
            analysis.append("🔠 Много заглавных букв")
        
        if features['has_quotes']:
            analysis.append("💬 Есть цитаты/ссылки на источники")
        
        if not analysis:
            analysis.append("✅ Текст выглядит нейтральным")
        
        return analysis