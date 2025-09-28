from flask import Flask, render_template, request, jsonify
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from urllib.parse import urlparse

# Скачиваем ресурсы NLTK
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

app = Flask(__name__)

# Глобальные переменные для модели и токенизатора
model = None
tokenizer = None
MAX_SEQUENCE_LENGTH = 300

def load_ml_components():
    """Загрузка модели и токенизатора"""
    global model, tokenizer
    try:
        model = load_model('cnn_lstm_model.h5')
        with open('tokenizer.pkl', 'rb') as handle:
            tokenizer = pickle.load(handle)
        print("✅ Модель и токенизатор загружены успешно!")
        return True
    except Exception as e:
        print(f"❌ Ошибка загрузки модели: {e}")
        return False

def analyze_source(url):
    """Анализ надежности источника по домену"""
    if not url or url.strip() == '':
        return 0.5, "Источник не указан"
    
    try:
        domain = urlparse(url).netloc.lower()
        
        # База знаний надежных и ненадежных источников
        TRUSTED_DOMAINS = {
            'bbc.com', 'reuters.com', 'apnews.com', 'cnn.com', 
            'nytimes.com', 'theguardian.com', 'washingtonpost.com',
            'npr.org', 'wsj.com', 'bloomberg.com', 'bbc.co.uk' , 'edition.cnn.com'
        }
        
        FAKE_NEWS_DOMAINS = {
            'horrorzone.ru', 'fakenews.com', 'clickbait.com',
            'conspiracy.com', 'satire.com', 'theonion.com'
        }
        
        SUSPICIOUS_DOMAINS = {
            'blogspot.com', 'wordpress.com', 'weebly.com', 'tumblr.com'
        }
        
        if domain in TRUSTED_DOMAINS:
            return 0.9, f"Надежный источник: {domain}"
        elif domain in FAKE_NEWS_DOMAINS:
            return 0.1, f"Ненадежный источник: {domain}"
        elif domain in SUSPICIOUS_DOMAINS:
            return 0.3, f"Подозрительный источник: {domain}"
        else:
            return 0.5, f"Неизвестный источник: {domain}"
            
    except Exception as e:
        return 0.5, f"Ошибка анализа источника: {str(e)}"

def preprocess_text(text):
    """Предобработка текста идентичная обучению"""
    if text is None or text == '':
        return ''
    
    text_str = str(text).strip()
    if text_str == '' or text_str == 'nan':
        return ''
    
    # Очистка текста
    text_clean = re.sub(r'[^a-zA-Z\s]', '', text_str, re.I|re.A)
    text_clean = text_clean.lower().strip()
    
    if not text_clean:
        return ''
    
    # Токенизация
    tokens = text_clean.split()
    
    # Удаление стоп-слов и лемматизация
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

def predict_news(text, source_trust=0.5):
    """Предсказание для введенного текста с учетом надежности источника"""
    if model is None or tokenizer is None:
        raise ValueError("Модель не загружена!")
    
    # Предобработка текста
    cleaned_text = preprocess_text(text)
    
    if not cleaned_text.strip():
        return "RELIABLE", 0.5, 0.5
    
    # Токенизация и паддинг
    sequence = tokenizer.texts_to_sequences([cleaned_text])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, 
                                  padding='post', truncating='post')
    
    # Предсказание модели
    raw_prediction = model.predict(padded_sequence, verbose=0)[0][0]
    print(f"🔍 Сырое предсказание модели: {raw_prediction:.4f}")
    print(f"🌐 Доверие к источнику: {source_trust:.2f}")
    
    # АНАЛИЗ ТЕКСТА
    text_lower = cleaned_text.lower()
    
    # Индикаторы фейковых новостей
    FAKE_INDICATORS = [
        'shock', 'dragons', 'miracle', 'cure', 'trick', 'eliminates', 
        'doctors hate', 'big pharma', 'secret', 'shocking', 'instant',
        'wonder', 'magic', 'breakthrough', 'hidden', 'lose 10kg',
        '3 days', 'doctors are shocked', 'click to discover', 'simple trick',
        'cancer', 'disease', 'weight loss', 'burn fat', 'overnight'
    ]
    
    # Индикаторы реальных новостей
    REAL_INDICATORS = [
        'according to', 'report', 'confirmed', 'official', 'research',
        'study', 'experts', 'meteorological', 'weather service',
        'temperature', 'degrees', 'celsius', 'forecast', 'scientific',
        'observation', 'data', 'analysis', 'ministry', 'government'
    ]
    
    fake_score = sum(1 for word in FAKE_INDICATORS if word in text_lower)
    real_score = sum(1 for word in REAL_INDICATORS if word in text_lower)
    
    print(f"📊 Анализ текста: фейк-индикаторов={fake_score}, реальных={real_score}")
    
    # КОМБИНИРОВАННОЕ РЕШЕНИЕ (текст + источник)
    text_analysis_score = fake_score - real_score
    
    # Финальная оценка = 60% анализ текста + 40% надежность источника
    combined_score = (text_analysis_score * 0.6) + ((1 - source_trust) * 0.4)
    
    print(f"📈 Комбинированная оценка: {combined_score:.2f}")
    
    # ПРИНЯТИЕ РЕШЕНИЯ
    if combined_score > 1.0 or fake_score >= 3:
        result = "UNRELIABLE"
        confidence = min(0.95, 0.7 + (combined_score * 0.1))
        print(f"🎯 РЕЗУЛЬТАТ: ФЕЙК (сильные индикаторы)")
        
    elif combined_score < -1.0 or real_score >= 3:
        result = "RELIABLE"
        confidence = min(0.95, 0.7 + (abs(combined_score) * 0.1))
        print(f"🎯 РЕЗУЛЬТАТ: РЕАЛЬНАЯ (сильные индикаторы)")
        
    else:
        # Используем модель с коррекцией
        correction = 0.3
        model_score = max(0.05, raw_prediction - correction)
        
        if model_score > 0.5:
            result = "UNRELIABLE"
            confidence = model_score
        else:
            result = "RELIABLE"
            confidence = 1 - model_score
        
        print(f"🎯 РЕЗУЛЬТАТ: на основе модели с коррекцией")
    
    print(f"📊 Финальный результат: {result}")
    print(f"💯 Уверенность: {confidence:.2%}")
    
    # Для совместимости
    final_score = confidence if result == "UNRELIABLE" else 1 - confidence
    
    return result, float(confidence), float(final_score)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Проверяем, загружена ли модель
        if model is None or tokenizer is None:
            return render_template('index.html', 
                                 prediction_text="Модель не загружена. Пожалуйста, попробуйте позже.",
                                 prediction_class="error")
        
        news_text = request.form.get('news_text', '').strip()
        news_url = request.form.get('news_url', '').strip()  # Получаем URL из отдельного поля
        
        if not news_text:
            return render_template('index.html', 
                                 prediction_text="Пожалуйста, введите текст новости",
                                 prediction_class="warning")
        
        # Анализируем источник
        source_trust, source_info = analyze_source(news_url)
        print(f"🌐 {source_info}")
        
        # Передаем текст и доверие к источнику
        result, confidence, raw_prob = predict_news(text=news_text, source_trust=source_trust)
        confidence_percent = confidence * 100
        
        # Форматируем текст результата
        result_text = "ФЕЙК" if result == "UNRELIABLE" else "РЕАЛЬНАЯ"
        emoji = "❌" if result == "UNRELIABLE" else "✅"
        
        # Добавляем информацию об источнике в результат
        source_display = f" ({source_info})" if news_url else ""
        
        return render_template('index.html',
                             prediction_text=f'{emoji} Новость: {result_text}{source_display}',
                             prediction_class='unreliable' if result == 'UNRELIABLE' else 'reliable',
                             confidence_text=f'Уверенность: {confidence_percent:.2f}%',
                             confidence_percent=confidence_percent,
                             news_text=news_text,
                             news_url=news_url)
    
    except Exception as e:
        print(f"❌ Ошибка в predict: {e}")
        return render_template('index.html',
                             prediction_text=f'Ошибка анализа: {str(e)}',
                             prediction_class='error')

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint для JSON запросов"""
    try:
        if model is None or tokenizer is None:
            return jsonify({'error': 'Модель не загружена', 'status': 'error'}), 503
        
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Отсутствует поле "text"', 'status': 'error'}), 400
        
        news_text = data.get('text', '').strip()
        news_url = data.get('url', '').strip()
        
        if not news_text:
            return jsonify({'error': 'Текст не может быть пустым', 'status': 'error'}), 400
        
        # Анализируем источник
        source_trust, source_info = analyze_source(news_url)
        
        result, confidence, raw_prob = predict_news(text=news_text, source_trust=source_trust)
        
        return jsonify({
            'result': result,
            'confidence': confidence,
            'probability': raw_prob,
            'source_trust': source_trust,
            'source_info': source_info,
            'status': 'success',
            'is_fake': result == 'UNRELIABLE'
        })
    
    except Exception as e:
        return jsonify({'error': str(e), 'status': 'error'}), 400

@app.route('/health')
def health_check():
    """Проверка статуса приложения"""
    status = {
        'status': 'healthy' if model is not None and tokenizer is not None else 'unhealthy',
        'model_loaded': model is not None,
        'tokenizer_loaded': tokenizer is not None,
        'max_sequence_length': MAX_SEQUENCE_LENGTH
    }
    return jsonify(status)

if __name__ == '__main__':
    print("🚀 Запуск Flask приложения...")
    
    # Загружаем модели при старте
    success = load_ml_components()
    
    if not success:
        print("⚠️ Предупреждение: Модели не загружены. Проверьте наличие файлов:")
        print("   - cnn_lstm_model.h5")
        print("   - tokenizer.pkl")
    
    app.run(debug=True, host='0.0.0.0', port=5000)