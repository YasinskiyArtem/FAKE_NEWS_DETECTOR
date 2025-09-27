from flask import Flask, render_template, request, jsonify
from data_collector import ImprovedDataCollector as DataCollector
from news_analyzer import NewsAnalyzer
from reputation_system import ReputationSystem
import os
import json

app = Flask(__name__)

# Инициализация компонентов
data_collector = DataCollector()
news_analyzer = NewsAnalyzer()
reputation_system = ReputationSystem()

@app.route('/')
def index():
    """Главная страница"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_news():
    """Анализ новости"""
    data = request.get_json()
    news_text = data.get('text', '').strip()
    news_url = data.get('url', '').strip()
    
    if not news_text:
        return jsonify({'error': 'Введите текст новости'}), 400
    
    # Анализ AI
    ai_result = news_analyzer.analyze_news(news_text)
    
    # Анализ источника
    trust_score = 50.0
    source_info = None
    
    if news_url:
        trust_score = reputation_system.get_trust_score(news_url)
        source_info = reputation_system.get_source_info(news_url)
    
    # Комбинированный вердикт
    final_verdict = get_final_verdict(ai_result, trust_score)
    
    response = {
        'ai_analysis': ai_result,
        'source_analysis': {
            'trust_score': trust_score,
            'source_info': source_info,
            'url_provided': bool(news_url)
        },
        'final_verdict': final_verdict
    }
    
    return jsonify(response)

@app.route('/verify', methods=['POST'])
def verify_news():
    """Верификация новости пользователем"""
    data = request.get_json()
    url = data.get('url', '')
    is_fake = data.get('is_fake', False)
    
    if not url:
        return jsonify({'error': 'URL обязателен'}), 400
    
    new_score = reputation_system.update_trust_score(url, is_fake)
    
    return jsonify({
        'message': 'Репутация обновлена',
        'new_trust_score': new_score,
        'domain': reputation_system.extract_domain(url)
    })

@app.route('/admin/update-dataset', methods=['POST'])
def update_dataset():
    """Обновление датасета"""
    try:
        dataset = data_collector.create_dataset()
        news_analyzer.train_model()
        
        return jsonify({
            'success': True,
            'message': f'Датасет обновлен! Записей: {len(dataset)}',
            'dataset_size': len(dataset)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/admin/status')
def system_status():
    """Статус системы"""
    dataset_exists = data_collector.dataset_exists()
    model_exists = os.path.exists('models/news_model.pkl')
    
    status = {
        'dataset_ready': dataset_exists,
        'model_ready': model_exists,
        'system_status': 'operational' if model_exists else 'training_required'
    }
    
    return jsonify(status)

def get_final_verdict(ai_result, trust_score):
    """Определение финального вердикта"""
    ai_confidence = ai_result['confidence']
    ai_prediction = ai_result['prediction']
    
    # Весовые коэффициенты
    ai_weight = 0.7
    trust_weight = 0.3
    
    trust_normalized = trust_score / 100.0
    
    if ai_prediction == 'fake':
        ai_score = ai_result['fake_prob']
    else:
        ai_score = ai_result['real_prob']
    
    combined_score = (ai_score * ai_weight) + (trust_normalized * trust_weight)
    
    if combined_score >= 0.8:
        return {
            'verdict': '✅ ВЫСОКАЯ ДОСТОВЕРНОСТЬ',
            'confidence': 'high',
            'explanation': 'Новость выглядит достоверной. Стиль нейтральный, источник надежный.',
            'score': combined_score
        }
    elif combined_score >= 0.6:
        return {
            'verdict': '⚠️ ВЕРОЯТНО ДОСТОВЕРНА',
            'confidence': 'medium',
            'explanation': 'Новость вероятно достоверна. Рекомендуется проверка в других источниках.',
            'score': combined_score
        }
    elif combined_score >= 0.4:
        return {
            'verdict': '🔍 ТРЕБУЕТ ПРОВЕРКИ',
            'confidence': 'low',
            'explanation': 'Новость вызывает вопросы. Рекомендуется осторожность.',
            'score': combined_score
        }
    else:
        return {
            'verdict': '❌ ВЕРОЯТНЫЙ ФЕЙК',
            'confidence': 'very_low',
            'explanation': 'Высокая вероятность фейковой новости. Обнаружены маркеры недостоверности.',
            'score': combined_score
        }

if __name__ == '__main__':
    # Создание папок
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Проверка и инициализация системы
    print("🚀 Запуск Fake News Detector Pro...")
    
    if not data_collector.dataset_exists():
        print("📊 Создание начального датасета...")
        data_collector.create_balanced_dataset(100)

    print("🤖 Обучение модели...")
    news_analyzer.train_model()
    
    print("🌐 Система готова! Запуск сервера...")
    app.run(debug=True, host='0.0.0.0', port=5000)