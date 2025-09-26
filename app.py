from flask import Flask, render_template, request, jsonify
from ai_module import NewsAIAnalyzer
from reputation_db import ReputationSystem
import json

app = Flask(__name__)
ai_analyzer = NewsAIAnalyzer()
reputation_system = ReputationSystem()

# Загрузка модели при старте
ai_analyzer.load_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_news():
    """Основной endpoint для анализа новости"""
    data = request.json
    news_text = data.get('text', '')
    news_url = data.get('url', '')
    
    if not news_text:
        return jsonify({'error': 'Текст новости обязателен'}), 400
    
    # 1. Анализ текста AI-моделью
    ai_result = ai_analyzer.analyze_text(news_text)
    
    # 2. Проверка репутации источника
    trust_score = 50.0
    source_info = None
    
    if news_url:
        trust_score = reputation_system.get_trust_score(news_url)
        source_info = reputation_system.get_source_stats(news_url)
    
    # 3. Комбинированная оценка
    final_verdict = combine_assessments(ai_result, trust_score)
    
    response = {
        'ai_analysis': ai_result,
        'source_analysis': {
            'trust_score': trust_score,
            'source_info': source_info,
            'url_provided': bool(news_url)
        },
        'final_verdict': final_verdict,
        'recommendation': generate_recommendation(final_verdict, trust_score)
    }
    
    return jsonify(response)

@app.route('/verify', methods=['POST'])
def verify_news():
    """Endpoint для верификации новости пользователями"""
    data = request.json
    url = data.get('url', '')
    is_fake = data.get('is_fake', False)
    fact_checker = data.get('fact_checker', 'user')
    
    if not url:
        return jsonify({'error': 'URL обязателен'}), 400
    
    new_score = reputation_system.update_trust_score(url, is_fake, fact_checker)
    
    return jsonify({
        'message': 'Репутация источника обновлена',
        'new_trust_score': new_score,
        'domain': reputation_system.extract_domain(url)
    })

@app.route('/stats/<path:url>')
def get_stats(url):
    """Получение статистики по источнику"""
    stats = reputation_system.get_source_stats(url)
    return jsonify(stats if stats else {'error': 'Источник не найден'})

def combine_assessments(ai_result, trust_score):
    """Комбинирование оценки AI и репутации источника"""
    ai_confidence = ai_result['confidence']
    ai_prediction = ai_result['prediction']
    
    # Весовые коэффициенты
    ai_weight = 0.7
    trust_weight = 0.3
    
    # Нормализация балла доверия (0-1)
    trust_normalized = trust_score / 100.0
    
    if ai_prediction == 'fake':
        ai_score = ai_result['fake_prob']
    else:
        ai_score = ai_result['real_prob']
    
    # Комбинированная оценка
    combined_score = (ai_score * ai_weight) + (trust_normalized * trust_weight)
    
    # Определение финального вердикта
    if combined_score >= 0.7:
        return 'highly_trustworthy'
    elif combined_score >= 0.5:
        return 'likely_real'
    elif combined_score >= 0.3:
        return 'suspicious'
    else:
        return 'likely_fake'

def generate_recommendation(verdict, trust_score):
    """Генерация рекомендации для пользователя"""
    recommendations = {
        'highly_trustworthy': '✅ Новость выглядит достоверной. Источник имеет хорошую репутацию.',
        'likely_real': '✅ Новость вероятно достоверна. Рекомендуется проверка в других источниках.',
        'suspicious': '⚠️ Новость вызывает подозрения. Рекомендуется осторожность и дополнительная проверка.',
        'likely_fake': '❌ Высокая вероятность фейковой новости. Не рекомендуем делиться этой информацией.'
    }
    
    base_recommendation = recommendations.get(verdict, 'Требуется дополнительный анализ.')
    
    if trust_score < 30:
        base_recommendation += ' ⚠️ Источник имеет низкую репутацию.'
    elif trust_score > 80:
        base_recommendation += ' 👍 Источник надежный.'
    
    return base_recommendation

if __name__ == '__main__':
    # Создаем папку для моделей
    import os
    os.makedirs('models', exist_ok=True)
    
    app.run(debug=True, port=5000)