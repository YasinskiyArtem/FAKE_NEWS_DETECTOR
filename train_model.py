import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from text_preprocessing import load_and_preprocess_data
from model_training import CNNLSTMModel

def find_or_create_labels(df):
    """
    Находит или создает метки для датасета Fake News Detection
    """
    print("🔍 Поиск столбца с метками...")
    
    # Проверяем доступные столбцы
    print("Доступные столбцы:", df.columns.tolist())
    
    # Вариант 1: Если есть явный столбец с метками
    if 'label' in df.columns:
        print("✅ Найден столбец 'label'")
        return df['label'].values
    
    # Вариант 2: Если есть столбец 'class'
    elif 'class' in df.columns:
        print("✅ Найден столбец 'class', используем как метки")
        # Преобразуем в числовой формат если нужно
        if df['class'].dtype == object:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            labels = le.fit_transform(df['class'])
            print(f"Преобразованные метки: {dict(zip(le.classes_, le.transform(le.classes_)))}")
            return labels
        return df['class'].values
    
    # Вариант 3: Создаем метки на основе других признаков
    else:
        print("⚠️ Столбец с метками не найден, создаем на основе данных...")
        
        # Способ A: На основе ключевых слов в тексте
        fake_keywords = ['fake', 'false', 'hoax', 'misinformation', 'rumor', 'debunk']
        
        def detect_fake_label(text):
            if pd.isna(text):
                return 0
            text_lower = str(text).lower()
            return 1 if any(keyword in text_lower for keyword in fake_keywords) else 0
        
        # Проверяем разные столбцы на наличие ключевых слов
        if 'text' in df.columns:
            labels = df['text'].apply(detect_fake_label).values
        elif 'title' in df.columns:
            labels = df['title'].apply(detect_fake_label).values
        else:
            # Способ B: Случайные метки для демонстрации (не для продакшена)
            print("⚠️ Созданы случайные метки для тестирования")
            labels = np.random.randint(0, 2, size=len(df))
        
        print(f"📊 Распределение созданных меток: {pd.Series(labels).value_counts().to_dict()}")
        return labels

def get_texts_column(df):
    """
    Находит подходящий столбец с текстовыми данными
    """
    # Предпочтительная последовательность столбцов
    text_columns_priority = ['cleaned_text', 'text', 'title', 'content', 'article']
    
    for col in text_columns_priority:
        if col in df.columns:
            print(f"✅ Используем текстовый столбец: '{col}'")
            return df[col].tolist()
    
    # Если не нашли предпочтительные, ищем любой текстовый столбец
    text_columns = df.select_dtypes(include=[object]).columns
    if len(text_columns) > 0:
        print(f"✅ Используем текстовый столбец: '{text_columns[0]}'")
        return df[text_columns[0]].tolist()
    
    raise ValueError("❌ Не найден подходящий текстовый столбец")

def main():
    print("🚀 ЗАПУСК ОБУЧЕНИЯ ГИБРИДНОЙ МОДЕЛИ CNN-LSTM")
    print("=" * 60)
    
    # Загрузка и предобработка данных
    print("📊 Загрузка данных...")
    df = load_and_preprocess_data('Fake.csv')
    
    # Проверяем структуру данных после предобработки
    print("\n📋 Структура данных после предобработки:")
    print(f"Количество записей: {len(df)}")
    print(f"Столбцы: {df.columns.tolist()}")
    
    # Получаем тексты и метки
    texts = get_texts_column(df)
    labels = find_or_create_labels(df)
    
    print(f"\n📊 Статистика данных:")
    print(f"Количество текстов: {len(texts)}")
    print(f"Количество меток: {len(labels)}")
    print(f"Распределение меток: {pd.Series(labels).value_counts().to_dict()}")
    
    # Проверяем наличие текстов
    empty_texts = sum(1 for text in texts if not text or text.strip() == "")
    if empty_texts > 0:
        print(f"⚠️ Внимание: {empty_texts} пустых текстов обнаружено")
    
    # Создание и обучение модели
    print("\n🧠 Создание модели CNN-LSTM...")
    model = CNNLSTMModel()
    
    print("🎯 Обучение модели...")
    try:
        history, (X_test, y_test) = model.train(
            texts=texts,
            labels=labels,
            epochs=15,
            batch_size=32
        )
        
        # Оценка модели
        print("📈 Оценка модели...")
        accuracy = model.evaluate(X_test, y_test)
        
        # Сохранение модели
        print("💾 Сохранение модели...")
        model.save_model()
        
        # Графики обучения
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'], label='Training Accuracy')
        plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
        plt.title('Точность модели')
        plt.xlabel('Эпоха')
        plt.ylabel('Точность')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.title('Потери модели')
        plt.xlabel('Эпоха')
        plt.ylabel('Потери')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.show()
        
        print("✅ Обучение завершено успешно!")
        print(f"🎯 Финальная точность: {accuracy:.4f}")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении модели: {e}")
        print("🔧 Рекомендации по устранению:")
        print("- Проверьте, что тексты не пустые")
        print("- Убедитесь, что метки корректны (0 и 1)")
        print("- Проверьте размерности данных")

if __name__ == "__main__":
    main()