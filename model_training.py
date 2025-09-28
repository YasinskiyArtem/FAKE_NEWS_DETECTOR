import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

class CNNLSTMModel:
    def __init__(self, max_vocab_size=10000, max_sequence_length=300, embedding_dim=100):
        self.max_vocab_size = max_vocab_size
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim
        self.tokenizer = None
        self.model = None
        
    def create_model(self, vocab_size):
        """Создание гибридной модели CNN + LSTM"""
        model = Sequential([
            # Embedding слой
            Embedding(input_dim=vocab_size, 
                     output_dim=self.embedding_dim, 
                     input_length=self.max_sequence_length,
                     name='embedding'),
            
            # Первый сверточный блок
            Conv1D(128, 5, activation='relu', padding='same', name='conv1d_1'),
            MaxPooling1D(2, name='maxpool_1'),
            
            # Второй сверточный блок
            Conv1D(64, 3, activation='relu', padding='same', name='conv1d_2'),
            MaxPooling1D(2, name='maxpool_2'),
            
            # LSTM слой
            LSTM(64, dropout=0.2, recurrent_dropout=0.2, name='lstm'),
            
            # Полносвязные слои
            Dense(64, activation='relu', name='dense_1'),
            Dropout(0.5, name='dropout_1'),
            
            Dense(32, activation='relu', name='dense_2'),
            Dropout(0.3, name='dropout_2'),
            
            # Выходной слой
            Dense(1, activation='sigmoid', name='output')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return model
    
    def prepare_data(self, texts, labels):
        """Подготовка данных для обучения"""
        # Токенизация
        self.tokenizer = Tokenizer(num_words=self.max_vocab_size, oov_token='<OOV>')
        self.tokenizer.fit_on_texts(texts)
        
        # Преобразование в последовательности
        sequences = self.tokenizer.texts_to_sequences(texts)
        
        # Паддинг
        X = pad_sequences(sequences, maxlen=self.max_sequence_length, 
                         padding='post', truncating='post')
        y = np.array(labels)
        
        return X, y
    
    def train(self, texts, labels, epochs=15, batch_size=32, validation_split=0.2):
        """Обучение модели"""
        # Подготовка данных
        X, y = self.prepare_data(texts, labels)
        
        # Разделение на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Создание модели
        vocab_size = min(self.max_vocab_size, len(self.tokenizer.word_index) + 1)
        self.model = self.create_model(vocab_size)
        
        print("Архитектура модели:")
        self.model.summary()
        
        # Callback для ранней остановки
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True,
            verbose=1
        )
        
        # Обучение
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[early_stopping],
            verbose=1
        )
        
        return history, (X_test, y_test)
    
    def evaluate(self, X_test, y_test):
        """Оценка модели"""
        if self.model is None:
            raise ValueError("Модель не обучена!")
        
        # Базовая оценка
        test_loss, test_accuracy, test_precision, test_recall = self.model.evaluate(
            X_test, y_test, verbose=0
        )
        
        # Детальная оценка
        y_pred = (self.model.predict(X_test) > 0.5).astype("int32")
        
        print("\n" + "="*50)
        print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
        print("="*50)
        print(f"Точность (Accuracy): {test_accuracy:.4f}")
        print(f"Потери (Loss): {test_loss:.4f}")
        print(f"Precision: {test_precision:.4f}")
        print(f"Recall: {test_recall:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Матрица ошибок
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Матрица ошибок')
        plt.ylabel('Истинные значения')
        plt.xlabel('Предсказанные значения')
        plt.show()
        
        return test_accuracy
    
    def save_model(self, model_path='cnn_lstm_model.h5', tokenizer_path='tokenizer.pkl'):
        """Сохранение модели и токенизатора"""
        if self.model is not None:
            self.model.save(model_path)
            print(f"Модель сохранена как {model_path}")
        
        if self.tokenizer is not None:
            with open(tokenizer_path, 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Токенизатор сохранен как {tokenizer_path}")
    
    def load_model(self, model_path='cnn_lstm_model.h5', tokenizer_path='tokenizer.pkl'):
        """Загрузка модели и токенизатора"""
        self.model = tf.keras.models.load_model(model_path)
        
        with open(tokenizer_path, 'rb') as handle:
            self.tokenizer = pickle.load(handle)
        
        print("Модель и токенизатор загружены успешно!")