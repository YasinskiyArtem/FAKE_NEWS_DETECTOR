import json
import sqlite3
from datetime import datetime
import hashlib

class ReputationSystem:
    def __init__(self, db_path='reputation.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Инициализация базы данных для хранения репутации"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                domain TEXT UNIQUE,
                initial_trust_score REAL DEFAULT 50.0,
                current_trust_score REAL DEFAULT 50.0,
                total_checks INTEGER DEFAULT 0,
                fake_flags INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS verifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_domain TEXT,
                verdict TEXT,
                fact_checker TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (source_domain) REFERENCES sources (domain)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def get_source_hash(self, url):
        """Создание хэша для идентификации источника"""
        domain = self.extract_domain(url)
        return hashlib.md5(domain.encode()).hexdigest()
    
    def extract_domain(self, url):
        """Извлечение домена из URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc
    
    def get_trust_score(self, url):
        """Получение текущего балла доверия источника"""
        domain = self.extract_domain(url)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT current_trust_score FROM sources WHERE domain = ?', 
            (domain,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        else:
            # Если источник новый, добавляем его с начальным баллом
            self.add_new_source(domain)
            return 50.0  # Средний начальный балл
    
    def add_new_source(self, domain):
        """Добавление нового источника в базу"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'INSERT OR IGNORE INTO sources (domain) VALUES (?)',
            (domain,)
        )
        conn.commit()
        conn.close()
    
    def update_trust_score(self, url, is_fake, fact_checker="user"):
        """Обновление балла доверия на основе верификации"""
        domain = self.extract_domain(url)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Получаем текущие данные
        cursor.execute(
            'SELECT current_trust_score, total_checks, fake_flags FROM sources WHERE domain = ?',
            (domain,)
        )
        result = cursor.fetchone()
        
        if not result:
            self.add_new_source(domain)
            current_score, total_checks, fake_flags = 50.0, 0, 0
        else:
            current_score, total_checks, fake_flags = result
        
        # Обновляем счетчики
        total_checks += 1
        if is_fake:
            fake_flags += 1
            penalty = -10  # Штраф за фейк
        else:
            penalty = 2    # Небольшое повышение за достоверность
        
        # Рассчитываем новый балл (с ограничениями 0-100)
        new_score = max(0, min(100, current_score + penalty))
        
        # Обновляем базу
        cursor.execute('''
            UPDATE sources 
            SET current_trust_score = ?, total_checks = ?, fake_flags = ?, last_updated = ?
            WHERE domain = ?
        ''', (new_score, total_checks, fake_flags, datetime.now(), domain))
        
        # Записываем верификацию
        verdict = "fake" if is_fake else "real"
        cursor.execute('''
            INSERT INTO verifications (source_domain, verdict, fact_checker)
            VALUES (?, ?, ?)
        ''', (domain, verdict, fact_checker))
        
        conn.commit()
        conn.close()
        
        return new_score
    
    def get_source_stats(self, url):
        """Получение статистики по источнику"""
        domain = self.extract_domain(url)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT domain, current_trust_score, total_checks, fake_flags, last_updated
            FROM sources WHERE domain = ?
        ''', (domain,))
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'domain': result[0],
                'trust_score': result[1],
                'total_checks': result[2],
                'fake_flags': result[3],
                'last_updated': result[4]
            }
        return None