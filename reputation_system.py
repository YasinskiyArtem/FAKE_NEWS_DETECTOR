import sqlite3
import hashlib
from urllib.parse import urlparse

class ReputationSystem:
    def __init__(self, db_path='reputation.db'):
        self.db_path = db_path
        self.trusted_domains = {
            'gazeta.ru': 75, 'ria.ru': 80, 'tass.ru': 85, 'rbc.ru': 70,
            'lenta.ru': 65, 'kommersant.ru': 80, 'interfax.ru': 75,
            'vedomosti.ru': 75, 'rt.com': 60, 'bbc.com': 85
        }
        self.init_database()
    
    def init_database(self):
        """Инициализация базы данных"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sources (
                domain TEXT PRIMARY KEY,
                trust_score REAL DEFAULT 50.0,
                total_checks INTEGER DEFAULT 0,
                fake_flags INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Добавление известных источников
        for domain, score in self.trusted_domains.items():
            cursor.execute('''
                INSERT OR REPLACE INTO sources (domain, trust_score) 
                VALUES (?, ?)
            ''', (domain, score))
        
        conn.commit()
        conn.close()
    
    def extract_domain(self, url):
        """Извлечение домена из URL"""
        try:
            if not url.startswith(('http://', 'https://')):
                url = 'https://' + url
            return urlparse(url).netloc.lower()
        except:
            return "unknown"
    
    def get_trust_score(self, url):
        """Получение балла доверия источника"""
        domain = self.extract_domain(url)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT trust_score FROM sources WHERE domain = ?', 
            (domain,)
        )
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return result[0]
        else:
            # Новый источник
            base_score = self.calculate_base_score(domain)
            self.add_source(domain, base_score)
            return base_score
    
    def calculate_base_score(self, domain):
        """Расчет базового балла по домену"""
        if domain.endswith(('.gov', '.gov.ru')): return 80
        elif domain.endswith(('.edu', '.ac.ru')): return 75
        elif any(trusted in domain for trusted in ['bbc', 'reuters', 'apnews']): return 85
        elif domain.endswith('.ru'): return 60
        else: return 50
    
    def add_source(self, domain, score=50):
        """Добавление нового источника"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            'INSERT OR REPLACE INTO sources (domain, trust_score) VALUES (?, ?)',
            (domain, score)
        )
        conn.commit()
        conn.close()
    
    def update_trust_score(self, url, is_fake):
        """Обновление балла доверия"""
        domain = self.extract_domain(url)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            'SELECT trust_score, total_checks, fake_flags FROM sources WHERE domain = ?',
            (domain,)
        )
        result = cursor.fetchone()
        
        if not result:
            current_score, total_checks, fake_flags = 50, 0, 0
        else:
            current_score, total_checks, fake_flags = result
        
        # Обновление счетчиков
        total_checks += 1
        if is_fake:
            fake_flags += 1
            penalty = -15  # Штраф за фейк
        else:
            penalty = 5    # Поощрение за достоверность
        
        new_score = max(0, min(100, current_score + penalty))
        
        cursor.execute('''
            UPDATE sources 
            SET trust_score = ?, total_checks = ?, fake_flags = ?, last_updated = CURRENT_TIMESTAMP
            WHERE domain = ?
        ''', (new_score, total_checks, fake_flags, domain))
        
        conn.commit()
        conn.close()
        
        return new_score
    
    def get_source_info(self, url):
        """Получение информации об источнике"""
        domain = self.extract_domain(url)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT domain, trust_score, total_checks, fake_flags, last_updated
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