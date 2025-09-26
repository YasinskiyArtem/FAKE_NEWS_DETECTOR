import pandas as pd
import requests
import time
from datetime import datetime
import os
import re
from bs4 import BeautifulSoup

class DataCollector:
    def __init__(self):
        self.collected_data = []
        
    def collect_news_from_rss(self):
        """Сбор реальных новостей через RSS"""
        rss_feeds = {
            "ria": "https://ria.ru/export/rss2/index.xml",
            "tass": "https://tass.ru/rss/v2.xml", 
            "lenta": "https://lenta.ru/rss/news",
            "rbc": "https://rssexport.rbc.ru/rbcnews/news/30/full.rss"
        }
        
        real_news = []
        
        for source, url in rss_feeds.items():
            try:
                print(f"📡 Сбор новостей с {source}...")
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'xml')
                    
                    for item in soup.find_all('item')[:15]:  # Берем 15 последних
                        title = item.title.text if item.title else ""
                        description = item.description.text if item.description else ""
                        
                        if title and description:
                            real_news.append({
                                "title": title,
                                "text": f"{title}. {description}",
                                "is_fake": 0,
                                "source": source,
                                "date": datetime.now().strftime("%Y-%m-%d")
                            })
                    
                    time.sleep(1)  # Пауза между запросами
                    
            except Exception as e:
                print(f"❌ Ошибка при сборе с {source}: {e}")
        
        return real_news
    
    def generate_fake_news(self, count=100):
        """Генерация фейковых новостей"""
        fake_templates = [
            "ШОК! Ученые обнаружили, что {product} вызывает {disease}",
            "СРОЧНО! {politician} попал в больницу с {diagnosis}",
            "ВРАЧИ В УЖАСЕ! Новое исследование о {topic}",
            "СКАНДАЛ! {organization} скрывает правду о {secret}",
            "ЭКСКЛЮЗИВ: Раскрыта тайна {mystery}"
        ]
        
        products = ["сахар", "глютен", "5G", "вакцины", "ГМО"]
        diseases = ["рак", "диабет", "аллергия", "бесплодие"]
        politicians = ["Путин", "Трамп", "Зеленский", "Макрон"]
        diagnoses = ["инфаркт", "инсульт", "COVID-19", "отравление"]
        topics = ["вакцинах", "5G", "излучении", "питании"]
        organizations = ["правительство", "фармацевты", "ученые", "врачи"]
        secrets = ["инопланетянах", "бессмертии", "лекарстве от рака"]
        mysteries = ["бермудского треугольника", "египетских пирамид", "снежного человека"]
        
        fake_news = []
        
        for i in range(count):
            template = fake_templates[i % len(fake_templates)]
            
            text = template.format(
                product=products[i % len(products)],
                disease=diseases[i % len(diseases)],
                politician=politicians[i % len(politicians)],
                diagnosis=diagnoses[i % len(diagnoses)],
                topic=topics[i % len(topics)],
                organization=organizations[i % len(organizations)],
                secret=secrets[i % len(secrets)],
                mystery=mysteries[i % len(mysteries)]
            )
            
            # Добавляем эмоциональное описание
            emotional_endings = [
                "Эта информация шокировала весь мир!",
                "Власти пытаются скрыть эту правду!",
                "Прочтите, пока эту новость не удалили!",
                "Эта тайна раскрыта эксклюзивно для наших читателей!"
            ]
            
            full_text = f"{text}. {emotional_endings[i % len(emotional_endings)]}"
            
            fake_news.append({
                "title": text,
                "text": full_text,
                "is_fake": 1,
                "source": "generated",
                "date": datetime.now().strftime("%Y-%m-%d")
            })
        
        return fake_news
    
    def create_dataset(self, min_size=200):
        """Создание сбалансированного датасета"""
        print("🔄 Создание датасета...")
        
        # Сбор реальных новостей
        real_news = self.collect_news_from_rss()
        print(f"📊 Собрано реальных новостей: {len(real_news)}")
        
        # Генерация фейковых новостей
        fake_news = self.generate_fake_news(max(min_size, len(real_news)))
        print(f"📊 Сгенерировано фейковых новостей: {len(fake_news)}")
        
        # Балансировка
        min_samples = min(len(real_news), len(fake_news), min_size)
        
        # Создание DataFrame
        real_df = pd.DataFrame(real_news[:min_samples])
        fake_df = pd.DataFrame(fake_news[:min_samples])
        
        combined_df = pd.concat([real_df, fake_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Сохранение
        os.makedirs('data', exist_ok=True)
        combined_df.to_csv('data/russian_news_dataset.csv', index=False, encoding='utf-8')
        
        print(f"✅ Датасет создан! Размер: {len(combined_df)} записей")
        return combined_df
    
    def dataset_exists(self):
        """Проверка существования датасета"""
        return os.path.exists('data/russian_news_dataset.csv')