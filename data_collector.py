import pandas as pd
import requests
import time
from datetime import datetime, timedelta
import os
import re
from bs4 import BeautifulSoup
import random
import json

class ImprovedDataCollector:
    def __init__(self):
        self.collected_data = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        
    def dataset_exists(self, filename='data/russian_news_dataset_improved.csv'):
        return os.path.exists(filename)
        
    def collect_real_news(self, news_per_source=100):
        """Сбор реальных новостей из проверенных источников"""
        rss_feeds = {
            "ria": "https://ria.ru/export/rss2/index.xml",
            "tass": "https://tass.ru/rss/v2.xml", 
            "lenta": "https://lenta.ru/rss/news",
            "rbc": "https://rssexport.rbc.ru/rbcnews/news/30/full.rss",
            "kommersant": "https://www.kommersant.ru/RSS/news.xml",
            "rg": "https://rg.ru/xml/index.xml",
            "interfax": "https://www.interfax.ru/rss.asp"
        }
        
        real_news = []
        
        for source, url in rss_feeds.items():
            try:
                print(f"📡 Сбор реальных новостей с {source}...")
                response = self.session.get(url, timeout=20)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, 'xml')
                    items = soup.find_all('item')[:news_per_source]
                    
                    for item in items:
                        title = item.title.text.strip() if item.title else ""
                        description = item.description.text.strip() if item.description else ""
                        pub_date = item.pubDate.text if item.pubDate else datetime.now().strftime("%Y-%m-%d")
                        
                        if title and len(title) > 15:
                            text = f"{title}. {description}" if description else title
                            
                            real_news.append({
                                "title": title,
                                "text": text,
                                "is_fake": 0,
                                "source": source,
                                "date": pub_date,
                                "category": "real"
                            })
                    
                    print(f"✅ {source}: собрано {len(items)} новостей")
                    time.sleep(1)
                    
            except Exception as e:
                print(f"❌ Ошибка при сборе с {source}: {e}")
        
        return real_news
    
    def collect_fake_news_examples(self):
        """Сбор примеров фейковых новостей из разоблачающих ресурсов"""
        fake_news_examples = []
        
        # Источники разоблачений фейков
        factcheck_sources = [
            # Пастилы с разоблачениями (примеры)
            self.get_fake_news_from_factcheck()
        ]
        
        # База известных фейков
        known_fakes = self.get_known_fake_news()
        fake_news_examples.extend(known_fakes)
        
        return fake_news_examples
    
    def get_known_fake_news(self):
        """База известных фейковых новостей, которые были разоблачены"""
        known_fakes = [
            {
                "title": "Путин подписал указ о раздаче квартир всем россиянам",
                "text": "Владимир Путин подписал указ о предоставлении бесплатного жилья каждому гражданину России. По словам источников, программа начнет действовать с следующего месяца.",
                "is_fake": 1,
                "source": "known_fake",
                "date": "2024-01-15",
                "category": "politics"
            },
            {
                "title": "Ученые доказали, что COVID-19 создан в лаборатории для сокращения населения",
                "text": "Международная группа ученых опубликовала доклад, доказывающий искусственное происхождение коронавируса. По их данным, вирус был создан для контроля численности населения Земли.",
                "is_fake": 1,
                "source": "known_fake", 
                "date": "2024-02-20",
                "category": "health"
            },
            {
                "title": "Россия начинает добычу полезных ископаемых на Луне",
                "text": "Роскосмос запустил секретную миссию по добыче гелия-3 на Луне. Первая партия ценного ресурса уже доставлена на Землю.",
                "is_fake": 1,
                "source": "known_fake",
                "date": "2024-03-10",
                "category": "science"
            }
        ]
        
        return known_fakes
    
    def get_fake_news_from_factcheck(self):
        """Парсинг сайтов-фактчекеров для сбора разоблаченных фейков"""
        # Здесь можно добавить парсинг реальных фактчек-сайтов
        # Например: stopfake.org, factcheck.org и др.
        
        # Временные примеры
        factcheck_examples = [
            {
                "title": "Вакцины от COVID-19 содержат микрочипы для слежки",
                "text": "Новое исследование показывает, что в состав вакцин включены наночипы, позволяющие отслеживать перемещения людей. Фармацевтические компании скрывают эту информацию.",
                "is_fake": 1,
                "source": "factcheck",
                "date": "2024-01-20",
                "category": "health"
            },
            {
                "title": "5G сети вызывают рак и распространяют вирусы",
                "text": "Врачи бьют тревогу: излучение вышек 5G приводит к онкологическим заболеваниям и ослабляет иммунитет. В Европе уже зафиксированы случаи массовых заболеваний.",
                "is_fake": 1, 
                "source": "factcheck",
                "date": "2024-02-15",
                "category": "technology"
            }
        ]
        
        return factcheck_examples
    
    def generate_plausible_fake_news(self, real_news_samples, count=1000):
        """Генерация правдоподобных фейков на основе реальных новостей"""
        fake_news = []
        
        for _ in range(count):
            # Берем реальную новость за основу
            real_sample = random.choice(real_news_samples)
            
            # Создаем фейковую версию
            fake_title = self.modify_title(real_sample['title'])
            fake_text = self.modify_text(real_sample['text'])
            
            fake_news.append({
                "title": fake_title,
                "text": fake_text,
                "is_fake": 1,
                "source": "generated_plausible",
                "date": real_sample['date'],
                "category": real_sample.get('category', 'general')
            })
        
        return fake_news
    
    def modify_title(self, title):
        """Модификация заголовка для создания фейка"""
        modifications = [
            lambda t: t.replace("заявил", "шокирующе признался"),
            lambda t: t.replace("сообщил", "тайно рассказал"),
            lambda t: t + " - РАЗОБЛАЧЕНИЕ",
            lambda t: "СКАНДАЛ: " + t,
            lambda t: "СЕНСАЦИЯ: " + t,
            lambda t: t.replace("обсудили", "скрыли правду о"),
            lambda t: re.sub(r'(\w+)\s(\w+)', r'\1 СЕКРЕТНО \2', t)
        ]
        
        modified = random.choice(modifications)(title)
        return modified
    
    def modify_text(self, text):
        """Модификация текста для создания фейка"""
        # Добавляем маркеры фейковости более тонко
        fake_phrases = [
            "По данным анонимных источников, ",
            "Эксперты, пожелавшие остаться неизвестными, утверждают, что ",
            "Внутренние документы, попавшие в наше распоряжение, свидетельствуют: ",
            "Бывший сотрудник организации, рискнувший рассказать правду, сообщает: ",
            "Расследование, проведенное нашими журналистами, выявило: "
        ]
        
        # Разбиваем текст на предложения
        sentences = re.split(r'[.!?]', text)
        if len(sentences) > 1:
            # Заменяем первое предложение
            sentences[0] = random.choice(fake_phrases) + sentences[0].lower()
            
            # Добавляем "сенсационное" окончание
            sensational_endings = [
                " Эта информация тщательно скрывается властями.",
                " Источник утверждает, что за правду могут жестоко наказать.",
                " Официальные лица отрицают эти факты, но доказательства неопровержимы.",
                " Читатели должны знать правду, пока её не удалили из интернета."
            ]
            
            sentences.append(random.choice(sensational_endings))
            
            return '. '.join([s for s in sentences if s.strip()])
        
        return text
    
    def augment_dataset(self, news_list, target_count, is_fake=0):
        """Аугментация данных с сохранением стиля"""
        if len(news_list) >= target_count:
            return news_list[:target_count]
            
        augmented = news_list.copy()
        
        while len(augmented) < target_count:
            original = random.choice(news_list)
            
            # Создаем вариации через синонимизацию
            augmented_text = self.synonymize_text(original['text'])
            augmented_title = self.synonymize_text(original['title'])
            
            augmented.append({
                "title": augmented_title,
                "text": augmented_text,
                "is_fake": is_fake,
                "source": f"{original['source']}_augmented",
                "date": original['date'],
                "category": original.get('category', 'general')
            })
        
        return augmented
    
    def synonymize_text(self, text):
        """Замена слов на синонимы для аугментации"""
        synonyms = {
            'новость': ['сообщение', 'информация', 'известие', 'сводка'],
            'заявил': ['сообщил', 'отметил', 'подчеркнул', 'декларировал'],
            'российский': ['русский', 'россия', 'нашей страны', 'отечественный'],
            'мировой': ['глобальный', 'международный', 'всемирный', 'планетарный'],
            'экономика': ['хозяйство', 'финансы', 'экономическая система'],
            'политика': ['госуправление', 'власть', 'политическая система']
        }
        
        words = text.split()
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word.lower())
            if clean_word in synonyms:
                words[i] = word.replace(clean_word, random.choice(synonyms[clean_word]))
        
        return ' '.join(words)
    
    def create_balanced_dataset(self, target_size=10000):
        """Создание сбалансированного и качественного датасета"""
        print("🔄 Создание улучшенного датасета...")
        
        # Сбор реальных новостей
        real_news = self.collect_real_news(news_per_source=150)
        print(f"📊 Собрано реальных новостей: {len(real_news)}")
        
        # Сбор примеров фейков
        fake_examples = self.collect_fake_news_examples()
        print(f"📊 Собрано примеров фейков: {len(fake_examples)}")
        
        # Генерация правдоподобных фейков на основе реальных новостей
        plausible_fakes = self.generate_plausible_fake_news(real_news, 2000)
        print(f"📊 Сгенерировано правдоподобных фейков: {len(plausible_fakes)}")
        
        # Аугментация данных
        target_real = target_size // 2
        target_fake = target_size // 2
        
        real_augmented = self.augment_dataset(real_news, target_real, 0)
        fake_combined = fake_examples + plausible_fakes
        fake_augmented = self.augment_dataset(fake_combined, target_fake, 1)
        
        print(f"📊 После аугментации - Реальные: {len(real_augmented)}, Фейковые: {len(fake_augmented)}")
        
        # Создание финального датасета
        real_df = pd.DataFrame(real_augmented)
        fake_df = pd.DataFrame(fake_augmented)
        
        combined_df = pd.concat([real_df, fake_df], ignore_index=True)
        combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        # Сохранение
        os.makedirs('data', exist_ok=True)
        filename = 'data/russian_news_dataset_improved.csv'
        combined_df.to_csv(filename, index=False, encoding='utf-8')
        
        # Сохранение метаданных
        metadata = {
            "created_date": datetime.now().isoformat(),
            "total_samples": len(combined_df),
            "real_samples": len(real_df),
            "fake_samples": len(fake_df),
            "sources": list(set(combined_df['source'].tolist()))
        }
        
        with open('data/dataset_metadata.json', 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Улучшенный датасет создан! Размер: {len(combined_df)} записей")
        print(f"📁 Сохранен в: {filename}")
        
        # Статистика
        self.print_dataset_stats(combined_df)
        
        return combined_df
    
    def print_dataset_stats(self, df):
        """Печать статистики датасета"""
        print("\n📊 Детальная статистика датасета:")
        print(f"Всего записей: {len(df)}")
        print(f"Реальные новости: {len(df[df['is_fake'] == 0])}")
        print(f"Фейковые новости: {len(df[df['is_fake'] == 1])}")
        
        print("\n📋 Распределение по источникам:")
        source_stats = df['source'].value_counts()
        for source, count in source_stats.items():
            print(f"  {source}: {count}")
        
        print("\n🎯 Качество данных:")
        avg_title_len = df['title'].str.len().mean()
        avg_text_len = df['text'].str.len().mean()
        print(f"Средняя длина заголовка: {avg_title_len:.1f} символов")
        print(f"Средняя длина текста: {avg_text_len:.1f} символов")

# Дополнительная функция для проверки качества
def analyze_dataset_quality(df):
    """Анализ качества собранного датасета"""
    print("\n🔍 Анализ качества датасета:")
    
    # Проверка на дубликаты
    duplicates = df.duplicated(subset=['title', 'text']).sum()
    print(f"Дубликаты: {duplicates} ({duplicates/len(df)*100:.1f}%)")
    
    # Проверка распределения длин
    title_lengths = df['title'].str.len()
    text_lengths = df['text'].str.len()
    
    print(f"Заголовки: мин={title_lengths.min()}, макс={title_lengths.max()}, среднее={title_lengths.mean():.1f}")
    print(f"Тексты: мин={text_lengths.min()}, макс={text_lengths.max()}, среднее={text_lengths.mean():.1f}")
    
    # Проверка баланса
    real_ratio = len(df[df['is_fake'] == 0]) / len(df)
    print(f"Баланс классов: реальные={real_ratio*100:.1f}%, фейковые={(1-real_ratio)*100:.1f}%")

# Использование
if __name__ == "__main__":
    collector = ImprovedDataCollector()
    
    if not collector.dataset_exists():
        dataset = collector.create_balanced_dataset(target_size=10000)
        analyze_dataset_quality(dataset)
    else:
        print("✅ Датасет уже существует!")
        dataset = pd.read_csv('data/russian_news_dataset_improved.csv')
        analyze_dataset_quality(dataset)