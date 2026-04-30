import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import requests
import feedparser
import yfinance as yf
import os
import json
from sentence_transformers import SentenceTransformer
from datetime import datetime, timedelta

class XenonDataset(Dataset):
    def __init__(self, data, targets):
        self.data = torch.FloatTensor(data)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class RealDataCollector:
    """
    محرك XenonBrain المتطور (V3):
    1. ذاكرة طويلة الأمد (Long-Term Memory): يحفظ الأنماط السابقة في HISTORY.json.
    2. تحليل السياق المتقدم: يربط الأخبار العالمية بحركة السوق اللحظية.
    """
    def __init__(self, raw_data_path="data/raw"):
        self.raw_data_path = raw_data_path
        self.history_file = "HISTORY.json"
        os.makedirs(raw_data_path, exist_ok=True)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')

    def fetch_tech_news(self):
        print("جاري جلب أخبار التقنية الحقيقية...")
        feeds = [
            "https://news.mit.edu/rss/topic/artificial-intelligence2",
            "https://www.theverge.com/rss/index.xml",
            "https://techcrunch.com/feed/"
        ]
        news_items = []
        for url in feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    news_items.append(entry.title)
            except: continue
        return news_items

    def fetch_market_data(self):
        print("جاري جلب بيانات السوق الحقيقية...")
        ticker = "^GSPC"
        data = yf.download(ticker, period="3mo", interval="1d", progress=False) # زيادة المدى لـ 3 أشهر
        market_features = data[['Open', 'High', 'Low', 'Close', 'Volume']].pct_change().dropna().values
        return market_features

    def update_history(self, news_summary, prediction):
        """تحديث ذاكرة النظام"""
        history = []
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r', encoding='utf-8') as f:
                history = json.load(f)
        
        history.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": news_summary[:100] + "...",
            "prediction": int(prediction)
        })
        
        # الاحتفاظ بآخر 50 سجل فقط للذاكرة النشطة
        history = history[-50:]
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

    def prepare_real_world_batch(self):
        news = self.fetch_tech_news()
        market = self.fetch_market_data()
        
        news_embeddings = self.text_model.encode(news)
        avg_news_emb = np.mean(news_embeddings, axis=0)
        
        seq_len = 5
        # إضافة ميزة الذاكرة (Memory Feature): متوسط التوقعات السابقة إذا وجدت
        memory_val = 0.5
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                h = json.load(f)
                if len(h) > 0:
                    memory_val = np.mean([i['prediction'] for i in h])

        X, y = [], []
        for i in range(len(market) - seq_len):
            market_seq = market[i:i+seq_len]
            combined_seq = []
            for m_step in market_seq:
                # دمج (الأخبار + السوق + الذاكرة)
                combined_step = np.concatenate([avg_news_emb, m_step, [memory_val]])
                combined_seq.append(combined_step)
            
            X.append(combined_seq)
            target = 1 if market[i+seq_len][3] > 0 else 0
            y.append(target)
            
        return np.array(X), np.array(y), news[0] if news else "No News"

def get_dataloader(batch_size=8):
    collector = RealDataCollector()
    try:
        X, y, latest_news = collector.prepare_real_world_batch()
        if len(X) == 0: raise ValueError("Insufficient data")
    except Exception as e:
        print(f"Error: {e}. Using fallback data.")
        X = np.random.randn(10, 5, 390) # 384 + 5 + 1 (Memory)
        y = np.random.randint(0, 2, 10)
    
    y_one_hot = np.eye(2)[y]
    dataset = XenonDataset(X, y_one_hot)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
