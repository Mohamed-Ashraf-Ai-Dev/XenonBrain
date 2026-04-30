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
    محرك XenonBrain المتطور (V4.5):
    1. مصادر بيانات متعددة (Tech, Finance, Global News).
    2. تقسيم منطقي للبيانات (Logic Partitioning).
    3. ذاكرة تاريخية متكاملة.
    """
    def __init__(self, raw_data_path="data/raw"):
        self.raw_data_path = raw_data_path
        self.history_file = "HISTORY.json"
        os.makedirs(raw_data_path, exist_ok=True)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')

    def fetch_all_sources(self):
        print("جاري جلب البيانات من المصادر العالمية المتعددة...")
        sources = {
            "tech": [
                "https://news.mit.edu/rss/topic/artificial-intelligence2",
                "https://techcrunch.com/feed/",
                "https://www.wired.com/feed/category/science/latest/rss",
                "https://thenextweb.com/feed"
            ],
            "finance": [
                "https://www.ft.com/?format=rss",
                "https://www.economist.com/business/rss.xml"
            ]
        }
        
        all_news = {"tech": [], "finance": []}
        for category, urls in sources.items():
            for url in urls:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:5]:
                        all_news[category].append(entry.title)
                except: continue
        return all_news

    def fetch_market_indicators(self):
        print("جاري جلب المؤشرات المالية العالمية...")
        tickers = ["^GSPC", "^IXIC", "BTC-USD"] # S&P 500, Nasdaq, Bitcoin
        market_data = {}
        for t in tickers:
            try:
                data = yf.download(t, period="3mo", interval="1d", progress=False)
                market_data[t] = data[['Close']].pct_change().dropna().values[-20:] # آخر 20 يوم
            except: continue
        return market_data

    def update_history(self, news_summary, prediction):
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except: pass
        
        history.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": news_summary[:100],
            "prediction": int(prediction)
        })
        history = history[-100:] # زيادة سعة الذاكرة لـ 100 سجل
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

    def prepare_data(self):
        news = self.fetch_all_sources()
        markets = self.fetch_market_indicators()
        
        # تحويل الأخبار لـ Embeddings وتقسيمها
        tech_emb = np.mean(self.text_model.encode(news['tech']), axis=0) if news['tech'] else np.zeros(384)
        fin_emb = np.mean(self.text_model.encode(news['finance']), axis=0) if news['finance'] else np.zeros(384)
        
        # دمج أخبار التقنية والمالية (متوسط)
        combined_news_emb = (tech_emb + fin_emb) / 2
        
        # استخدام مؤشر S&P 500 كمرجع أساسي للتدريب
        main_market = markets.get("^GSPC", np.zeros((20, 1)))
        
        seq_len = 5
        memory_val = 0.5
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    h = json.load(f)
                    if h: memory_val = np.mean([i['prediction'] for i in h])
            except: pass

        X, y = [], []
        for i in range(len(main_market) - seq_len):
            market_seq = main_market[i:i+seq_len]
            combined_seq = []
            for m_step in market_seq:
                # 384 (News) + 1 (Market) + 1 (Memory) + 4 (Padding for model dim 390)
                combined_step = np.concatenate([combined_news_emb, m_step, [memory_val], [0,0,0,0]])
                combined_seq.append(combined_step)
            
            X.append(combined_seq)
            target = 1 if main_market[i+seq_len][0] > 0 else 0
            y.append(target)
            
        return np.array(X), np.array(y), news['tech'][0] if news['tech'] else "Global Update"

def get_dataloader(batch_size=8):
    collector = RealDataCollector()
    try:
        X, y, _ = collector.prepare_data()
        if len(X) == 0: raise ValueError("No Data")
    except:
        X = np.random.randn(10, 5, 390)
        y = np.random.randint(0, 2, 10)
    
    y_one_hot = np.eye(2)[y]
    dataset = XenonDataset(X, y_one_hot)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
