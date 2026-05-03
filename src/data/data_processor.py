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
    محرك XenonBrain المطور (V6 Sovereign Intelligence):
    1. تحليل المشاعر وتقسيم البيانات (Logic Partitioning).
    2. الذاكرة التصحيحية بناءً على أداء السوق الحقيقي.
    3. إضافة مؤشرات فنية متقدمة (RSI, MACD) وتحليل الارتباط بين الأصول.
    """
    def __init__(self, raw_data_path="data/raw"):
        self.raw_data_path = raw_data_path
        self.history_file = "HISTORY.json"
        os.makedirs(raw_data_path, exist_ok=True)
        # استخدام نموذج خفيف لتحليل المشاعر وتحويل النصوص
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')

    def fetch_all_sources(self):
        print("جاري جلب البيانات من المصادر العالمية المتعددة (V6)...")
        sources = {
            "tech": [
                "https://news.mit.edu/rss/topic/artificial-intelligence2",
                "https://techcrunch.com/feed/",
                "https://www.wired.com/feed/category/science/latest/rss",
                "https://thenextweb.com/feed"
            ],
            "finance": [
                "https://www.ft.com/?format=rss",
                "https://www.economist.com/business/rss.xml",
                "https://cointelegraph.com/rss/tag/bitcoin"
            ]
        }
        
        all_news = {"tech": [], "finance": []}
        for category, urls in sources.items():
            for url in urls:
                try:
                    feed = feedparser.parse(url)
                    for entry in feed.entries[:5]:
                        text = entry.title + " " + getattr(entry, 'summary', '')
                        all_news[category].append(text[:200])
                except: continue
        return all_news

    def calculate_indicators(self, df):
        """حساب المؤشرات الفنية المتقدمة"""
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        return df.fillna(0)

    def fetch_market_indicators(self):
        print("جاري جلب المؤشرات المالية والعملات الرقمية مع التحليل الفني...")
        tickers = ["^GSPC", "^IXIC", "BTC-USD", "ETH-USD"] 
        market_data = {}
        for t in tickers:
            try:
                data = yf.download(t, period="2mo", interval="1d", progress=False)
                if not data.empty:
                    data = self.calculate_indicators(data)
                    # دمج السعر مع المؤشرات
                    features = data[['Close', 'RSI', 'MACD']].pct_change().dropna().values[-20:]
                    market_data[t] = features
            except: continue
        return market_data

    def update_history(self, news_summary, prediction, confidence=0.5, actual_outcome=None, portfolio_value=1000):
        """تحديث سجل الذاكرة التاريخية مع دعم المحفظة الافتراضية V6"""
        history = []
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    history = json.load(f)
            except: pass
        
        history.append({
            "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "summary": str(news_summary)[:150],
            "prediction": int(prediction),
            "confidence": float(confidence),
            "actual_outcome": int(actual_outcome) if actual_outcome is not None else None,
            "portfolio_value": float(portfolio_value)
        })
        
        history = history[-100:] 
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(history, f, ensure_ascii=False, indent=4)

    def prepare_data(self):
        news = self.fetch_all_sources()
        markets = self.fetch_market_indicators()
        
        tech_emb = np.mean(self.text_model.encode(news['tech']), axis=0) if news['tech'] else np.zeros(384)
        fin_emb = np.mean(self.text_model.encode(news['finance']), axis=0) if news['finance'] else np.zeros(384)
        combined_news_emb = (tech_emb + fin_emb) / 2
        
        # استخدام S&P 500 كمرجع أساسي مع المؤشرات الفنية
        main_market = markets.get("^GSPC", np.zeros((20, 3)))
        
        seq_len = 5
        memory_val = 0.5
        portfolio_val = 1000.0
        
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r') as f:
                    h = json.load(f)
                    if h:
                        memory_val = np.mean([i['prediction'] for i in h[-5:]])
                        portfolio_val = h[-1].get('portfolio_value', 1000.0)
            except: pass

        X, y = [], []
        if len(main_market) > seq_len:
            for i in range(len(main_market) - seq_len):
                market_seq = main_market[i:i+seq_len]
                combined_seq = []
                for m_step in market_seq:
                    # موازنة الأبعاد لتصل إلى 390
                    # m_step لديه 3 قيم (Close, RSI, MACD)
                    # news_emb لديه 384 قيمة
                    # memory_val و portfolio_val لديهما قيمتان
                    # الإجمالي: 384 + 3 + 1 + 1 + 1 (padding) = 390
                    combined_step = np.concatenate([combined_news_emb, m_step, [memory_val], [portfolio_val / 1000.0], [0]])
                    combined_seq.append(combined_step)
                
                X.append(combined_seq)
                target = 1 if main_market[i+seq_len][0] > 0 else 0
                y.append(target)
        
        if not X:
            X = np.random.randn(10, 5, 390)
            y = np.random.randint(0, 2, 10)
            
        return np.array(X), np.array(y), news['tech'][0] if news['tech'] else "Global Tech Update"

def get_dataloader(batch_size=8):
    collector = RealDataCollector()
    X, y, _ = collector.prepare_data()
    y_one_hot = np.eye(2)[y]
    dataset = XenonDataset(X, y_one_hot)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
