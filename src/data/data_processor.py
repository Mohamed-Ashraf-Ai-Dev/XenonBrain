import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import requests
import feedparser
import yfinance as yf
import os
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
    محرك XenonBrain الحقيقي لجلب البيانات:
    1. يجلب أخبار الذكاء الاصطناعي والتقنية عبر RSS.
    2. يجلب بيانات السوق (S&P 500) كمؤشر اقتصادي للنمط.
    3. يدمج النصوص مع الأرقام لإنشاء "وعي سياقي".
    """
    def __init__(self, raw_data_path="data/raw"):
        self.raw_data_path = raw_data_path
        os.makedirs(raw_data_path, exist_ok=True)
        # استخدام نموذج صغير لتحويل النصوص إلى أرقام (Embeddings)
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')

    def fetch_tech_news(self):
        """جلب أحدث أخبار التقنية"""
        print("جاري جلب أخبار التقنية الحقيقية...")
        feeds = [
            "https://news.mit.edu/rss/topic/artificial-intelligence2",
            "https://www.theverge.com/rss/index.xml",
            "https://techcrunch.com/feed/"
        ]
        news_items = []
        for url in feeds:
            feed = feedparser.parse(url)
            for entry in feed.entries[:10]:
                news_items.append(entry.title)
        return news_items

    def fetch_market_data(self):
        """جلب بيانات السوق الحقيقية (آخر 30 يوم)"""
        print("جاري جلب بيانات السوق الحقيقية...")
        ticker = "^GSPC" # S&P 500
        data = yf.download(ticker, period="1mo", interval="1d", progress=False)
        # تحويل البيانات إلى ميزات رقمية
        market_features = data[['Open', 'High', 'Low', 'Close', 'Volume']].pct_change().dropna().values
        return market_features

    def prepare_real_world_batch(self):
        """دمج الأخبار مع بيانات السوق لإنشاء نمط منطقي حقيقي"""
        news = self.fetch_tech_news()
        market = self.fetch_market_data()
        
        # تحويل النصوص إلى Embeddings (384 dimension)
        news_embeddings = self.text_model.encode(news)
        
        # توحيد الأبعاد (هذا مثال للتبسيط، في النظام الحقيقي نستخدم أبعاد ثابتة)
        # سنقوم بدمج متوسط تضمين الأخبار مع ميزات السوق
        avg_news_emb = np.mean(news_embeddings, axis=0) # (384,)
        
        # إنشاء تسلسل بيانات (Sequence) للتدريب
        # سنفترض أن XenonBrain يحاول التنبؤ باتجاه السوق بناءً على الأخبار والأنماط السابقة
        seq_len = 5
        input_dim = 384 + 5 # (أخبار + ميزات السوق)
        
        X = []
        y = []
        
        # بناء تسلسلات حقيقية
        for i in range(len(market) - seq_len):
            market_seq = market[i:i+seq_len] # (5, 5)
            # دمج الأخبار مع كل خطوة زمنية في السوق
            combined_seq = []
            for m_step in market_seq:
                combined_step = np.concatenate([avg_news_emb, m_step])
                combined_seq.append(combined_step)
            
            X.append(combined_seq)
            # الهدف: هل سيصعد السوق في الخطوة القادمة؟ (1 صعود، 0 هبوط)
            target = 1 if market[i+seq_len][3] > 0 else 0
            y.append(target)
            
        return np.array(X), np.array(y)

def get_dataloader(batch_size=8):
    collector = RealDataCollector()
    try:
        X, y = collector.prepare_real_world_batch()
        if len(X) == 0:
            raise ValueError("لم يتم العثور على بيانات كافية.")
    except Exception as e:
        print(f"فشل جلب البيانات الحقيقية: {e}. سيتم استخدام مولد بيانات طوارئ.")
        # مولد طوارئ في حالة انقطاع الإنترنت أو فشل الـ API
        X = np.random.randn(10, 5, 389)
        y = np.random.randint(0, 2, 10)
    
    y_one_hot = np.eye(2)[y]
    dataset = XenonDataset(X, y_one_hot)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

if __name__ == "__main__":
    loader = get_dataloader()
    for batch_x, batch_y in loader:
        print(f"Batch X shape (Real Data): {batch_x.shape}")
        print(f"Batch Y shape (Real Targets): {batch_y.shape}")
        break
    print("Real Data Engine is Online!")
