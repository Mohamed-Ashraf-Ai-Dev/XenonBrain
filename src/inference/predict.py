import torch
import torch.nn.functional as F
import sys
import os
import yaml
import json
from sentence_transformers import SentenceTransformer
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.xenon_model import XenonModel

def load_config():
    with open('config/config.yaml', 'r') as f:
        return yaml.safe_load(f)

class XenonInference:
    def __init__(self, model_path='models/xenon_brain_latest.pth'):
        self.config = load_config()
        m_cfg = self.config['model']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = XenonModel(
            input_dim=m_cfg['input_dim'],
            hidden_dim=m_cfg['hidden_dim'],
            output_dim=m_cfg['output_dim'],
            num_layers=m_cfg['num_layers'],
            nhead=m_cfg['nhead']
        ).to(self.device)
        
        if os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded XenonBrain model from {model_path}")
        else:
            print("Warning: Model file not found. Please run training first.")
        
        self.model.eval()
        self.text_model = SentenceTransformer('all-MiniLM-L6-v2')

    def run_logic(self, news_text, market_data_list):
        # تحويل النص إلى embedding
        text_emb = self.text_model.encode([news_text])[0]
        
        # جلب الذاكرة التاريخية إذا وجدت
        memory_val = 0.5
        history_file = "HISTORY.json"
        if os.path.exists(history_file):
            try:
                with open(history_file, 'r') as f:
                    h = json.load(f)
                    if len(h) > 0:
                        memory_val = np.mean([i['prediction'] for i in h])
            except: pass

        # دمج البيانات (الأخبار + السوق + الذاكرة) لتطابق أبعاد التدريب (390)
        combined_seq = []
        for m_step in market_data_list:
            combined_step = np.concatenate([text_emb, m_step, [memory_val]])
            combined_seq.append(combined_step)
        
        input_tensor = torch.FloatTensor([combined_seq]).to(self.device)
        
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = F.softmax(output, dim=1)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()
            
        return {
            "prediction": "Positive Pattern" if prediction == 1 else "Negative/Neutral Pattern",
            "confidence": f"{confidence*100:.2f}%"
        }

if __name__ == "__main__":
    engine = XenonInference()
    sample_market = [[0.01, 0.02, -0.01, 0.01, 1000]] * 5
    result = engine.run_logic("AI breakthroughs in logic processing are accelerating.", sample_market)
    print(f"Inference Result: {result}")
