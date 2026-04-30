import torch
import torch.nn.functional as F
import sys
import os
import yaml
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
        """
        تشغيل الاستنتاج المنطقي بناءً على نص وخلفية بيانات.
        news_text: نص الخبر الحالي.
        market_data_list: قائمة بـ 5 خطوات زمنية لبيانات السوق [Open, High, Low, Close, Volume].
        """
        # تحويل النص إلى embedding
        text_emb = self.text_model.encode([news_text])[0]
        
        # دمج البيانات
        combined_seq = []
        for m_step in market_data_list:
            combined_step = np.concatenate([text_emb, m_step])
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
    # تجربة استدلال سريعة
    engine = XenonInference()
    sample_market = [[0.01, 0.02, -0.01, 0.01, 1000]] * 5
    result = engine.run_logic("AI breakthroughs in logic processing are accelerating.", sample_market)
    print(f"Inference Result: {result}")
