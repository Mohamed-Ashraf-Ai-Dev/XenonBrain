import torch
import torch.nn as nn
import torch.nn.functional as F

class XenonModel(nn.Module):
    """
    XenonBrain Sovereign Intelligence Engine (V6.5):
    تمت ترقية النموذج بإضافة طبقات انتباه متعددة الرؤوس (Multi-Head Attention) لفهم الارتباطات المعقدة بين الأصول،
    وتعميق بوابة المنطق لاتخاذ قرارات أكثر سيادية. يدعم الآن التعلم المحسن من الأخطاء التاريخية
    ومشاعر مجتمع Reddit المدمجة في بيانات التدريب.
    تمت إزالة Sigmoid النهائية لتحسين استقرار دالة الخسارة CrossEntropyLoss.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=8, num_heads=8):
        super(XenonModel, self).__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        self.cross_asset_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_dropout = nn.Dropout(0.2)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.2,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.logic_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = x.float()
        x = self.input_norm(x)
        x = self.embedding(x)
        
        attn_output, _ = self.cross_asset_attention(x, x, x)
        x = x + self.attention_dropout(attn_output)
        x = self.attention_norm(x)
        
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1) 
        logits = self.logic_gate(x)
        return logits

if __name__ == "__main__":
    model = XenonModel(input_dim=390, hidden_dim=128, output_dim=2)
    sample_input = torch.randn(8, 5, 390) 
    output = model(sample_input)
    print(f"XenonBrain Model V6.5 (Sovereign Intelligence) Initialized.")
    print(f"Input: {sample_input.shape} -> Output: {output.shape}")
