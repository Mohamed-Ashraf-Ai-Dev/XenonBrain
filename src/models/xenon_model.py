import torch
import torch.nn as nn
import torch.nn.functional as F

class XenonModel(nn.Module):
    """
    XenonBrain Sovereign Intelligence Engine (V6):
    تمت ترقية النموذج بإضافة طبقات انتباه متعددة الرؤوس (Multi-Head Attention) لفهم الارتباطات المعقدة بين الأصول،
    وتعميق بوابة المنطق لاتخاذ قرارات أكثر سيادية.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=8, nhead=8):
        super(XenonModel, self).__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # طبقة انتباه إضافية لتعزيز فهم الارتباطات الأولية بين المدخلات
        self.cross_asset_attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=nhead, batch_first=True)
        self.attention_norm = nn.LayerNorm(hidden_dim)
        self.attention_dropout = nn.Dropout(0.2)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.2,
            batch_first=True,
            activation=\'gelu\'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # بوابة منطقية أكثر عمقاً للتعامل مع الأنماط التاريخية والقرارات السيادية
        self.logic_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid() # إضافة Sigmoid هنا لضمان أن المخرجات بين 0 و 1 مباشرة
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.embedding(x)
        
        # تطبيق طبقة الانتباه العابر للأصول
        # نحتاج إلى تغيير شكل x ليتناسب مع MultiheadAttention (seq_len, batch_size, embed_dim)
        # ولكن بما أننا نستخدم batch_first=True، الشكل هو (batch_size, seq_len, embed_dim)
        # لذا، x هو بالفعل بالشكل الصحيح (batch_size, seq_len, hidden_dim)
        attn_output, _ = self.cross_asset_attention(x, x, x)
        x = x + self.attention_dropout(attn_output) # Residual connection
        x = self.attention_norm(x)

        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1) # تجميع المعلومات من التسلسل الزمني
        logits = self.logic_gate(x)
        return logits

if __name__ == "__main__":
    model = XenonModel(input_dim=390, hidden_dim=128, output_dim=2)
    sample_input = torch.randn(8, 5, 390) # (batch_size, sequence_length, input_dim)
    output = model(sample_input)
    print(f"XenonBrain Model V6 (Sovereign Intelligence) Initialized.")
    print(f"Input: {sample_input.shape} -> Output: {output.shape}")
