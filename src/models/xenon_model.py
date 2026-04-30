import torch
import torch.nn as nn
import torch.nn.functional as F

class XenonModel(nn.Module):
    """
    XenonBrain Production Engine: Advanced Logic Processing.
    تم تحديثه ليتعامل مع أبعاد البيانات الحقيقية واستخدام تقنيات استقرار التدريب.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=6, nhead=8):
        super(XenonModel, self).__init__()
        
        # طبقة الإسقاط الخطي مع LayerNorm لاستقرار البيانات الحقيقية
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        # طبقات Transformer Encoder المتقدمة
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.2,
            batch_first=True,
            activation='gelu' # GELU أفضل للنماذج العميقة
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # محرك اتخاذ القرار المنطقي
        self.logic_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )

    def forward(self, x):
        # x: (batch, seq, input_dim)
        x = self.input_norm(x)
        x = self.embedding(x)
        
        # معالجة الأنماط عبر الزمن والسياق
        x = self.transformer_encoder(x)
        
        # استخدام الـ Attention Pooling (المتوسط) لتمثيل التسلسل بالكامل
        x = torch.mean(x, dim=1)
        
        # استنتاج المخرج النهائي
        logits = self.logic_gate(x)
        
        return logits

if __name__ == "__main__":
    # اختبار البنية الجديدة
    model = XenonModel(input_dim=389, hidden_dim=128, output_dim=2)
    sample_input = torch.randn(8, 5, 389)
    output = model(sample_input)
    print(f"XenonBrain Model V2 Initialized.")
    print(f"Input: {sample_input.shape} -> Output: {output.shape}")
