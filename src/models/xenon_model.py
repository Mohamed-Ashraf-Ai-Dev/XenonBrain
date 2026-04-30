import torch
import torch.nn as nn
import torch.nn.functional as F

class XenonModel(nn.Module):
    """
    XenonBrain Evolution Engine (V3):
    تمت زيادة العمق وإضافة Residual Connections لتمكين التعلم من الذاكرة الطويلة.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=8, nhead=8):
        super(XenonModel, self).__init__()
        
        self.input_norm = nn.LayerNorm(input_dim)
        self.embedding = nn.Linear(input_dim, hidden_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=hidden_dim * 4,
            dropout=0.2,
            batch_first=True,
            activation='gelu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # بوابة منطقية أكثر عمقاً للتعامل مع الأنماط التاريخية
        self.logic_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        x = self.input_norm(x)
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = torch.mean(x, dim=1)
        logits = self.logic_gate(x)
        return logits

if __name__ == "__main__":
    model = XenonModel(input_dim=390, hidden_dim=128, output_dim=2)
    sample_input = torch.randn(8, 5, 390)
    output = model(sample_input)
    print(f"XenonBrain Model V3 (Evolution) Initialized.")
    print(f"Input: {sample_input.shape} -> Output: {output.shape}")
