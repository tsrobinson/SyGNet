from .requirements import *

class gHead(nn.Module):
  ''' '''
  def __init__(self, head_size):
     super().__init__()
     self.key = nn.Linear(n_lin, head_size, bias=False)
     self.query = nn.Linear(n_lin, head_size, bias=False)
     self.value = nn.Linear(n_lin, head_size, bias=False)
     self.dropout = nn.Dropout(dropout)

  def forward(self, x): 
    T,C = x.shape
    k = self.key(x) 
    q = self.query(x)
    att_score = q @ k.T * C**-5 # (T,C) @ (C,T) -> (T,T)
    att_score = F.softmax(att_score, dim=-1)
    att_score = self.dropout(att_score)
    v = self.value(x) 
    out = att_score @ v # (T,T) @ (T,C) = (T,C)
    return out

class gMultiHeadAttention(nn.Module):
  ''' '''
  def __init__(self, n_heads, head_size):
    super().__init__()
    self.heads = nn.ModuleList([gHead(head_size) for _ in range(n_heads)])
    self.proj = nn.Linear(n_heads * head_size, n_lin)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1) #
    out = self.proj(out)
    out = self.dropout(out)
    return out