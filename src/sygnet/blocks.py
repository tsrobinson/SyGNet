from .requirements import *

class gHead(nn.Module):
  ''' 
  Single attention-head module
  '''
  def __init__(self, head_size, n_lin, d_p):
     super().__init__()
     self.key = nn.Linear(n_lin, head_size, bias=False)
     self.query = nn.Linear(n_lin, head_size, bias=False)
     self.value = nn.Linear(n_lin, head_size, bias=False)
     self.dropout = nn.Dropout(d_p)

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
  ''' 
  Multi-headed attention block
  '''
  def __init__(self, n_heads, head_size, n_lin, d_p):
    super().__init__()
    self.heads = nn.ModuleList([gHead(head_size, n_lin, d_p) for _ in range(n_heads)])
    self.proj = nn.Linear(n_heads * head_size, n_lin)
    self.dropout = nn.Dropout(d_p)

  def forward(self, x):
    out = torch.cat([h(x) for h in self.heads], dim=-1) #
    out = self.proj(out)
    out = self.dropout(out)
    return out
  
class gLN1(nn.Module):
  '''
  SyGNet-LN1 block option when you do not want MHA
  '''
  def __init__(self, n_lin, d_p, r_a):
    super().__init__()
    self.lin = nn.Linear(n_lin, n_lin)
    self.ln = nn.LayerNorm(n_lin)
    self.relu = nn.LeakyReLU(r_a)
    self.dp = nn.Dropout(d_p)

  def forward(self, x):
    x = self.lin(x)
    x = self.ln(x)
    x = self.relu(x)
    x = self.dp(x)
    return x


  
class LgBlock(nn.Module):
  ''' 
  Linear no residual connection block for generator module
  '''
  def __init__(self, n_heads, n_lin, d_p):
    super().__init__()
    head_size = n_lin // n_heads
    self.sa = gMultiHeadAttention(n_heads, head_size, n_lin, d_p)
    self.norm1 = nn.BatchNorm1d(n_lin)
    self.relu = nn.LeakyReLU()

  def forward(self, x):
    x = x + self.sa(x)
    x = self.norm1(x)
    x = self.relu(x)
    return x

class LcBlock(nn.Module):
  ''' 
  Linear no residual connection block for critic module
  '''
  def __init__(self, n_lin, d_p, r_a):
    super().__init__()
    self.lin = nn.Linear(n_lin, n_lin)
    self.dp = nn.Dropout(d_p)
    self.relu = nn.LeakyReLU(r_a)

  def forward(self, x):
    x = self.lin(x)
    x = self.relu(x)
    x = self.dp(x)
    return x