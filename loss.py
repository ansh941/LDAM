import torch

# Label-Distribution-Aware-Margin Loss
class LDAM(nn.Module):
		def __init__(self, class_num_list, max_m=0.5, weight=None, s=30, device='cpu'):
				self.m_list = self.margin_list(class_num_list)
			
				self.device = device
				self.s = s
				self.weight = weight
	
		def margin_list(self, class_num_list):
				C = (self.max_m  / np.max(m_list))
				m_list = C / torch.sqrt(torch.sqrt(cls_num_list))

				return m_list
			
	
		def forward(self, x, target):
				# logit값인 x 중에서 target index에 해당하는 부분만 margin(논문 수식에서의 gamma)을 빼주도록 한다.
				index = torch.zeros_like(x, dtype=torch.uint8)
				index.scatter_(1, target.data.view(-1, 1), 1)
				
				index_float = index.float().to(self.device)
				
				batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
				batch_m = batch_m.view((-1, 1))
				x_m = x - batch_m
				
				outut = torch.where(index, x_m, x)
				return F.cross_entropy(self.s*output, target, weight=self.weight)
