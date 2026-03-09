import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F


# DeepSeekMOE components
class AddAuxiliaryLoss(torch.autograd.Function):
    """
    修正后的辅助损失函数，确保领域专家参数通过 aux_loss 更新。
    """

    @staticmethod
    def forward(ctx, x, loss):
        ctx.save_for_backward(loss)
        return x  # 正向传播不修改输出

    @staticmethod
    def backward(ctx, grad_output):
        loss, = ctx.saved_tensors
        # 主任务梯度：grad_output 直接传递给前层
        # 辅助损失梯度：loss 的梯度为 1（因为总损失是 main_loss + loss）
        grad_loss = torch.ones_like(loss) if loss.numel() == 1 else torch.ones_like(loss).mean()
        return grad_output, grad_loss  # 返回主梯度和辅助梯度


class MoEGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.n_routed_experts = config.n_routed_experts
        self.num_experts = config.n_routed_experts
        self.scoring_func = config.scoring_func
        self.alpha = config.aux_loss_alpha
        self.register_buffer('expert_counts', torch.zeros(config.n_routed_experts))
        self.register_buffer('total_steps', torch.tensor(0))
        self.norm_topk_prob = config.norm_topk_prob
        self.gating_dim = config.input_size
        self.weight = nn.Parameter(torch.empty((self.n_routed_experts, self.gating_dim)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, hidden_states):
        bsz, h = hidden_states.shape
        hidden_states = hidden_states.view(-1, h)
        logits = F.linear(hidden_states, self.weight, None)
        scores = logits.softmax(dim=-1)
        topk_weight, topk_idx = torch.topk(scores, k=self.top_k, dim=-1, sorted=False)
        if self.top_k > 1 and self.norm_topk_prob:
            denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
            topk_weight = topk_weight / denominator

        aux_loss = None
        if self.training and self.alpha > 0.0:
            Pi = scores.mean(dim=0)
            expert_counts = F.one_hot(topk_idx, num_classes=self.num_experts).sum(dim=[0, 1]).float()
            fi = expert_counts * self.num_experts / (bsz * self.top_k)
            load_balance_loss = torch.sum(Pi * fi)
            aux_loss = self.alpha * load_balance_loss
            if aux_loss.numel() > 1:
                aux_loss = aux_loss.mean()
        return topk_idx, topk_weight, aux_loss


class MLPExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout: float = 0.1):
        super().__init__()
        self.n_hidden_layers = 2
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)  # 替换为LayerNorm
            # nn.BatchNorm1d(hidden_dim)
        )

        self.hidden = nn.ModuleList()
        for i in range(self.n_hidden_layers):
            self.hidden.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Dropout(dropout),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim)  # 替换为LayerNorm，避免分到某个专家的样本数为1，导致无法使用BatchNorm
                # nn.BatchNorm1d(hidden_dim)
            ))
        self.output = nn.Sequential(*[
            nn.Linear(hidden_dim, output_dim),
        ])

    def forward(self, x):
        o = self.input(x)
        for hidden_layer in self.hidden:
            o = hidden_layer(o)
        o = self.output(o)
        return o

class LightExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)
class DeepseekMoE(nn.Module):
    def __init__(self, config, num_experts=None, num_shared_experts=1):
        super().__init__()
        self.num_experts = num_experts if num_experts is not None else config.n_routed_experts
        self.expert_dim = config.output_dim
        self.num_shared_experts = num_shared_experts
        self.gate = MoEGate(config)
        self.experts = nn.ModuleList([
            LightExpert(input_dim=config.input_size, hidden_dim=config.hidden_dim, output_dim=self.expert_dim) 
            for _ in range(self.num_experts)
        ])
        self.num_experts_per_tok = config.num_experts_per_tok
        self.shared_experts = MLPExpert(input_dim=config.input_size, hidden_dim=config.hidden_dim,
                                        output_dim=self.expert_dim)

    def forward(self, combined):
        topk_indices, topk_weights, aux_loss = self.gate(combined)
        identity = combined

        if self.training:
            combined = combined.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.zeros(combined.size(0), self.expert_dim, device=combined.device)
            flat_topk_idx = topk_indices.view(-1)

            for i, expert in enumerate(self.experts):
                expert_input = combined[flat_topk_idx == i]
                if expert_input.size(0) > 0:  # Only process if there are samples
                    expert_output = expert(expert_input)
                    y[flat_topk_idx == i] = expert_output

            y = y.view(*topk_weights.shape, -1)
            y = (y * topk_weights.unsqueeze(-1)).sum(dim=1)
            y = AddAuxiliaryLoss.apply(y, aux_loss) if aux_loss is not None else y

        else:
            expert_mask = F.one_hot(topk_indices, num_classes=self.num_experts).float()
            weights_sum = (expert_mask * topk_weights.unsqueeze(-1)).sum(dim=1)
            expert_outputs = torch.stack([expert(combined) for expert in self.experts], dim=1)
            moe_output = (expert_outputs * weights_sum.unsqueeze(-1)).sum(dim=1)
            y = moe_output

        shared_feature = self.shared_experts(identity)
        final_output = y + shared_feature
        return final_output

