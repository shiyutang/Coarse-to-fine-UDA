import torch
import torch.nn.functional as F
from torch import nn, autograd


class HM(autograd.Function):
    @staticmethod
    def forward(ctx, inputs, index, average_center, features, momentum):
        ctx.features = features
        ctx.momentum = momentum
        ctx.save_for_backward(index, average_center)
        outputs = inputs.mm(ctx.features.t())  # 1*91*161, 2048; 19, 2048

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        index, average_center = ctx.saved_tensors
        grad_inputs = None
        # ctx.needs_input_grad[0] = True if the first input to forward() needs gradient computated
        # w.r.t. the output
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(ctx.features)

        # momentum update 每一个样本，滑动平均后归一化， 减少更新的样本个数，
        for idx, feat in zip(index, average_center):
            ctx.features[idx] = ctx.momentum * ctx.features[idx] + (1. - ctx.momentum) * feat
            ctx.features[idx] /= (ctx.features[idx].norm() if ctx.features[idx].norm() != 0 else 1)

        return grad_inputs, None, None, None, None


def hm(inputs, indexes, average_center, features, momentum=0.5):
    return HM.apply(inputs, indexes, average_center, features, torch.Tensor([momentum]).to(inputs.device))


class HybridMemory(nn.Module):
    def __init__(self, num_features, num_samples, temp=0.05, momentum=0.2):
        super(HybridMemory, self).__init__()
        self.num_features = num_features
        self.num_samples = num_samples

        self.momentum = momentum
        self.temp = temp

        self.register_buffer('features', torch.zeros(num_samples, dtype=torch.float16))  # 特征是（样本数，样本特征长度）
        self.register_buffer('labels', torch.arange(num_samples).cuda().long())

    def forward(self, inputs, labels, index, average_center):
        """
        compute contrastive loss
        """
        # inputs: 4*160*320, 256, features: 2965*160*320, 256
        inputs = F.normalize(inputs, dim=1)
        inputs = hm(inputs, index, average_center, self.features, self.momentum)  #
        inputs /= self.temp  # 4*160*320,  2965*160*320
        B = inputs.size(0)  # 4*160*320

        def masked_softmax(vec, mask, dim=1, epsilon=1e-6):
            exps = torch.exp(vec - vec.mean())  # 20, 4*160*320
            masked_exps = exps * mask.float().clone()
            masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon  # 1, 4*160*320
            return masked_exps / masked_sums

        sumlabels = self.labels.clone()  # 2965*160*320, 1

        # 独立分开的对比损失、融合的对比损失

        sim = torch.zeros(sumlabels.max() + 1, B).float().cuda()  # 20,  4*160*320
        # 把第三项的内容依次根据第二项的映射加到 sim 中，MB中一个样本和所有batch样本的内积按照这个样本的标签归类到一起
        sim.index_add_(0, sumlabels, inputs.t().contiguous())  # 2965*160*320, 4*160*320 inputs==sim.t()

        nums = torch.zeros(sumlabels.max() + 1, 1).float().cuda()
        nums.index_add_(0, sumlabels, torch.ones(self.num_samples, 1).float().cuda())  # MB 中每个label有多少样本

        # 同一个label的内积求平均 mean(x*MB_d_{i}) s.t: d_{i} has same label,
        mask = (nums > 0).float()
        sim /= (mask * nums + (1 - mask)).clone().expand_as(sim)  # 20, 4*160*320, 一个batch 每个特征和原型的内积

        mask = mask.expand_as(sim)
        masked_sim = masked_softmax(sim.t().contiguous(), mask.t().contiguous())

        return F.nll_loss(torch.log(masked_sim + 1e-6), labels, ignore_index=-1)  # nan in masked_sim



