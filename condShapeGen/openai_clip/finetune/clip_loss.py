import torch
import torch.nn.functional as F

def contrastiveloss_f(i_feats, t_feats, logit_scale, reduction='mean'):
    
    logits_per_image = logit_scale.exp() * i_feats @ t_feats.t()
    logits_per_text = logit_scale.exp() * t_feats @ i_feats.t()
    ground_truth = torch.arange(len(logits_per_image)).long().cuda()
    total_loss = (
        F.cross_entropy(logits_per_image, ground_truth, reduction=reduction)
        + 
        F.cross_entropy(logits_per_text, ground_truth, reduction=reduction)
    ) / 2
    return total_loss

def single_contrastiveloss_f(i_feats, t_feats, logit_scale, reduction='mean'):
    
    logits_per_image = logit_scale.exp() * i_feats @ t_feats.t()
    logits_per_text = logit_scale.exp() * t_feats @ i_feats.t()
    ground_truth = torch.arange(len(logits_per_image)).long().cuda()
    total_loss = (
        # F.cross_entropy(logits_per_image, ground_truth, reduction=reduction)
        # + 
        F.cross_entropy(logits_per_text, ground_truth, reduction=reduction)
    ) #/ 2
    return total_loss
