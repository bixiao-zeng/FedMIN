import torch
import torch.nn.functional as F
import numpy as np
import pdb


def KL(alpha):
    K = 10
    beta = torch.tensor(np.ones((1, K)), dtype=torch.float32)
    beta = beta.to(torch.device('cuda:0'))
    S_alpha = torch.sum(alpha, axis=1, keepdim=True)
    S_beta = torch.sum(beta, axis=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - torch.sum(torch.lgamma(alpha), axis=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), axis=1, keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), axis=1, keepdim=True) + lnB + lnB_uni
    return kl


def mse_loss(output,target, global_step=1, annealing_step=10):
    E = F.relu(output)
    alpha = E + 1
    S = torch.sum(alpha, dim=1, keepdim=True)
    m = alpha / S

    A = torch.sum((target - m) ** 2, axis=1, keepdim=True)
    B = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), axis=1, keepdim=True)

    annealing_coef = min(1,global_step / annealing_step)

    alp = E * (1 - target) + 1
    C = annealing_coef * KL(alp)
    return (A + B) + C

def one_hot_embedding(labels, num_classes=10):
    y = torch.eye(num_classes)
    neg = labels < 0 # negative labels
    labels[neg] = 0  # placeholder label to class-0
    y = y[labels] # create one hot embedding
    y[neg, 0] = 0 # remove placeholder label
    return y

def relu_evidence(y):
    return F.relu(y)


def exp_evidence(y):
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    return F.softplus(y)

def rce_loss(outputs,labels,device='cuda',reduce=True):
    sfm_probs = F.softmax(outputs)  # 添加激活层取正数
    loss_ce = F.nll_loss(F.log_softmax(outputs), labels,reduce=reduce)
    q = one_hot_embedding(labels)
    q = q.to(device)
    for i in range(len(q)):
        for j in range(len(q[i])):
            if int(q[i][j]) == 1:
                q[i][j] = 0
            else:
                q[i][j] = -4

    multi = q.mul(sfm_probs)
    # np.multiply(q,sfm_probs.cpu().detach().numpy())
    sum_Forrow = torch.sum(multi, dim=1)
    if reduce:
        rce = torch.mean(sum_Forrow, dim=0)
    else:
        rce = sum_Forrow
    loss_rce = (-1) * rce
    loss = 0.01 * loss_ce.detach() + loss_rce
    return loss

def edl_mse_loss(output, target, device='cuda'):
    target = one_hot_embedding(target,num_classes=output.shape[1])
    evidence = F.relu(output)
    alpha = evidence + 1

    target = target.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    err = torch.sum(
        (target - (alpha / S)) ** 2, dim=1, keepdim=True)
    var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)

    loss = err + var
    loss = torch.squeeze(loss)
    return loss

def edl_mae_loss(output, target, device='cuda'):
    evidence = F.relu(output)
    alpha = evidence + 1
    target = target.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    err = torch.sum(
        torch.abs(target - (alpha / S)), dim=1, keepdim=True)
    var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    return err + var

def edl_soft_mse_loss(output, target, device='cuda'):
    alpha = F.softmax(output)
    target = target.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    err = torch.sum(
        (target - (alpha / S)) ** 2, dim=1, keepdim=True)
    var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    return err + var
