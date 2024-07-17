import torch
import torch.nn.functional as F
import numpy as np
import torch.nn as nn

class ConditionalEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(ConditionalEntropyLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = b.sum(dim=1)
        return -1.0 * b.mean(dim=0)


class NTXentLoss(torch.nn.Module):

    def __init__(self, config, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.configs = config
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.projection_head = nn.Sequential(
            nn.Linear(self.configs.AR_hid_dim, self.configs.final_out_channels // 2),
            nn.BatchNorm1d(self.configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.configs.dropout),
            nn.Linear(self.configs.final_out_channels // 2, self.configs.final_out_channels // 4),
        )

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        # print("x.unsqueeze(1): ", x.unsqueeze(1).shape)
        # print("y.unsqueeze(0): ", y.unsqueeze(0).shape)
        # print("v: ", v.shape)
        return v

    def forward(self, zis, zjs):
        # print("zis: ", zis.shape)
        # print("zjs: ", zjs.shape)
        # zis = self.projection_head(zis)
        # zjs = self.projection_head(zjs)
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)
        # print("similarity_matrix_shape:",similarity_matrix.shape)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        # print("l_pos_shape:",l_pos.shape)
        # print("r_pos_shape:",r_pos.shape)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # print("positives_shape:",positives.shape)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)

class NTXentLoss2(torch.nn.Module):

    def __init__(self, config, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss2, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.configs = config
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")
        self.projection_head = nn.Sequential(
            nn.Linear(self.configs.AR_hid_dim, self.configs.final_out_channels // 2),
            nn.BatchNorm1d(self.configs.final_out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(self.configs.dropout),
            nn.Linear(self.configs.final_out_channels // 2, self.configs.final_out_channels // 4),
        )

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

        # diag = np.eye(3 * self.batch_size)  # Assuming the input is now 3D, so we triple the size
        # l1 = np.eye((3 * self.batch_size), 3 * self.batch_size, k=-self.batch_size)
        # l2 = np.eye((3 * self.batch_size), 3 * self.batch_size, k=self.batch_size)
        # mask = torch.from_numpy((diag + l1 + l2))
        # mask = (1 - mask).type(torch.bool)
        # return mask.to(self.device)

    # @staticmethod
    # def _dot_simililarity(x, y):
    #     v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
    #     # x shape: (N, 1, C)
    #     # y shape: (1, C, 2N)
    #     # v shape: (N, 2N)
    #     return v

    @staticmethod
    def _dot_simililarity(x, y):
        # v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        v = torch.tensordot(x.unsqueeze(1), y.permute(0, 2, 1), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    # def _cosine_simililarity(self, x, y):
    #     # x shape: (N, 1, C)
    #     # y shape: (1, 2N, C)
    #     # v shape: (N, 2N)
    #     v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
    #     print("x.unsqueeze(1): ", x.unsqueeze(1).shape)
    #     print("y.unsqueeze(0): ", y.unsqueeze(0).shape)
    #     print("v: ", v.shape)
    #     return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, N)
        x_flat = x.view(x.size(0), -1)  # Flatten x to be 2D
        y_flat = y.view(y.size(0), -1)  # Flatten y to be 2D and adjust the size
        # print("x.unsqueeze(1): ", x_flat.shape)
        # print("y.unsqueeze(0): ", y_flat.shape)
        v = F.cosine_similarity(x_flat.unsqueeze(1), y_flat.unsqueeze(0), dim=2)  # Compute cosine similarity
        # print("v: ", v.shape)
        return v

    # def _cosine_simililarity(self, x, y):
    #     # Assuming x and y are 3D tensors
    #     print("xxx: ", x.shape)
    #     v = F.cosine_similarity(x.view(-1, x.size(-1)), y.view(-1, y.size(-1)), dim=1)
    #     return v.view(x.size(0), x.size(1), y.size(1))

    def forward(self, zis, zjs):
        # print("zis: ", zis.shape)
        # print("zjs: ", zjs.shape)
        # zis = self.projection_head(zis)
        # zjs = self.projection_head(zjs)
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)
        # print("similarity_matrix_shape:",similarity_matrix.shape)
        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        # print("l_pos_shape:",l_pos.shape)
        # print("r_pos_shape:",r_pos.shape)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)
        # print("positives_shape:",positives.shape)
        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature

        labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        loss = self.criterion(logits, labels)

        return loss / (2 * self.batch_size)


class CrossEntropyLabelSmooth(nn.Module):
    def __init__(self, num_classes, device, epsilon=0.1):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.device = device
    def forward(self, inputs, targets):
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).to(self.device).scatter_(1, targets.unsqueeze(1), 1)
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()

        return loss
def EntropyLoss(input_):
    mask = input_.ge(0.0000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = - (torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))


def evidential_uncertainty(predictions, target, num_classes, device):
    predictions = predictions.to(device)
    target = target.to(device)

    # one hot encoding
    eye = torch.eye(num_classes).to(torch.float).to(device)
    labels = eye[target]
    # Calculate evidence
    evidence = F.softplus(predictions)

    # Dirichlet distribution paramter
    alpha = evidence + 1

    # Dirichlet strength
    strength = alpha.sum(dim=-1)

    # expected probability
    p = alpha / strength[:, None]

    # calculate error
    error = (labels - p) ** 2

    # calculate variance

    var = p * (1 - p) / (strength[:, None] + 1)

    # loss function
    loss = (error + var).sum(dim=-1)

    return loss.mean()

def evident_dl(predictions):
    # Calculate evidence
    evidence = F.softplus(predictions)

    # Dirichlet distribution paramter
    alpha = evidence + 1

    # Dirichlet strength
    strength = alpha.sum(dim=-1)

    # expected probability
    p = alpha / strength[:, None]

    var = p * (1 - p) / (strength[:, None] + 1)

    evident_entropy = torch.mean(EntropyLoss(p))
    evident_var = torch.mean(var)

    return p, evident_var, evident_entropy


class SinkhornDistance(nn.Module):
    r"""
    Given two empirical measures each with :math:`P_1` locations
    :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
    outputs an approximation of the regularized OT cost for point clouds.
    Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
            'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
            'mean': the sum of the output will be divided by the number of
            elements in the output, 'sum': the output will be summed. Default: 'none'
    Shape:
        - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
        - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps, max_iter, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, x, y):
        # The Sinkhorn algorithm takes as input three variables :
        C = self._cost_matrix(x, y)  # Wasserstein cost function
        x_points = x.shape[-2]
        y_points = y.shape[-2]
        if x.dim() == 2:
            batch_size = 1
        else:
            batch_size = x.shape[0]

        # both marginals are fixed with equal weights
        mu = torch.empty(batch_size, x_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / x_points).squeeze().to('cuda')
        nu = torch.empty(batch_size, y_points, dtype=torch.float,
                         requires_grad=False).fill_(1.0 / y_points).squeeze().to('cuda')

        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            # print(mu.device, self.M(C,u,v).device)
            u = self.eps * (torch.log(mu + 1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * (torch.log(nu + 1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(self.M(C, U, V))
        # Sinkhorn distance
        cost = torch.sum(pi * C, dim=(-2, -1))

        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        return cost, pi, C

    def M(self, C, u, v):
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"

        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

    @staticmethod
    def _cost_matrix(x, y, p=2):
        "Returns the matrix of $|x_i-y_j|^p$."
        x_col = x.unsqueeze(-2)
        y_lin = y.unsqueeze(-3)
        C = torch.sum((torch.abs(x_col - y_lin)) ** p, -1)
        return C

    @staticmethod
    def ave(u, u1, tau):
        "Barycenter subroutine, used by kinetic acceleration through extrapolation."
        return tau * u + (1 - tau) * u1




