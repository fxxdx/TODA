import torch
from torch import nn
from einops import rearrange
from models.attention import *

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]

class CNN2d(nn.Module):
    def __init__(self, configs):
        super(CNN2d, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(configs.final_out_channels, configs.final_out_channels*2, kernel_size=(configs.kernel_size, 1),
                      stride=(configs.stride, 1), bias=False, padding=(configs.kernel_size // 2, 0)),
            nn.BatchNorm2d(configs.final_out_channels*2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(1, 0)),
            nn.Dropout(configs.dropout)
        )
        self.ca1 = CA(configs.final_out_channels*2)
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(configs.final_out_channels*2, configs.final_out_channels * 4, kernel_size=(8, 1), stride=(1, 1), bias=False, padding=(4, 0)),
            nn.BatchNorm2d(configs.final_out_channels * 4),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(1, 0))
        )
        self.ca2 = CA(configs.final_out_channels * 4)
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(configs.final_out_channels * 4, configs.final_out_channels, kernel_size=(8, 1), stride=(1, 1), bias=False,
                      padding=(4, 0)),
            nn.BatchNorm2d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1), padding=(1, 0)),
        )
        self.ca3 = CA(configs.final_out_channels)
        self.aap = nn.AdaptiveAvgPool2d((configs.features_len, 1))

    def forward(self, x_in):
        # print("2d x_in: ", x_in.shape)
        x = self.conv_block1(x_in)
        # print("x_in  111: ", x.shape)
        x = self.ca1(x)
        x = self.conv_block2(x)
        x = self.ca2(x)
        x = self.conv_block3(x)
        x = self.ca3(x)
        # print("before x:", x.shape)

        # print("after x:", x.shape)
        x_flat = self.aap(x).reshape(x.size(0), -1)
        # print("x:", x.shape)
        return x_flat, x

## Feature Extractor
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.Dropout(configs.dropout)
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.aap = nn.AdaptiveAvgPool1d(configs.features_len)
    def forward(self, x_in):
        # print("x_in: ", x_in.shape)
        x = self.conv_block1(x_in)
        # print("x: ", x.shape)

        x = self.conv_block2(x)

        x = self.conv_block3(x)
        x_flat = self.aap(x).view(x.shape[0], -1)
        return x_flat, x
##  Classifier
class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions
## Temporal Imputer
class Temporal_Imputer(nn.Module):
    def __init__(self, configs):
        super(Temporal_Imputer, self).__init__()
        self.seq_length = configs.features_len
        self.num_channels = configs.final_out_channels
        self.hid_dim = configs.AR_hid_dim
        # input size: batch_size, 128 channel, 18 seq_length
        self.rnn = nn.LSTM(input_size=self.num_channels, hidden_size=self.hid_dim)

    def forward(self, x):
        x = x.view(x.size(0), -1, self.num_channels)
        out, (h, c) = self.rnn(x)
        out = out.view(x.size(0), self.num_channels, -1)
        # take the last time step
        return out

# temporal masking    时序遮码的过程
def masking(x, num_splits=8, num_masked=4):
    # num_masked = int(masking_ratio * num_splits)
    patches = rearrange(x, 'a b (p l) -> a b p l', p=num_splits)
    masked_patches = patches.clone()  # deepcopy(patches)
    # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked
    rand_indices = torch.rand(x.shape[1], num_splits).argsort(dim=-1)
    selected_indices = rand_indices[:, :num_masked]
    masks = []
    for i in range(masked_patches.shape[1]):
        masks.append(masked_patches[:, i, (selected_indices[i, :]), :])
        masked_patches[:, i, (selected_indices[i, :]), :] = 0
        # orig_patches[:, i, (selected_indices[i, :]), :] =
    mask = rearrange(torch.stack(masks), 'b a p l -> a b (p l)')
    masked_x = rearrange(masked_patches, 'a b p l -> a b (p l)', p=num_splits)

    return masked_x, mask


class ResClassifier_MME(nn.Module):
    def __init__(self, configs):
        super(ResClassifier_MME, self).__init__()
        self.norm = True
        self.tmp = 0.02
        num_classes = configs.num_classes
        input_size = configs.out_dim

        self.fc = nn.Linear(input_size, num_classes, bias=False)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, dropout=False, return_feat=False):
        if return_feat:
            return x
        x = self.fc(x) / self.tmp
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True)
        self.fc.weight.data = w.div(norm.expand_as(w))

    def weights_init(self):
        self.fc.weight.data.normal_(0.0, 0.1)