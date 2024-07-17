import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(sys.path[0]))
from models.attention import Seq_Transformer,Seq_TransformerCA
from models.models import classifier, Temporal_Imputer, masking, CNN2d
from models.loss import EntropyLoss, CrossEntropyLabelSmooth, evidential_uncertainty, evident_dl, NTXentLoss, NTXentLoss2
from scipy.spatial.distance import cdist
from dataloader.augmentations import *
from torch.optim.lr_scheduler import StepLR
from copy import deepcopy
from utils import *

def get_algorithm_class(algorithm_name):
    """Return the algorithm class with the given name."""
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]


class Algorithm(torch.nn.Module):
    """
    A subclass of Algorithm implements a domain adaptation algorithm.
    Subclasses should implement the update() method.
    """

    def __init__(self, configs):
        super(Algorithm, self).__init__()
        self.configs = configs
        self.cross_entropy = nn.CrossEntropyLoss()

    def update(self, *args, **kwargs):
        raise NotImplementedError

class TODA(Algorithm):

    def __init__(self, backbone, configs, hparams, device, args):
        super(TODA, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.feature_extractor2 = CNN2d(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)
        self.seq_transformer = Seq_Transformer(patch_size=configs.final_out_channels, dim=configs.AR_hid_dim, depth=4,
                                               heads=4, mlp_dim=64)
        self.seq_transformerca = Seq_TransformerCA(patch_size=configs.features_len, dim=configs.AR_hid_dim, depth=4,
                                                   heads=4, mlp_dim=64)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.trans_optimizer = torch.optim.Adam(
            self.seq_transformer.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.transca_optimizer = torch.optim.Adam(
            self.seq_transformerca.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.fea2_optimizer = torch.optim.Adam(
            self.feature_extractor2.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams
        self.num_splits = args.splits
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # print("enumurate src.shape:", src_x.shape)
                if src_x.shape[0]!= self.hparams['batch_size']:
                    continue
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.fea2_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()
                self.transca_optimizer.zero_grad()
                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)
                # print("src_feat:",src_feat.shape)
                # print("seq_src_feat:",seq_src_feat.shape)
                # masking the input_sequences
                # print(src_x.shape)
                masked_data, mask = masking(src_x, num_splits=self.num_splits, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(src_x, max_segments=self.configs.max_seg), self.configs.jitter_ratio).cpu().float().to(self.device)
                aug_data2 = jitter(permutation(src_x, max_segments=5),0.001).cpu().float().to(self.device)

                src_feat_aug, seq_src_feat_aug = self.feature_extractor(aug_data)
                src_feat_aug2, seq_src_feat_aug2 = self.feature_extractor(aug_data2)
                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                '''
                内容对比模块
                '''
                # print(seq_src_feat_aug.shape)
                transform_seq_src_feat_aug = self.seq_transformer(trans(self.configs.timesteps,seq_src_feat_aug))
                transform_seq_src_feat = self.seq_transformer(trans(self.configs.timesteps,seq_src_feat))
                # print("transform_seq_src_feat_aug:",transform_seq_src_feat_aug.shape)
                nt_xent_criterion = NTXentLoss(self.configs, self.device, self.configs.batch_size, self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss = nt_xent_criterion(transform_seq_src_feat_aug, transform_seq_src_feat)

                '''
                CA提取前后关系
                '''
                strong = torch.stack((seq_src_feat_aug, seq_src_feat_aug2), dim=1).permute(0, 2, 1, 3)
                weak = torch.stack((seq_src_feat, seq_src_feat), dim=1).permute(0, 2, 1, 3)
                fea_strong, seq_fea_strong = self.feature_extractor2(strong)
                fea_weak, seq_fea_weak = self.feature_extractor2(weak)
                transform_seq_src_feat_aug = self.seq_transformerca(seq_fea_strong)
                transform_seq_src_feat = self.seq_transformerca(seq_fea_weak)
                # transform_seq_src_feat_aug = self.seq_transformerca(strong)
                # transform_seq_src_feat = self.seq_transformerca(weak)
                nt_xent_criterion2 = NTXentLoss2(self.configs, self.device, self.configs.batch_size,
                                               self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss2 = nt_xent_criterion2(transform_seq_src_feat_aug, transform_seq_src_feat)


                # classifier predictions
                # print("#####src_feat: ", src_feat.shape)
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss + content_loss + content_loss2
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()
                losses = {'entropy_loss': src_cls_loss.detach().item(), 'Masking_loss': tov_loss.detach().item(),'content_loss': content_loss.detach().item(),'content_loss2': content_loss2.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False
        for k, v in self.seq_transformer.named_parameters():
            v.requires_grad = False
        for k, v in self.seq_transformerca.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                if trg_x.shape[0]!= self.hparams['batch_size']:
                    continue
                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.fea2_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()
                self.transca_optimizer.zero_grad()
                # extract features
                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)

                masked_data, mask = masking(trg_x, num_splits=self.num_splits, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(trg_x, max_segments=self.configs.max_seg),
                                  self.configs.jitter_ratio).float().to(self.device)
                aug_data2 = jitter(permutation(trg_x, max_segments=5), 0.001).cpu().float().to(self.device)


                trg_feat_aug2, seq_trg_feat_aug2 = self.feature_extractor(aug_data2)
                trg_feat_aug, seq_trg_feat_aug = self.feature_extractor(aug_data)
                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                '''
                内容对比模块
                '''
                transform_seq_trg_feat_aug = self.seq_transformer(trans(self.configs.timesteps, seq_trg_feat_aug))
                transform_seq_trg_feat = self.seq_transformer(trans(self.configs.timesteps, trg_feat_seq))
                # print("transform_seq_src_feat_aug:", transform_seq_trg_feat_aug.shape)
                nt_xent_criterion = NTXentLoss(self.configs, self.device, self.configs.batch_size,
                                               self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss = nt_xent_criterion(transform_seq_trg_feat_aug, transform_seq_trg_feat)



                strong = torch.stack((seq_trg_feat_aug, seq_trg_feat_aug2), dim=1).permute(0, 2, 1, 3)
                weak = torch.stack((trg_feat_seq, trg_feat_seq), dim=1).permute(0, 2, 1, 3)
                fea_strong, seq_fea_strong = self.feature_extractor2(strong)
                fea_weak, seq_fea_weak = self.feature_extractor2(weak)
                transform_seq_src_feat_aug = self.seq_transformerca(seq_fea_strong)
                transform_seq_src_feat = self.seq_transformerca(seq_fea_weak)
                # transform_seq_src_feat_aug = self.seq_transformerca(strong)
                # transform_seq_src_feat = self.seq_transformerca(weak)
                nt_xent_criterion2 = NTXentLoss2(self.configs, self.device, self.configs.batch_size,
                                                 self.configs.temperature,
                                                 self.configs.use_cosine_similarity)
                content_loss2 = nt_xent_criterion2(transform_seq_src_feat_aug, transform_seq_src_feat)

                # prediction scores
                trg_pred = self.classifier(trg_feat)

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))

                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                '''
                Overall objective loss
                '''
                # removing trg ent
                loss = trg_ent + self.hparams['TOV_wt'] * tov_loss +content_loss+content_loss2

                loss.backward()
                self.optimizer.step()
                self.fea2_optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()
                self.transca_optimizer.step()
                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item(),'content_loss': content_loss.detach().item(),'content_loss2': content_loss2.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

class SHOT(Algorithm):

    def __init__(self, backbone, configs, hparams, device, args):
        super(SHOT, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger, dataset):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # Freeze the classifier
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # obtain pseudo labels for each epoch
            pseudo_labels = self.obtain_pseudo_labels(trg_dataloader)

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                # print("step:",step)
                # print("trg_x:",trg_x.shape)
                # print("trg_idx.shape:",trg_idx.shape)
                # print("trg_idx:", trg_idx)
                trg_x = trg_x.float().to(self.device)

                # prevent gradient accumulation
                self.optimizer.zero_grad()

                # Extract features
                trg_feat, _ = self.feature_extractor(trg_x)
                trg_pred = self.classifier(trg_feat)

                # pseudo labeling loss
                # print("trg_pre:", trg_pred)
                # if dataset =='EEG':
                inn = 0
                for i in range(trg_idx.shape[0]):
                    if trg_idx[i].long() < len(pseudo_labels):
                        la_idx = trg_pred[i].unsqueeze(0)
                        inn = i+1
                        break
                    else:
                        trg_idx[i] = -1
                # print(la_idx)
                for i in range(inn,trg_idx.shape[0], 1):
                    if trg_idx[i].long() >= len(pseudo_labels):
                        trg_idx[i] = -1
                    else:
                        la_idx = torch.cat((la_idx, trg_pred[i].unsqueeze(0)), dim=0)
                        # print("after:",la_idx)
                mask1 = trg_idx != -1
                trg_idx = torch.masked_select(trg_idx, mask1)
                trg_pred = la_idx
                # print("trg_pred1:", len(trg_pred))
                # print(trg_pred)


                pseudo_label = pseudo_labels[trg_idx.long()].to(self.device)
                # print("trg_pred2:",len(trg_pred))
                # print("pseudo_label:",len(pseudo_label))

                target_loss = F.cross_entropy(trg_pred.squeeze(), pseudo_label.long())

                # Entropy loss
                softmax_out = nn.Softmax(dim=1)(trg_pred)
                entropy_loss = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(softmax_out))

                #  Information maximization loss
                entropy_loss -= self.hparams['im'] * torch.sum(
                    -softmax_out.mean(dim=0) * torch.log(softmax_out.mean(dim=0) + 1e-5))

                # Total loss
                loss = entropy_loss + self.hparams['target_cls_wt'] * target_loss

                # self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                losses = {'Total_loss': loss.item(), 'Target_loss': target_loss.item(),
                          'Ent_loss': entropy_loss.detach().item()}

                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model


    def obtain_pseudo_labels(self, trg_loader):
        self.feature_extractor.eval()
        self.classifier.eval()
        preds, feas = [], []
        with torch.no_grad():
            for inputs, labels, _ in trg_loader:
                inputs = inputs.float().to(self.device)
                features, _ = self.feature_extractor(inputs)
                predictions = self.classifier(features)
                preds.append(predictions)
                feas.append(features)
                # print("preds:", len(preds))

        preds = torch.cat((preds))
        feas = torch.cat((feas))
        preds = nn.Softmax(dim=1)(preds)
        _, predict = torch.max(preds, 1)

        all_features = torch.cat((feas, torch.ones(feas.size(0), 1).to(self.device)), 1)
        all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()
        all_features = all_features.float().cpu().numpy()

        K = preds.size(1)
        aff = preds.float().cpu().numpy()
        initc = aff.transpose().dot(all_features)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_features, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = torch.from_numpy(pred_label)

        #多次更新伪标签
        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_features)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_features, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
            pred_label = torch.from_numpy(pred_label)
        return pred_label

class AaD(Algorithm):
    """
    (NeurIPS 2022 Spotlight) Attracting and Dispersing: A Simple Approach for Source-free Domain Adaptation
    https://github.com/Albert0147/AaD_SFDA
    """

    def __init__(self, backbone, configs, hparams, device, args):
        super(AaD, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            # inilize alpha value

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                # Extract features
                features, _ = self.feature_extractor(trg_x)
                predictions = self.classifier(features)
                num_samples = len(trg_dataloader.dataset)
                fea_bank = torch.randn(num_samples, self.configs.final_out_channels * self.configs.features_len)
                score_bank = torch.randn(num_samples, self.configs.num_classes).to(self.device)
                softmax_out = nn.Softmax(dim=1)(predictions)

                alpha = (1 + 10 * step / self.hparams["num_epochs"] * len(trg_dataloader)) ** (-self.hparams['beta']) * \
                        self.hparams['alpha']
                with torch.no_grad():
                    output_f_norm = F.normalize(features)
                    output_f_ = output_f_norm.cpu().detach().clone()

                    fea_bank[trg_idx] = output_f_.detach().clone().cpu()
                    score_bank[trg_idx] = softmax_out.detach().clone()

                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             k=5 + 1)
                    idx_near = idx_near[:, 1:]  # batch x K
                    score_near = score_bank[idx_near]  # batch x K x C

                # start gradients
                softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C

                loss = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)).sum(1))

                mask = torch.ones((trg_x.shape[0], trg_x.shape[0]))
                diag_num = torch.diag(mask)
                mask_diag = torch.diag_embed(diag_num)
                mask = mask - mask_diag
                copy = softmax_out.T  # .detach().clone()#

                dot_neg = softmax_out @ copy  # batch x batch

                dot_neg = (dot_neg * mask.to(self.device)).sum(-1)  # batch
                neg_pred = torch.mean(dot_neg)
                loss += neg_pred * alpha

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # meter updates
                avg_meter['Total_loss'].update(loss.item(), 32)

            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model


class NRC(Algorithm):
    """
    Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation (NIPS 2021)
    https://github.com/Albert0147/NRC_SFDA
    """

    def __init__(self, backbone, configs, hparams, device, args):
        super(NRC, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()

                # extract features
                src_feat, _ = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                src_cls_loss.backward()

                # update weights
                self.pre_optimizer.step()

                # acculate loss
                avg_meter['Src_cls_loss'].update(src_cls_loss.item(), 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)
                # Extract features
                features, _ = self.feature_extractor(trg_x)
                predictions = self.classifier(features)
                num_samples = len(trg_dataloader.dataset)
                fea_bank = torch.randn(num_samples, self.configs.final_out_channels * self.configs.features_len)
                score_bank = torch.randn(num_samples, self.configs.num_classes).to(self.device)
                softmax_out = nn.Softmax(dim=1)(predictions)

                with torch.no_grad():
                    output_f_norm = F.normalize(features)
                    output_f_ = output_f_norm.cpu().detach().clone()

                    fea_bank[trg_idx] = output_f_.detach().clone().cpu()
                    score_bank[trg_idx] = softmax_out.detach().clone()

                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             k=5 + 1)
                    idx_near = idx_near[:, 1:]  # batch x K
                    score_near = score_bank[idx_near]  # batch x K x C

                    fea_near = fea_bank[idx_near]  # batch x K x num_dim
                    fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
                    distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
                    _, idx_near_near = torch.topk(distance_, dim=-1, largest=True,
                                                  k=5 + 1)  # M near neighbors for each of above K ones
                    idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
                    trg_idx_ = trg_idx.unsqueeze(-1).unsqueeze(-1)
                    match = (
                            idx_near_near == trg_idx_).sum(-1).float()  # batch x K
                    weight = torch.where(
                        match > 0., match,
                        torch.ones_like(match).fill_(0.1))  # batch x K

                    weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                            5)  # batch x K x M
                    weight_kk = weight_kk.fill_(0.1)

                    # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
                    # weight_kk[idx_near_near == trg_idx_]=0

                    score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
                    # print(weight_kk.shape)
                    weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                            -1)  # batch x KM

                    score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                                                    self.configs.num_classes)  # batch x KM x C

                    score_self = score_bank[trg_idx]

                # start gradients
                output_re = softmax_out.unsqueeze(1).expand(-1, 5 * 5,
                                                            -1)  # batch x C x 1
                const = torch.mean(
                    (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
                     weight_kk.to(self.device)).sum(
                        1))  # kl_div here equals to dot product since we do not use log for score_near_kk
                loss = torch.mean(const)

                # nn
                softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C

                loss += torch.mean(
                    (F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.to(self.device)).sum(1))

                # self, if not explicitly removing the self feature in expanded neighbor then no need for this
                # loss += -torch.mean((softmax_out * score_self).sum(-1))

                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(msoftmax *
                                          torch.log(msoftmax + self.hparams['epsilon']))
                loss += gentropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # meter updates
                avg_meter['Total_loss'].update(loss.item(), 32)

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model


class MAPU(Algorithm):

    def __init__(self, backbone, configs, hparams, device):
        super(MAPU, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=10, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # extract features
                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)

                masked_data, mask = masking(trg_x, num_splits=10, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                # prediction scores
                trg_pred = self.classifier(trg_feat)

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))

                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                '''
                Overall objective loss
                '''
                # removing trg ent
                loss = trg_ent + self.hparams['TOV_wt'] * tov_loss

                loss.backward()
                self.optimizer.step()
                self.tov_optimizer.step()

                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model



class TODAwoShallow(Algorithm):

    def __init__(self, backbone, configs, hparams, device, args):
        super(TODAwoShallow, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.feature_extractor2 = CNN2d(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)
        self.seq_transformer = Seq_Transformer(patch_size=configs.final_out_channels, dim=configs.AR_hid_dim, depth=4,
                                               heads=4, mlp_dim=64)
        self.seq_transformerca = Seq_TransformerCA(patch_size=configs.features_len, dim=configs.AR_hid_dim, depth=4,
                                                   heads=4, mlp_dim=64)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.trans_optimizer = torch.optim.Adam(
            self.seq_transformer.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.transca_optimizer = torch.optim.Adam(
            self.seq_transformerca.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.fea2_optimizer = torch.optim.Adam(
            self.feature_extractor2.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams
        self.num_splits = args.splits
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # print("enumurate src.shape:", src_x.shape)
                if src_x.shape[0]!= self.hparams['batch_size']:
                    continue
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.fea2_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()
                self.transca_optimizer.zero_grad()
                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)
                # print("src_feat:",src_feat.shape)
                # print("seq_src_feat:",seq_src_feat.shape)
                # masking the input_sequences
                # print(src_x.shape)
                masked_data, mask = masking(src_x, num_splits=self.num_splits, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(src_x, max_segments=self.configs.max_seg), self.configs.jitter_ratio).cpu().float().to(self.device)
                aug_data2 = jitter(permutation(src_x, max_segments=5),0.001).cpu().float().to(self.device)

                src_feat_aug, seq_src_feat_aug = self.feature_extractor(aug_data)
                src_feat_aug2, seq_src_feat_aug2 = self.feature_extractor(aug_data2)
                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                '''
                内容对比模块
                '''
                # print(seq_src_feat_aug.shape)

                '''
                CA提取前后关系
                '''
                strong = torch.stack((seq_src_feat_aug, seq_src_feat_aug2), dim=1).permute(0, 2, 1, 3)
                weak = torch.stack((seq_src_feat, seq_src_feat), dim=1).permute(0, 2, 1, 3)
                fea_strong, seq_fea_strong = self.feature_extractor2(strong)
                fea_weak, seq_fea_weak = self.feature_extractor2(weak)
                transform_seq_src_feat_aug = self.seq_transformerca(seq_fea_strong)
                transform_seq_src_feat = self.seq_transformerca(seq_fea_weak)
                # transform_seq_src_feat_aug = self.seq_transformerca(strong)
                # transform_seq_src_feat = self.seq_transformerca(weak)
                nt_xent_criterion2 = NTXentLoss2(self.configs, self.device, self.configs.batch_size,
                                               self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss2 = nt_xent_criterion2(transform_seq_src_feat_aug, transform_seq_src_feat)


                # classifier predictions
                # print("#####src_feat: ", src_feat.shape)
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss + content_loss2
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()
                losses = {'entropy_loss': src_cls_loss.detach().item(), 'Masking_loss': tov_loss.detach().item(),'content_loss2': content_loss2.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False
        for k, v in self.seq_transformer.named_parameters():
            v.requires_grad = False
        for k, v in self.seq_transformerca.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                if trg_x.shape[0]!= self.hparams['batch_size']:
                    continue
                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.fea2_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()
                self.transca_optimizer.zero_grad()
                # extract features
                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)

                masked_data, mask = masking(trg_x, num_splits=self.num_splits, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(trg_x, max_segments=self.configs.max_seg),
                                  self.configs.jitter_ratio).float().to(self.device)
                aug_data2 = jitter(permutation(trg_x, max_segments=5), 0.001).cpu().float().to(self.device)


                trg_feat_aug2, seq_trg_feat_aug2 = self.feature_extractor(aug_data2)
                trg_feat_aug, seq_trg_feat_aug = self.feature_extractor(aug_data)
                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                '''
                内容对比模块
                '''



                strong = torch.stack((seq_trg_feat_aug, seq_trg_feat_aug2), dim=1).permute(0, 2, 1, 3)
                weak = torch.stack((trg_feat_seq, trg_feat_seq), dim=1).permute(0, 2, 1, 3)
                fea_strong, seq_fea_strong = self.feature_extractor2(strong)
                fea_weak, seq_fea_weak = self.feature_extractor2(weak)
                transform_seq_src_feat_aug = self.seq_transformerca(seq_fea_strong)
                transform_seq_src_feat = self.seq_transformerca(seq_fea_weak)
                # transform_seq_src_feat_aug = self.seq_transformerca(strong)
                # transform_seq_src_feat = self.seq_transformerca(weak)
                nt_xent_criterion2 = NTXentLoss2(self.configs, self.device, self.configs.batch_size,
                                                 self.configs.temperature,
                                                 self.configs.use_cosine_similarity)
                content_loss2 = nt_xent_criterion2(transform_seq_src_feat_aug, transform_seq_src_feat)

                # prediction scores
                trg_pred = self.classifier(trg_feat)

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))

                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                '''
                Overall objective loss
                '''
                # removing trg ent
                loss = trg_ent + self.hparams['TOV_wt'] * tov_loss +content_loss2

                loss.backward()
                self.optimizer.step()
                self.fea2_optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()
                self.transca_optimizer.step()
                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item(),'content_loss2': content_loss2.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

class TODAwoTC(Algorithm):

    def __init__(self, backbone, configs, hparams, device, args):
        super(TODAwoTC, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.feature_extractor2 = CNN2d(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)
        self.seq_transformer = Seq_Transformer(patch_size=configs.final_out_channels, dim=configs.AR_hid_dim, depth=4,
                                               heads=4, mlp_dim=64)
        self.seq_transformerca = Seq_TransformerCA(patch_size=configs.features_len, dim=configs.AR_hid_dim, depth=4,
                                                   heads=4, mlp_dim=64)
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.trans_optimizer = torch.optim.Adam(
            self.seq_transformer.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.transca_optimizer = torch.optim.Adam(
            self.seq_transformerca.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.fea2_optimizer = torch.optim.Adam(
            self.feature_extractor2.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams
        self.num_splits = args.splits
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # print("enumurate src.shape:", src_x.shape)
                if src_x.shape[0]!= self.hparams['batch_size']:
                    continue
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.fea2_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()
                self.transca_optimizer.zero_grad()
                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)
                # print("src_feat:",src_feat.shape)
                # print("seq_src_feat:",seq_src_feat.shape)
                # masking the input_sequences
                # print(src_x.shape)
                # masked_data, mask = masking(src_x, num_splits=self.num_splits, num_masked=1)
                # src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(src_x, max_segments=self.configs.max_seg), self.configs.jitter_ratio).cpu().float().to(self.device)
                aug_data2 = jitter(permutation(src_x, max_segments=5),0.001).cpu().float().to(self.device)

                src_feat_aug, seq_src_feat_aug = self.feature_extractor(aug_data)
                src_feat_aug2, seq_src_feat_aug2 = self.feature_extractor(aug_data2)
                ''' Temporal order verification  '''
                # pass the data with and without detach
                # tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                # tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                '''
                内容对比模块
                '''
                # print(seq_src_feat_aug.shape)
                transform_seq_src_feat_aug = self.seq_transformer(trans(self.configs.timesteps,seq_src_feat_aug))
                transform_seq_src_feat = self.seq_transformer(trans(self.configs.timesteps,seq_src_feat))
                # print("transform_seq_src_feat_aug:",transform_seq_src_feat_aug.shape)
                nt_xent_criterion = NTXentLoss(self.configs, self.device, self.configs.batch_size, self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss = nt_xent_criterion(transform_seq_src_feat_aug, transform_seq_src_feat)

                '''
                CA提取前后关系
                '''
                strong = torch.stack((seq_src_feat_aug, seq_src_feat_aug2), dim=1).permute(0, 2, 1, 3)
                weak = torch.stack((seq_src_feat, seq_src_feat), dim=1).permute(0, 2, 1, 3)
                fea_strong, seq_fea_strong = self.feature_extractor2(strong)
                fea_weak, seq_fea_weak = self.feature_extractor2(weak)
                transform_seq_src_feat_aug = self.seq_transformerca(seq_fea_strong)
                transform_seq_src_feat = self.seq_transformerca(seq_fea_weak)
                # transform_seq_src_feat_aug = self.seq_transformerca(strong)
                # transform_seq_src_feat = self.seq_transformerca(weak)
                nt_xent_criterion2 = NTXentLoss2(self.configs, self.device, self.configs.batch_size,
                                               self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss2 = nt_xent_criterion2(transform_seq_src_feat_aug, transform_seq_src_feat)


                # classifier predictions
                # print("#####src_feat: ", src_feat.shape)
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss  + content_loss + content_loss2
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()
                losses = {'entropy_loss': src_cls_loss.detach().item(),'content_loss': content_loss.detach().item(),'content_loss2': content_loss2.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False
        for k, v in self.seq_transformer.named_parameters():
            v.requires_grad = False
        for k, v in self.seq_transformerca.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                if trg_x.shape[0]!= self.hparams['batch_size']:
                    continue
                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.fea2_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()
                self.transca_optimizer.zero_grad()
                # extract features
                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)

                # masked_data, mask = masking(trg_x, num_splits=self.num_splits, num_masked=1)
                # trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(trg_x, max_segments=self.configs.max_seg),
                                  self.configs.jitter_ratio).float().to(self.device)
                aug_data2 = jitter(permutation(trg_x, max_segments=5), 0.001).cpu().float().to(self.device)


                trg_feat_aug2, seq_trg_feat_aug2 = self.feature_extractor(aug_data2)
                trg_feat_aug, seq_trg_feat_aug = self.feature_extractor(aug_data)
                # tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                # tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                '''
                内容对比模块
                '''
                transform_seq_trg_feat_aug = self.seq_transformer(trans(self.configs.timesteps, seq_trg_feat_aug))
                transform_seq_trg_feat = self.seq_transformer(trans(self.configs.timesteps, trg_feat_seq))
                # print("transform_seq_src_feat_aug:", transform_seq_trg_feat_aug.shape)
                nt_xent_criterion = NTXentLoss(self.configs, self.device, self.configs.batch_size,
                                               self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss = nt_xent_criterion(transform_seq_trg_feat_aug, transform_seq_trg_feat)



                strong = torch.stack((seq_trg_feat_aug, seq_trg_feat_aug2), dim=1).permute(0, 2, 1, 3)
                weak = torch.stack((trg_feat_seq, trg_feat_seq), dim=1).permute(0, 2, 1, 3)
                fea_strong, seq_fea_strong = self.feature_extractor2(strong)
                fea_weak, seq_fea_weak = self.feature_extractor2(weak)
                transform_seq_src_feat_aug = self.seq_transformerca(seq_fea_strong)
                transform_seq_src_feat = self.seq_transformerca(seq_fea_weak)
                # transform_seq_src_feat_aug = self.seq_transformerca(strong)
                # transform_seq_src_feat = self.seq_transformerca(weak)
                nt_xent_criterion2 = NTXentLoss2(self.configs, self.device, self.configs.batch_size,
                                                 self.configs.temperature,
                                                 self.configs.use_cosine_similarity)
                content_loss2 = nt_xent_criterion2(transform_seq_src_feat_aug, transform_seq_src_feat)

                # prediction scores
                trg_pred = self.classifier(trg_feat)

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))

                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                '''
                Overall objective loss
                '''
                # removing trg ent
                loss = trg_ent + content_loss + content_loss2

                loss.backward()
                self.optimizer.step()
                self.fea2_optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()
                self.transca_optimizer.step()
                losses = {'entropy_loss': trg_ent.detach().item(), 'content_loss': content_loss.detach().item(),'content_loss2': content_loss2.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model


class TODAwoDeep(Algorithm):

    def __init__(self, backbone, configs, hparams, device, args):
        super(TODAwoDeep, self).__init__(configs)

        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)
        self.seq_transformer = Seq_Transformer(patch_size=configs.final_out_channels, dim=configs.AR_hid_dim, depth=4,
                                               heads=4, mlp_dim=64)

        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.trans_optimizer = torch.optim.Adam(
            self.seq_transformer.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        # device
        self.device = device
        self.hparams = hparams
        self.num_splits = args.splits
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # print("enumurate src.shape:", src_x.shape)
                if src_x.shape[0]!= self.hparams['batch_size']:
                    continue
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()

                # forward pass correct sequences
                src_feat, seq_src_feat = self.feature_extractor(src_x)
                # print("src_feat:",src_feat.shape)
                # print("seq_src_feat:",seq_src_feat.shape)
                # masking the input_sequences
                # print(src_x.shape)
                masked_data, mask = masking(src_x, num_splits=self.num_splits, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(src_x, max_segments=self.configs.max_seg), self.configs.jitter_ratio).cpu().float().to(self.device)
                src_feat_aug, seq_src_feat_aug = self.feature_extractor(aug_data)

                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                '''
                内容对比模块
                '''
                # print(seq_src_feat_aug.shape)
                transform_seq_src_feat_aug = self.seq_transformer(trans(self.configs.timesteps,seq_src_feat_aug))
                transform_seq_src_feat = self.seq_transformer(trans(self.configs.timesteps,seq_src_feat))
                # print("transform_seq_src_feat_aug:",transform_seq_src_feat_aug.shape)
                nt_xent_criterion = NTXentLoss(self.configs, self.device, self.configs.batch_size, self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss = nt_xent_criterion(transform_seq_src_feat_aug, transform_seq_src_feat)

                # classifier predictions
                # print("#####src_feat: ", src_feat.shape)
                src_pred = self.classifier(src_feat)

                # normal cross entropy
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss + content_loss
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()
                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')
        src_only_model = deepcopy(self.network.state_dict())
        return src_only_model

    def update(self, trg_dataloader, avg_meter, logger):

        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False
        for k, v in self.seq_transformer.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                if trg_x.shape[0]!= self.hparams['batch_size']:
                    continue
                trg_x = trg_x.float().to(self.device)

                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()
                # extract features
                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)

                masked_data, mask = masking(trg_x, num_splits=self.num_splits, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(trg_x, max_segments=self.configs.max_seg),
                                  self.configs.jitter_ratio).float().to(self.device)
                trg_feat_aug, seq_trg_feat_aug = self.feature_extractor(aug_data)

                '''
                内容对比模块
                '''
                transform_seq_trg_feat_aug = self.seq_transformer(trans(self.configs.timesteps, seq_trg_feat_aug))
                transform_seq_trg_feat = self.seq_transformer(trans(self.configs.timesteps, trg_feat_seq))
                # print("transform_seq_src_feat_aug:", transform_seq_trg_feat_aug.shape)
                nt_xent_criterion = NTXentLoss(self.configs, self.device, self.configs.batch_size,
                                               self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss = nt_xent_criterion(transform_seq_trg_feat_aug, transform_seq_trg_feat)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                # prediction scores
                trg_pred = self.classifier(trg_feat)

                # select evidential vs softmax probabilities
                trg_prob = nn.Softmax(dim=1)(trg_pred)

                # Entropy loss
                trg_ent = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(trg_prob))

                # IM loss
                trg_ent -= self.hparams['im'] * torch.sum(
                    -trg_prob.mean(dim=0) * torch.log(trg_prob.mean(dim=0) + 1e-5))

                '''
                Overall objective loss
                '''
                # removing trg ent
                loss = trg_ent + self.hparams['TOV_wt'] * tov_loss +content_loss

                loss.backward()
                self.optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()
                losses = {'entropy_loss': trg_ent.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model



class SHOTTODA(Algorithm):

    def __init__(self, backbone, configs, hparams, device, args):
        super(SHOTTODA, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.feature_extractor2 = CNN2d(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)
        self.seq_transformer = Seq_Transformer(patch_size=configs.final_out_channels, dim=configs.AR_hid_dim, depth=4,
                                               heads=4, mlp_dim=64)
        self.seq_transformerca = Seq_TransformerCA(patch_size=configs.features_len, dim=configs.AR_hid_dim, depth=4,
                                                   heads=4, mlp_dim=64)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.trans_optimizer = torch.optim.Adam(
            self.seq_transformer.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.transca_optimizer = torch.optim.Adam(
            self.seq_transformerca.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.fea2_optimizer = torch.optim.Adam(
            self.feature_extractor2.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])
        self.num_splits = args.splits
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                if src_x.shape[0] != self.hparams['batch_size']:
                    continue
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()
                self.fea2_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()
                self.transca_optimizer.zero_grad()

                # extract features
                src_feat, seq_src_feat = self.feature_extractor(src_x)
                masked_data, mask = masking(src_x, num_splits=self.num_splits, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(src_x, max_segments=self.configs.max_seg),
                                  self.configs.jitter_ratio).cpu().float().to(self.device)
                aug_data2 = jitter(permutation(src_x, max_segments=5), 0.001).cpu().float().to(self.device)

                src_feat_aug, seq_src_feat_aug = self.feature_extractor(aug_data)
                src_feat_aug2, seq_src_feat_aug2 = self.feature_extractor(aug_data2)
                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                '''
                                内容对比模块
                                '''
                # print(seq_src_feat_aug.shape)
                transform_seq_src_feat_aug = self.seq_transformer(trans(self.configs.timesteps, seq_src_feat_aug))
                transform_seq_src_feat = self.seq_transformer(trans(self.configs.timesteps, seq_src_feat))
                # print("transform_seq_src_feat_aug:",transform_seq_src_feat_aug.shape)
                nt_xent_criterion = NTXentLoss(self.configs, self.device, self.configs.batch_size,
                                               self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss = nt_xent_criterion(transform_seq_src_feat_aug, transform_seq_src_feat)

                '''
                CA提取前后关系
                '''
                strong = torch.stack((seq_src_feat_aug, seq_src_feat_aug2), dim=1).permute(0, 2, 1, 3)
                weak = torch.stack((seq_src_feat, seq_src_feat), dim=1).permute(0, 2, 1, 3)
                fea_strong, seq_fea_strong = self.feature_extractor2(strong)
                fea_weak, seq_fea_weak = self.feature_extractor2(weak)
                transform_seq_src_feat_aug = self.seq_transformerca(seq_fea_strong)
                transform_seq_src_feat = self.seq_transformerca(seq_fea_weak)
                nt_xent_criterion2 = NTXentLoss2(self.configs, self.device, self.configs.batch_size,
                                                 self.configs.temperature,
                                                 self.configs.use_cosine_similarity)
                content_loss2 = nt_xent_criterion2(transform_seq_src_feat_aug, transform_seq_src_feat)

                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss + content_loss+content_loss2

                # calculate gradients
                total_loss.backward()

                # update weights
                self.pre_optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger, dataset):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False
        for k, v in self.seq_transformer.named_parameters():
            v.requires_grad = False
        for k, v in self.seq_transformerca.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # obtain pseudo labels for each epoch
            pseudo_labels = self.obtain_pseudo_labels(trg_dataloader)

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                if trg_x.shape[0] != self.hparams['batch_size']:
                    continue
                trg_x = trg_x.float().to(self.device)

                # prevent gradient accumulation
                self.optimizer.zero_grad()
                self.fea2_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()
                self.transca_optimizer.zero_grad()

                # Extract features
                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)
                trg_pred = self.classifier(trg_feat)

                # pseudo labeling loss
                # print("trg_pre:", trg_pred)
                # if dataset == 'EEG':
                inn = 0
                for i in range(trg_idx.shape[0]):
                    if trg_idx[i].long() < len(pseudo_labels):
                        la_idx = trg_pred[i].unsqueeze(0)
                        inn = i + 1
                        break
                    else:
                        trg_idx[i] = -1
                # print(la_idx)
                for i in range(inn, trg_idx.shape[0], 1):
                    if trg_idx[i].long() >= len(pseudo_labels):
                        trg_idx[i] = -1
                    else:
                        la_idx = torch.cat((la_idx, trg_pred[i].unsqueeze(0)), dim=0)
                        # print("after:",la_idx)
                mask1 = trg_idx != -1
                trg_idx = torch.masked_select(trg_idx, mask1)
                trg_pred = la_idx
                    # print("trg_pred1:", len(trg_pred))
                    # print(trg_pred)

                pseudo_label = pseudo_labels[trg_idx.long()].to(self.device)
                # print("trg_pred2:",len(trg_pred))
                # print("pseudo_label:",len(pseudo_label))

                target_loss = F.cross_entropy(trg_pred.squeeze(), pseudo_label.long())

                # Entropy loss
                softmax_out = nn.Softmax(dim=1)(trg_pred)
                entropy_loss = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(softmax_out))

                #  Information maximization loss
                entropy_loss -= self.hparams['im'] * torch.sum(
                    -softmax_out.mean(dim=0) * torch.log(softmax_out.mean(dim=0) + 1e-5))

                masked_data, mask = masking(trg_x, num_splits=self.num_splits, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(trg_x, max_segments=self.configs.max_seg),
                                  self.configs.jitter_ratio).float().to(self.device)
                aug_data2 = jitter(permutation(trg_x, max_segments=5), 0.001).cpu().float().to(self.device)

                trg_feat_aug2, seq_trg_feat_aug2 = self.feature_extractor(aug_data2)
                trg_feat_aug, seq_trg_feat_aug = self.feature_extractor(aug_data)
                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                '''
                内容对比模块
                '''
                transform_seq_trg_feat_aug = self.seq_transformer(trans(self.configs.timesteps, seq_trg_feat_aug))
                transform_seq_trg_feat = self.seq_transformer(trans(self.configs.timesteps, trg_feat_seq))
                # print("transform_seq_src_feat_aug:", transform_seq_trg_feat_aug.shape)
                nt_xent_criterion = NTXentLoss(self.configs, self.device, self.configs.batch_size,
                                               self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss = nt_xent_criterion(transform_seq_trg_feat_aug, transform_seq_trg_feat)

                strong = torch.stack((seq_trg_feat_aug, seq_trg_feat_aug2), dim=1).permute(0, 2, 1, 3)
                weak = torch.stack((trg_feat_seq, trg_feat_seq), dim=1).permute(0, 2, 1, 3)
                fea_strong, seq_fea_strong = self.feature_extractor2(strong)
                fea_weak, seq_fea_weak = self.feature_extractor2(weak)
                transform_seq_src_feat_aug = self.seq_transformerca(seq_fea_strong)
                transform_seq_src_feat = self.seq_transformerca(seq_fea_weak)
                nt_xent_criterion2 = NTXentLoss2(self.configs, self.device, self.configs.batch_size,
                                                 self.configs.temperature,
                                                 self.configs.use_cosine_similarity)
                content_loss2 = nt_xent_criterion2(transform_seq_src_feat_aug, transform_seq_src_feat)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                # Total loss
                loss = entropy_loss + self.hparams['target_cls_wt'] * target_loss + content_loss + content_loss2 +self.hparams[
                    'TOV_wt'] * tov_loss

                # self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.fea2_optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()
                self.transca_optimizer.step()

                losses = {'Total_loss': loss.item(), 'Target_loss': target_loss.item(),
                          'Ent_loss': entropy_loss.detach().item()}

                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()
            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

    def obtain_pseudo_labels(self, trg_loader):
        self.feature_extractor.eval()
        self.classifier.eval()
        preds, feas = [], []
        with torch.no_grad():
            for inputs, labels, _ in trg_loader:
                inputs = inputs.float().to(self.device)
                features, _ = self.feature_extractor(inputs)
                predictions = self.classifier(features)
                preds.append(predictions)
                feas.append(features)
                # print("preds:", len(preds))

        preds = torch.cat((preds))
        feas = torch.cat((feas))
        preds = nn.Softmax(dim=1)(preds)
        _, predict = torch.max(preds, 1)

        all_features = torch.cat((feas, torch.ones(feas.size(0), 1).to(self.device)), 1)
        all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()
        all_features = all_features.float().cpu().numpy()

        K = preds.size(1)
        aff = preds.float().cpu().numpy()
        initc = aff.transpose().dot(all_features)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_features, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = torch.from_numpy(pred_label)

        # 多次更新伪标签
        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_features)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_features, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
            pred_label = torch.from_numpy(pred_label)
        return pred_label


class AaDTODA(Algorithm):
    """
    (NeurIPS 2022 Spotlight) Attracting and Dispersing: A Simple Approach for Source-free Domain Adaptation
    https://github.com/Albert0147/AaD_SFDA
    """

    def __init__(self, backbone, configs, hparams, device, args):
        super(AaDTODA, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.feature_extractor2 = CNN2d(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)
        self.seq_transformer = Seq_Transformer(patch_size=configs.final_out_channels, dim=configs.AR_hid_dim, depth=4,
                                               heads=4, mlp_dim=64)
        self.seq_transformerca = Seq_TransformerCA(patch_size=configs.features_len, dim=configs.AR_hid_dim, depth=4,
                                                   heads=4, mlp_dim=64)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.trans_optimizer = torch.optim.Adam(
            self.seq_transformer.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.transca_optimizer = torch.optim.Adam(
            self.seq_transformerca.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.fea2_optimizer = torch.optim.Adam(
            self.feature_extractor2.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.num_splits = args.splits
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )
        self.mse_loss = nn.MSELoss()
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                if src_x.shape[0] != self.hparams['batch_size']:
                    continue
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()
                self.fea2_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()
                self.transca_optimizer.zero_grad()

                # extract features
                src_feat, seq_src_feat = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=self.num_splits, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(src_x, max_segments=self.configs.max_seg),
                                  self.configs.jitter_ratio).cpu().float().to(self.device)
                aug_data2 = jitter(permutation(src_x, max_segments=5), 0.001).cpu().float().to(self.device)

                src_feat_aug, seq_src_feat_aug = self.feature_extractor(aug_data)
                src_feat_aug2, seq_src_feat_aug2 = self.feature_extractor(aug_data2)
                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                '''
                内容对比模块
                '''
                # print(seq_src_feat_aug.shape)
                transform_seq_src_feat_aug = self.seq_transformer(trans(self.configs.timesteps,seq_src_feat_aug))
                transform_seq_src_feat = self.seq_transformer(trans(self.configs.timesteps,seq_src_feat))
                # print("transform_seq_src_feat_aug:",transform_seq_src_feat_aug.shape)
                nt_xent_criterion = NTXentLoss(self.configs, self.device, self.configs.batch_size, self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss = nt_xent_criterion(transform_seq_src_feat_aug, transform_seq_src_feat)

                '''
                CA提取前后关系
                '''
                strong = torch.stack((seq_src_feat_aug, seq_src_feat_aug2), dim=1).permute(0, 2, 1, 3)
                weak = torch.stack((seq_src_feat, seq_src_feat), dim=1).permute(0, 2, 1, 3)
                fea_strong, seq_fea_strong = self.feature_extractor2(strong)
                fea_weak, seq_fea_weak = self.feature_extractor2(weak)
                transform_seq_src_feat_aug = self.seq_transformerca(seq_fea_strong)
                transform_seq_src_feat = self.seq_transformerca(seq_fea_weak)
                nt_xent_criterion2 = NTXentLoss2(self.configs, self.device, self.configs.batch_size,
                                               self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss2 = nt_xent_criterion2(transform_seq_src_feat_aug, transform_seq_src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # calculate gradients
                total_loss = src_cls_loss + tov_loss + content_loss + content_loss2
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item(), 'content_loss': content_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)


            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False
        for k, v in self.seq_transformer.named_parameters():
            v.requires_grad = False
        for k, v in self.seq_transformerca.named_parameters():
            v.requires_grad = False

        for epoch in range(1, self.hparams["num_epochs"] + 1):
            # inilize alpha value

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                if trg_x.shape[0] != self.hparams['batch_size']:
                    continue
                trg_x = trg_x.float().to(self.device)

                # self.optimizer.zero_grad()
                # self.tov_optimizer.zero_grad()
                # self.trans_optimizer.zero_grad()

                # Extract features
                features, trg_feat_seq = self.feature_extractor(trg_x)
                predictions = self.classifier(features)
                num_samples = len(trg_dataloader.dataset)
                fea_bank = torch.randn(num_samples, self.configs.final_out_channels * self.configs.features_len)
                score_bank = torch.randn(num_samples, self.configs.num_classes).to(self.device)
                softmax_out = nn.Softmax(dim=1)(predictions)

                alpha = (1 + 10 * step / self.hparams["num_epochs"] * len(trg_dataloader)) ** (-self.hparams['beta']) * \
                        self.hparams['alpha']
                with torch.no_grad():
                    output_f_norm = F.normalize(features)
                    output_f_ = output_f_norm.cpu().detach().clone()

                    fea_bank[trg_idx] = output_f_.detach().clone().cpu()
                    score_bank[trg_idx] = softmax_out.detach().clone()

                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             k=5 + 1)
                    idx_near = idx_near[:, 1:]  # batch x K
                    score_near = score_bank[idx_near]  # batch x K x C

                # start gradients
                softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C

                loss = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)).sum(1))

                mask = torch.ones((trg_x.shape[0], trg_x.shape[0]))
                diag_num = torch.diag(mask)
                mask_diag = torch.diag_embed(diag_num)
                mask = mask - mask_diag
                copy = softmax_out.T  # .detach().clone()#

                dot_neg = softmax_out @ copy  # batch x batch

                dot_neg = (dot_neg * mask.to(self.device)).sum(-1)  # batch
                neg_pred = torch.mean(dot_neg)
                loss += neg_pred * alpha

                masked_data, mask = masking(trg_x, num_splits=self.num_splits, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(trg_x, max_segments=self.configs.max_seg),
                                  self.configs.jitter_ratio).float().to(self.device)
                aug_data2 = jitter(permutation(trg_x, max_segments=5), 0.001).cpu().float().to(self.device)

                trg_feat_aug2, seq_trg_feat_aug2 = self.feature_extractor(aug_data2)
                trg_feat_aug, seq_trg_feat_aug = self.feature_extractor(aug_data)
                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                '''
                内容对比模块
                '''
                transform_seq_trg_feat_aug = self.seq_transformer(trans(self.configs.timesteps, seq_trg_feat_aug))
                transform_seq_trg_feat = self.seq_transformer(trans(self.configs.timesteps, trg_feat_seq))
                # print("transform_seq_src_feat_aug:", transform_seq_trg_feat_aug.shape)
                nt_xent_criterion = NTXentLoss(self.configs, self.device, self.configs.batch_size,
                                               self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss = nt_xent_criterion(transform_seq_trg_feat_aug, transform_seq_trg_feat)

                strong = torch.stack((seq_trg_feat_aug, seq_trg_feat_aug2), dim=1).permute(0, 2, 1, 3)
                weak = torch.stack((trg_feat_seq, trg_feat_seq), dim=1).permute(0, 2, 1, 3)
                fea_strong, seq_fea_strong = self.feature_extractor2(strong)
                fea_weak, seq_fea_weak = self.feature_extractor2(weak)
                transform_seq_src_feat_aug = self.seq_transformerca(seq_fea_strong)
                transform_seq_src_feat = self.seq_transformerca(seq_fea_weak)
                nt_xent_criterion2 = NTXentLoss2(self.configs, self.device, self.configs.batch_size,
                                                 self.configs.temperature,
                                                 self.configs.use_cosine_similarity)
                content_loss2 = nt_xent_criterion2(transform_seq_src_feat_aug, transform_seq_src_feat)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                total_loss = loss + self.hparams['TOV_wt'] * tov_loss +content_loss+content_loss2

                self.optimizer.zero_grad()
                self.fea2_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()
                self.transca_optimizer.zero_grad()

                total_loss.backward()
                self.optimizer.step()
                self.fea2_optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()
                self.transca_optimizer.step()

                # meter updates
                losses = {'Total loss': total_loss.detach().item(), 'loss':loss.detach().item(), 'masking_loss': tov_loss.detach().item(),
                      'content_loss':content_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)
            self.lr_scheduler.step()
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model


class NRCTODA(Algorithm):
    """
    Exploiting the Intrinsic Neighborhood Structure for Source-free Domain Adaptation (NIPS 2021)
    https://github.com/Albert0147/NRC_SFDA
    """

    def __init__(self, backbone, configs, hparams, device, args):
        super(NRCTODA, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.feature_extractor2 = CNN2d(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)
        self.seq_transformer = Seq_Transformer(patch_size=configs.final_out_channels, dim=configs.AR_hid_dim, depth=4,
                                               heads=4, mlp_dim=64)
        self.seq_transformerca = Seq_TransformerCA(patch_size=configs.features_len, dim=configs.AR_hid_dim, depth=4,
                                                   heads=4, mlp_dim=64)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.trans_optimizer = torch.optim.Adam(
            self.seq_transformer.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.transca_optimizer = torch.optim.Adam(
            self.seq_transformerca.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.fea2_optimizer = torch.optim.Adam(
            self.feature_extractor2.parameters(),
            lr=hparams["lr"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.num_splits = args.splits
        self.tov = args.TOV
        self.con1 = args.CON
        self.con2 = args.CON2

        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                if src_x.shape[0]!= self.hparams['batch_size']:
                    continue
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()
                self.fea2_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()
                self.transca_optimizer.zero_grad()

                # extract features
                src_feat, seq_src_feat = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                masked_data, mask = masking(src_x, num_splits=self.num_splits, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(src_x, max_segments=self.configs.max_seg),
                                  self.configs.jitter_ratio).cpu().float().to(self.device)
                aug_data2 = jitter(permutation(src_x, max_segments=5), 0.001).cpu().float().to(self.device)

                src_feat_aug, seq_src_feat_aug = self.feature_extractor(aug_data)
                src_feat_aug2, seq_src_feat_aug2 = self.feature_extractor(aug_data2)
                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                '''
                内容对比模块
                '''
                transform_seq_src_feat_aug = self.seq_transformer(trans(self.configs.timesteps, seq_src_feat_aug))
                transform_seq_src_feat = self.seq_transformer(trans(self.configs.timesteps, seq_src_feat))
                # print("transform_seq_src_feat_aug:",transform_seq_src_feat_aug.shape)
                nt_xent_criterion = NTXentLoss(self.configs, self.device, self.configs.batch_size,
                                               self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss = nt_xent_criterion(transform_seq_src_feat_aug, transform_seq_src_feat)

                '''
                               CA提取前后关系
                               '''
                strong = torch.stack((seq_src_feat_aug, seq_src_feat_aug2), dim=1).permute(0, 2, 1, 3)
                weak = torch.stack((seq_src_feat, seq_src_feat), dim=1).permute(0, 2, 1, 3)
                fea_strong, seq_fea_strong = self.feature_extractor2(strong)
                fea_weak, seq_fea_weak = self.feature_extractor2(weak)
                transform_seq_src_feat_aug = self.seq_transformerca(seq_fea_strong)
                transform_seq_src_feat = self.seq_transformerca(seq_fea_weak)
                nt_xent_criterion2 = NTXentLoss2(self.configs, self.device, self.configs.batch_size,
                                                 self.configs.temperature,
                                                 self.configs.use_cosine_similarity)
                content_loss2 = nt_xent_criterion2(transform_seq_src_feat_aug, transform_seq_src_feat)

                # calculate gradients
                total_loss = src_cls_loss + tov_loss + content_loss + content_loss2

                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()
                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False
        for k, v in self.seq_transformer.named_parameters():
            v.requires_grad = False
        for k, v in self.seq_transformerca.named_parameters():
            v.requires_grad = False

        for epoch in range(1, self.hparams["num_epochs"] + 1):

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                if trg_x.shape[0]!= self.hparams['batch_size']:
                    continue
                trg_x = trg_x.float().to(self.device)
                # Extract features
                features, trg_feat_seq= self.feature_extractor(trg_x)
                predictions = self.classifier(features)
                num_samples = len(trg_dataloader.dataset)
                fea_bank = torch.randn(num_samples, self.configs.final_out_channels * self.configs.features_len)
                score_bank = torch.randn(num_samples, self.configs.num_classes).to(self.device)
                softmax_out = nn.Softmax(dim=1)(predictions)

                with torch.no_grad():
                    output_f_norm = F.normalize(features)
                    output_f_ = output_f_norm.cpu().detach().clone()

                    fea_bank[trg_idx] = output_f_.detach().clone().cpu()
                    score_bank[trg_idx] = softmax_out.detach().clone()

                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             k=5 + 1)
                    idx_near = idx_near[:, 1:]  # batch x K
                    score_near = score_bank[idx_near]  # batch x K x C

                    fea_near = fea_bank[idx_near]  # batch x K x num_dim
                    fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
                    distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
                    _, idx_near_near = torch.topk(distance_, dim=-1, largest=True,
                                                  k=5 + 1)  # M near neighbors for each of above K ones
                    idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
                    trg_idx_ = trg_idx.unsqueeze(-1).unsqueeze(-1)
                    match = (
                            idx_near_near == trg_idx_).sum(-1).float()  # batch x K
                    weight = torch.where(
                        match > 0., match,
                        torch.ones_like(match).fill_(0.1))  # batch x K

                    weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                            5)  # batch x K x M
                    weight_kk = weight_kk.fill_(0.1)

                    # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
                    # weight_kk[idx_near_near == trg_idx_]=0

                    score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
                    # print(weight_kk.shape)
                    weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                            -1)  # batch x KM

                    score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                                                    self.configs.num_classes)  # batch x KM x C

                    score_self = score_bank[trg_idx]

                # start gradients
                output_re = softmax_out.unsqueeze(1).expand(-1, 5 * 5,
                                                            -1)  # batch x C x 1
                const = torch.mean(
                    (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
                     weight_kk.to(self.device)).sum(
                        1))  # kl_div here equals to dot product since we do not use log for score_near_kk
                loss = torch.mean(const)

                # nn
                softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C

                loss += torch.mean(
                    (F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.to(self.device)).sum(1))

                # self, if not explicitly removing the self feature in expanded neighbor then no need for this
                # loss += -torch.mean((softmax_out * score_self).sum(-1))

                msoftmax = softmax_out.mean(dim=0)
                gentropy_loss = torch.sum(msoftmax *
                                          torch.log(msoftmax + self.hparams['epsilon']))
                loss += gentropy_loss

                masked_data, mask = masking(trg_x, num_splits=self.num_splits, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                # 时序数据增强
                aug_data = jitter(permutation(trg_x, max_segments=self.configs.max_seg),
                                  self.configs.jitter_ratio).float().to(self.device)
                aug_data2 = jitter(permutation(trg_x, max_segments=5), 0.001).cpu().float().to(self.device)
                trg_feat_aug2, seq_trg_feat_aug2 = self.feature_extractor(aug_data2)
                trg_feat_aug, seq_trg_feat_aug = self.feature_extractor(aug_data)
                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                '''
                内容对比模块
                '''
                transform_seq_trg_feat_aug = self.seq_transformer(trans(self.configs.timesteps, seq_trg_feat_aug))
                transform_seq_trg_feat = self.seq_transformer(trans(self.configs.timesteps, trg_feat_seq))
                # print("transform_seq_src_feat_aug:", transform_seq_trg_feat_aug.shape)
                nt_xent_criterion = NTXentLoss(self.configs, self.device, self.configs.batch_size,
                                               self.configs.temperature,
                                               self.configs.use_cosine_similarity)
                content_loss = nt_xent_criterion(transform_seq_trg_feat_aug, transform_seq_trg_feat)
                strong = torch.stack((seq_trg_feat_aug, seq_trg_feat_aug2), dim=1).permute(0, 2, 1, 3)
                weak = torch.stack((trg_feat_seq, trg_feat_seq), dim=1).permute(0, 2, 1, 3)
                fea_strong, seq_fea_strong = self.feature_extractor2(strong)
                fea_weak, seq_fea_weak = self.feature_extractor2(weak)
                transform_seq_src_feat_aug = self.seq_transformerca(seq_fea_strong)
                transform_seq_src_feat = self.seq_transformerca(seq_fea_weak)
                nt_xent_criterion2 = NTXentLoss2(self.configs, self.device, self.configs.batch_size,
                                                 self.configs.temperature,
                                                 self.configs.use_cosine_similarity)
                content_loss2 = nt_xent_criterion2(transform_seq_src_feat_aug, transform_seq_src_feat)




                total_loss = loss + self.tov * tov_loss + content_loss*self.con1+ content_loss2*self.con2

                self.optimizer.zero_grad()
                self.fea2_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                self.trans_optimizer.zero_grad()
                self.transca_optimizer.zero_grad()

                total_loss.backward()

                self.optimizer.step()
                self.fea2_optimizer.step()
                self.tov_optimizer.step()
                self.trans_optimizer.step()
                self.transca_optimizer.step()

                losses = {'loss': loss.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            # saving the best model based on src risk
            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

class SHOTTODA(Algorithm):

    def __init__(self, backbone, configs, hparams, device, args):
        super(SHOTTODA, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.hparams = hparams
        self.device = device
        self.num_splits = args.splits
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()
                # extract features
                src_feat, seq_src_feat = self.feature_extractor(src_x)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=self.num_splits, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # classifier predictions
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                total_loss = src_cls_loss + tov_loss
                total_loss.backward()
                self.pre_optimizer.step()
                self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')


    def update(self, trg_dataloader, avg_meter, logger, dataset):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False

        # obtain pseudo labels
        for epoch in range(1, self.hparams["num_epochs"] + 1):

            # obtain pseudo labels for each epoch
            pseudo_labels = self.obtain_pseudo_labels(trg_dataloader)

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):

                trg_x = trg_x.float().to(self.device)
                # prevent gradient accumulation
                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # Extract features
                trg_feat, trg_feat_seq = self.feature_extractor(trg_x)
                trg_pred = self.classifier(trg_feat)

                # if dataset =='EEG':
                inn = 0
                for i in range(trg_idx.shape[0]):
                    if trg_idx[i].long() < len(pseudo_labels):
                        la_idx = trg_pred[i].unsqueeze(0)
                        inn = i+1
                        break
                    else:
                        trg_idx[i] = -1
                # print(la_idx)
                for i in range(inn,trg_idx.shape[0], 1):
                    if trg_idx[i].long() >= len(pseudo_labels):
                        trg_idx[i] = -1
                    else:
                        la_idx = torch.cat((la_idx, trg_pred[i].unsqueeze(0)), dim=0)
                        # print("after:",la_idx)
                mask1 = trg_idx != -1
                trg_idx = torch.masked_select(trg_idx, mask1)
                trg_pred = la_idx
                    # print("trg_pred1:", len(trg_pred))
                    # print(trg_pred)

                # pseudo labeling loss
                pseudo_label = pseudo_labels[trg_idx.long()].to(self.device)
                target_loss = F.cross_entropy(trg_pred.squeeze(), pseudo_label.long())

                # Entropy loss
                softmax_out = nn.Softmax(dim=1)(trg_pred)
                entropy_loss = self.hparams['ent_loss_wt'] * torch.mean(EntropyLoss(softmax_out))

                #  Information maximization loss
                entropy_loss -= self.hparams['im'] * torch.sum(
                    -softmax_out.mean(dim=0) * torch.log(softmax_out.mean(dim=0) + 1e-5))

                masked_data, mask = masking(trg_x, num_splits=self.num_splits, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                # Total loss
                loss = entropy_loss + self.hparams['target_cls_wt'] * target_loss + self.hparams['TOV_wt'] * tov_loss

                # self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.tov_optimizer.step()

                losses = {'Total_loss': loss.item(), 'Target_loss': target_loss.item(),
                          'Ent_loss': entropy_loss.detach().item(), 'tov_loss': tov_loss.detach().item()}

                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model


    def obtain_pseudo_labels(self, trg_loader):
        self.feature_extractor.eval()
        self.classifier.eval()
        preds, feas = [], []
        with torch.no_grad():
            for inputs, labels, _ in trg_loader:
                inputs = inputs.float().to(self.device)
                features, _ = self.feature_extractor(inputs)
                predictions = self.classifier(features)
                preds.append(predictions)
                feas.append(features)
                # print("preds:", len(preds))

        preds = torch.cat((preds))
        feas = torch.cat((feas))
        preds = nn.Softmax(dim=1)(preds)
        _, predict = torch.max(preds, 1)

        all_features = torch.cat((feas, torch.ones(feas.size(0), 1).to(self.device)), 1)
        all_features = (all_features.t() / torch.norm(all_features, p=2, dim=1)).t()
        all_features = all_features.float().cpu().numpy()

        K = preds.size(1)
        aff = preds.float().cpu().numpy()
        initc = aff.transpose().dot(all_features)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_features, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        pred_label = torch.from_numpy(pred_label)

        #多次更新伪标签
        for round in range(1):
            aff = np.eye(K)[pred_label]
            initc = aff.transpose().dot(all_features)
            initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
            dd = cdist(all_features, initc, 'cosine')
            pred_label = dd.argmin(axis=1)
            pred_label = torch.from_numpy(pred_label)
        return pred_label


class AaDTODA(Algorithm):
    """
    (NeurIPS 2022 Spotlight) Attracting and Dispersing: A Simple Approach for Source-free Domain Adaptation
    https://github.com/Albert0147/AaD_SFDA
    """

    def __init__(self, backbone, configs, hparams, device, args):
        super(AaDTODA, self).__init__(configs)
        self.feature_extractor = backbone(configs)
        self.classifier = classifier(configs)
        self.temporal_verifier = Temporal_Imputer(configs)
        # construct sequential network
        self.network = nn.Sequential(self.feature_extractor, self.classifier)

        # optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.pre_optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=hparams["pre_learning_rate"],
            weight_decay=hparams["weight_decay"]
        )
        self.tov_optimizer = torch.optim.Adam(
            self.temporal_verifier.parameters(),
            lr=hparams["learning_rate"],
            weight_decay=hparams["weight_decay"]
        )

        self.hparams = hparams
        self.device = device
        self.num_splits = args.splits
        self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

        # losses
        self.mse_loss = nn.MSELoss()
        self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

    def pretrain(self, src_dataloader, avg_meter, logger):
        # pretrain
        for epoch in range(1, self.hparams["num_epochs"] + 1):
            for step, (src_x, src_y, _) in enumerate(src_dataloader):
                # input src data
                src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                # optimizer zero_grad
                self.pre_optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                # extract features
                src_feat, seq_src_feat = self.feature_extractor(src_x)
                src_pred = self.classifier(src_feat)

                # classification loss
                src_cls_loss = self.cross_entropy(src_pred, src_y)

                # masking the input_sequences
                masked_data, mask = masking(src_x, num_splits=self.num_splits, num_masked=1)
                src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                ''' Temporal order verification  '''
                # pass the data with and without detach
                tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                # calculate gradients
                total_loss = src_cls_loss + tov_loss
                total_loss.backward()

                self.pre_optimizer.step()
                self.tov_optimizer.step()

                losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
                # acculate loss
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)


            # logging
            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

    def update(self, trg_dataloader, avg_meter, logger):
        # defining best and last model
        best_src_risk = float('inf')
        best_model = self.network.state_dict()
        last_model = self.network.state_dict()

        # freeze both classifier and ood detector
        for k, v in self.classifier.named_parameters():
            v.requires_grad = False
        for k, v in self.temporal_verifier.named_parameters():
            v.requires_grad = False


        for epoch in range(1, self.hparams["num_epochs"] + 1):
            # inilize alpha value

            for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                trg_x = trg_x.float().to(self.device)


                # Extract features
                features, trg_feat_seq = self.feature_extractor(trg_x)
                predictions = self.classifier(features)
                num_samples = len(trg_dataloader.dataset)
                fea_bank = torch.randn(num_samples, self.configs.final_out_channels * self.configs.features_len)
                score_bank = torch.randn(num_samples, self.configs.num_classes).to(self.device)
                softmax_out = nn.Softmax(dim=1)(predictions)

                alpha = (1 + 10 * step / self.hparams["num_epochs"] * len(trg_dataloader)) ** (-self.hparams['beta']) * \
                        self.hparams['alpha']
                with torch.no_grad():
                    output_f_norm = F.normalize(features)
                    output_f_ = output_f_norm.cpu().detach().clone()

                    fea_bank[trg_idx] = output_f_.detach().clone().cpu()
                    score_bank[trg_idx] = softmax_out.detach().clone()

                    distance = output_f_ @ fea_bank.T
                    _, idx_near = torch.topk(distance,
                                             dim=-1,
                                             largest=True,
                                             k=5 + 1)
                    idx_near = idx_near[:, 1:]  # batch x K
                    score_near = score_bank[idx_near]  # batch x K x C

                # start gradients
                softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C

                loss = torch.mean((F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1)).sum(1))

                mask = torch.ones((trg_x.shape[0], trg_x.shape[0]))
                diag_num = torch.diag(mask)
                mask_diag = torch.diag_embed(diag_num)
                mask = mask - mask_diag
                copy = softmax_out.T  # .detach().clone()#

                dot_neg = softmax_out @ copy  # batch x batch

                dot_neg = (dot_neg * mask.to(self.device)).sum(-1)  # batch
                neg_pred = torch.mean(dot_neg)
                loss += neg_pred * alpha

                masked_data, mask = masking(trg_x, num_splits=self.num_splits, num_masked=1)
                trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                total_loss = loss + self.hparams['TOV_wt'] * tov_loss

                self.optimizer.zero_grad()
                self.tov_optimizer.zero_grad()

                total_loss.backward()

                self.optimizer.step()
                self.tov_optimizer.step()

                # meter updates
                losses = {'loss': loss.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                for key, val in losses.items():
                    avg_meter[key].update(val, 32)

            self.lr_scheduler.step()

            if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                best_src_risk = avg_meter['Src_cls_loss'].avg
                best_model = deepcopy(self.network.state_dict())

            logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
            for key, val in avg_meter.items():
                logger.debug(f'{key}\t: {val.avg:2.4f}')
            logger.debug(f'-------------------------------------')

        return last_model, best_model

class NRCTODA(Algorithm):
        def __init__(self, backbone, configs, hparams, device, args):
            super(NRCTODA, self).__init__(configs)
            self.feature_extractor = backbone(configs)
            self.classifier = classifier(configs)
            self.temporal_verifier = Temporal_Imputer(configs)
            # construct sequential network
            self.network = nn.Sequential(self.feature_extractor, self.classifier)

            # optimizer
            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=hparams["learning_rate"],
                weight_decay=hparams["weight_decay"]
            )
            self.pre_optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=hparams["pre_learning_rate"],
                weight_decay=hparams["weight_decay"]
            )
            self.tov_optimizer = torch.optim.Adam(
                self.temporal_verifier.parameters(),
                lr=hparams["learning_rate"],
                weight_decay=hparams["weight_decay"]
            )

            self.hparams = hparams
            self.device = device
            self.num_splits = args.splits
            self.lr_scheduler = StepLR(self.optimizer, step_size=hparams['step_size'], gamma=hparams['lr_decay'])

            # losses
            self.mse_loss = nn.MSELoss()
            self.cross_entropy = CrossEntropyLabelSmooth(self.configs.num_classes, device, epsilon=0.1, )

        def pretrain(self, src_dataloader, avg_meter, logger):
            # pretrain
            for epoch in range(1, self.hparams["num_epochs"] + 1):
                for step, (src_x, src_y, _) in enumerate(src_dataloader):
                    # input src data
                    src_x, src_y = src_x.float().to(self.device), src_y.long().to(self.device)

                    # optimizer zero_grad
                    self.pre_optimizer.zero_grad()
                    self.tov_optimizer.zero_grad()

                    # extract features
                    src_feat, seq_src_feat = self.feature_extractor(src_x)
                    src_pred = self.classifier(src_feat)

                    # classification loss
                    src_cls_loss = self.cross_entropy(src_pred, src_y)

                    # masking the input_sequences
                    masked_data, mask = masking(src_x, num_splits=self.num_splits, num_masked=1)
                    src_feat_mask, seq_src_feat_mask = self.feature_extractor(masked_data)

                    ''' Temporal order verification  '''
                    # pass the data with and without detach
                    tov_predictions = self.temporal_verifier(seq_src_feat_mask.detach())
                    tov_loss = self.mse_loss(tov_predictions, seq_src_feat)

                    # calculate gradients
                    total_loss = src_cls_loss + tov_loss

                    total_loss.backward()
                    self.pre_optimizer.step()
                    self.tov_optimizer.step()

                    losses = {'cls_loss': src_cls_loss.detach().item(), 'making_loss': tov_loss.detach().item()}
                    # acculate loss
                    for key, val in losses.items():
                        avg_meter[key].update(val, 32)

                # logging
                logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                for key, val in avg_meter.items():
                    logger.debug(f'{key}\t: {val.avg:2.4f}')
                logger.debug(f'-------------------------------------')

        def update(self, trg_dataloader, avg_meter, logger):
            # defining best and last model
            best_src_risk = float('inf')
            best_model = self.network.state_dict()
            last_model = self.network.state_dict()

            # freeze both classifier and ood detector
            for k, v in self.classifier.named_parameters():
                v.requires_grad = False
            for k, v in self.temporal_verifier.named_parameters():
                v.requires_grad = False

            for epoch in range(1, self.hparams["num_epochs"] + 1):

                for step, (trg_x, _, trg_idx) in enumerate(trg_dataloader):
                    trg_x = trg_x.float().to(self.device)
                    # Extract features
                    features, trg_feat_seq = self.feature_extractor(trg_x)
                    predictions = self.classifier(features)
                    num_samples = len(trg_dataloader.dataset)
                    fea_bank = torch.randn(num_samples, self.configs.final_out_channels * self.configs.features_len)
                    score_bank = torch.randn(num_samples, self.configs.num_classes).to(self.device)
                    softmax_out = nn.Softmax(dim=1)(predictions)

                    with torch.no_grad():
                        output_f_norm = F.normalize(features)
                        output_f_ = output_f_norm.cpu().detach().clone()

                        fea_bank[trg_idx] = output_f_.detach().clone().cpu()
                        score_bank[trg_idx] = softmax_out.detach().clone()

                        distance = output_f_ @ fea_bank.T
                        _, idx_near = torch.topk(distance,
                                                 dim=-1,
                                                 largest=True,
                                                 k=5 + 1)
                        idx_near = idx_near[:, 1:]  # batch x K
                        score_near = score_bank[idx_near]  # batch x K x C

                        fea_near = fea_bank[idx_near]  # batch x K x num_dim
                        fea_bank_re = fea_bank.unsqueeze(0).expand(fea_near.shape[0], -1, -1)  # batch x n x dim
                        distance_ = torch.bmm(fea_near, fea_bank_re.permute(0, 2, 1))  # batch x K x n
                        _, idx_near_near = torch.topk(distance_, dim=-1, largest=True,
                                                      k=5 + 1)  # M near neighbors for each of above K ones
                        idx_near_near = idx_near_near[:, :, 1:]  # batch x K x M
                        trg_idx_ = trg_idx.unsqueeze(-1).unsqueeze(-1)
                        match = (
                                idx_near_near == trg_idx_).sum(-1).float()  # batch x K
                        weight = torch.where(
                            match > 0., match,
                            torch.ones_like(match).fill_(0.1))  # batch x K

                        weight_kk = weight.unsqueeze(-1).expand(-1, -1,
                                                                5)  # batch x K x M
                        weight_kk = weight_kk.fill_(0.1)

                        # removing the self in expanded neighbors, or otherwise you can keep it and not use extra self regularization
                        # weight_kk[idx_near_near == trg_idx_]=0

                        score_near_kk = score_bank[idx_near_near]  # batch x K x M x C
                        # print(weight_kk.shape)
                        weight_kk = weight_kk.contiguous().view(weight_kk.shape[0],
                                                                -1)  # batch x KM

                        score_near_kk = score_near_kk.contiguous().view(score_near_kk.shape[0], -1,
                                                                        self.configs.num_classes)  # batch x KM x C

                        score_self = score_bank[trg_idx]

                    # start gradients
                    output_re = softmax_out.unsqueeze(1).expand(-1, 5 * 5,
                                                                -1)  # batch x C x 1
                    const = torch.mean(
                        (F.kl_div(output_re, score_near_kk, reduction='none').sum(-1) *
                         weight_kk.to(self.device)).sum(
                            1))  # kl_div here equals to dot product since we do not use log for score_near_kk
                    loss = torch.mean(const)

                    # nn
                    softmax_out_un = softmax_out.unsqueeze(1).expand(-1, 5, -1)  # batch x K x C

                    loss += torch.mean(
                        (F.kl_div(softmax_out_un, score_near, reduction='none').sum(-1) * weight.to(self.device)).sum(1))

                    # self, if not explicitly removing the self feature in expanded neighbor then no need for this
                    # loss += -torch.mean((softmax_out * score_self).sum(-1))

                    msoftmax = softmax_out.mean(dim=0)
                    gentropy_loss = torch.sum(msoftmax *
                                              torch.log(msoftmax + self.hparams['epsilon']))
                    loss += gentropy_loss

                    masked_data, mask = masking(trg_x, num_splits=self.num_splits, num_masked=1)
                    trg_feat_mask, seq_trg_feat_mask = self.feature_extractor(masked_data)

                    tov_predictions = self.temporal_verifier(seq_trg_feat_mask)
                    tov_loss = self.mse_loss(tov_predictions, trg_feat_seq)

                    total_loss = loss + self.hparams['TOV_wt'] * tov_loss

                    self.optimizer.zero_grad()
                    self.tov_optimizer.zero_grad()
                    total_loss.backward()
                    self.optimizer.step()
                    self.tov_optimizer.step()

                    losses = {'eloss': loss.detach().item(), 'Masking_loss': tov_loss.detach().item()}
                    for key, val in losses.items():
                        avg_meter[key].update(val, 32)

                self.lr_scheduler.step()
                # saving the best model based on src risk
                if (epoch + 1) % 10 == 0 and avg_meter['Src_cls_loss'].avg < best_src_risk:
                    best_src_risk = avg_meter['Src_cls_loss'].avg
                    best_model = deepcopy(self.network.state_dict())

                logger.debug(f'[Epoch : {epoch}/{self.hparams["num_epochs"]}]')
                for key, val in avg_meter.items():
                    logger.debug(f'{key}\t: {val.avg:2.4f}')
                logger.debug(f'-------------------------------------')

            return last_model, best_model