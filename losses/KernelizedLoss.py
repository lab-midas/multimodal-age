"""File for loss calculation"""
from cmath import isinf
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import logging


class KernelizedLoss(nn.Module):
    """
    Supervised contrastive loss based on: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    Based on: https://github.com/HobbitLong/SupContrast

    Adapted for multimodal training.
    """
    def __init__(self, method: str = None, temperature: float=0.07,
                 base_temperature: float=0.07, kernel: callable=None, delta_reduction: str='sum', loss_mode = "Cos", larger_matrix = False, split_matrix = False, cross_weight = 1):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.method = method
        self.kernel = kernel
        self.delta_reduction = delta_reduction
        self.loss_mode = loss_mode
        self.larger_matrix = larger_matrix 
        self.split_matrix = split_matrix
        self.cross_weight = cross_weight


    def forward(self, projections, labels=None, labels_t = None):
        """
        Compute loss for model. If `labels` and `labels_t` is None, 
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Adapted for multimodal training.
        """
        if self.larger_matrix == True and projections.ndim != 3:
            raise Exception(f"larger Loss is only available with multimodal model")
        if self.larger_matrix == True:
            return self.calc_Loss(projections, labels, labels_t)
        else:
            if projections.ndim == 3:
                loss1 = self.calc_Loss(projections, labels, labels_t)
                swapped_projections = projections.clone()
                swapped_projections[:, 0, :], swapped_projections[:, 1, :] = (projections[:, 1, :],projections[:, 0, :],)
                loss2 = self.calc_Loss(swapped_projections, labels_t, labels)
                return ((loss1 + loss2) / 2)
            else:
                loss = self.calc_Loss(projections, labels)
                return loss

    def compute_quadrant_loss(self, contrast, weights, method, temperature, base_temperature):
        """
        This method calculates the Loss for one quadrant of the loss-matrix (image-tabular, tabular-image, image-image, tabular-tabular)
        """
        weighted_contrast = contrast

        zero_rows = (weights == 0).all(dim=1)
        if zero_rows.any():
            logging.info(f"zero row detected in quadrant: {zero_rows}")
            weights = weights.clone()
            weights[zero_rows] = 1.0

        if method == 'expw':
            weighted_contrast = weighted_contrast * (1 - weights)
            N = weighted_contrast.shape[1]

            masked_stack = []

            for j in range(N):
                masked = torch.cat([weighted_contrast[:, :j], weighted_contrast[:, j+1:]], dim=1)
                masked_stack.append(masked.unsqueeze(0))

            masked_contrast  = torch.cat(masked_stack, dim=0)
            logsumexp_per_j = torch.logsumexp(masked_contrast, dim=2).transpose(0, 1)
            log_prob = contrast - logsumexp_per_j
            log_prob = (weights * log_prob).sum(1) / weights.sum(1)
            loss = - (temperature / base_temperature) * log_prob
            return loss.mean()

        # loss calculation
        uniformity = torch.logsumexp(weighted_contrast, dim=1, keepdim=True)
        log_prob = contrast - uniformity
        log_prob = (weights * log_prob).sum(1) / weights.sum(1)
        
        loss = - (temperature / base_temperature) * log_prob
        return loss.mean()

        
    def calc_Loss(self, projections, labels=None, labels_t = None):
        """
        This method calculates the loss for the given projections.
        """
        batch_size = projections.shape[0]
        projection_size = projections.ndim

        if projection_size == 3:
            # Projection_size is 3 if it's a multimodal training
            projection_1, projection_2 = projections[:, 0, :], projections[:, 1, :]
            if self.larger_matrix == True:
                features = torch.cat([projection_1, projection_2], dim=0)
        else:
            features = projections

        device = projections.device


        if labels is None or (labels is None and labels_t is None):
            weights = torch.eye(batch_size, device=device)
        
        else:
            labels = labels.view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            
            if self.kernel is None:
                weights = torch.eq(labels, labels.T)

            else:
                # calculation of weights based on labeldifference
                if self.larger_matrix == True and self.split_matrix == False:
                    if labels_t is None:
                        labels = torch.cat([labels, labels], dim=0)
                    else:
                        labels = torch.cat([labels, labels_t], dim=0)
                if  projection_size == 3 and self.larger_matrix != True:
                    weights = self.kernel(labels, labels_t)
                elif self.larger_matrix == True and self.split_matrix == True:
                    weights_cm_1 = self.kernel(labels, labels_t)
                    weights_cm_2 = self.kernel(labels_t, labels)
                    weights_im_image = self.kernel(labels, labels)
                    weights_im_tabular = self.kernel(labels_t, labels_t)
                    top = torch.cat([weights_im_image, weights_cm_1], dim=1)
                    bottom = torch.cat([weights_cm_2, weights_im_tabular], dim=1) 

                    weights = torch.cat([top, bottom], dim=0)

                else:
                    weights = self.kernel(labels, labels)

        if self.loss_mode == "Cos":
            # similarity calculation
            if self.larger_matrix == False and projection_size == 3:
                features_1 = F.normalize(projection_1, p=2, dim=1)
                features_2 = F.normalize(projection_2, p=2, dim=1)
                contrast = torch.div(
                    torch.matmul(features_1, features_2.T),
                    self.temperature
                )
            else:
                features = F.normalize(features, p=2, dim=1)
                contrast = torch.div(
                    torch.matmul(features, features.T),
                    self.temperature
                )
        elif self.loss_mode == "L2":
            if self.larger_matrix == False and projection_size == 3:
                contrast = - (features_1[:, None, :] - features_2[None, :, :]).norm(2, dim=-1)
            else:
                contrast = - (features[:, None, :] - features[None, :, :]).norm(2, dim=-1)

        elif self.loss_mode == "Cos and L2":
            if self.larger_matrix == False and projection_size == 3:
                features_1 = F.normalize(projection_1, p=2, dim=1)
                features_2 = F.normalize(projection_2, p=2, dim=1)
            
            else:
                features_1 = F.normalize(features, p=2, dim=1)
                features_2 = F.normalize(features, p=2, dim=1)

            contrast_cos = torch.div(
                torch.matmul(features_1, features_2.T),
                self.temperature
            )
            contrast_l2 = - (features_1[:, None, :] - features_2[None, :, :]).pow(2).mean(dim=-1)
            contrast_l2 = torch.div(
                contrast_l2,
                self.temperature / 10
            )
            contrast = contrast_cos + contrast_l2


        if (projection_size == 3 and self.larger_matrix == True) or projection_size == 2:

            mask = ~torch.eye(batch_size * (projection_size-1), dtype=torch.bool, device=features.device)
            contrast = contrast[mask].view(batch_size * (projection_size-1), -1)
            weights = weights[mask].view(batch_size * (projection_size-1), -1)


        if self.split_matrix ==  True and projection_size != 2 and self.larger_matrix == True:
            logging.info(f"splitting matrix")
            contrast_intra_modal_image = contrast[:batch_size, :batch_size-1]
            contrast_cross_modal_1 = contrast[:batch_size, batch_size-1:]
            contrast_cross_modal_2 = contrast[batch_size:, :batch_size]
            contrast_intra_modal_tabular = contrast[batch_size:, batch_size:]

            weights_intra_modal_image = weights[:batch_size, :batch_size-1]
            weights_cross_modal_1 = weights[:batch_size, batch_size-1:]
            weights_cross_modal_2 = weights[batch_size:, :batch_size]
            weights_intra_modal_tabular = weights[batch_size:, batch_size:]

            loss_intra_modal_image = self.compute_quadrant_loss(contrast_intra_modal_image, weights_intra_modal_image, self.method, self.temperature, self.base_temperature)
            loss_cross_modal_1 = self.compute_quadrant_loss(contrast_cross_modal_1, weights_cross_modal_1, self.method, self.temperature, self.base_temperature)
            loss_cross_modal_2 = self.compute_quadrant_loss(contrast_cross_modal_2, weights_cross_modal_2, self.method, self.temperature, self.base_temperature)
            loss_intra_modal_tabular = self.compute_quadrant_loss(contrast_intra_modal_tabular, weights_intra_modal_tabular, self.method, self.temperature, self.base_temperature)
            logging.info(f"loss image: {loss_intra_modal_image}")
            logging.info(f"loss tabular: {loss_intra_modal_tabular}")
            logging.info(f"loss image-tabular{loss_cross_modal_1}")
            logging.info(f"loss tabular-image{loss_cross_modal_2}")

            loss = (loss_intra_modal_image +  loss_cross_modal_1 + loss_cross_modal_2 + loss_intra_modal_tabular) / 4
        elif (self.split_matrix ==  True and projection_size == 2) or (self.split_matrix ==  True and self.larger_matrix != True):
            raise Exception('splitting matrix is available for multimodal network with intramodallity')
        else:
            loss = self.compute_quadrant_loss(contrast, weights, self.method, self.temperature, self.base_temperature)
        return loss
