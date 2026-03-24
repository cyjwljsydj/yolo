import torch
import torch.nn as nn

class LossFunction(nn.Module):
    def __init__(self, S=7, B=2, C=20, lambda_coord=5, lambda_noobj=0.5):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj
        self.regression_loss = nn.MSELoss(reduction="sum")  # Sum over all elements for regression loss
        class_weights = torch.ones(self.C)  # You can adjust these weights based on class imbalance
        self.classification_loss = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=class_weights)
        conf_weights = torch.tensor([1.0] + [self.lambda_noobj] * (self.B - 1))  # Higher weight for the first box, lower for the second box
        self.confidence_loss = nn.BCEWithLogitsLoss(reduction="sum", pos_weight=conf_weights)

    def forward(self, predictions, targets):
        # predictions and targets are of shape (batch_size, S, S, B*5 + C)
        # Implement the YOLO loss calculation here
        # predictions and targets are of shape (batch_size, S, S, B*5 + C)
        batch_size = predictions.shape[0]
        # ========== classification loss ========== 
        targets_classification = targets[:, :, :, self.B*5:]  # (batch_size, S, S, C)
        predictions_classification = predictions[:, :, :, self.B*5:]  # (batch_size, S, S, C)
        classification_loss = self.classification_loss(predictions_classification, targets_classification).mean()  # Average over the batch
        # ========== objectness loss(regression) ==========
        # bbox1: index 0-4, bbox2: index 5-9, class: index 10-29
        targets_regressionbbox1 = targets[:, :, :, 1:5]  # (batch_size, S, S, 4)
        predictions_regressionbbox1 = predictions[:, :, :, 1:5]  # (batch_size, S, S, 4)
        regression_loss_bbox1 = self.regression_loss(predictions_regressionbbox1, targets_regressionbbox1).sum() / batch_size
        
        targets_regressionbbox2 = targets[:, :, :, 6:10]  # (batch_size, S, S, 4)
        predictions_regressionbbox2 = predictions[:, :, :, 6:10]  # (batch_size , S, S, 4)
        regression_loss_bbox2 = self.regression_loss(predictions_regressionbbox2, targets_regressionbbox2).sum() / batch_size

        regression_loss = self.lambda_coord * (regression_loss_bbox1 + regression_loss_bbox2)
        # ========== confidence loss ==========
        conf_pred_bbox1 = predictions[:, :, :, 0]  # (batch_size, S, S)
        conf_target_bbox1 = targets[:, :, :, 0]  # (batch_size, S, S)

        conf_pred_bbox2 = predictions[:, :, :, 5]  # (batch_size, S, S)
        conf_target_bbox2 = targets[:, :, :, 5]  # (batch_size,

        # mask for object and no-object for both bbox1 and bbox2
        obj_mask_bbox1 = conf_target_bbox1  # 1 where there is an object, 0 otherwise
        noobj_mask_bbox1 = 1 - conf_target_bbox1

        obj_mask_bbox2 = conf_target_bbox2
        noobj_mask_bbox2 = 1 - conf_target_bbox2

        # objectness loss for bbox1 and bbox2, applying different weights for object and no-object cases
        loss_conf_bbox1_obj = obj_mask_bbox1 * self.confidence_loss(conf_pred_bbox1, conf_target_bbox1)
        loss_conf_bbox1_noobj = self.lambda_noobj * noobj_mask_bbox1 * self.confidence_loss(conf_pred_bbox1, conf_target_bbox1)
        loss_conf_bbox1 = loss_conf_bbox1_obj + loss_conf_bbox1_noobj

        loss_conf_bbox1_obj = obj_mask_bbox2 * self.confidence_loss(conf_pred_bbox2, conf_target_bbox2)
        loss_conf_bbox1_noobj = self.lambda_noobj * noobj_mask_bbox2 * self.confidence_loss(conf_pred_bbox2, conf_target_bbox2)
        loss_conf_bbox2 = loss_conf_bbox1_obj + loss_conf_bbox1_noobj

        confidence_loss = (loss_conf_bbox1.sum() + loss_conf_bbox2.sum()) / batch_size
        
        # ========== total loss ==========
        total_loss = classification_loss + regression_loss + confidence_loss
        return total_loss
