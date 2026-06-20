from .detector3d_template import Detector3DTemplate
from ..model_utils.calculate_utils import *

class PointPillar(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        if not self.training:
            #batch_dict, per_mod, total = profile_modules(self.module_list,self.module_topology,batch_dict)
            for cur_module in self.module_list:
                batch_dict = cur_module(batch_dict)
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            #print_total_profile(total)
            #print_profile(per_mod)
            return pred_dicts, recall_dicts
            
            
        for i, cur_module in enumerate(self.module_list):
            batch_dict = cur_module(batch_dict)
            module_name = self.module_topology[i]
            save_feature(self.vis_feat, batch_dict, module_name)

        loss, tb_dict, disp_dict = self.get_training_loss(batch_dict)
        ret_dict = {'loss': loss}
        return ret_dict, tb_dict, disp_dict

    def get_training_loss(self,batch_dict):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
            loss = loss_rpn + loss_point
        else:
            loss = loss_rpn
        return loss, tb_dict, disp_dict
