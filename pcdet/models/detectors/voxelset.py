from .detector3d_template import Detector3DTemplate
from ..model_utils.calculate_utils import *

class VoxelSet(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_topology = [
            'vfe', 'point_head', 'map_to_bev_module', 
            'backbone_2d', 'dense_head'
        ]
        self.module_list = self.build_networks()
        self.vis_feat = model_cfg.get('VIS_FEAT',None)

    def forward(self, batch_dict):
        if not self.training:
            #prof_list = []
            #prof_topo = []

            #for m, name in zip(self.module_list, self.module_topology):
            #    if m.__class__.__name__ == "PointHeadSimple":
            #        continue
            #    prof_list.append(m)
            #    prof_topo.append(name)

            #batch_dict, per_mod, total = profile_modules(
            #    prof_list, prof_topo, batch_dict
            #)
            for cur_module in self.module_list:
                if cur_module.__class__.__name__== 'PointHeadSimple':
                    continue
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

        loss_trans, tb_dict = batch_dict['loss'],batch_dict['tb_dict']
        tb_dict = {
            'loss_trans': loss_trans.item(),
            **tb_dict
        }

        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss(tb_dict)
            loss = loss_trans + loss_point
        else:
            loss = loss_trans
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict

