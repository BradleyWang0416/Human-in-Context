import numpy as np
import torch
import torch.nn as nn
from timm.layers import trunc_normal_

from lib.utils.utils_smpl import SMPL
from lib.utils.utils_mesh import rotation_matrix_to_angle_axis, rot6d_to_rotmat
from lib.model.MotionAGFormer import MotionAGM


class XFusionNet(nn.Module):
    def __init__(self, args, model_config):
        super().__init__()
        act_mapper = {'gelu': nn.GELU, 'relu': nn.ReLU}
        model_config['act_layer'] = act_mapper[model_config['act_layer']]
        model_config['layer_scale_init_value'] = float(model_config['layer_scale_init_value'])

        num_frame = model_config['n_frames']
        num_joints = model_config['num_joints']
        in_chans = model_config['dim_in']
        
        hidden_dim = model_config['dim_feat']
        dim_rep = model_config['dim_rep']


        self.spatial_pos_embed = nn.ParameterDict({'query': nn.Parameter(torch.zeros(num_joints, hidden_dim)),
                                                   'prompt': nn.Parameter(torch.zeros(num_joints, hidden_dim))})
        
        self.temporal_pos_embed = nn.ParameterDict({'query': nn.Parameter(torch.zeros(1, num_frame, 1, hidden_dim)),
                                                    'prompt': nn.Parameter(torch.zeros(1, num_frame, 1, hidden_dim))})

        self.spatial_patch_to_embedding = nn.ModuleDict({'query': nn.Linear(in_chans+3, hidden_dim),
                                                         'prompt': nn.Linear(in_chans+3, hidden_dim)})

        self.MotionAGFormer = nn.ModuleDict({'query': MotionAGM(**model_config),
                                             'prompt': MotionAGM(**model_config)})

        self.smpl_head = SMPLRegressor(data_root='data/support_data/mesh', dim_rep=dim_rep, num_joints=num_joints, hidden_dim=1024, dropout_ratio=0.1)
        
        self.V2J_Regressor = nn.Parameter(self.smpl_head.smpl.J_regressor_h36m.clone())    # [17,6890]

        num_prompt = args.num_prompt
        self.TUP_ST = nn.Parameter(torch.zeros(num_prompt, num_frame, num_joints))   # (P,16,17)
        self.TUP_C = nn.Parameter(torch.zeros(num_prompt, hidden_dim))   # (P,512)
        trunc_normal_(self.TUP_ST, std=.01, a=-0.1, b=0.1)
        trunc_normal_(self.TUP_C, std=.01, a=-0.1, b=0.1)


    def joint_head(self, x):
        return torch.einsum('jv,btvc->btjc', self.V2J_Regressor.to(x.device), x)  # [B,T,6890,3] --> [B,T,17,3]

    def encode_joint(self, x_in, x_out, key):
        x = torch.cat([x_in, x_out], dim=-1)
        x = self.spatial_patch_to_embedding[key](x)   # [B,T,17,512]
        x += self.spatial_pos_embed[key].unsqueeze(0).unsqueeze(0)
        return x
    
    @staticmethod
    def get_target_joint_and_smpl(target_dict):
        target_joint = target_dict['joint'].clone()
        target_smpl = {}
        if 'smpl_vertex' in target_dict.keys():
            target_smpl['theta'] = torch.cat([target_dict['smpl_pose'], target_dict['smpl_shape']], dim=-1).clone()
            target_smpl['verts'] = target_dict['smpl_vertex'].clone() * 1000
            target_smpl['kp_3d'] = target_dict['joint'].clone() * 1000
        return target_joint, target_smpl

    def forward(self, query_input_tensor, prompt_input_tensor, input_mask,
                      prompt_target_tensor,
                      query_target_dict, prompt_chunk_id):
        
        
        B, T, J, C = query_input_tensor.shape
        query_input = query_input_tensor        # [B,T,24,3]

        prompt_input = prompt_input_tensor      # [B,T,24,3]
        prompt_target = prompt_target_tensor    # [B,T,24,3]

        # PROMPT BRANCH
        prompt = self.encode_joint(prompt_input, prompt_target, 'prompt')       # [B,T,24,3] || [B,T,24,3] --> [B,T,24,6] --> [B,T,24,512]
        prompt = prompt + self.temporal_pos_embed['prompt']
        PROMPTS = (prompt,)

        for layer in self.MotionAGFormer['prompt'].layers:
            prompt = layer(prompt, input_mask=input_mask)    # TODO: design spatial adj
            PROMPTS = PROMPTS + (prompt,)


        # QUERY BRANCH
        query = self.encode_joint(query_input, prompt_target, 'query')     # [B,T,24,3] || [B,T,24,3] --> [B,T,24,6] --> [B,T,24,512]

        prompt_chunk_id = prompt_chunk_id.long()
        TUP = torch.einsum('ptj,pc->ptjc', self.TUP_ST, self.TUP_C)
        soft_prompt = torch.gather(TUP, 0, prompt_chunk_id[:, None, None].expand(-1, TUP.shape[1], TUP.shape[2], TUP.shape[3]))
        query += soft_prompt * 0.1

        query = query + self.temporal_pos_embed['query']
        query += PROMPTS[0]

        for i, layer in enumerate(self.MotionAGFormer['query'].layers):
            query = layer(query, input_mask=input_mask)
            query += PROMPTS[1+i]
        
        query = self.MotionAGFormer['query'].norm(query)
        query = self.MotionAGFormer['query'].rep_logit(query)
        # [B,T,24,512]



        # MERGE
        output_smpl = self.smpl_head(query)
        for s in output_smpl:
            s['theta'] = s['theta'].reshape(B, T, -1)
            s['verts'] = s['verts'].reshape(B, T, -1, 3)
            s['kp_3d'] = s['kp_3d'].reshape(B, T, -1, 3)
        # [
        #   {
        #       'theta': [B,T,82],
        #       'verts': [B,T,6890,3],
        #       'kp_3d': [B,T,17,3]
        #   }
        # ]
        output_joint = self.joint_head(output_smpl[-1]['verts'])    # [B,T,17,3]
        output_joint = output_joint / 1000

        target_joint, target_smpl = self.get_target_joint_and_smpl(query_target_dict)

        return output_joint, output_smpl, target_joint, target_smpl


class SMPLRegressor(nn.Module):
    def __init__(self, data_root='', dim_rep=512, num_joints=17, hidden_dim=2048, dropout_ratio=0.):
        super(SMPLRegressor, self).__init__()
        param_pose_dim = 24 * 6
        self.dropout = nn.Dropout(p=dropout_ratio)
        self.fc1 = nn.Linear(num_joints*dim_rep, hidden_dim)
        self.pool2 = nn.AdaptiveAvgPool2d((None, 1))
        self.fc2 = nn.Linear(num_joints*dim_rep, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.bn2 = nn.BatchNorm1d(hidden_dim, momentum=0.1)
        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)
        self.head_pose = nn.Linear(hidden_dim, param_pose_dim)
        self.head_shape = nn.Linear(hidden_dim, 10)
        nn.init.xavier_uniform_(self.head_pose.weight, gain=0.01)
        nn.init.xavier_uniform_(self.head_shape.weight, gain=0.01)
        self.smpl = SMPL(
            data_root,
            batch_size=64,
            create_transl=False,
        )
        mean_params = np.load(self.smpl.smpl_mean_params)
        init_pose = torch.from_numpy(mean_params['pose'][:]).unsqueeze(0)       # [1,144]
        init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32')).unsqueeze(0)   # (1,10)
        self.register_buffer('init_pose', init_pose)
        self.register_buffer('init_shape', init_shape)
        self.J_regressor = self.smpl.J_regressor_h36m

    @staticmethod
    def get_mesh_from_init_pose_shape_only(init_pose=None, init_shape=None):
        smpl = SMPL(
            'data/support_data/mesh',
            batch_size=64,
            create_transl=False,
        )
        J_regressor = smpl.J_regressor_h36m
        mean_params = np.load(smpl.smpl_mean_params)
        if init_pose is None:
            init_pose = torch.from_numpy(mean_params['pose'][:])    # (144,)
        if init_shape is None: 
            init_shape = torch.from_numpy(mean_params['shape'][:].astype('float32'))    # (10,)
        pred_pose = init_pose.unsqueeze(0)   # (1,144)
        pred_shape = init_shape.unsqueeze(0)  # (1,10)
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(-1, 24, 3, 3)
        pred_output = smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, :1],
            pose2rot=False
        )
        pred_vertices = pred_output.vertices * 1000.0
        J_regressor_batch = J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
        pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        output = [{
            'theta'  : torch.cat([pose, pred_shape], dim=1),    # (1, 72+10)
            'verts'  : pred_vertices,                           # (1, 6890, 3)
            'kp_3d'  : pred_joints,                             # (1, 17, 3)
        }]
        return output
        

    def forward(self, feat):
        N, T, J, C = feat.shape
        NT = N * T
        feat = feat.reshape(N, T, -1)

        feat_pose = feat.reshape(NT, -1)     # (N*T, J*C)

        feat_pose = self.dropout(feat_pose)
        feat_pose = self.fc1(feat_pose)
        feat_pose = self.bn1(feat_pose)
        feat_pose = self.relu1(feat_pose)    # (NT, C)

        feat_shape = feat.permute(0,2,1)     # (N, T, J*C) -> (N, J*C, T)
        feat_shape = self.pool2(feat_shape).reshape(N, -1)          # (N, J*C)

        feat_shape = self.dropout(feat_shape)
        feat_shape = self.fc2(feat_shape)
        feat_shape = self.bn2(feat_shape)
        feat_shape = self.relu2(feat_shape)     # (N, C)

        pred_pose = self.init_pose.expand(NT, -1)   # (NT, C)
        pred_shape = self.init_shape.expand(N, -1)  # (N, C)

        pred_pose = self.head_pose(feat_pose) + pred_pose
        pred_shape = self.head_shape(feat_shape) + pred_shape
        pred_shape = pred_shape.expand(T, N, -1).permute(1, 0, 2).reshape(NT, -1)
        pred_rotmat = rot6d_to_rotmat(pred_pose).view(-1, 24, 3, 3)
        pred_output = self.smpl(
            betas=pred_shape,
            body_pose=pred_rotmat[:, 1:],
            global_orient=pred_rotmat[:, 0].unsqueeze(1),
            pose2rot=False
        )
        pred_vertices = pred_output.vertices * 1000.0
        assert self.J_regressor is not None
        J_regressor_batch = self.J_regressor[None, :].expand(pred_vertices.shape[0], -1, -1).to(pred_vertices.device)
        pred_joints = torch.matmul(J_regressor_batch, pred_vertices)
        pose = rotation_matrix_to_angle_axis(pred_rotmat.reshape(-1, 3, 3)).reshape(-1, 72)
        output = [{
            'theta'  : torch.cat([pose, pred_shape], dim=1),    # (N*T, 72+10)
            'verts'  : pred_vertices,                           # (N*T, 6890, 3)
            'kp_3d'  : pred_joints,                             # (N*T, 17, 3)
        }]
        return output