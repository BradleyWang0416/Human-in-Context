import torch

def prepare_task_data(data_dict, dataset, task, clip_len, num_pose_joint, num_mesh_joint):
    chunk_dict = data_dict[dataset]
    N = chunk_dict['joint3d'].shape[0]
    num_joint = max(num_pose_joint, num_mesh_joint)
    input_tensor = torch.zeros((N, clip_len, num_joint, 3))
    target_tensor = torch.zeros((N, clip_len, num_joint, 3))
    input_mask = torch.zeros(N, num_mesh_joint)
    target_dict = {}
    if task == 'Motion Prediction (pose)':
        input_frames = slice(None, clip_len)
        input_joints = slice(None, num_pose_joint)
        input_modality = 'joint3d'
        input_num_joint = num_pose_joint

        target_frames = slice(clip_len, None)
        target_joints = slice(None, num_pose_joint)
        target_modality = 'joint3d'
        target_num_joint = num_pose_joint
    elif task == 'Future Pose Estimation':
        input_frames = slice(None, clip_len)
        input_joints = slice(None, num_pose_joint)
        input_modality = 'joint2d'
        input_num_joint = num_pose_joint

        target_frames = slice(clip_len, None)
        target_joints = slice(None, num_pose_joint)
        target_modality = 'joint3d'
        target_num_joint = num_pose_joint
    elif task == 'Pose Estimation':
        input_frames = slice(None, clip_len)
        input_joints = slice(None, num_pose_joint)
        input_modality = 'joint2d'
        input_num_joint = num_pose_joint

        target_frames = slice(None, clip_len)
        target_joints = slice(None, num_pose_joint)
        target_modality = 'joint3d'
        target_num_joint = num_pose_joint
    elif task == 'Joint Completion (pose)':
        input_frames = slice(None, clip_len)
        input_joints = slice(None, num_pose_joint)
        input_modality = 'joint3d'
        input_num_joint = num_pose_joint

        target_frames = slice(None, clip_len)
        target_joints = slice(None, num_pose_joint)
        target_modality = 'joint3d'
        target_num_joint = num_pose_joint
    elif task == 'Motion In-Between (pose)':
        input_frames = slice(None, clip_len)
        input_joints = slice(None, num_pose_joint)
        input_modality = 'joint3d'
        input_num_joint = num_pose_joint

        target_frames = slice(None, clip_len)
        target_joints = slice(None, num_pose_joint)
        target_modality = 'joint3d'
        target_num_joint = num_pose_joint
    elif task == 'Mesh Recovery':
        input_frames = slice(None, clip_len)
        input_joints = slice(None, num_pose_joint)
        input_modality = 'joint2d'
        input_num_joint = num_pose_joint

        target_frames = slice(None, clip_len)
        target_joints = slice(None, num_mesh_joint)
        target_modality = 'smpl_pose'
        target_num_joint = num_mesh_joint
    elif task == 'Future Mesh Recovery':
        input_frames = slice(None, clip_len)
        input_joints = slice(None, num_pose_joint)
        input_modality = 'joint2d'
        input_num_joint = num_pose_joint

        target_frames = slice(clip_len, None)
        target_joints = slice(None, num_mesh_joint)
        target_modality = 'smpl_pose'
        target_num_joint = num_mesh_joint
    elif task == 'Motion Prediction (mesh)':
        input_frames = slice(None, clip_len)
        input_joints = slice(None, num_mesh_joint)
        input_modality = 'smpl_pose'
        input_num_joint = num_mesh_joint

        target_frames = slice(clip_len, None)
        target_joints = slice(None, num_mesh_joint)
        target_modality = 'smpl_pose'
        target_num_joint = num_mesh_joint
    elif task == 'Joint Completion (mesh)':
        input_frames = slice(None, clip_len)
        input_joints = slice(None, num_mesh_joint)
        input_modality = 'smpl_pose'
        input_num_joint = num_mesh_joint

        target_frames = slice(None, clip_len)
        target_joints = slice(None, num_mesh_joint)
        target_modality = 'smpl_pose'
        target_num_joint = num_mesh_joint
    elif task == 'Motion In-Between (mesh)':
        input_frames = slice(None, clip_len)
        input_joints = slice(None, num_mesh_joint)
        input_modality = 'smpl_pose'
        input_num_joint = num_mesh_joint

        target_frames = slice(None, clip_len)
        target_joints = slice(None, num_mesh_joint)
        target_modality = 'smpl_pose'
        target_num_joint = num_mesh_joint
    

    input_tensor[:, :, input_joints, :] = chunk_dict[input_modality][:, input_frames].clone().reshape(N, clip_len, input_num_joint, 3)
    if input_modality == 'joint3d' or input_modality == 'joint2d':
        input_tensor[:, :, input_joints, :] = input_tensor[:, :, input_joints, :] - input_tensor[:, :, 0:1, :]
    input_mask[:, input_joints] = 1

    target_tensor[:, :, target_joints, :] = chunk_dict[target_modality][:, target_frames].clone().reshape(N, clip_len, target_num_joint, 3)
    target_dict['joint'] = chunk_dict['joint3d'][:, target_frames].clone()
    target_dict['smpl_pose'] = chunk_dict['smpl_pose'][:, target_frames].clone().reshape(N, clip_len, 72)
    target_dict['smpl_shape'] = chunk_dict['smpl_shape'][:, target_frames].clone()
    return input_tensor, target_tensor, target_dict, input_mask

def compute_smpl_vertex(data_dict, SMPL_MODEL=None):
    if len(data_dict['smpl_shape'].shape) == 3:
        B, T, _ = data_dict['smpl_shape'].shape
        B_ = B*T
    elif len(data_dict['smpl_shape'].shape) == 2:
        T, _ = data_dict['smpl_shape'].shape
        B_ = T
    else:
        raise ValueError(f"Shape length must be 2 or 3, but got {data_dict['smpl_shape'].shape}")
    shape = data_dict['smpl_shape'].reshape(B_, 10)
    pose = data_dict['smpl_pose'].reshape(B_, 72)
    motion_smpl = SMPL_MODEL(
        betas=shape,
        body_pose=pose[:, 3:],
        global_orient=pose[:, :3],
        pose2rot=True
        # pose2rot: bool, optional
        #   Flag on whether to convert the input pose tensor to rotation matrices. 
        #   The default value is True. If False, then the pose tensor should already contain rotation matrices and have a size of Bx(J + 1)x9
    )
    vertex = motion_smpl.vertices.detach()
    if len(data_dict['smpl_shape'].shape) == 3:
        return vertex.reshape(B, T, -1, 3)
    elif len(data_dict['smpl_shape'].shape) == 2:
        return vertex.reshape(T, -1, 3)
    else:
        raise ValueError(f"Shape length must be 2 or 3, but got {data_dict['smpl_shape'].shape}")
    
def evaluate_mpve(output_smpl, target_smpl):
    with torch.no_grad():
        pred_verts = output_smpl[0]['verts'].detach().reshape(-1, 6890, 3)
        pred_j3ds = output_smpl[0]['kp_3d'].detach().reshape(-1, 17, 3)

        target_verts = target_smpl['verts'].detach().reshape(-1, 6890, 3)
        target_j3ds = target_smpl['kp_3d'].detach().reshape(-1, 17, 3)

        pred_verts = pred_verts - pred_j3ds[:, :1, :]
        target_verts = target_verts - target_j3ds[:, :1, :]
        mpve = torch.mean(torch.sqrt(torch.square(pred_verts - target_verts).sum(dim=2)), dim=1)
    return mpve.cpu().numpy()

def apply_mask(input_tensor, task, joint_mask=None, frame_mask=None, mesh_joint_mask=None):
        N = input_tensor.shape[0]
        if task == 'Joint Completion (pose)':
            input_tensor[:, :, joint_mask] = 0
        elif task == 'Joint Completion (mesh)':
            input_tensor[:, :, mesh_joint_mask] = 0
        elif task == 'Motion In-Between (pose)':
            input_tensor[:, frame_mask] = 0
        elif task == 'Motion In-Between (mesh)':
            input_tensor[:, frame_mask] = 0
        return input_tensor

def get_anchor(data_dict, dataset, clip_len, anchor_collection):
    if 'joint3d' in data_dict[dataset]:
        similarity = torch.norm((data_dict[dataset]['joint3d'][:, :clip_len]-data_dict[dataset]['joint3d'][:, :clip_len, 0:1, :]) - (anchor_collection[dataset]['joint3d'][:, :clip_len]-anchor_collection[dataset]['joint3d'][:, :clip_len, 0:1, :]), dim=-1)
        similarity = similarity.mean(-1).mean(-1)
        _, anchor_id = torch.topk(similarity, 1, dim=-1, largest=False) # Motion Prediction (pose): 481; Future Pose Estimation: 788
    elif 'joint2d' in data_dict[dataset]:
        similarity = torch.norm((data_dict[dataset]['joint2d'][:, :clip_len]-data_dict[dataset]['joint2d'][:, :clip_len, 0:1, :]) - (anchor_collection[dataset]['joint2d'][:, :clip_len]-anchor_collection[dataset]['joint2d'][:, :clip_len, 0:1, :]), dim=-1)
        similarity = similarity.mean(-1).mean(-1)
        _, anchor_id = torch.topk(similarity, 1, dim=-1, largest=False) # Motion Prediction (pose): 481; Future Pose Estimation: 788
    elif 'smpl_pose' in data_dict[dataset]:
        Q = data_dict[dataset]['smpl_pose'][:, :clip_len]   # [1,T,72]
        A = anchor_collection[dataset]['smpl_pose'][:, :clip_len]   # [num_anchors,T,72]
        Q = Q.reshape(-1, clip_len, 24, 3) # [1,T,24,3]
        Q = Q / torch.norm(Q, dim=-1, keepdim=True) # [1,T,24,3]
        A = A.reshape(-1, clip_len, 24, 3) # [num_anchors,T,24,3]
        A = A / torch.norm(A, dim=-1, keepdim=True) # [num_anchors,T,24,3]
        similarity = torch.einsum('qtjc,ptjc->ptj', Q, A) # [num_anchors,T,24]
        similarity = similarity.mean(-1).mean(-1)
        _, anchor_id = torch.topk(similarity, 1, dim=-1, largest=True) # Motion Prediction (pose): 481; Future Pose Estimation: 788
    return anchor_id