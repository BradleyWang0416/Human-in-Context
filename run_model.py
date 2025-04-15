import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import gradio as gr
import sys
import numpy as np
import torch
import joblib
from typing import Tuple, Dict

from lib.utils.utils_smpl import SMPL
from lib.utils.tools import get_config
from lib.model.xfusionnet import XFusionNet
from lib.utils.viz_motion import viz_motion
from lib.data.data_utils import prepare_task_data, compute_smpl_vertex, evaluate_mpve, apply_mask, get_anchor

def run_inference(task: str) -> Tuple[Dict[str, str], np.ndarray]:
    args = get_config("config/config.yaml")
    model_config = {
        'args': args,
        'model_config': {**args.model_config},
    }
    MODEL = XFusionNet(**model_config)
    if torch.cuda.is_available():
        MODEL = MODEL.cuda()
    ckpt_pth = "ckpt/nominee_069.bin"
    checkpoint = torch.load(ckpt_pth, weights_only=True)
    MODEL.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_pos'].items()}, strict=True)
    MODEL.eval()

    SMPL_MODEL = SMPL('data/support_data/mesh', batch_size=1)
    if torch.cuda.is_available():
        SMPL_MODEL = SMPL_MODEL.cuda()

    anchor_collection = joblib.load('data/support_data/anchor_collection/anchor_collection.pkl')
    anchor_collection = {k: {m: torch.from_numpy(d).float() for m, d in v.items()} for k, v in anchor_collection.items()}

    query_task = task
    query_dataset = 'H36M_MESH'
    query_joint3d = np.load(f"data/H36M_MESH_test02947/npy/query_pose3d.npy")
    query_joint2d = np.load(f"data/H36M_MESH_test02947/npy/query_pose2d.npy")
    query_smpl_pose = np.load(f"data/H36M_MESH_test02947/npy/query_smpl_pose.npy")
    query_smpl_shape = np.load(f"data/H36M_MESH_test02947/npy/query_smpl_shape.npy")
    query_data_dict ={
        'H36M_MESH': {
            'joint3d': torch.from_numpy(query_joint3d).float().unsqueeze(0), # [1,32,17,3]
            'joint2d': torch.from_numpy(query_joint2d).float().unsqueeze(0), # [1,32,17,3]
            'smpl_pose': torch.from_numpy(query_smpl_pose).float().unsqueeze(0), # [1,32,72]
            'smpl_shape': torch.from_numpy(query_smpl_shape).float().unsqueeze(0)   # [1,32,10]
        }
    }
    chunk_len = query_joint3d.shape[0]
    clip_len = chunk_len // 2
    num_pose_joint = args.num_pose_joint
    num_mesh_joint = args.num_mesh_joint

    query_input_tensor, query_target_tensor, query_target_dict, query_input_mask = prepare_task_data(query_data_dict, query_dataset, query_task, clip_len, num_pose_joint, num_mesh_joint)
    anchor_input_tensors, anchor_target_tensors, _, _ = prepare_task_data(anchor_collection, query_dataset, query_task, clip_len, num_pose_joint, num_mesh_joint)
    anchor_id = get_anchor(query_data_dict, query_dataset, clip_len, anchor_collection)

    prompt_input_tensor, prompt_target_tensor = anchor_input_tensors[anchor_id], anchor_target_tensors[anchor_id]

    if query_task == 'Joint Completion (pose)':
        joint_mask = [10, 2, 9, 15, 11, 6]
        query_input_tensor = apply_mask(query_input_tensor, query_task, joint_mask=joint_mask)
        prompt_input_tensor = apply_mask(prompt_input_tensor, query_task, joint_mask=joint_mask)
    if query_task == 'Motion In-Between (pose)' or query_task == 'Motion In-Between (mesh)':
        frame_mask = [9, 1, 8, 14, 10, 5]
        query_input_tensor = apply_mask(query_input_tensor, query_task, frame_mask=frame_mask)
        prompt_input_tensor = apply_mask(prompt_input_tensor, query_task, frame_mask=frame_mask)
    if query_task == 'Joint Completion (mesh)':
        mesh_joint_mask = [2,  5, 8, 11,14, 17, 19, 21, 23]
        query_input_tensor = apply_mask(query_input_tensor, query_task, mesh_joint_mask=mesh_joint_mask)
        prompt_input_tensor = apply_mask(prompt_input_tensor, query_task, mesh_joint_mask=mesh_joint_mask)

    if torch.cuda.is_available():
        query_input_tensor = query_input_tensor.cuda()
        query_target_tensor = query_target_tensor.cuda()
        query_target_dict = {k: v.cuda() for k, v in query_target_dict.items()}
        prompt_input_tensor = prompt_input_tensor.cuda()
        prompt_target_tensor = prompt_target_tensor.cuda()
        query_input_mask = query_input_mask.cuda()
        anchor_id = anchor_id.unsqueeze(0).cuda()

    query_target_vertex = compute_smpl_vertex(query_target_dict, SMPL_MODEL=SMPL_MODEL)
    query_target_dict['smpl_vertex'] = query_target_vertex
    query_target_dict['smpl_vertex'] = query_target_dict['smpl_vertex'] - query_target_dict['joint'][..., 0:1, :]
    query_target_dict['joint'] = query_target_dict['joint'] - query_target_dict['joint'][..., 0:1, :]

    with torch.no_grad():
        output_joint, output_smpl, target_joint, target_smpl = MODEL(query_input_tensor, prompt_input_tensor, query_input_mask, prompt_target_tensor, query_target_dict, anchor_id)

        if task in ['Motion Prediction (pose)', 'Future Pose Estimation', 'Pose Estimation', 'Joint Completion (pose)', 'Motion In-Between (pose)']:
            mpjpe = torch.norm(output_joint*1000-target_joint*1000, dim=-1).mean(-1).cpu().numpy()   # [B,T]
            output_joint = output_joint - output_joint[..., 0:1, :]
            query_output = output_joint[0].cpu().numpy()
            gif_path = viz_motion(query_output)
            return {
                "Mean Per Joint Position Error (MPJPE) (mm)": f"{mpjpe.mean().item():.2f} mm"
            }, gif_path
        elif task in ['Mesh Recovery' ,'Future Mesh Recovery' ,'Motion Prediction (mesh)' ,'Joint Completion (mesh)' ,'Motion In-Between (mesh)']:
            mpve = evaluate_mpve(output_smpl, target_smpl)  # (BT,)
            query_output = output_smpl[0]['verts'][0].cpu().numpy()/1000
            gif_path = viz_motion(query_output)
            return {
                "Mean Per Vertex Error (MPVE) (mm)": f"{mpve.mean().item():.2f} mm"
            }, gif_path