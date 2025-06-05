import os
os.environ["CUDA_VISIBLE_DEVICES"] = '6'
import gradio as gr
import sys
import numpy as np
import torch
import joblib
from typing import Tuple, Dict

from run_model import run_inference

TASK_MAP = {
    "2d pose": ["Future Pose Estimation", "Pose Estimation", "Mesh Recovery", "Future Mesh Recovery"],
    "3d pose": ["Motion Prediction (pose)", "Joint Completion (pose)", "Motion In-Between (pose)"],
    "mesh": ["Motion Prediction (mesh)", "Joint Completion (mesh)", "Motion In-Between (mesh)"]
    }

TASK_NICKNAME = {
    "Future Pose Estimation"    : "fpe",
    "Pose Estimation"           : "pe",
    "Mesh Recovery"             : "mr",
    "Future Mesh Recovery"      : "fmr",
    "Motion Prediction (pose)"  : "mp",
    "Joint Completion (pose)"   : "mc",
    "Motion In-Between (pose)"  : "mib",
    "Motion Prediction (mesh)"  : "meshpred",
    "Joint Completion (mesh)"   : "meshc",
    "Motion In-Between (mesh)"  : "meshib"
    }

INPUT_VIDEO_MAP = {
    "2d pose": "media/query/query_2d_pose.mp4",
    "3d pose": "media/query/query_3d_pose.mp4",
    "mesh": "media/query/query_mesh.mp4"
}

PROMPT_VIDEO_MAP = {
    task: f"media/prompt/prompt_{TASK_NICKNAME[task]}.mp4"
    for tasks in TASK_MAP.values()
    for task in tasks
}

with gr.Blocks() as demo:
    gr.Markdown("## 🧠 Human-in-Context Demo")

    # Step 1: Select input type
    input_type = gr.Dropdown(label="Select an query input (2d pose / 3D pose / mesh)", choices=list(TASK_MAP.keys()))
    show_input_btn = gr.Button("Visualize query input")
    input_video = gr.Video(visible=False, height=500)

    # Step 2: Select task (based on input type)
    task_dropdown = gr.Dropdown(label="Select the task you want to do", choices=[], visible=False)
    prompt_video = gr.Video(label="Visualize task prompt", visible=False, height=700)

    # Step 3: Run inference
    run_btn = gr.Button("Run Inference")
    result_json = gr.JSON(label="Error Metrics")
    output_gif = gr.Image(label="Output Visualization", visible=True, type="filepath")

    # Update input video
    def show_input_video(input_type):
        path = INPUT_VIDEO_MAP.get(input_type)
        return gr.update(value=path, visible=True)

    show_input_btn.click(fn=show_input_video, inputs=input_type, outputs=input_video)

    # Update task options
    def update_tasks(input_type):
        tasks = TASK_MAP.get(input_type, [])
        return gr.update(choices=tasks, visible=True)

    input_type.change(fn=update_tasks, inputs=input_type, outputs=task_dropdown)

    # Show prompt video based on task
    def show_prompt_video(task):
        return gr.update(value=PROMPT_VIDEO_MAP.get(task), visible=True)

    task_dropdown.change(fn=show_prompt_video, inputs=task_dropdown, outputs=prompt_video)

    # Run model
    def run(task: str) -> Tuple[Dict[str, str], str]:
        errors, gif_path = run_inference(task)
        return errors, gif_path

    run_btn.click(fn=run, inputs=task_dropdown, outputs=[result_json, output_gif])

demo.launch()