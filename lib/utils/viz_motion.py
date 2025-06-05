import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

from lib.utils.utils_smpl import SMPL

def viz_motion(data, save=True):
    
    proj3d = True

    if data.shape[-2] == 6890:
        SMPL_MODEL = SMPL('data/support_data/mesh', batch_size=1)
        data = {'vertex': data, 'face': SMPL_MODEL.faces}
        init_pose = {'vertex': data['vertex'][0],
                     'face': data['face']}
        frame_n = data['vertex'].shape[0]
    else:
        if data.shape[-1] == 2:
            proj3d = False
        init_pose = data[0]
        frame_n = data.shape[0]
    
    if proj3d:
        fig, axes = plt.subplots(1, 1, squeeze=False, figsize=(10, 10), subplot_kw=dict(projection='3d'))
    else:
        fig, axes = plt.subplots(1, 1, squeeze=False, figsize=(10, 10))
    ax = axes[0, 0]
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    if ax.name == '3d':
        ax.set_zlabel('Z')
        ax.view_init(azim=90, elev=95)

    plot = []
    plot = create_pose(ax, plot, init_pose, update=False)

    line_anim = animation.FuncAnimation(fig, update, frame_n, fargs=(data, plot, fig, ax), interval=120, blit=False)

    if not save:
        plt.show()
    else:
        gif_folder = "media/output"
        if not os.path.exists(gif_folder):
            os.makedirs(gif_folder)
        gif_path = os.path.join(gif_folder, "query_output.gif")
        line_anim.save(gif_path, writer='pillow')
        plt.close()
        return gif_path

def create_pose(ax, plot, pose, update=False):
    if isinstance(pose, dict):
        if update:
            plot[-1].remove()
        plot.append(ax.plot_trisurf(pose['vertex'][:, 0], pose['vertex'][:, 1], pose['vertex'][:, 2], triangles=pose['face'], color='#DBDBDB', shade=True, edgecolor='none', linewidth=0))
        return plot

    else:
        connect = [(0,1), (1,2), (2,3),       (0,4), (4,5), (5,6),      (0,7), (7,8), (8,9), (9,10),
                (8,14), (14,15), (15,16),      (8,11), (11,12), (12,13), ]
        RightFlag = [True, True, True, False, False, False, True, True, True, True,
                         True, True, True, False, False, False]
        colors = []
        for bone_idx in range(len(connect)):
            if RightFlag[bone_idx]:
                colors.append(np.array([0.18,0.80,0.44]))
            else:
                colors.append(np.array([0.61,0.35,0.71]))

        edge_st = [tuple[0] for tuple in connect]
        edge_ed = [tuple[1] for tuple in connect]

        for bone_idx in np.arange(len(edge_st)):
            x = np.array([pose[edge_st[bone_idx], 0], pose[edge_ed[bone_idx], 0]])
            y = np.array([pose[edge_st[bone_idx], 1], pose[edge_ed[bone_idx], 1]])
            if ax.name == '3d':
                z = np.array([pose[edge_st[bone_idx], 2], pose[edge_ed[bone_idx], 2]])
            if not update:
                if ax.name == '3d':
                    plot.append(ax.plot(x, y, z, lw=4, c=colors[bone_idx]))
                else:
                    plot.append(ax.plot(x, y, lw=4, c=colors[bone_idx]))
            else:
                plot[bone_idx][0].set_xdata(x)
                plot[bone_idx][0].set_ydata(y)
                if ax.name == '3d':
                    plot[bone_idx][0].set_3d_properties(z)
                plot[bone_idx][0].set_color(colors[bone_idx])

        if ax.name == '3d':
            if not update:
                plot.append(ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c='black', s=80, alpha=1))
            else:
                plot[-1]._offsets3d = (pose[:, 0], pose[:, 1], pose[:, 2])
        else:
            if not update:
                plot.append(ax.scatter(pose[:, 0], pose[:, 1], c='black', s=80, alpha=1))
            else:
                plot[-1].set_offsets(pose[:, :2])
        return plot

def update(num, data, plot, fig, ax):
    if ax.name == '3d':
        ax.set_xlim3d([-1, 1])
        ax.set_ylim3d([-1, 1])
        ax.set_zlim3d([-1, 1])
    elif ax.name == '2d':
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])

    if isinstance(data, np.ndarray):
        pose = data[num]
        plot = create_pose(ax, plot, pose, update=True)
    elif isinstance(data, dict):
        pose = {'vertex': data['vertex'][num], 'face': data['face']}
        plot = create_pose(ax, plot, pose, update=True)
    return plot
