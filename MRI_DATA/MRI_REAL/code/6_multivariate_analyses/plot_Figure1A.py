#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Figure 1a: schematic of non-local representation learning (simulated points, no data).
#
# Author: Shany Grossman

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.lines import Line2D

save_dir = '/Users/grossman/Volumes/STATE_FORMATION/MRI_DATA/MRI_REAL/'
# ---------- PARAMETERS ----------
same_direction = True
min_shift = 3
jitter_scale = 1
single_color = True
non_feedback_color = 'cyan'
feedbacked_color = 'red'

np.random.seed(86)
n_points = 8
feedbacked = 2
arrow_head_size = 0.25  # this controls the size of arrow head

# ---------- Draw 3D sphere ----------
def draw_sphere(ax, center, radius, color, alpha):
    u = np.linspace(0, 2*np.pi, 24)
    v = np.linspace(0, np.pi, 24)
    x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
    z = radius * np.outer(np.ones_like(u), np.cos(v)) + center[2]
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)

# ---------- Generate positions ----------
positions_before = np.random.randn(n_points, 3) * 1.5

# ---------- Colors ----------
if single_color:
    colors = np.array([non_feedback_color]*n_points)
    colors[feedbacked] = feedbacked_color
else:
    colors = cm.viridis(np.linspace(0,1,n_points))
    colors[feedbacked] = np.array([1,0,0,1])

# ---------- Axis limits ----------
mins = positions_before.min(axis=0) - 1.0
maxs = positions_before.max(axis=0) + 1.0

# ---------- Generate random shifts ----------
shifts = np.random.normal(0, 2, size=(n_points, 3))
shifts[feedbacked] = np.array([1.5, -1.2, 1.0])

# ---------- Direction constraint + jitter ----------
if same_direction:
    fb_dir = shifts[feedbacked] / np.linalg.norm(shifts[feedbacked])
    for i in range(n_points):
        if i != feedbacked:
            proj_len = np.dot(shifts[i], fb_dir)
            projected = proj_len * fb_dir
            jitter = np.random.normal(0, jitter_scale, 3)
            shifts[i] = projected + jitter

# ---------- Minimum shift ----------
for i in range(n_points):
    norm = np.linalg.norm(shifts[i])
    if norm < min_shift:
        if norm == 0:
            shifts[i] = np.random.randn(3)
            norm = np.linalg.norm(shifts[i])
        shifts[i] *= (min_shift / norm)

# ---------- Prevent exiting axis bounds ----------
positions_after = positions_before + shifts
for i in range(n_points):
    for dim in range(3):
        if positions_after[i, dim] < mins[dim]:
            shifts[i, dim] = mins[dim] - positions_before[i, dim]
        elif positions_after[i, dim] > maxs[dim]:
            shifts[i, dim] = maxs[dim] - positions_before[i, dim]
positions_after = positions_before + shifts

# ---------- Plot function ----------
def plot_positions(ax, current, previous=None, title=""):
    if previous is not None:
        for i in range(n_points):
            draw_sphere(ax, previous[i], radius=0.2, color=colors[i], alpha=0.35)

    for i in range(n_points):
        draw_sphere(ax, current[i], radius=0.29, color=colors[i], alpha=1.0)

    if previous is not None:
        for i in range(n_points):
            vec = current[i] - previous[i]
            ax.quiver(previous[i,0], previous[i,1], previous[i,2],
                      vec[0], vec[1], vec[2],
                      pivot='tail',
                      arrow_length_ratio=arrow_head_size,
                      linewidth=1.5, color='k', alpha=0.85)

    ax.set_xlim(mins[0], maxs[0])
    ax.set_ylim(mins[1], maxs[1])
    ax.set_zlim(mins[2], maxs[2])
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.view_init(elev=20, azim=45)
    ax.set_title(title, fontsize=14)

# ---------- Plot ----------
fig = plt.figure(figsize=(7,3))
ax1 = fig.add_subplot(121, projection='3d')
plot_positions(ax1, positions_before, title="Before Feedback")
ax2 = fig.add_subplot(122, projection='3d')
plot_positions(ax2, positions_after, previous=positions_before, title="After Feedback")

# ---------- Legend with 2D circles ----------
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=feedbacked_color, markersize=10, label='Feedbacked'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=non_feedback_color, markersize=10, label='Non-feedbacked')
]
ax2.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05,1), frameon=False, title='')

plt.tight_layout()

fig.savefig(save_dir + "Figure1A.pdf", dpi=600, transparent=True, bbox_inches="tight")
plt.show()
plt.show()
