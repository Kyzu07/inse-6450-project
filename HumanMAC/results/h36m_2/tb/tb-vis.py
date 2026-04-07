"""
visualize_tb.py
---------------
Reads a TensorBoard event file and produces multiple training analysis plots.
Usage: python visualize_tb.py --event_file <path_to_tfevents_file> [--out_dir <dir>]
"""

import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MaxNLocator
from tensorboard.backend.event_processing import event_accumulator

# ── CLI args ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--event_file', type=str, required=True,
                    help='Path to the tfevents file')
parser.add_argument('--out_dir', type=str, default='tb_plots',
                    help='Directory to save output plots')
parser.add_argument('--smooth', type=float, default=0.6,
                    help='Exponential smoothing factor (0 = no smoothing, 0.99 = heavy)')
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)

# ── Load event file ───────────────────────────────────────────────────────────
print(f"Loading: {args.event_file}")
ea = event_accumulator.EventAccumulator(args.event_file)
ea.Reload()

tags = ea.Tags()['scalars']
print(f"Tags found: {tags}")

# Extract into numpy arrays
data = {}
for tag in tags:
    events = ea.Scalars(tag)
    data[tag] = {
        'step':  np.array([e.step  for e in events]),
        'value': np.array([e.value for e in events]),
        'wall':  np.array([e.wall_time for e in events]),
    }

train = data.get('Loss/train', data.get('Loss_train', list(data.values())[0]))
val   = data.get('Loss/val',   data.get('Loss_val',   list(data.values())[-1]))

steps       = train['step']
train_loss  = train['value']
val_loss    = val['value'][:len(steps)]  # align length

# ── Exponential moving average ────────────────────────────────────────────────
def ema(values, alpha):
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i-1] + (1 - alpha) * values[i]
    return smoothed

train_smooth = ema(train_loss, args.smooth)
val_smooth   = ema(val_loss,   args.smooth)

# ── Rolling statistics (window = 10 epochs) ───────────────────────────────────
W = 10
def rolling(arr, fn, w):
    return np.array([fn(arr[max(0, i-w):i+1]) for i in range(len(arr))])

train_std = rolling(train_loss, np.std, W)
val_std   = rolling(val_loss,   np.std, W)

# ── Convergence rate (epoch-over-epoch % improvement) ────────────────────────
conv_rate = np.diff(train_smooth) / (train_smooth[:-1] + 1e-9) * -100  # positive = improving

# ── Wall time per step ────────────────────────────────────────────────────────
wall       = train['wall']
step_times = np.diff(wall)  # seconds between logged steps

# ── Generalization gap ────────────────────────────────────────────────────────
gap = train_loss - val_loss

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 1 — Train & Val loss with smoothing and confidence band
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 4.5))

# raw (faint)
ax.plot(steps, train_loss, color='steelblue',  alpha=0.2, lw=1.0)
ax.plot(steps, val_loss,   color='darkorange', alpha=0.2, lw=1.0)
# smoothed
ax.plot(steps, train_smooth, color='steelblue',  lw=2.0, label='Train (smoothed)')
ax.plot(steps, val_smooth,   color='darkorange', lw=2.0, label='Val (smoothed)',   ls='--')
# rolling std band
ax.fill_between(steps, train_smooth - train_std, train_smooth + train_std,
                color='steelblue', alpha=0.10)
ax.fill_between(steps, val_smooth - val_std,   val_smooth + val_std,
                color='darkorange', alpha=0.10)

ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss (DCT space)', fontsize=12)
ax.set_title('Train & Validation Loss with EMA Smoothing', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.xaxis.set_major_locator(MaxNLocator(integer=True))
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, '01_loss_curves.png'), dpi=150)
plt.close()
print("Saved: 01_loss_curves.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 2 — Generalization gap
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.plot(steps, gap, color='purple', lw=1.5)
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.fill_between(steps, gap, 0, where=(gap > 0), alpha=0.15, color='purple', label='Train > Val (healthy)')
ax.fill_between(steps, gap, 0, where=(gap < 0), alpha=0.15, color='red',    label='Val > Train')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Train − Val Loss', fontsize=12)
ax.set_title('Generalization Gap', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, '02_generalization_gap.png'), dpi=150)
plt.close()
print("Saved: 02_generalization_gap.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 3 — Convergence rate (% loss improvement per epoch)
# ─────────────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(11, 3.5))
ax.bar(steps[1:], conv_rate, color=np.where(conv_rate > 0, 'seagreen', 'tomato'),
       alpha=0.75, width=0.8)
ax.axhline(0, color='black', lw=0.8)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('% Loss Improvement', fontsize=12)
ax.set_title('Epoch-over-Epoch Convergence Rate (Train, EMA)', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, '03_convergence_rate.png'), dpi=150)
plt.close()
print("Saved: 03_convergence_rate.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 4 — Loss distribution: early vs mid vs late training
# ─────────────────────────────────────────────────────────────────────────────
n = len(steps)
thirds = [
    ('Early  (0–33%)',  train_loss[:n//3],          'steelblue'),
    ('Mid    (33–66%)', train_loss[n//3:2*n//3],    'darkorange'),
    ('Late   (66–100%)',train_loss[2*n//3:],         'seagreen'),
]

fig, ax = plt.subplots(figsize=(9, 4))
for label, values, color in thirds:
    ax.hist(values, bins=25, alpha=0.55, color=color, label=label, edgecolor='white')
ax.set_xlabel('Train Loss', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Train Loss Distribution: Early vs Mid vs Late Training', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, '04_loss_distribution.png'), dpi=150)
plt.close()
print("Saved: 04_loss_distribution.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 5 — Val loss: best checkpoint marker + rolling minimum
# ─────────────────────────────────────────────────────────────────────────────
rolling_min_val = np.minimum.accumulate(val_loss)
best_epoch      = steps[np.argmin(val_loss)]
best_val        = val_loss.min()

fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(steps, val_loss,       color='darkorange', lw=1.5, alpha=0.5, label='Val Loss')
ax.plot(steps, rolling_min_val, color='red',        lw=1.8, ls='--',  label='Running Best')
ax.axvline(best_epoch, color='gray', lw=1.2, ls=':')
ax.scatter([best_epoch], [best_val], color='red', zorder=5, s=60,
           label=f'Best: {best_val:.5f} @ epoch {best_epoch}')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Val Loss', fontsize=12)
ax.set_title('Validation Loss — Best Checkpoint Tracking', fontsize=13)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(args.out_dir, '05_best_checkpoint.png'), dpi=150)
plt.close()
print("Saved: 05_best_checkpoint.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 6 — Wall-clock time between logged steps
# ─────────────────────────────────────────────────────────────────────────────
if len(step_times) > 1:
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.5))

    axes[0].plot(steps[1:], step_times, color='teal', lw=1.2, alpha=0.8)
    axes[0].axhline(step_times.mean(), color='red', ls='--', lw=1.3,
                    label=f'Mean: {step_times.mean():.1f}s')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Seconds')
    axes[0].set_title('Time Between Logged Steps'); axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].hist(step_times, bins=20, color='teal', alpha=0.75, edgecolor='white')
    axes[1].axvline(step_times.mean(),   color='red',    ls='--', lw=1.5, label=f'Mean:   {step_times.mean():.1f}s')
    axes[1].axvline(np.median(step_times),color='orange', ls='--', lw=1.5, label=f'Median: {np.median(step_times):.1f}s')
    axes[1].set_xlabel('Seconds'); axes[1].set_ylabel('Frequency')
    axes[1].set_title('Step Time Distribution'); axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, '06_step_timing.png'), dpi=150)
    plt.close()
    print("Saved: 06_step_timing.png")

# ─────────────────────────────────────────────────────────────────────────────
# PLOT 7 — Summary dashboard (all-in-one)
# ─────────────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle('HumanMAC Training Dashboard', fontsize=15, fontweight='bold', y=0.98)
gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3)

# 7a — loss curves
ax1 = fig.add_subplot(gs[0, :])
ax1.plot(steps, train_loss,   color='steelblue',  alpha=0.2, lw=1.0)
ax1.plot(steps, val_loss,     color='darkorange',  alpha=0.2, lw=1.0)
ax1.plot(steps, train_smooth, color='steelblue',  lw=2.0, label='Train')
ax1.plot(steps, val_smooth,   color='darkorange', lw=2.0, label='Val', ls='--')
ax1.set_xlabel('Epoch'); ax1.set_ylabel('MSE Loss')
ax1.set_title('Loss Curves (EMA smoothed)'); ax1.legend(); ax1.grid(True, alpha=0.3)

# 7b — gap
ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(steps, gap, color='purple', lw=1.5)
ax2.axhline(0, color='black', lw=0.8, ls='--')
ax2.fill_between(steps, gap, 0, where=(gap > 0), alpha=0.15, color='purple')
ax2.fill_between(steps, gap, 0, where=(gap < 0), alpha=0.15, color='red')
ax2.set_xlabel('Epoch'); ax2.set_ylabel('Train − Val')
ax2.set_title('Generalization Gap'); ax2.grid(True, alpha=0.3)

# 7c — rolling best val
ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(steps, val_loss,        color='darkorange', lw=1.5, alpha=0.5)
ax3.plot(steps, rolling_min_val, color='red',        lw=2.0, ls='--')
ax3.scatter([best_epoch], [best_val], color='red', zorder=5, s=60,
            label=f'Best: {best_val:.5f} @ ep {best_epoch}')
ax3.set_xlabel('Epoch'); ax3.set_ylabel('Val Loss')
ax3.set_title('Best Checkpoint Tracking'); ax3.legend(fontsize=9); ax3.grid(True, alpha=0.3)

# 7d — convergence rate
ax4 = fig.add_subplot(gs[2, 0])
ax4.bar(steps[1:], conv_rate, color=np.where(conv_rate > 0, 'seagreen', 'tomato'),
        alpha=0.75, width=0.8)
ax4.axhline(0, color='black', lw=0.8)
ax4.set_xlabel('Epoch'); ax4.set_ylabel('% Improvement')
ax4.set_title('Convergence Rate'); ax4.grid(True, alpha=0.3, axis='y')

# 7e — loss distribution
ax5 = fig.add_subplot(gs[2, 1])
for label, values, color in thirds:
    ax5.hist(values, bins=20, alpha=0.55, color=color, label=label.split()[0], edgecolor='white')
ax5.set_xlabel('Train Loss'); ax5.set_ylabel('Frequency')
ax5.set_title('Loss Distribution by Phase'); ax5.legend(fontsize=9); ax5.grid(True, alpha=0.3)

plt.savefig(os.path.join(args.out_dir, '07_dashboard.png'), dpi=150, bbox_inches='tight')
plt.close()
print("Saved: 07_dashboard.png")

# ─────────────────────────────────────────────────────────────────────────────
# Console summary
# ─────────────────────────────────────────────────────────────────────────────
print()
print('=' * 45)
print('Summary')
print('=' * 45)
print(f'  Epochs logged       : {len(steps)}')
print(f'  Initial train loss  : {train_loss[0]:.5f}')
print(f'  Final train loss    : {train_loss[-1]:.5f}')
print(f'  Initial val loss    : {val_loss[0]:.5f}')
print(f'  Final val loss      : {val_loss[-1]:.5f}')
print(f'  Best val loss       : {best_val:.5f}  @ epoch {best_epoch}')
print(f'  Avg convergence rate: {conv_rate[conv_rate > 0].mean():.3f}% / epoch')
if len(step_times) > 1:
    print(f'  Avg step time       : {step_times.mean():.1f}s')
    print(f'  Total wall time     : {step_times.sum()/3600:.2f} hrs')
print(f'  Plots saved to      : {args.out_dir}/')
print('=' * 45)