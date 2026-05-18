import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from scipy import stats

ch_names = ['Fz', 'C3', 'Cz', 'C4', 'CP1', 'CPz', 'CP2', 'Pz']

FILES = {
    'P1_high1': r'D:\Users\Utilizator\Desktop\neuro\P1_high1.mat',
    'P1_high2': r'D:\Users\Utilizator\Desktop\neuro\P1_high2.mat',
    'P1_low1':  r'D:\Users\Utilizator\Desktop\neuro\P1_low1.mat',
    'P1_low2':  r'D:\Users\Utilizator\Desktop\neuro\P1_low2.mat',
    'P2_high1': r'D:\Users\Utilizator\Desktop\neuro\P2_high1.mat',
    'P2_high2': r'D:\Users\Utilizator\Desktop\neuro\P2_high2.mat',
    'P2_low1':  r'D:\Users\Utilizator\Desktop\neuro\P2_low1.mat',
    'P2_low2':  r'D:\Users\Utilizator\Desktop\neuro\P2_low2.mat',
}



def load_epochs(path, t_min_ms=-100, t_max_ms=700):
    mat = sio.loadmat(path)
    fs   = int(mat['fs'].flat[0])
    y    = mat['y']         
    trig = mat['trig'].flatten()

    t_start = int(t_min_ms / 1000 * fs)   
    t_end   = int(t_max_ms / 1000 * fs)   
    bl_end  = -t_start                    

    epochs = {'target': [], 'nontarget': [], 'distractor': []}
    label_map = {2: 'target', 1: 'nontarget', -1: 'distractor'}

    for i, val in enumerate(trig):
        if val in label_map:
            s, e = i + t_start, i + t_end
            if s >= 0 and e <= len(y):
                ep = y[s:e, :]
                ep = ep - ep[:bl_end, :].mean(axis=0)   
                epochs[label_map[val]].append(ep)

    for k in epochs:
        epochs[k] = np.array(epochs[k]) if epochs[k] else np.zeros((0, t_end - t_start, 8))

    time_ms = np.linspace(t_min_ms, t_max_ms, t_end - t_start)
    return epochs, fs, time_ms



def lda_classify(epochs, n_splits=5, random_state=42):
    X_pos = epochs['nontarget'].reshape(len(epochs['nontarget']), -1)
    X_neg = epochs['distractor'].reshape(len(epochs['distractor']), -1)

    rng = np.random.RandomState(random_state)
    idx = rng.choice(len(X_neg), len(X_pos), replace=False)
    X = np.vstack([X_pos, X_neg[idx]])
    y = np.array([1] * len(X_pos) + [0] * len(X_pos))

    lda  = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    skf  = StratifiedKFold(n_splits, shuffle=True, random_state=random_state)
    accs = [accuracy_score(y[te], lda.fit(X[tr], y[tr]).predict(X[te]))
            for tr, te in skf.split(X, y)]
    return np.mean(accs) * 100, np.std(accs) * 100



def p300_snr(epochs, time_ms, ch=2, t_min=250, t_max=500):
    win = (time_ms >= t_min) & (time_ms <= t_max)
    nt_erp  = epochs['nontarget'][:, win, ch].mean(axis=0)
    dist_erp = epochs['distractor'][:, win, ch].mean(axis=0)
    nt_peak  = nt_erp.max()
    d_peak   = dist_erp.max()
    return nt_peak - d_peak, nt_peak, d_peak



def baseline_noise(epochs, t_min_ms=-100, fs=256):
    bl_samples = int(abs(t_min_ms) / 1000 * fs)
    nt = epochs['nontarget'][:, :bl_samples, :]   
    return nt.std(axis=(0, 1)).mean()              



def p300_latency(epochs, time_ms, ch=2, t_min=200, t_max=600):
    win = (time_ms >= t_min) & (time_ms <= t_max)
    erp = epochs['nontarget'][:, win, ch].mean(axis=0)
    return time_ms[win][erp.argmax()]



def cohens_d(epochs, time_ms, ch=2, t_min=250, t_max=500):
    win = (time_ms >= t_min) & (time_ms <= t_max)
    nt   = epochs['nontarget'][:, win, ch].mean(axis=1)  
    dist = epochs['distractor'][:, win, ch].mean(axis=1)
    pooled_std = np.sqrt((nt.std()**2 + dist.std()**2) / 2)
    if pooled_std == 0:
        return 0.0
    return (nt.mean() - dist.mean()) / pooled_std



results = {}
print(f"{'Session':<12} {'LDA%':>6} {'±std':>6} {'SNR(µV)':>9} {'CohenD':>8} {'Latency':>9} {'Noise':>7}")
print("-" * 65)
for name, path in FILES.items():
    epochs, fs, t = load_epochs(path)
    acc, acc_std   = lda_classify(epochs)
    snr, nt_pk, d_pk = p300_snr(epochs, t)
    d              = cohens_d(epochs, t)
    lat            = p300_latency(epochs, t)
    noise          = baseline_noise(epochs, fs=fs)
    results[name]  = dict(epochs=epochs, time=t, acc=acc, acc_std=acc_std,
                          snr=snr, cohens_d=d, latency=lat, noise=noise,
                          nt_peak=nt_pk, d_peak=d_pk)
    print(f"{name:<12} {acc:>6.1f} {acc_std:>6.1f} {snr:>9.2f} {d:>8.3f} {lat:>8.1f}ms {noise:>7.2f}")



fig, axes = plt.subplots(2, 4, figsize=(16, 6))
for ax, (name, r) in zip(axes.flat, results.items()):
    t = r['time']
    for cond, col in [('distractor', 'red'), ('nontarget', 'blue'), ('target', 'green')]:
        m = r['epochs'][cond][:, :, 2].mean(axis=0)   
        ax.plot(t, m, color=col, lw=1.8, label=cond)
    ax.axvspan(250, 500, alpha=0.1, color='yellow')
    ax.axvline(0, color='k', lw=0.8, ls='--')
    ax.set_title(f"{name}  ({r['acc']:.0f}%)", fontsize=8)
    ax.set_xlim(-100, 700)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('µV')

axes[0, 0].legend(fontsize=7)
plt.suptitle('ERP at Cz — Attended (nontarget) vs Distractor vs Non-attended (target)')
plt.tight_layout()
plt.savefig('erp_quick.png', dpi=120)
plt.close()
print("\nSaved: erp_quick.png")



patients = ['P1', 'P2']
metrics  = ['acc', 'snr', 'cohens_d', 'noise']
labels   = ['LDA Accuracy (%)', 'P300 SNR (µV)', "Cohen's d", 'Baseline Noise (µV)']

fig, axes = plt.subplots(2, 4, figsize=(16, 7))
fig.suptitle('High vs Low Session Comparison — What Drives Accuracy?', fontsize=13, fontweight='bold')

for row, patient in enumerate(patients):
    high_keys = [k for k in results if k.startswith(patient) and 'high' in k]
    low_keys  = [k for k in results if k.startswith(patient) and 'low' in k]

    for col, (metric, mlabel) in enumerate(zip(metrics, labels)):
        ax = axes[row, col]
        high_vals = [results[k][metric] for k in high_keys]
        low_vals  = [results[k][metric] for k in low_keys]

        x = np.array([0, 1])
        bar_w = 0.35
        b1 = ax.bar(x - bar_w/2, high_vals, bar_w, color='steelblue', label='High', alpha=0.85)
        b2 = ax.bar(x + bar_w/2, low_vals,  bar_w, color='tomato',    label='Low',  alpha=0.85)

        for bar in b1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 * abs(bar.get_height()),
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)
        for bar in b2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 * abs(bar.get_height()),
                    f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=7)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{k.split("_")[1]}' for k in high_keys], fontsize=8)
        ax.set_xlabel('Run', fontsize=8)
        ax.set_ylabel(mlabel, fontsize=8)
        ax.set_title(f'{patient} — {mlabel}', fontsize=8, fontweight='bold')
        if col == 0:
            ax.set_ylabel(mlabel, fontsize=8)
        if row == 0 and col == 0:
            ax.legend(fontsize=7)

plt.tight_layout()
plt.savefig('high_vs_low_comparison.png', dpi=130)
plt.close()
print("Saved: high_vs_low_comparison.png")



all_acc   = [results[k]['acc']      for k in results]
all_snr   = [results[k]['snr']      for k in results]
all_d     = [results[k]['cohens_d'] for k in results]
all_noise = [results[k]['noise']    for k in results]

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
fig.suptitle('What Predicts LDA Accuracy?', fontsize=12, fontweight='bold')

predictors = [(all_snr, 'P300 SNR (µV)'), (all_d, "Cohen's d"), (all_noise, 'Baseline Noise (µV)')]
colors = ['steelblue' if 'high' in k else 'tomato' for k in results]

for ax, (pred, plabel) in zip(axes, predictors):
    r_val, p_val = stats.pearsonr(pred, all_acc)
    ax.scatter(pred, all_acc, c=colors, s=80, zorder=3, edgecolors='k', linewidths=0.5)

    m, b = np.polyfit(pred, all_acc, 1)
    xline = np.linspace(min(pred), max(pred), 100)
    ax.plot(xline, m * xline + b, 'k--', lw=1.2, alpha=0.6)

    for k, xv, yv in zip(results.keys(), pred, all_acc):
        ax.annotate(k.replace('P', '').replace('_', '\n'),
                    (xv, yv), textcoords='offset points', xytext=(5, 3), fontsize=6)

    ax.set_xlabel(plabel)
    ax.set_ylabel('LDA Accuracy (%)')
    ax.set_title(f'r = {r_val:.2f},  p = {p_val:.3f}', fontsize=9)
    ax.grid(alpha=0.3)

from matplotlib.patches import Patch
axes[0].legend(handles=[Patch(color='steelblue', label='High session'),
                         Patch(color='tomato',    label='Low session')], fontsize=8)

plt.tight_layout()
plt.savefig('accuracy_predictors.png', dpi=130)
plt.close()
print("Saved: accuracy_predictors.png")



fig, ax = plt.subplots(figsize=(13, 4))
ax.axis('off')

col_labels = ['Session', 'Condition', 'LDA Acc (%)', '±Std', 'P300 SNR (µV)',
              "Cohen's d", 'P300 Latency (ms)', 'Baseline Noise (µV)']
table_data = []
for k, r in results.items():
    cond = 'HIGH' if 'high' in k else 'LOW'
    table_data.append([
        k, cond,
        f"{r['acc']:.1f}", f"{r['acc_std']:.1f}",
        f"{r['snr']:.2f}", f"{r['cohens_d']:.3f}",
        f"{r['latency']:.0f}", f"{r['noise']:.2f}"
    ])

tbl = ax.table(cellText=table_data, colLabels=col_labels,
               loc='center', cellLoc='center')
tbl.auto_set_font_size(False)
tbl.set_fontsize(9)
tbl.scale(1.1, 1.6)

for i, row in enumerate(table_data):
    colour = '#d6eaf8' if row[1] == 'HIGH' else '#fadbd8'
    for j in range(len(col_labels)):
        tbl[(i + 1, j)].set_facecolor(colour)

ax.set_title('Full Session Metrics Summary', fontweight='bold', fontsize=11, pad=12)
plt.tight_layout()
plt.savefig('summary_table.png', dpi=130, bbox_inches='tight')
plt.close()
print("Saved: summary_table.png")

print("\n✓ All figures saved.")
