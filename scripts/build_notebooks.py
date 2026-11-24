"""Write notebooks/01_MedSAM_basic_demo.ipynb and notebooks/02_extension_GluCEST.ipynb (then execute with nbconvert)."""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / 'notebooks'
md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell

SETUP = '''import sys, time, os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
sys.path.insert(0, '..')
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from medsam_glucest import data as DT
from medsam_glucest.model import MedSAM, to_rgb
from medsam_glucest.metrics import box_from_mask, dice, hd95, largest_component
from medsam_glucest.experiments import best_axial_slice
plt.rcParams.update({'figure.dpi': 110, 'font.size': 9, 'axes.titlesize': 9})

def show_box(ax, box, **kw):
    x0, y0, x1, y1 = box
    ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=kw.get('c', 'yellow'), lw=kw.get('lw', 1.2)))

def crop_to(m, pad=25):
    r, c = np.nonzero(m)
    return slice(max(r.min() - pad, 0), r.max() + pad), slice(max(c.min() - pad, 0), c.max() + pad)'''

nb1 = nbf.v4.new_notebook()
nb1.cells = [
    md('''# MedSAM on 7T brain MRI: basic demo

**Tool.** MedSAM (Ma *et al.*, *Nature Communications* 15, 654, 2024; github.com/bowang-lab/MedSAM) is the Segment
Anything Model (SAM, ViT-B) fine-tuned on 1.57 million medical image–mask pairs from 10 imaging modalities. It is a
*promptable* segmenter: draw a bounding box around an object and it returns the object's outline. One model, no
task-specific training.

**Data.** Our own 7T research data (not a tutorial dataset): whole-brain MP2RAGE T1-weighted images (0.69 × 0.69 × 0.8 mm,
skull-stripped with HD-BET) from the participants of our GluCEST white-matter ageing study (17 healthy adults, 23–73 y).
Reference masks come from FSL FIRST (subcortical structures) and the JHU white-matter atlas warped to each brain.

**This notebook** walks through one participant step by step:
1. load the image and reference masks
2. what MedSAM does to an image before segmenting it
3. one box → one structure, scored against FSL FIRST
4. many boxes on one slice (the expensive encoder runs once)
5. three imaging planes, including the classic mid-sagittal corpus callosum
6. a pre-processing pitfall we found and fixed

Notebook 02 scales this to all 17 participants and adds three extensions (box robustness, single-click 3D, and a GluCEST
application).'''),
    md('## 0. Setup\nEnvironment: `requirements.txt` (CPU PyTorch is enough; ~10 s per slice on a 12-core laptop). '
       'The MedSAM weights download automatically from Hugging Face on first use (`wanglab/medsam-vit-base`, 375 MB).'),
    code(SETUP + '''
t = time.time()
model = MedSAM(threads=4)
print(f'MedSAM loaded on {model.device} in {time.time() - t:.0f} s;',
      f'{sum(p.numel() for p in model.model.parameters()) / 1e6:.0f} M parameters')'''),
    md('## 1. Load one participant\nEverything is reoriented to RAS on load (`nibabel.as_closest_canonical`) so that '
       'axial / coronal / sagittal slicing is unambiguous; displayed slices have superior or anterior at the top.'),
    code('''S = 'JX'                                      # 25-year-old participant
nv = DT.native(S)
uni, first = nv['uni'], nv['first']
print('MP2RAGE', uni.shape, 'voxel (mm)', tuple(round(float(v), 3) for v in nv['spacing']))
print('FSL FIRST structures:', ', '.join(DT.FIRST[v] for v in np.unique(first) if v in DT.FIRST))

i, j, k = (np.array(uni.shape) / 2).astype(int)
k = best_axial_slice(first == 11)               # axial slice through the left caudate
fig, ax = plt.subplots(1, 3, figsize=(11, 4))
for a, img, lab, t in zip(ax, [DT.axial(uni, k), DT.coronal(uni, j), DT.sagittal(uni, i)],
                          [DT.axial(first, k), DT.coronal(first, j), DT.sagittal(first, i)], ['Axial', 'Coronal', 'Sagittal']):
    a.imshow(img, cmap='gray'); a.imshow(np.ma.masked_equal(lab, 0), cmap='tab20', alpha=0.45, interpolation='nearest')
    a.set_title(f'{t} MP2RAGE + FSL FIRST'); a.axis('off')
plt.tight_layout(); plt.show()'''),
    md('''## 2. What MedSAM sees
MedSAM's official pre-processing: intensity window (0.5–99.5th percentile of brain voxels) → [0, 1] → 3 channels →
resize to 1024 × 1024. Our slices are not square (396 × 208 voxels axially), so we zero-pad to a square first; a
straight resize would stretch the anatomy almost 2:1 (section 6 shows why that matters). The ViT-B encoder turns the
1024² image into a 64 × 64 × 256 embedding; the prompt encoder and mask decoder then work on that embedding.'''),
    code('''img = DT.axial(uni, k)
rgb = to_rgb(img)
n = max(img.shape); sq = np.zeros((n, n, 3)); oy, ox = (n - img.shape[0]) // 2, (n - img.shape[1]) // 2
sq[oy:oy + img.shape[0], ox:ox + img.shape[1]] = rgb
from skimage.transform import resize
fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
ax[0].imshow(img, cmap='gray'); ax[0].set_title(f'raw slice {img.shape}')
ax[1].imshow(rgb); ax[1].set_title('windowed, 3-channel')
ax[2].imshow(resize(rgb, (1024, 1024))); ax[2].set_title('straight resize (distorted)')
ax[3].imshow(resize(sq, (1024, 1024))); ax[3].set_title('pad to square, then 1024²')
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()
t = time.time(); model.set_image(img); te = time.time() - t
print(f'image encoder: {te:.1f} s  ->  embedding {tuple(model._emb.shape)}')'''),
    md('## 3. One box → one structure\nThe box is the tight bounding box of the FSL FIRST left caudate (+2 voxels), '
       'standing in for a box a radiologist would draw. MedSAM returns a probability map; > 0.5 is the mask.'),
    code('''ref = largest_component(DT.axial(first == 11, k))
box = box_from_mask(ref, pad=2)
t = time.time(); prob = model.predict(box, return_prob=True)[0]; td = time.time() - t
pred = prob > 0.5
sp = (nv['spacing'][1], nv['spacing'][0])
print(f'mask decoder: {td * 1000:.0f} ms | Dice vs FIRST {dice(pred, ref):.3f} | HD95 {hd95(pred, ref, sp):.2f} mm')
r, c = crop_to(ref)
fig, ax = plt.subplots(1, 3, figsize=(10, 4))
ax[0].imshow(img[r, c], cmap='gray'); show_box(ax[0], [box[0] - c.start, box[1] - r.start, box[2] - c.start, box[3] - r.start])
ax[0].set_title('prompt: one box')
ax[1].imshow(img[r, c], cmap='gray'); ax[1].imshow(prob[r, c], cmap='magma', alpha=0.6, vmin=0, vmax=1); ax[1].set_title('MedSAM probability')
ax[2].imshow(img[r, c], cmap='gray'); ax[2].contour(ref[r, c], [0.5], colors='lime'); ax[2].contour(pred[r, c], [0.5], colors='red')
ax[2].set_title('FIRST (green) vs MedSAM (red)')
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()'''),
    md('## 4. Many boxes, one encoder pass\nOn a slice through the basal ganglia, six structures are prompted at once. '
       'The encoder cost is paid once; each extra box costs only the decoder.'),
    code('''labs = [11, 50, 12, 51, 10, 49]                  # L/R caudate, putamen, thalamus
k2 = best_axial_slice(first == 12)
img2 = DT.axial(uni, k2)
refs = [largest_component(DT.axial(first == l, k2)) for l in labs]
keep = [(l, m) for l, m in zip(labs, refs) if m.sum() > 20]
boxes = [box_from_mask(m, 2) for _, m in keep]
t = time.time(); model.set_image(img2); t_enc = time.time() - t
t = time.time(); preds = model.predict(boxes); t_dec = time.time() - t
rows = [dict(structure=DT.FIRST[l], dice=dice(p, m), hd95_mm=hd95(p, m, sp)) for (l, m), p in zip(keep, preds)]
print(f'encoder {t_enc:.1f} s once + decoder {t_dec * 1000:.0f} ms for {len(boxes)} boxes')
display(pd.DataFrame(rows).round(3))
allref = np.any(refs, axis=0); r, c = crop_to(allref, 15)
fig, ax = plt.subplots(1, 2, figsize=(10, 5))
ax[0].imshow(img2[r, c], cmap='gray'); [show_box(ax[0], [b[0] - c.start, b[1] - r.start, b[2] - c.start, b[3] - r.start]) for b in boxes]
ax[0].set_title('six boxes'); ax[1].imshow(img2[r, c], cmap='gray')
for (l, m), p in zip(keep, preds):
    ax[1].contour(m[r, c], [0.5], colors='lime', linewidths=0.8); ax[1].contour(p[r, c], [0.5], colors='red', linewidths=0.8)
ax[1].set_title('FIRST (green) vs MedSAM (red)')
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()'''),
    md('''## 5. Three planes
The same model works in any plane: hippocampus on a coronal slice, and the corpus callosum on the mid-sagittal slice
(the classic CC segmentation task; reference = JHU genu + body + splenium warped to this brain), where a single box
does worse than three boxes along the arch.'''),
    code('''fig, ax0 = plt.subplots(figsize=(5, 4.5)); ax = [ax0]
# coronal hippocampus
jh = int(np.argmax((first == 17).sum(axis=(0, 2))))
imgc, refc = DT.coronal(uni, jh), largest_component(DT.coronal(first == 17, jh))
pc = model.segment(imgc, box_from_mask(refc, 2))[0]
r, c = crop_to(refc, 30)
ax[0].imshow(imgc[r, c], cmap='gray'); ax[0].contour(refc[r, c], [0.5], colors='lime'); ax[0].contour(pc[r, c], [0.5], colors='red')
ax[0].set_title(f'Coronal: left hippocampus, Dice {dice(pc, refc):.2f}')
# mid-sagittal corpus callosum. Two details matter:
#  * use the true midline (centre of the brain), where the CC is a bright arch with clear edges; a few mm lateral it
#    merges into the hemispheric white matter and there is no edge to find
#  * the CC is an arch: one box around it is mostly ventricle, so split it into 3 boxes along the anterior-posterior axis
cc3 = np.isin(nv['jhu'], [3, 4, 5])
im_ = int(round(np.argwhere(uni > 0)[:, 0].mean()))
imgs, refs_ = DT.sagittal(uni, im_), largest_component(DT.sagittal(cc3, im_))
def ap_boxes(ref, n):
    cols = np.nonzero(ref)[1]; edges = np.linspace(cols.min(), cols.max() + 1, n + 1).astype(int)
    out = []
    for a_, b_ in zip(edges[:-1], edges[1:]):
        part = ref.copy(); part[:, :a_] = False; part[:, b_:] = False
        if part.sum() > 5:
            out.append(box_from_mask(largest_component(part), 2))
    return out
model.set_image(imgs)
p_one = model.predict(ap_boxes(refs_, 1)).any(axis=0)
boxes3 = ap_boxes(refs_, 3)
p_three = model.predict(boxes3).any(axis=0)
ax[0].axis('off'); plt.tight_layout(); plt.show()
r, c = crop_to(refs_, 25)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for a, p_, bx, t in [(ax[0], p_one, ap_boxes(refs_, 1), 'one box'), (ax[1], p_three, boxes3, 'three boxes')]:
    a.imshow(imgs[r, c], cmap='gray'); a.contour(refs_[r, c], [0.5], colors='lime'); a.contour(p_[r, c], [0.5], colors='red')
    for b_ in bx:
        show_box(a, [b_[0] - c.start, b_[1] - r.start, b_[2] - c.start, b_[3] - r.start], lw=0.8)
    a.set_title(f'Mid-sagittal corpus callosum, {t}: Dice {dice(p_, refs_):.2f}'); a.axis('off')
plt.tight_layout(); plt.show()'''),
    md('''## 6. A pre-processing pitfall
The Hugging Face processor's defaults (ImageNet mean/std, pad to square) are not MedSAM's training pre-processing, and
the official recipe (straight resize to 1024²) distorts non-square slices. We compared all three on eight structures.'''),
    code('''from medsam_glucest.experiments import structure_eval
res = []
for name, pp, kw in [('official, padded (ours)', 'medsam', {}), ('Hugging Face defaults', 'hf', {})]:
    m_ = model if pp == 'medsam' else MedSAM(threads=4, preprocess='hf')
    res.append(structure_eval(m_, S).assign(preprocessing=name))
    if m_ is not model:
        del m_                                   # free the second model copy (memory)
res = pd.concat(res)
display(res.pivot(index='structure', columns='preprocessing', values='dice').round(3))
print(res.groupby('preprocessing').dice.mean().round(3).to_string())'''),
    md('''## Take-aways
* One box per structure is enough for MedSAM to outline structures with a **visible boundary** (caudate next to the
  ventricle, hippocampus) closely; where MP2RAGE contrast is weak (thalamus / putamen against white matter) it
  segments only what it can see.
* FSL FIRST is a shape model and not ground truth: part of the disagreement is the reference extending past the
  visible edge.
* Box prompts are ambiguous for curved, hollow shapes: one box around the corpus callosum arch is mostly ventricle;
  three boxes along the arch work better. And the slice matters: at the true midline the CC has clear edges, a few mm
  lateral it merges into the white matter and MedSAM has nothing to follow.
* The encoder runs once per slice (~10-17 s on CPU); extra boxes are almost free (milliseconds).
* Pre-processing matters: matching MedSAM's training recipe (and not distorting the slice) changes Dice noticeably.

→ Notebook 02: all 17 participants, robustness, single-click 3D and a GluCEST application.'''),
]

nb2 = nbf.v4.new_notebook()
nb2.cells = [
    md('''# Extension: MedSAM at cohort scale, and for GluCEST

Built on the `medsam_glucest` package (see README). The heavy compute is done by `scripts/run_batch.py`
(~1–2 h on CPU for 17 participants); this notebook loads its results from `outputs/` and makes the figures.

1. **Cohort evaluation**: 8 subcortical structures × 17 participants vs FSL FIRST
2. **Robustness**: how sloppy can the user's box be?
3. **Single-click 3D**: one box on one slice → whole structure, by propagating MedSAM's own masks slice to slice
4. **GluCEST application**: MedSAM refines the warped JHU corpus-callosum atlas on the GluCEST slab; does the GluCEST
   value, and its age trend, change?'''),
    code(SETUP + '''
from scipy import stats
OUT = __import__('pathlib').Path('../outputs')
st = pd.read_csv(OUT / 'structures.csv'); rb = pd.read_csv(OUT / 'box_robustness.csv')
gc = pd.read_csv(OUT / 'glucest_cc.csv'); pr = pd.read_csv(OUT / 'propagation.csv')
print('participants:', st.Subject.nunique(), '| structure evaluations:', len(st))'''),
    md('## 1. Cohort evaluation'),
    code('''st['side'] = st.structure.str[0]; st['struct'] = st.structure.str[2:]
summ = st.groupby('struct').agg(dice_mean=('dice', 'mean'), dice_sd=('dice', 'std'), hd95_mm=('hd95_mm', 'median'),
                                pred_over_ref_area=('area_pred_mm2', 'sum')).round(3)
summ['pred_over_ref_area'] = (st.groupby('struct').area_pred_mm2.sum() / st.groupby('struct').area_ref_mm2.sum()).round(2)
display(summ.sort_values('dice_mean', ascending=False))
order = summ.sort_values('dice_mean', ascending=False).index
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for i, s_ in enumerate(order):
    v = st[st.struct == s_].dice
    ax[0].scatter(np.full(len(v), i) + np.random.default_rng(i).uniform(-0.15, 0.15, len(v)), v, s=10, alpha=0.6)
    ax[0].plot([i - 0.25, i + 0.25], [v.median()] * 2, 'k', lw=2)
ax[0].set_xticks(range(len(order)), order); ax[0].set_ylabel('Dice vs FSL FIRST'); ax[0].set_ylim(0, 1)
ax[0].set_title('Per structure (17 participants x 2 sides)')
pm = st.groupby(['Subject', 'Age']).dice.mean().reset_index()
r, p = stats.pearsonr(pm.Age, pm.dice)
ax[1].scatter(pm.Age, pm.dice); ax[1].set_xlabel('Age (years)'); ax[1].set_ylabel('mean Dice')
ax[1].set_title(f'Does performance depend on age? r = {r:.2f}, P = {p:.2f}')
plt.tight_layout(); plt.show()'''),
    md('## 2. Robustness to the box\nEach box edge is moved randomly by up to x % of the box size (5 draws per level), '
       'for the left caudate (high contrast) and right thalamus (low contrast).'),
    code('''g = rb.groupby(['structure', 'jitter']).dice.agg(['mean', 'sem']).reset_index()
fig, ax = plt.subplots(figsize=(5, 3.6))
for s_, d in g.groupby('structure'):
    ax.errorbar(100 * d.jitter, d['mean'], yerr=d['sem'], marker='o', capsize=2, label=s_)
ax.set_xlabel('box perturbation (% of box size)'); ax.set_ylabel('Dice vs FSL FIRST'); ax.legend(); ax.set_ylim(0, 1)
ax.set_title('How precise must the user be?'); plt.tight_layout(); plt.show()
display(g.pivot(index='jitter', columns='structure', values='mean').round(3))'''),
    md('## 3. Single-click 3D\nOne box on the slice where the left caudate is largest; every other slice is prompted '
       'with the box of MedSAM\'s own mask on the neighbouring slice. No reference is used after the first click.'),
    code('''display(pr[['Subject', 'Age', 'dice3d', 'vol_ref_ml', 'vol_pred_ml', 'slices_ref', 'slices_pred', 'encoder_calls']].round(3))
s0 = pr.sort_values('dice3d').iloc[len(pr) // 2].Subject              # median participant
z = np.load(OUT / f'prop_{s0}_Lcaudate.npz'); pred3, ref3 = z['pred'], z['ref']
nv = DT.native(s0)
jc = int(np.argmax(ref3.sum(axis=(0, 2)))); ic = int(np.argmax(ref3.sum(axis=(1, 2))))
fig, ax = plt.subplots(1, 2, figsize=(9, 4.5))
for a, img, rf, pd_, t in [(ax[0], DT.coronal(nv['uni'], jc), DT.coronal(ref3, jc), DT.coronal(pred3, jc), 'coronal'),
                           (ax[1], DT.sagittal(nv['uni'], ic), DT.sagittal(ref3, ic), DT.sagittal(pred3, ic), 'sagittal')]:
    r, c = crop_to(rf | pd_, 25)
    a.imshow(img[r, c], cmap='gray'); a.contour(rf[r, c], [0.5], colors='lime'); a.contour(pd_[r, c], [0.5], colors='red')
    a.set_title(f'{s0}: left caudate, {t} view of the 3D result'); a.axis('off')
plt.tight_layout(); plt.show()'''),
    md('''## 4. GluCEST application: the corpus callosum
In the GluCEST study, white-matter regions come from the JHU atlas warped to each brain, which is coarse at the 1 mm
slab resolution. Here each connected piece of the atlas CC on each slab slice becomes a MedSAM box prompt on the slab's
MP2RAGE; GluCEST is then averaged inside MedSAM's mask instead of the atlas mask.'''),
    code('''display(gc[['Subject', 'Age', 'dice_medsam_vs_atlas', 'n_vox_atlas', 'n_vox_medsam', 'wm_frac_atlas', 'wm_frac_medsam',
           'glucest_atlas', 'glucest_medsam']].round(3))
t_ = stats.ttest_rel(gc.glucest_medsam, gc.glucest_atlas)
print(f"MedSAM vs atlas mask: Dice {gc.dice_medsam_vs_atlas.mean():.2f}; WM fraction {gc.wm_frac_atlas.mean():.2f} -> {gc.wm_frac_medsam.mean():.2f};",
      f"GluCEST {gc.glucest_atlas.mean():.2f} -> {gc.glucest_medsam.mean():.2f} % (paired P = {t_.pvalue:.3g})")
fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
ax[0].scatter(gc.glucest_atlas, gc.glucest_medsam); lim = [gc[['glucest_atlas', 'glucest_medsam']].min().min() - 0.3, gc[['glucest_atlas', 'glucest_medsam']].max().max() + 0.3]
ax[0].plot(lim, lim, 'k:'); ax[0].set_xlabel('GluCEST, atlas CC (%)'); ax[0].set_ylabel('GluCEST, MedSAM CC (%)'); ax[0].set_title('Per participant')
for col, lab in [('glucest_atlas', 'atlas'), ('glucest_medsam', 'MedSAM')]:
    r, p = stats.pearsonr(gc.Age, gc[col]); b = np.polyfit(gc.Age, gc[col], 1)
    ax[1].scatter(gc.Age, gc[col], s=14); ax[1].plot(gc.Age.sort_values(), np.polyval(b, gc.Age.sort_values()),
                                               label=f'{lab}: {10 * b[0]:+.2f} %/decade, r = {r:.2f}, P = {p:.2f}')
ax[1].set_xlabel('Age (years)'); ax[1].set_ylabel('CC GluCEST (%)'); ax[1].legend(fontsize=7); ax[1].set_title('Age trend')
plt.tight_layout(); plt.show()'''),
    code('''# what the refinement looks like on one slab slice
from medsam_glucest.experiments import glucest_cc
model = MedSAM(threads=8)
s1 = gc.sort_values('Age').iloc[len(gc) // 2].Subject
masks, _ = glucest_cc(model, s1)
sl = DT.slab(s1); cc = DT.slab_cc(sl)
z = max(masks, key=lambda q: masks[q].sum())
fig, ax = plt.subplots(1, 3, figsize=(12, 4))
ax[0].imshow(sl['uni'][:, :, z], cmap='gray'); ax[0].contour(cc[:, :, z], [0.5], colors='lime'); ax[0].set_title(f'{s1} slab slice {z}: JHU atlas CC')
ax[1].imshow(sl['uni'][:, :, z], cmap='gray'); ax[1].contour(masks[z], [0.5], colors='red'); ax[1].set_title('MedSAM CC')
g_ = np.where(DT.valid_glucest(sl)[:, :, z], sl['glucest'][:, :, z], np.nan)
ax[2].imshow(sl['uni'][:, :, z], cmap='gray'); ax[2].imshow(np.where(masks[z], g_, np.nan), cmap='magma', vmin=0, vmax=12)
ax[2].set_title('GluCEST inside the MedSAM CC')
for a in ax: a.axis('off')
plt.tight_layout(); plt.show()'''),
    md('''## Take-aways for the pitch
* **Scale**: one generic model, no training, segments 8 structures in 17 brains from boxes alone.
* **Robust**: performance holds up when the box is drawn sloppily (see section 2 for where it breaks).
* **Single click → 3D**: one box gives a whole structure in 3D.
* **Plugs into real research**: MedSAM can refine atlas-derived ROIs for quantitative imaging such as GluCEST.
* **Limits**: driven by visible contrast; low-contrast boundaries (thalamus / putamen on MP2RAGE) are under-segmented;
  2D model (3D only by propagation); CPU inference ~10 s per slice.'''),
]

if __name__ == '__main__':
    nbf.write(nb1, NB / '01_MedSAM_basic_demo.ipynb')
    nbf.write(nb2, NB / '02_extension_GluCEST.ipynb')
    print('notebooks written')
