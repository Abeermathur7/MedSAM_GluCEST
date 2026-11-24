"""Data access for the 17 GluCEST participants (RG excluded, as in the paper).

native(s): whole-brain 7T MP2RAGE (0.69 x 0.69 x 0.8 mm, HD-BET skull-stripped) with FSL FIRST subcortical labels,
           JHU white-matter labels and FAST tissue classes, all reoriented to RAS so slicing is unambiguous.
slab(s):   the 24 mm GluCEST slab (240 x 196 x 12, 1 x 1 x 2 mm): MP2RAGE resampled into it, JHU atlas, tissue mask,
           ROI mask and the GluCEST map, from the paper pipeline's cache (orientation-checked 2026-11-19).
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np

PROC = Path(r'C:/Users/Abeer/OneDrive/Desktop/GluCEST_processing_code')
CACHE = PROC / 'nature_figures_agent3' / '_cache'
CACHE_SEG = PROC / 'nature_figures_agent3' / '_cache_niftiseg'      # NIfTI (FAST) tissue masks
sys.path.insert(0, str(PROC))
import utils_main as U  # noqa: E402  (subject table + data-drive location)

EXCLUDE = {'RG'}
FIRST = {10: 'L thalamus', 11: 'L caudate', 12: 'L putamen', 13: 'L pallidum', 16: 'Brainstem', 17: 'L hippocampus',
         18: 'L amygdala', 26: 'L accumbens', 49: 'R thalamus', 50: 'R caudate', 51: 'R putamen', 52: 'R pallidum',
         53: 'R hippocampus', 54: 'R amygdala', 58: 'R accumbens'}
CC_LABELS = {'Genu of corpus callosum', 'Body of corpus callosum', 'Splenium of corpus callosum'}   # JHU names


def subjects():
    """[(initials, age)] youngest first."""
    out = [(k.split(' – ')[0], a) for k, (f, a) in U.SUBJECT_MAP.items()]
    return sorted([(s, a) for s, a in out if s not in EXCLUDE], key=lambda x: x[1])


def _native_dir(s):
    folder = U.SUBJECT_MAP[f'{s} – Scan 1'][0]
    hits = sorted((U.BASE / folder).rglob('MP2RAGE_UNI_hdbet_first.nii.gz'))
    if not hits:
        raise FileNotFoundError(f'no FIRST segmentation for {s} under {U.BASE / folder} (is the data drive connected?)')
    # several export folders for some participants: take the one whose FIRST file is populated
    return max(hits, key=lambda h: (np.asarray(nib.load(str(h)).dataobj) > 0).sum()).parent


@lru_cache(maxsize=1)   # whole-brain volumes are ~300 MB per participant
def native(s):
    d = _native_dir(s)

    def ras(name, dtype):
        img = nib.as_closest_canonical(nib.load(str(d / name)))
        return np.asarray(img.dataobj, dtype), img

    uni, img = ras('MP2RAGE_UNI_hdbet.nii.gz', np.float32)
    out = dict(uni=uni, first=ras('MP2RAGE_UNI_hdbet_first.nii.gz', np.int16)[0],
               fast=ras('MP2RAGE_UNI_hdbet_fast3_seg.nii.gz', np.int16)[0], spacing=img.header.get_zooms()[:3],
               folder=str(d))
    if (d / 'MP2RAGE_UNI_hdbet_WMrois.nii.gz').exists():
        out['jhu'] = ras('MP2RAGE_UNI_hdbet_WMrois.nii.gz', np.int16)[0]
    return out


# RAS volume -> 2D display slice, superior/anterior at the top
def axial(v, k):
    return np.rot90(v[:, :, k])


def coronal(v, j):
    return np.rot90(v[:, j, :])


def sagittal(v, i):
    return np.rot90(v[i, :, :])


def slab(s):
    sb = np.load(CACHE / f'sub_{s}.npz')
    vl = np.load(CACHE_SEG / f'vol_{s}.npz')
    lbl = {int(k): v for k, v in __import__('json').loads((CACHE / f'labels_{s}.json').read_text())['WM'].items()}
    return dict(uni=sb['uni'], wm_atlas=sb['wm_atlas'].astype(int), seg=vl['seg'], roi=vl['roi'],
                glucest=vl['CESTmap'].astype(float), labels=lbl)


def slab_cc(sl):
    """corpus callosum (genu + body + splenium) from the JHU atlas in the slab."""
    keys = [k for k, n in sl['labels'].items() if n in CC_LABELS]
    return np.isin(sl['wm_atlas'], keys)


def valid_glucest(sl):
    c = sl['glucest']
    return np.isfinite(c) & (sl['roi'] > 0) & (c > -10) & (c < 30) & (c != 0)
