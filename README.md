# MedSAM on 7T brain MRI: demo + GluCEST extension

Course project (IMgVultr Capital investor pitch). Tool: **MedSAM** (Ma *et al.*, *Nat. Commun.* 15, 654, 2024;
github.com/bowang-lab/MedSAM), the Segment Anything Model fine-tuned on 1.57 M medical image–mask pairs. Draw a box,
get the organ.

Data: our own 7T research scans: whole-brain MP2RAGE and 3D GluCEST from 17 healthy adults (23–73 y). Reference masks:
FSL FIRST (subcortical) and the JHU white-matter atlas warped to each brain.

## Contents

| | |
|---|---|
| `notebooks/01_MedSAM_basic_demo.ipynb` | **Basic demo**, one participant, step by step: data, MedSAM pre-processing, one box, many boxes, three planes, a pre-processing pitfall |
| `notebooks/02_extension_GluCEST.ipynb` | **Extension**: 17-participant evaluation, box robustness, single-click 3D, GluCEST corpus-callosum application |
| `medsam_glucest/model.py` | MedSAM wrapper (official pre-processing, encoder cached per slice, multi-box decoding) |
| `medsam_glucest/data.py` | loaders for the native MP2RAGE (RAS) and the GluCEST slab |
| `medsam_glucest/experiments.py` | the four extension experiments |
| `medsam_glucest/metrics.py` | Dice, HD95, box prompts (with jitter), largest component |
| `scripts/run_batch.py` | runs the experiments for all participants → `outputs/*.csv` (resumable) |
| `scripts/build_notebooks.py` | regenerates the two notebooks |
| `tests/test_core.py` | fast checks of the metric / prompt logic |

## Setup

```bash
python -m venv medsam && medsam\Scripts\activate          # Windows
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m ipykernel install --user --name medsam
python tests/test_core.py                                  # sanity check
python scripts/run_batch.py                                # ~1-2 h on CPU; then run notebook 02
```

The MedSAM weights (375 MB) download from Hugging Face (`wanglab/medsam-vit-base`) on first use. A GPU is not needed:
the image encoder takes ~10 s per slice on a 12-core CPU; each extra box prompt takes milliseconds.

The loaders expect the GluCEST data drive (`GlucestData_withNifit`) and the paper pipeline's cache
(`GluCEST_processing_code/nature_figures_agent3/_cache*`); paths are in `medsam_glucest/data.py`.

## Notes

* **Pre-processing.** We follow MedSAM's own inference recipe (percentile window, [0, 1], resize to 1024², no
  ImageNet normalisation) but zero-pad each slice to a square first; the Hugging Face processor's defaults and the
  straight resize both lose accuracy on non-square MRI slices (notebook 01, section 6).
* **Reference masks are not ground truth.** FSL FIRST is a shape model and the JHU atlas is a warped template; Dice
  against them measures agreement, not accuracy.
* **Data sharing.** The images are human research data. Check with the PI / IRB before uploading any of them (or
  derived images) to Canvas; the notebooks can be submitted with outputs only.
