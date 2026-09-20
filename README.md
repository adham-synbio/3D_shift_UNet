# Parameter-efficient 3D liver and liver-tumour segmentation

We propose a 3D U-Net variant for volumetric medical image segmentation built around
depthwise separable convolution, together with the analysis code used to decide
where in the network learnable spatial filtering is actually needed.

|Convolution|2D|3D|growth|
|-|-:|-:|-:|
|Dense|17,266,371|51,772,227|3.00×|
|Depthwise separable|1,978,252|2,073,886|1.05×|

*Matched configuration: 1 input channel, 3 classes, base width 64.*



\---

## Status

**This repository accompanies a manuscript in preparation. Results below are
current but not final**,  see `docs/NUMBER\\\_AUDIT.md` for exactly which numbers
are verified, which come from run logs, and which are still open.

Main result on MSD Task03\_Liver, per-case Dice, held-out volume-level split of
19 volumes, three random initialisations, with post-processing:

||Parameters|Liver Dice|Tumour Dice|
|-|-:|-:|-:|
|Ours (hybrid)|536,990|0.9433 ± 0.0030|0.5762 ± 0.0438|
|Dense 3D U-Net, identical training|12,946,851|0.8733|0.4551|

For reference, nnU-Net reports 0.9411 liver / 0.6174 tumour for its 3D U-Net on
this task under five-fold cross-validation over all 131 cases
([arXiv:1809.10486](https://arxiv.org/abs/1809.10486), Table 2). **That is a
different evaluation protocol from ours and the numbers are not directly
comparable.**

### 

### What the parameter reduction does



||Dense 3D U-Net|Ours|Ratio|
|-|-:|-:|-:|
|Parameters|12,946,851|536,990|24× smaller|
|Training activation memory (128³, batch 1)|4.58 GB|6.06 GB|**1.32× larger**|
|CPU inference, s/patch (1 core, 128³)|20.6|16.8|1.2× faster|



\---

## Layout

```
src/     all code (flat,  the scripts import each other by module name)
paper/   LaTeX manuscript, bibliography, TikZ architecture figure
docs/    experiment log and number audit
```

### Models

|File|Purpose|
|-|-|
|`model3d.py`|3D U-Net with fixed-shift, learnable-depthwise and hybrid blocks|
|`model.py`|2D counterpart, plus L0 gating and the S²-MLPv2 shift operator|
|`l0\\\_depthwise.py`|Hard-concrete L0 gates for per-tap sparsity|

### Data

|File|Purpose|
|-|-|
|`prepare\\\_lits\\\_3d.py`|Resample MSD Task03\_Liver to isotropic spacing; reads the `.tar` directly|
|`lits3d\\\_data.py`|3D patch sampling with foreground oversampling; sliding-window inference|
|`augmentation.py`|Shared 2D augmentation|

### Training

|File|Purpose|
|-|-|
|`train\\\_lits3d.py`|3D training with per-case validation and resume|
|`crossval\\\_lits3d.py`|Five-fold cross-validation driver|

### Analysis

|File|Purpose|
|-|-|
|`analyze\\\_depthwise\\\_filters.py`|How far learned filters drift from the shift basis|
|`inspect\\\_lits3d\\\_cases.py`|Per-case failure diagnosis: contrast, liver coverage, recall, precision|
|`run\\\_sparse\\\_shift\\\_benchmark.py`|2D benchmark: all block variants, L0 sweep, freezing sweep|

\---

## Getting started

```bash
pip install torch numpy scipy nibabel pillow tqdm scikit-learn
```

### 1\. Prepare the data

Download MSD Task03\_Liver (about 27 GB) from
[medicaldecathlon.com](http://medicaldecathlon.com/), then:

```bash
python src/prepare\\\_lits\\\_3d.py \\\\
    --src /path/to/Task03\\\_Liver.tar \\\\
    --out\\\_dir /path/to/lits3d \\\\
    --target\\\_spacing 1.5 1.5 1.5
```

The archive is read case by case, so peak extra disk is one
volume pair rather than the full 27 GB. `--archive\\\_dir` moves each case to a
second location as it is written, which keeps local disk near zero. Output is
about 9 GB at 1.5 mm isotropic; run with `--max\\\_cases 3` first to measure the
cost on your data before committing.

### 2\. Train

```bash
python src/train\\\_lits3d.py \\\\
    --data\\\_dir /path/to/lits3d \\\\
    --arm hybrid --frozen\\\_blocks up1 \\\\
    --base\\\_channels 32 --patch\\\_size 128 128 128 \\\\
    --epochs 200 --val\\\_every 5 \\\\
    --checkpoint\\\_dir /path/to/checkpoints \\\\
    --results\\\_csv results.csv
```

Arms: `standard`, `frozen\\\_shift`, `fair\\\_shift`, `depthwise`, `hybrid`.

Training resumes from the last periodic checkpoint if interrupted, rerun the
identical command. Every run also writes a log file to `--checkpoint\\\_dir`, so
trajectories survive a disconnected session.

### 3\. Evaluate with post-processing

```bash
python src/train\\\_lits3d.py --data\\\_dir /path/to/lits3d \\\\
    --arm hybrid --frozen\\\_blocks up1 --epochs 0 \\\\
    --checkpoint\\\_dir /path/to/checkpoints \\\\
    --postprocess --min\\\_tumour\\\_size 50
```

### 4\. Cross-validation

```bash
python src/crossval\\\_lits3d.py --data\\\_dir /path/to/lits3d --make\\\_folds
python src/crossval\\\_lits3d.py --data\\\_dir /path/to/lits3d --run \\\\
    --arm hybrid --frozen\\\_blocks up1 --epochs 150 \\\\
    --postprocess --min\\\_tumour\\\_size 50 \\\\
    --checkpoint\\\_dir /path/to/cv\\\_ckpt --results\\\_csv cv.csv
```

\---

## Evaluation notes

Two choices differ from what is sometimes done and both matter.

**Split by volume, not by slice.** Pooling all slices and shuffling places
adjacent slices of the same patient in both training and test sets. In our own
development this inflated tumour Dice from 0.44 to roughly 0.90.

**Per-case Dice without smoothing.** With a smoothing term, a volume containing
no ground-truth tumour and several thousand false-positive voxels scores a small
positive value rather than zero, understating a categorical failure. Each volume
is additionally classified as *detected*, *missed*, *no overlap*, *false
positive* or *empty correct*.

We report the all-volume mean (the LiTS convention, in which a correctly-empty
volume scores 1.0) and, alongside it, the mean over tumour-bearing volumes only.
The two differ here by about 0.03 and the all-volume figure is sensitive to
binary outcomes on the two tumour-free test volumes.

\---

## Known limitations

* Results rest on three initialisations of a single 19-volume held-out split.
Seed standard deviation on tumour Dice is 0.044, dominated by the two
tumour-free volumes, each worth 1/19 of the mean.
* Resampling is 1.5 mm isotropic, coarser than nnU-Net's full-resolution
configuration, chosen for storage reasons.
* The mutually exclusive three-class softmax causes an identifiable failure:
large hypodense lesions are assigned to background, excluding them from the
predicted liver and preventing tumour detection. Two of nineteen test volumes
fail this way. A nested region formulation is implemented
(`--label\\\_mode region`) but not yet evaluated at length.
* The freezing-configuration sensitivity sweep was performed in 2D and has not
been repeated in 3D.

\---

## Citation

Manuscript in preparation. Please open an issue if you would like to be notified
when it appears.

## License

MIT

