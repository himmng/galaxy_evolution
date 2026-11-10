# galaxy_evolution

Analysis code for comparing a sample of extreme emission line galaxies (EELGs) with a matched control sample of galaxies in GOODS-South. The work stacks and measures rest-frame UV flux using HST images (FRESCO HLSP mosaics) and cross-matches the samples against JADES JWST catalogs.

The pipeline is a set of Jupyter notebooks plus one helper module. There is no package, test suite or command line entry point.

## Repository layout

```
galaxy_evolution/
├── notebooks/   Jupyter notebooks and segm_create.py (the actual analysis code)
├── codes/       only .ipynb_checkpoints (autosave copies of notebooks, no source of its own)
├── data/        intermediate products: FITS tables, .npy/.npz arrays, flux files (about 20 MB)
├── plots/       figures produced by the notebooks (.jpg/.png/.pdf)
└── output.png
```

## What the pipeline does

1. **Load the samples.** EELG and control catalogs are read from plain text tables (`eelgs_28aug2024.dat`, `control_28aug2024.dat`). Only galaxies with `zspec > 3.08` are kept, so that F336W samples the rest-frame UV below the Lyman limit.
2. **Load HST images.** F814W (ACS/WFC) and F336W (WFC3/UVIS) science and weight mosaics from the FRESCO HLSP release for GOODS-South.
3. **Make cutouts.** Cutouts are centred on the local flux peak within a small tolerance (0.1 arcsec) of each catalog RA/Dec, for both filters and both weight maps.
4. **Segment and mask.** Per cutout: subtract a 2D background (`Background2D` with `SExtractorBackground`), smooth with a Gaussian kernel, detect and deblend sources, then keep the segment closest to the cutout centre. Other segments are masked and refilled with Gaussian noise drawn from the background. Galaxies above the redshift limit can also be masked by position (`mask_highz_galaxies`).
5. **Cross-match catalogs.** Sample RA/Dec are matched to the JADES NIRCam photometry catalog (tolerances 0.0003 and 0.0005 deg are stored in `data/`), and photometric redshifts (EAZY, with confidence intervals) are compared against the spectroscopic ones using the JADES NIRSpec line flux catalogs.
6. **Stack and compare.** Weighted mean and inverse variance weighted stacks of source-only and noise-only images, F336W to F814W flux comparisons, and flux ratio histograms for EELGs versus control.

## Notebooks

| Notebook | Purpose |
|---|---|
| `data_run.ipynb` | Main exploratory run: sample loading, FITS loading, cutouts, segmentation, flux analysis, galaxy matching catalog |
| `find_galaxy_matching.ipynb` | Matches sample RA/Dec to the JADES NIRCam catalog and compares photometric and spectroscopic redshifts |
| `Data Load-EELGs.ipynb` | Cutouts, segmentation, masking and source/noise image generation for the EELG sample |
| `Data Load-control.ipynb` | Same as above for the control sample |
| `segm_testing.ipynb` | Tests of the segmentation parameters using `segm_create.py` |
| `final_seg_images_eelgs.ipynb` | Final segmentation images, stacks, flux comparison and flux ratio histograms for EELGs; writes `EELGs_data_final.fits` products |
| `final_seg_images_control.ipynb` | Same as above for the control sample |
| `EELGS_vs_Control_final_fits_load.ipynb` | Loads the final EELG and control FITS products and compares the stacks side by side |
| `JH_data_processing_.ipynb` | Processing of a separate set of cutouts (`anshu_gal*.fits`) with PSF and centroid tools from `photutils` |
| `background_remove.ipynb` | Small scratch notebook for background removal |

## `notebooks/segm_create.py`

Helper functions shared by the notebooks:

| Function | Role |
|---|---|
| `cutout_centered_on_local_peak_with_tolerance` | Cutouts for two filters and their weight maps, recentred on the peak flux near a position |
| `get_cutout_data_array_v2` | Runs the cutout step over a whole sample and returns the indices that failed |
| `get_segments_background` | Background subtraction, source detection, deblending, closest segment selection and noise filling |
| `apply_closest_mask` | Keeps the segment nearest the cutout centre and merges the rest into one mask |
| `mask_highz_galaxies` | Masks segments that contain galaxies above a redshift limit (default 3.08) |
| `fill_masked_with_noise`, `fill_masked_with_noise2`, `fill_masked_pixels` | Replace masked pixels with Gaussian noise matched to the unmasked image |
| `process_images_with_masks` | Applies the noise filling over a list of cutouts and returns noise-only and with-source images |
| `process_array`, `get_cleaned_data` | Small utilities for padding and indexing result arrays |

Note that `get_cutout_data_array_v2`, and `cutout_centered_on_local_peak` / `cutout_with_background_rms`, partly overlap with older variants in the same file. `get_cutout_data_array_v2` is defined twice with the same behaviour.

## Data

External inputs (not in the repo, must be downloaded separately):

- FRESCO HLSP HST mosaics for GOODS-South: `hlsp_fresco_hst_acs-wfc_goods-s_f814w_v1.0_{sci,wht}.fits` and `hlsp_fresco_hst_wfc3-uvis_goods-s_f336w_v1.0_{sci,wht}.fits`
- JADES NIRCam photometry catalog (`hlsp_jades_jwst_nircam_goods-s-deep_photometry_v2.0_catalog.fits`)
- JADES NIRSpec line flux catalogs (prism and gratings, v1.0 and v1.1)
- Sample tables `eelgs_28aug2024.dat` and `control_28aug2024.dat`

Products kept in `data/`:

- `EELGS_final.fits`, `EELGS_final_with_numpy_array.fits`: final EELG table
- `cat_matching_index_*.npy/.npz`: catalog match indices for each tolerance, for EELG and control
- `closest_mask_arr_*.npy`, `err_arr_*.npy`: segmentation masks and failed cutout indices
- `*_flux_F336W`, `*_flux_F814W`, `*_flux_F814W_wtd`, `f336_flux_*`, `f814_flux_*` (including `*_noise_*`): measured fluxes and noise
- `anshu_gal*.fits`: cutouts used by `JH_data_processing_.ipynb`

## Setup

The notebooks use Python 3 with:

```bash
pip install numpy scipy matplotlib astropy photutils jupyter
```

Several notebooks also import `GPy`, `emcee`, `nfft`, `lal`, `scikit-learn` and `tqdm`. These sit in a shared import block and are not all used by the analysis itself, so you can usually drop them if you do not need those cells.

## Before you run it

- **Paths are hardcoded.** The notebooks point at a local disk (`/media/himanshu/MyHDD/data_galaxy_evolution/...`) and at `/catalog/` or `/data/catalog/`. Edit `data_path` and the catalog paths near the top of each notebook.
- **Switches are manual.** Notebooks pick the EELG or control sample with a `use_control = True/False` flag, and the filters with `filt0='f814w'`, `filt1='f336w'`.
- **Run order.** Roughly: `find_galaxy_matching` and `data_run`, then `Data Load-EELGs` / `Data Load-control`, then `final_seg_images_*`, then `EELGS_vs_Control_final_fits_load`.
- Segmentation uses random noise filling, so results vary slightly between runs unless you seed NumPy.
- Folders named `.ipynb_checkpoints` and `__pycache__` are tracked in git and can be removed.
