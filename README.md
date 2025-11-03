# Arctic Coastal Ice Detection and Open-Water Dynamics (2019–2024)

**Author:** Ana Stringer  
**Course:** ENVIRON 514.01 – Geospatial Data Science (Fall 2025)  
**Instructor:** Dr. Johnny Ryan  

---

## Summary

This project develops a hybrid deep learning and rule-based system for classifying and tracking Arctic coastal ice using Sentinel-2 imagery (10 m RGB).  
It combines a U-Net convolutional neural network (CNN) trained with RGB plus distance-to-coast (meters) inputs, followed by post-processing and brightness-based classification to refine the detection of landfast ice, drift ice, and open water.  

The workflow enables temporal tracking of landfast-ice edge positioning and open-water fraction within coastal buffers (3 km, 5 km, 10 km) across key Arctic sites — **Wainwright**, **Utqiaġvik (Barrow)**, **Arviat**, and **Uelen** — between 2019 and 2024 during the spring months of March - June. 

---

## Motivation

Machine learning models often struggle to distinguish landfast ice from drift ice, since their spectral reflectance is nearly identical in visible and near-infrared bands.  
This similarity poses challenges for Arctic coastal monitoring, where small-scale changes in landfast ice and open water conditions can have strong implications for subsistence travel, ecosystem dynamics, and bowhead whale / marine hunting access.  

To overcome this limitation, this project integrates:
- A U-Net model to capture spatial texture and context.
- A distance-to-coast feature to supply geophysical grounding.
- A binary-tree thresholding step to refine open-water detection.  

Together, these components form a multi-site system for quantifying ice edge positioning and nearshore open-water variability during recent years. 

---

## Research Objectives

1. Classify landfast, drift, and open water across multiple Arctic coasts using Sentinel-2 data.  
2. Quantify open-water extent within 3 km, 5 km, and 10 km coastal and ice-edge buffers.  
3. Build time series (2019–2024) of landfast-ice edge position and open-water area for Wainwright, Arviat, Utqiaġvik, and Uelen.

---

## Datasets

- Sentinel-2 L2A RGB (10 m) – February–May, 2019–2024  
- Training polygons (GeoJSON / GPKG) – 4 classes (open water, landfast, drift, transition) for 75 image pairs
- Coastline shapefiles – U.S. and Canada landmasses  
- Derived:  
  - Land masks (binary rasters)  
  - Euclidean distance-to-coast rasters (meters)  
  - Stitched prediction mosaics and combined outputs

---

## Python Requirements

```bash
numpy
pandas
rasterio
geopandas
tensorflow
keras
opencv-python
matplotlib
scikit-learn
scipy
```

---

## Workflow Overview

| Step | Script                     | Description                                                                 |
|------|----------------------------|-----------------------------------------------------------------------------|
| 1    | `rasterize.py`             | Converts labeled polygons to raster masks (0–3 classes), land masked as NODATA. |
| 2    | `distance_to_coast_meters.py` | Computes Euclidean distance (in meters) from coastline using projected raster coordinates. |
| 3    | `tile_all.py`              | Tiles RGB, distance, and mask rasters (320×320, 25% overlap).              |
| 4    | `train.py`                 | Trains U-Net with RGB + distance inputs and weighted categorical cross-entropy. |
| 5    | `predict.py`               | Generates class predictions for all tiles.                                 |
| 6    | `stitch.py`                | Reconstructs full-scene mosaics from predicted tiles.                      |
| 7    | `fix_driftmisclass.py`     | Applies coastal connectivity filtering to ensure landfast ice remains anchored (300 m buffer). |
| 8    | `openwaterthresholding.py` | Brightness-based classification of open water, thin, and thick ice.        |
| 9    | `binarytree.py`            | Merges landfast (U-Net) and open-water (brightness) maps into a composite classification. |

---

## Model Configuration

- Architecture: U-Net  
- Input Channels: RGB + distance-to-coast (4 bands)  
- Loss: Weighted categorical cross-entropy  
- Weights: `[2.5, 5.0, 1.0, 0.5]`  
- Epochs: 25  
- Batch Size: 8  
- Validation Split: 15%  

---

## Post-Processing Logic

### 1. Coastal Connectivity Filter

The `fix_driftmisclass.py` step ensures physical realism by:
- Seeding landfast ice pixels within 300 m of the coastline.
- Propagating connectivity across adjacent pixels.
- Removing disconnected or small (<2000 px) components.

### 2. Binary-Tree Brightness Classifier

The `openwaterthresholding.py` script thresholds the red band brightness to classify open water, thin ice, and thick ice using scene-specific percentile ranges (10th, 40th, 80th).

### 3. Fusion of Outputs

`binarytree.py` merges:
- Landfast ice from U-Net (post-fixed)
- Open water from brightness thresholding

The final composite retains U-Net spatial accuracy and radiometric distinction from the binary rules.

---

## Distance-to-Coast Computation

The distance raster is generated by computing Euclidean distance from land pixels (derived from national coastline shapefiles) within the image’s projected CRS.  
Pixel size is extracted from the raster transform, ensuring distances are expressed in meters, not degrees.

This raster serves dual purposes:
- As an input feature in model training (4th channel).
- As a constraint in post-processing (for landfast connectivity filtering).

---

## Open-Water Buffer Analysis

Goal: Quantify open-water fraction near the coast and ice edge.

### Procedure:
1. Generate 3 km, 5 km, and 10 km seaward buffers from:
   - Coastline  
   - Landfast ice edge (from class 1 polygons)  
2. Mask and count open-water pixels (class 0) within each buffer.  
3. Summarize by site, year, and season (Feb–May).  

### Metrics:
- `open_water_pct_3km`  
- `open_water_pct_5km`  
- `open_water_pct_10km`  
- `edge_openwater_3km`  
- `edge_openwater_5km`  
- `edge_openwater_10km`  

---

## Time-Series Analysis (2019–2024)

### Study Sites:
- **Utqiaġvik (Barrow, AK)**  
- **Wainwright, AK**  
- **Arviat, Nunavut**  
- **Uelen, Chukotka**

### Analyses:
- Extract and vectorize landfast-ice edge per scene.  
- Compute mean edge distance from coastline.  
- Aggregate yearly and seasonal trends in edge position and open-water fraction.  
- Compare interannual variability across all four sites.

---

## Expected Outcomes

- Validated hybrid system (U-Net + Binary Tree) for landfast ice detection.  
- High-resolution time series of ice-edge movement and open-water fraction (2019–2024).  
- Quantitative metrics on coastal ice stability and polynya persistence.  
