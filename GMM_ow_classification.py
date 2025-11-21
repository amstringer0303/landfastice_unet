import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
import geopandas as gpd
from rasterio import features
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from sklearn.mixture import GaussianMixture
from scipy.ndimage import generic_filter

# =============================================================================
# PATHS (Uelen)
# =============================================================================
region_dir = r"D:\Ue"
landmask_dir = r"D:\UNET\landmask"
output_dir = r"D:\UNET\openwater_gmm"
os.makedirs(output_dir, exist_ok=True)
os.makedirs(landmask_dir, exist_ok=True)

coast_path = r"D:\GSHHS_shp\f\GSHHS_f_L1.shp"

# =============================================================================
# TEXTURE FUNCTION
# =============================================================================
def local_variance(arr, size=5):
    """Windowed local variance."""
    return generic_filter(arr, np.var, size=size)

# =============================================================================
# RASTERIZE COASTLINE
# =============================================================================
def rasterize_coastline(coast_path, reference_tif, save_path):
    with rasterio.open(reference_tif) as ref:
        transform = ref.transform
        crs = ref.crs
        shape = (ref.height, ref.width)
        meta = ref.meta.copy()

    coast = gpd.read_file(coast_path).to_crs(crs)

    landmask = features.rasterize(
        [(geom, 1) for geom in coast.geometry],
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype="uint8"
    )

    meta.update(dtype="uint8", count=1)
    with rasterio.open(save_path, "w", **meta) as dst:
        dst.write(landmask, 1)

    return landmask


# =============================================================================
# CLASSIFY SCENE — ADAPTIVE GMM (binary: water vs not-water)
# =============================================================================
def classify_scene(scene_name, rgb_path, landmask_path):

    print(f"\n=== Processing {scene_name} ===")

    out_tif = os.path.join(output_dir, f"{scene_name}_openwater.tif")
    out_png = os.path.join(output_dir, f"{scene_name}_openwater.png")

    # Remove old files (always reprocess)
    for f in [out_tif, out_png]:
        if os.path.exists(f):
            os.remove(f)

    # -------------------------------------------------------
    # Load RGB reflectance
    # -------------------------------------------------------
    with rasterio.open(rgb_path) as src:
        rgb = src.read([1, 2, 3]).astype(np.float32)
        transform = src.transform
        crs = src.crs
        meta = src.meta.copy()

    if rgb.max() > 1.5:
        rgb /= 10000.0

    rgb = np.transpose(rgb, (1, 2, 0))
    red, green, blue = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]

    # -------------------------------------------------------
    # Landmask
    # -------------------------------------------------------
    if os.path.exists(landmask_path):
        with rasterio.open(landmask_path) as lm:
            lm_arr = lm.read(1).astype(np.uint8)

        if lm_arr.shape != red.shape:
            aligned = np.zeros(red.shape, dtype=np.uint8)
            reproject(
                source=lm_arr,
                destination=aligned,
                src_transform=lm.transform,
                src_crs=lm.crs,
                dst_transform=transform,
                dst_crs=crs,
                resampling=Resampling.nearest,
            )
            landmask = aligned
        else:
            landmask = lm_arr
    else:
        landmask = rasterize_coastline(coast_path, rgb_path, landmask_path)

    ocean_mask = (landmask == 0)

    # -------------------------------------------------------
    # Features for GMM
    # -------------------------------------------------------
    texture = local_variance(red, size=5)

    # Stack features for ocean pixels only
    X = np.column_stack([
        red[ocean_mask],
        green[ocean_mask],
        blue[ocean_mask],
        texture[ocean_mask]
    ])

    # -------------------------------------------------------
    # Gaussian Mixture Model (2 classes)
    # -------------------------------------------------------
    gmm = GaussianMixture(n_components=2, covariance_type='full', random_state=0)
    gmm.fit(X)
    labels = gmm.predict(X)

    # -------------------------------------------------------
    # Identify which cluster is open water
    # -------------------------------------------------------
    # Open water cluster = darker + smoother
    cluster_means = gmm.means_
    water_cluster = np.argmin(cluster_means[:, 0] + 0.5 * cluster_means[:, 3])

    # -------------------------------------------------------
    # Build full-scene output
    # -------------------------------------------------------
    classification = np.zeros(red.shape, dtype=np.uint8)
    classification[ocean_mask] = (labels == water_cluster).astype(np.uint8)

    # Land stays 0

    # -------------------------------------------------------
    # Save GeoTIFF
    # -------------------------------------------------------
    meta.update(dtype="uint8", count=1)
    with rasterio.open(out_tif, "w", **meta) as dst:
        dst.write(classification, 1)

    # -------------------------------------------------------
    # PNG diagnostic
    # -------------------------------------------------------
    rgb_vis = np.clip(rgb, 0, 0.15)
    rgb_vis = (rgb_vis - rgb_vis.min()) / (rgb_vis.max() - rgb_vis.min() + 1e-6)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(rgb_vis)
    axes[0].set_title(f"{scene_name} – RGB")
    axes[0].axis("off")

    axes[1].imshow(classification, cmap="Blues", vmin=0, vmax=1)
    axes[1].set_title("Open Water (1 = water)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"{scene_name}: saved open-water output.")


# =============================================================================
# MAIN LOOP
# =============================================================================
if __name__ == "__main__":
    print("Running open-water GMM classification for Uelen...\n")

    for fname in os.listdir(region_dir):
        if fname.lower().endswith(".tif") and "training" not in fname.lower():
            scene_name = os.path.splitext(fname)[0]
            rgb_path = os.path.join(region_dir, fname)
            landmask_path = os.path.join(landmask_dir, f"{scene_name}_landmask.tif")
            classify_scene(scene_name, rgb_path, landmask_path)

    print("\nFinished GMM open-water classification.")
