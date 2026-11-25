"""CLI.

Two dates (change detection):
    python run_detection.py --before 2019.tif --after 2025.tif --model unet --weights unet.pt --out results
Single date (forest mask only):
    python run_detection.py --before scene.tif --model unet --weights unet.pt --out results
Pipeline test without trained weights:
    python run_detection.py --before 2019.tif --after 2025.tif --model ndvi --out results
"""
import argparse
import json

from deforestation import (NDVIPredictor, RandomForestPredictor, UNetPredictor,
                           run_change_pipeline, run_single_pipeline)


def main():
    ap = argparse.ArgumentParser(description="Deforestation detection & visualization")
    ap.add_argument("--before", required=True, help="6-band GeoTIFF (B2,B3,B4,B8,B11,B12)")
    ap.add_argument("--after", help="second-date GeoTIFF (omit for single-date mode)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--model", choices=["unet", "rf", "ndvi"], default="unet")
    ap.add_argument("--weights", help="U-Net .pt checkpoint or RandomForest .joblib")
    ap.add_argument("--tile", type=int, default=256)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=0.1,
                    help="hysteresis around the threshold; larger = fewer false positives")
    ap.add_argument("--min-area-ha", type=float, default=0.5)
    ap.add_argument("--no-cloud-mask", action="store_true")
    a = ap.parse_args()

    if a.model in ("unet", "rf") and not a.weights:
        ap.error(f"--weights is required for --model {a.model}")

    if a.model == "unet":
        predictor = UNetPredictor(a.weights, a.tile, a.overlap, a.batch_size, a.device)
    elif a.model == "rf":
        predictor = RandomForestPredictor(a.weights)
    else:
        predictor = NDVIPredictor()

    if a.after:
        stats = run_change_pipeline(a.before, a.after, a.out, predictor, a.threshold,
                                    a.margin, a.min_area_ha, not a.no_cloud_mask)
    else:
        stats = run_single_pipeline(a.before, a.out, predictor, a.threshold)
    print(json.dumps(stats, indent=2))
    print(f"\nOutputs written to: {a.out}/")


if __name__ == "__main__":
    main()