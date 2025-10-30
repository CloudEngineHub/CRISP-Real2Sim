from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal

from rich.progress import (
    BarColumn,
    Progress,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from neuralplane.frontend.planes.plane_excavator import PlaneExcavatorConfig, PlaneExcavator
from neuralplane.utils.yaml import load_yaml, update_yaml

# If you do not need Nerfstudio / ScanNet specific tools you can safely comment these two lines.
from nerfstudio.utils.rich_utils import CONSOLE
from nerfstudio.scripts.train import _set_random_seed

import h5py
import numpy as np
import torch
import tyro
import matplotlib.pyplot as plt
from PIL import Image


@dataclass
class Args:
    """Command‑line arguments."""

    # ↓↓↓ Existing YAML config is still supported, because it stores useful defaults
    # such as the random seed.  Feel free to point it to a minimal file if you do
    # not use Nerfstudio / ScanNet at all.
    config: Path = Path("config.yaml")
    """Path to a YAML configuration file (only .MACHINE.SEED is used here)."""

    image: Path = Path("image.png")
    """Path to the **single RGB image** to be processed."""

    vis: bool = False
    """Whether to save intermediate visualisations (needs matplotlib)."""

    skipping: bool = False
    """Skip excavation if a cache file for *this* image already exists."""


class LocalPlanarPrimitivesSingleImage:
    """Excavate local planar primitives from ONE image.

    This is a minimal rewrite of the original multi‑image pipeline so that it
    accepts a single --image argument instead of looking for a ScanNet scene.
    """

    def __init__(self, args: Args):
        # ── General set‑up ──────────────────────────────────────────────────────
        self.yaml_path = args.config
        self.conf = load_yaml(args.config)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.seed = getattr(self.conf.MACHINE, "SEED", 0)
        _set_random_seed(self.seed)

        # ── I/O paths ──────────────────────────────────────────────────────────
        self.image_path: Path = args.image.expanduser().resolve()
        assert self.image_path.exists(), f"Image not found: {self.image_path}"

        # Put results next to the image:

        self.output_dir: Path = (
            self.image_path.parent / "local_planar_primitives" / self.image_path.stem
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if args.vis:
            (self.output_dir / "vis").mkdir(parents=True, exist_ok=True)

        # ── Misc. ──────────────────────────────────────────────────────────────
        self.vis = args.vis
        self.skipping = args.skipping

        

    # -------------------------------------------------------------------------
    #  Main work
    # -------------------------------------------------------------------------
    def excavate_planes(self):
        """Run PlaneExcavator on *one* image and return its outputs."""
        # 1. Read image (RGB uint8[H,W,3]) ------------------------------------------------
        pil_img = Image.open(self.image_path).convert("RGB")
        # Resize so that the longer edge becomes exactly 400 px
        orig_w, orig_h = pil_img.size  # PIL order (W, H)
        max_side = max(orig_w, orig_h)
        if max_side != 800:
            scale = 800.0 / max_side
            new_w = int(round(orig_w * scale))
            new_h = int(round(orig_h * scale))
            pil_img = pil_img.resize((new_w, new_h), Image.BILINEAR)
        rgb_np = np.asarray(pil_img, dtype=np.uint8)  # (H, W, 3)
        img_height, img_width = rgb_np.shape[:2]


        # 2.   Setup excavator ---------------------------------------------------
        excavator: PlaneExcavator = PlaneExcavatorConfig().setup(
            device=self.device,
            img_height=img_height,
            img_width=img_width,
        )

        # 3.   Inference ---------------------------------------------------------
        results = excavator(rgb_np, c2w=None, vis=self.vis)

        # 4.   Optionally save a nice visualisation -----------------------------
        if self.vis:
            vis_batch = results["vis"]
            fig = plt.figure(figsize=(8, 6))
            gs = fig.add_gridspec(2, 2)
            ax1 = fig.add_subplot(gs[0, 0])
            ax1.imshow(vis_batch["image"])
            ax1.set_title("(1) Image")
            ax2 = fig.add_subplot(gs[0, 1])
            ax2.imshow(vis_batch["pred_norm"])
            ax2.set_title("(2) Estimated Normal Map")
            ax3 = fig.add_subplot(gs[1, 0])
            ax3.imshow(vis_batch["normal_mask"])
            ax3.set_title("(3) Normal Clusters")
            ax4 = fig.add_subplot(gs[1, 1])
            ax4.imshow(vis_batch["plane_mask"])
            ax4.set_title("(4) Plane Segmentation")
            plt.tight_layout()
            fig.savefig(self.output_dir / "vis" / f"{self.image_path.stem}.png")
            plt.close()

        return results["seg_mask"], results["normal"]

    # -------------------------------------------------------------------------
    def main(self):
        """Controller that handles caching and I/O."""
        cache_h5 = self.output_dir / "cache.h5"

        try:
            if not self.skipping:
                raise FileNotFoundError  # Force excavation unless --skipping.
            with h5py.File(cache_h5, "r") as f:
                CONSOLE.log("Cache found – skipping excavation …")
                pan_seg = f["pan_seg"][()]
                normals = f["normals"][()]
        except FileNotFoundError:
            pan_seg, normals = self.excavate_planes()
            CONSOLE.log(f"Saving cache to {cache_h5} …")
            with h5py.File(cache_h5, "w") as f:
                f.create_dataset("pan_seg", data=pan_seg)
                f.create_dataset("normals", data=normals)

        # Update YAML so downstream tasks know where the cache is (optional).
        self.conf.DATA.CACHE = cache_h5
        update_yaml(self.yaml_path, self.conf)
        CONSOLE.rule(f"[bold yellow]{self.image_path.stem}[/]: Plane excavation DONE")


# -----------------------------------------------------------------------------
#  Entrypoint
# -----------------------------------------------------------------------------

def entrypoint():
    """Entrypoint for `python local_planar_primitives_single_image.py …`"""

    tyro.extras.set_accent_color("bright_yellow")
    args = tyro.cli(Args)
    LocalPlanarPrimitivesSingleImage(args).main()


if __name__ == "__main__":
    entrypoint()
