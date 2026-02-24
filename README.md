# GeoSample

Reference implementation of **GeoSample** (geometry-guided symmetric sampling operator) and **Consensus Field** for 3D volumetric segmentation.

GeoSample is a drop-in local operator designed for encoder–decoder segmenters. It learns a voxel-wise geometric field (local 3D orientation + bounded step sizes) to **steer where features are sampled**, converts symmetric paired samples into compact **parity-separated differential cues** (gradient-/curvature-like tokens), and applies lightweight token gating + mixing.  
Consensus Field performs **rotation-consistent fusion** of geometry fields across skip connections to reduce cross-scale geometric mismatch.


## Key ideas (paper-aligned)

- **Unified stride 1 refinement and stride>1 downsampling** under one operator.
- **Symmetric sampling** makes odd/even components explicit and stabilizes directional cues.
- **SO(3) frame field + adaptive step sizes** provides structured, regularized sampling geometry.
- **Consensus Field** aligns geometry before skip fusion via quaternion-consistent interpolation.

## Repository structure

- `geosample/`: core modules
  - `operator.py`: `GeoSample3D` (main operator)
  - `field.py`: `GeoFieldHead3D` (geometry field prediction)
  - `consensus.py`: `ConsensusField3D` (skip alignment)
  - `mixer.py`: token gating + 1×1×1 mixing
- `models/`: a compact 3D U-Net example using GeoSample blocks

