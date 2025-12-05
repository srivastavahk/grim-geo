import numpy as np

class GRIMConfig:
    """
    Configuration for GRIM: Grasp Re-alignment via Iterative Matching.
    References:
        - Eq 4: Hybrid Alignment Cost weights
        - Eq 6: Grasp Scoring weights
    """
    # Hardware
    DEVICE = "cpu"  #

    # Models
    # Using specific versions to ensure consistency
    DINO_MODEL = "facebook/dinov2-small" # ViT-S/14 [cite: 425]
    CLIP_MODEL = "openai/clip-vit-base-patch32"

    # Alignment Hyperparameters (Eq 4, 8)
    # Weights for alignment cost: Geometric vs Feature
    W_G = 1.0       # Geometric weight [cite: 169]
    W_F = 1.0       # Feature weight. Paper suggests tuning, we start at 1.0.

    # PCA
    PCA_COMPONENTS = 3 # Reduce DINO features to 3D for alignment efficiency [cite: 150]

    # Coarse Alignment
    GRID_SEARCH_ANGLE_STEP = 30  # Degrees for Euler angle grid search [cite: 153]
    K_EVAL_NEIGHBORS = 5         # Number of neighbors for cost calculation [cite: 158]
    K_ORIENT_CANDIDATES = 5      # Top K initial transforms to refine with ICP [cite: 171]

    # Grasp Scoring Hyperparameters (Eq 5, 6)
    SIGMA_POS = 0.1  # Gaussian decay scaling factor [cite: 187]
    W_TASK = 0.95    # Weight for task compatibility [cite: 194]
    W_GEO = 0.05     # Weight for geometric stability [cite: 194]

    # Robot Specifics (Hello Robot Stretch 3)
    # Assumes gripper frame Z-axis is the approach direction [cite: 182]
    GRIPPER_APPROACH_VECTOR = np.array([0, 0, 1])
