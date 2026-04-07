# Human3.6M Dataset Metadata

**Generated**: 2026-03-12 20:17:53

## Dataset Overview
- **Total sequences (train)**: 209
- **Total sequences (test)**: 5168
- **Total frames (train)**: 525,774
- **Total frames (test)**: 646,000

## Skeleton Structure
- **Total joints in full dataset**: 32
- **Active joints (research standard)**: 17
- **Joint indices**: 0, 1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27
- **Skeleton connectivity**: 16 bones (parent-child connections)

### Joint Names (17-joint standard)
```
0: Hip (root), 1: RHip, 2: RKnee, 3: RFoot
6: LHip, 7: LKnee, 8: LFoot
12: Spine, 13: Thorax, 14: Neck/Nose, 15: Head
17: LShoulder, 18: LElbow, 19: LWrist
25: RShoulder, 26: RElbow, 27: RWrist
```

## Coordinate System
- **Format**: 3D Cartesian coordinates (X, Y, Z)
- **Units**: Millimeters (mm)
- **Origin**: Hip joint (index 0)
- **Axes**:
  - X: Left-Right (positive = right)
  - Y: Up-Down (positive = up)
  - Z: Forward-Backward (positive = forward)

## Temporal Information
- **Frame rate**: 50 Hz (fps)
- **Frame duration**: 20 ms
- **Temporal resolution**: 0.02 seconds/frame

## Data Split
- **Training subjects**: S1, S5, S6, S7, S8, S9, S11
- **Test subjects**: Preprocessed format (fixed-length sequences)
- **Unique base actions**: 15

## Sequence Statistics (Training Set)
- **Shortest sequence**: 992 frames (19.84s)
- **Longest sequence**: 6343 frames (126.86s)
- **Mean sequence length**: 2516 frames (50.31s)
- **Std sequence length**: 1102 frames

## Action Categories (15 total)
- Directions
- Discussion
- Eating
- Greeting
- Phoning
- Photo
- Posing
- Purchases
- Sitting
- SittingDown
- Smoking
- Waiting
- WalkDog
- WalkTogether
- Walking

## Data Format
### Training Data (data_3d_h36m.npz)
- Structure: Nested dictionary
- Format: subject → action → sequence
- Shape per sequence: (num_frames, 32, 3)

### Test Data (data_3d_h36m_test.npz)
- Structure: Preprocessed array
- Shape: (5168, 125, 48)
- Fixed-length sequences: 125 frames
- Inferred joints: 16 (features ÷ 3)

## Notes
- HumanMAC uses GSPS preprocessing convention
- Research standard uses 17-joint skeleton (subset of 32)
- Test data uses 16-joint representation
- Source: una-dinosauria/3d-pose-baseline for skeleton definition
