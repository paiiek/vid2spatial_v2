# Vid2Spatial Project Structure

Last updated: 2026-02-10

## Directory Layout

```
vid2spatial/
├── vid2spatial_pkg/          # Core Python package (19 modules)
│   │
│   │  ── Pipeline ──
│   ├── pipeline.py               # End-to-end pipeline orchestration
│   ├── config.py                 # Configuration management
│   ├── multi_source.py           # Multi-source FOA mixing
│   │
│   │  ── Detection & Tracking ──
│   ├── hybrid_tracker.py         # Adaptive-K hybrid tracker (DINO + YOLO/ByteTrack)
│   ├── vision.py                 # Camera geometry (pixel_to_ray, ray_to_angles)
│   ├── color_tracker.py          # Color-based tracking backend
│   ├── point_tracker.py          # Single-point tracking backend
│   ├── skeleton_tracker.py       # Human skeleton keypoint tracking
│   ├── ostrack_wrapper.py        # OSTrack integration wrapper
│   ├── sam2_adapter.py           # SAM2 mask refinement adapter
│   │
│   │  ── Trajectory Processing ──
│   ├── trajectory_stabilizer.py  # RTS smoother + depth stabilization
│   ├── temporal_smoother.py      # Temporal filtering utilities
│   │
│   │  ── Depth Estimation ──
│   ├── depth_metric.py           # Depth Anything V2 integration
│   ├── depth_utils.py            # Depth blending + confidence weighting
│   ├── depth_anything_adapter.py # Depth model backend adapter
│   │
│   │  ── Audio Rendering ──
│   ├── foa_render.py             # FOA AmbiX + HRTF binaural + stereo pan
│   ├── irgen.py                  # Room impulse response synthesis
│   │
│   │  ── Output & Utilities ──
│   ├── osc_sender.py             # OSC streaming for DAW
│   └── video_utils.py            # Video I/O, scene cut/zoom detection
│
├── experiments/              # Experiment scripts + results
│   ├── e2e_20_videos/            # 20 diverse real videos, E2E pipeline
│   ├── gt_eval_synthetic/        # 15 synthetic GT scenes, param accuracy
│   ├── sot_15_videos/            # SOT benchmark + HRTF binaural renders
│   └── synthetic_render.py       # 15 synthetic scenario renderer
│
├── evaluation/               # Evaluation code + results
│   ├── tracking_ablation/        # (A) Trajectory reliability ablation
│   ├── ablation_output/          # Renderer/baseline ablation
│   ├── comprehensive_results/    # (B-C) Control + depth stability results
│   ├── trajectory_videos/        # Trajectory visualization + reports
│   ├── listening_test/           # (D) Web-based perceptual evaluation
│   │   ├── index.html                # Listening test interface
│   │   ├── server.py                 # HTTP server + response saving
│   │   ├── prepare_stimuli.py        # Stimuli rendering script
│   │   ├── analyze_responses.py      # Response analysis
│   │   └── stimuli/config.json       # Test configuration
│   ├── tests/                    # Unit tests
│   ├── plots/                    # Evaluation plots (PNG)
│   ├── ISMAR_LISTENING_TEST_PLAN.md  # Listening test methodology
│   └── FOA_DISTANCE_REPORT.md       # Distance rendering validation
│
├── docs/                     # Documentation (dated versions)
│   ├── PROJECT_DOCUMENTATION_20260210.md  # Full project documentation (Korean)
│   ├── ARCHITECTURE_20260210.md           # System architecture
│   ├── PERCEPTUAL_EVALUATION_20260210.md  # Listening test design
│   ├── FINAL_EVALUATION_REPORT_20260210.md # Comprehensive evaluation
│   └── OSC_INTERFACE_SPEC_20260210.md     # OSC protocol spec
│
├── archive/                  # Old versions + unused modules (gitignored)
│   ├── pkg_unused/               # 13 deprecated vid2spatial_pkg modules
│   ├── scripts/                  # Old training/test scripts
│   ├── evaluation_old/           # Previous evaluation code
│   ├── docs_old/                 # Development documentation
│   └── results/                  # Historical experiment results
│
├── data/                     # Datasets (gitignored)
├── weights/                  # Model weights (gitignored)
│
├── .gitignore
├── README.md
├── STRUCTURE.md
├── pytest.ini
└── requirements.txt
```

## Key Files

### Core Implementation
- `vid2spatial_pkg/hybrid_tracker.py` — Adaptive K-frame detection + robustness layer
- `vid2spatial_pkg/trajectory_stabilizer.py` — RTS smoother, depth stabilization
- `vid2spatial_pkg/foa_render.py` — FOA AmbiX encoding + HRTF binaural (KEMAR SOFA) + stereo pan baseline
- `vid2spatial_pkg/pipeline.py` — Full pipeline orchestration
- `vid2spatial_pkg/depth_metric.py` — Depth Anything V2, confidence-weighted blending

### Evaluation (aligned with contributions)
- `evaluation/tracking_ablation/` — (A) Trajectory reliability: SAM2 3.4% → Adaptive-K 100% amplitude
- `evaluation/comprehensive_results/` — (B) Control stability: RTS 93-97% jerk reduction
- `evaluation/comprehensive_results/` — (C) Depth stability: 60% jitter reduction
- `evaluation/listening_test/` — (D) Perceptual evaluation: HRTF binaural vs stereo pan vs mono
