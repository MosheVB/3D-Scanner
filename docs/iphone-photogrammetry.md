# iPhone photogrammetry vs. D405 — comparison protocol

A procedure for getting a **numeric** answer to "could the phone replace or
supplement the D405?", measured on your objects with your existing measurement
harness, before committing to any iOS work.

Budget: an afternoon for the first object, ~30 min each after that.

## What this is testing

Not "is the iPhone a good 3D scanner" in the abstract. Two specific claims:

1. **Phone depth cannot replace the D405 for close-range work.** ARKit's
   `sceneDepth` map is 256×192 and the rear LiDAR's useful range starts around
   25 cm — against the D405's 1280×720 depth from 7 cm. Much of our working
   distance is inside the phone's blind zone, and the ARKit depth map is
   ML-completed from a sparse dToF return rather than measured per-pixel. This
   protocol does not test LiDAR; it's settled by the spec sheet.
2. **Phone RGB photogrammetry might beat the D405 on surface detail**, because
   its resolution ceiling is the 48 MP camera, not a depth sensor. This is the
   claim worth measuring, and it's what the protocol below runs.

The interesting outcome is not "phone wins" or "D405 wins" — it's finding the
object classes where each wins, since a hybrid (D405 geometry, phone texture)
is available if the split is clean.

## What you need

- Printed ChArUco board or scale bar small enough to sit on the platter
- Calipers
- A phone mount (tripod/clamp) that holds position for the whole capture
- COLMAP or GLOMAP on the Linux box, with GPU enabled
- The usual turntable rig

## Controls — hold these fixed across both runs

The comparison is only meaningful if the object and its environment don't move
between runs. Same lighting (diffuse, no specular hotspots, no auto-exposure
drift), same object placement, same background, same turntable speed, same room.
Lock the phone's exposure and focus before the spin — autofocus hunting mid-spin
is the single most common cause of a failed photogrammetry run.

## Test objects

Pick three, deliberately spanning the range where photogrammetry succeeds and
fails:

| Class | Example | Expectation |
|---|---|---|
| Texture-rich, matte | printed/painted part, worn tool | Photogrammetry should win on detail |
| Low-texture, matte | plain 3D print, white plastic | Photogrammetry degrades; D405 unaffected |
| Shiny or dark | polished metal, black rubber | Both struggle; photogrammetry likely fails outright |

The second and third rows are the point. A protocol run only on row one tells
you nothing you didn't already believe.

## Ground truth

Pick 3–5 features per object that calipers can measure unambiguously and that
survive meshing — overall width/height, a boss diameter, a hole spacing, a step
height. Avoid edges that round over in reconstruction.

Record them once, per object, and reuse for both runs. `scanner/measure.py`
(`measure_session`, `measure_fused_cloud`) is what reports the scan-side numbers.

## Run A — D405 baseline

Normal turntable capture, unchanged:

```bash
python -m scanner capture --mode turntable --name mug_d405
python -m scanner measure --session mug_d405     # or measure_session() directly
```

Record the same features, plus wall-clock capture time and total session size.

## Run B — phone

### 1. Capture

Phone on the mount, framing the object with a little headroom. **Put the ChArUco
board flat on the platter so it rotates with the object** — a scale reference
fixed to the room instead of the object is inconsistent across views and will
not give you a usable scale factor.

Lock exposure and focus (press-and-hold on iOS). Record one continuous video of a
full rotation at the slowest turntable speed that still finishes in reasonable
time. Slower is strictly better: less motion blur, more frames per degree bin.

Shoot at the highest resolution your phone offers at 30 fps. 4K is fine — the
frame selector streams and never loads the whole video.

### 2. Select frames

```bash
python scripts/phone_frames.py \
    --video IMG_1234.MOV \
    --out captures/mug_phone/images \
    --spin-start-s 2.5 --spin-end-s 34.0 \
    --pose-step-deg 5
```

`--spin-start-s`/`--spin-end-s` trim the lead-in and lead-out so angle labels
interpolate over the actual spin. 5° gives 72 images over a full turn, which is
a reasonable photogrammetry density; drop to 3° for small or detailed objects.

The script keeps the sharpest frame per degree bin and writes a `manifest.json`
alongside. Check the "degree bins filled" line — gaps mean the spin was too fast
for the frame rate.

### 3. Mask

The classic turntable-photogrammetry failure is a **static background with a
rotating object**: SfM tries to reconcile two contradictory motions and the
reconstruction collapses or wraps the object in background geometry. Masking is
not optional here.

`scanner/mask_rgb_video.py` ("Temporal RGB statistics from rotation video, no
depth") is the right starting point — it separates the rotating object from the
static background using exactly the signal a turntable capture provides, and it
needs no depth, so it transfers to phone frames as-is in principle.

COLMAP's convention: for image `foo.png`, the mask is `foo.png.png` in the mask
directory, where **black (0) pixels are ignored**.

### 4. Reconstruct

Quickest path:

```bash
colmap automatic_reconstructor \
    --workspace_path captures/mug_phone/ws \
    --image_path captures/mug_phone/images \
    --mask_path captures/mug_phone/masks \
    --quality high
```

Explicit pipeline if you want control over the dense stage — feature_extractor
(with `--SiftExtraction.use_gpu 1` and `--ImageReader.mask_path`),
exhaustive_matcher, mapper, image_undistorter, patch_match_stereo, stereo_fusion,
poisson_mesher.

Note this is the workload that justifies the GPU box doing double duty as a
worker: it's long, parallel, and entirely offline.

### 5. Scale

COLMAP output is **defined only up to scale** — this is the step people forget,
and it silently invalidates every measurement if skipped.

Detect the ChArUco board in the reconstructed images, compare its known square
size against the reconstructed distance, and apply the resulting factor
uniformly. `scripts/charuco_calibrate.py` already has the board detection
machinery to borrow.

Sanity check: scale factor should be consistent across several board squares. If
it drifts, the reconstruction has a distortion problem and the measurements
aren't trustworthy yet.

## Compare

Two levels:

**Feature measurements** — the caliper features, phone vs. D405 vs. ground truth.
This is the number that decides it.

**Mesh-to-mesh** — ICP-align the two reconstructions and look at the distance
distribution (Open3D, or CloudCompare's cloud-to-mesh). This shows you *where*
they differ, which is usually more informative than the scalar error: expect the
phone to win on flat textured faces and lose on edges, concavities, and anything
it couldn't see.

## Results template

| Object | Feature | Truth (mm) | D405 (mm) | Phone (mm) | D405 err | Phone err |
|---|---|---|---|---|---|---|
| | | | | | | |

Also record per run: capture time, processing time, whether reconstruction
succeeded at all, and a subjective note on surface detail.

## Deciding

- **Phone error within ~2× the D405's on row-one objects** → the texture path is
  real; hybrid (D405 geometry + phone texture) is worth building.
- **Phone fails or drifts badly on rows two and three** → expected, not a
  disqualifier; it defines the split between the two capture modes.
- **Phone competitive across all three rows** → surprising. Re-check the scale
  factor before believing it.

## Known failure modes

| Symptom | Cause |
|---|---|
| Reconstruction collapses / object wrapped in background | Masking missing or leaky |
| Everything measures uniformly wrong by a constant ratio | Scale step skipped or board mis-detected |
| Blurry/failed features on one side | Autofocus hunted mid-spin; lock focus |
| Few bins filled | Spin too fast for the frame rate |
| Low-texture object reconstructs as a blob | Expected — this is the row-two result |

## What this deliberately does not test

Handheld phone capture. Our handheld mode relies on RGB-D odometry at full frame
rate; a phone changes both the sensor and the registration strategy, and mixing
that into this comparison would confound it. Turntable mode isolates the
question to sensor and reconstruction quality, with pose supplied by the rig.
