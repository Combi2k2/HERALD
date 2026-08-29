# Object-level Reconstruction — Current Design

Detection-first, mask-free object mapping (`scripts/box_recon.py`, helpers in
`scripts/box_obb.py`). Per frame: YOLO-World boxes → GT-depth unproject →
geometric background removal → a per-detection object cloud, which is merged into
a persistent global object map by 3D voxel overlap. Values below reflect run
**912912** (Hospital/P0001, 1899 frames, 285 final objects).

> Terminology: there is **no segmentation mask** in this design. The per-detection
> "object cloud" plays the role a mask point cloud would — it is carved out
> *geometrically* inside the 2D detection box, not by a mask network.

## Pipeline at a glance

| Stage | Input | Operation | Output |
|---|---|---|---|
| Detect | RGB frame | YOLO-World open-vocab (`conf=0.25`) | boxes `xyxy` + class |
| Assign | boxes | paint into id image, largest-box-first | per-pixel box id |
| Lift | GT depth + box ids | `unproject_labeled` (stride 3) | frustum cloud per box |
| Ground cut | frustum cloud | keep `pt·up > (cam·up − eye) + margin` | above-floor points |
| Isolate | above-floor points | DBSCAN → nearest cluster to camera | 1 object cloud / detection |
| Merge | object cloud | voxel co-occupancy vs global map | assigned global id |
| Fit | accumulated cloud | yaw-only PCA about `up` | oriented box (OBB) |

---

## 1. How each object is detected from a frame

| Step | Method | Notes |
|---|---|---|
| Detector | **YOLO-World** (`yolov8x-worldv2`), open-vocab, prompt-conditioned on a fixed ~56-class vocab | Open-vocab but **label-bound**: only fires on vocab classes; not class-agnostic |
| Confidence | `conf = 0.25` | Precision-favoring (nav wants reliable landmarks, not recall) |
| Box → pixels | `box_label_image`: paint each box rectangle into a uint16 id image, **largest area first** so smaller boxes overwrite | Each pixel → at most one box (smallest box covering it) |
| Pixels → 3D | `unproject_labeled(depth, K, c2w, stride=3)` on `valid = finite & >0 & <max_depth` | GT depth; coarse (every 3rd pixel) frustum cloud per box |
| Background removal | (a) **floor cut**: drop points within `floor_margin` of the local floor `cam·up − eye`; (b) **DBSCAN** then keep the **cluster nearest the camera** with ≥`min_pts` points | Purely geometric; assumes object sits in front of its in-box background |

**Detection unit = one YOLO box per frame.** The object cloud for that box is the
nearest substantial DBSCAN cluster in the box frustum after the floor is removed.

---

## 2. How each object is represented — fine vs coarse

| Aspect | Level | Detail |
|---|---|---|
| Per-frame extraction | **medium/fine** | Raw unprojected points at pixel stride 3 (sub-pixel geometry, GT depth) |
| Stored object cloud | **coarse** | Voxel-downsampled at `merge_voxel = 0.15 m`: **one centroid per occupied voxel** (running mean of all raw points in that cell) → resolution 15 cm |
| Object shape model | **coarse** | A **yaw-only oriented box** (OBB): vertical sides along `up`, footprint rotated in the floor plane. Only 7 numbers (center 3, yaw+2 footprint extents, height) |
| Object identity | id + class label | Persistent global id; class = YOLO label of the merging detection |
| Whole-scene context | separate | `SceneCloud` background at `scene_voxel = 0.1 m`, colored, logged static (viz only, not used for objects) |

**Summary: fine at capture, coarse at storage.** The object is ultimately a
15 cm voxel cloud + a gravity-aligned box — enough for a *rough pose*, not shape.

---

## 3. How per-detection clouds are merged across frames

| Element | Rule |
|---|---|
| Voxel encoding | `floor(pt / merge_voxel)` → int key; object holds `{voxel → [Σx,Σy,Σz, n]}` |
| Association test | New detection's voxels vs global map's occupied voxels; count overlaps |
| Match | assign to the global object with the most overlapping voxels **if** overlap ≥ `merge_overlap × (#voxels in new cloud)`, else **start a new object** |
| Class gate | **off** in 912912 (`--no-class-gate`) — merge regardless of label (label flips were splitting objects) |
| Accumulation | matched voxels update the running centroid+count; new voxels are added and their ownership recorded |
| OBB update | box is **refit on the fly** from the accumulated voxel-centroids at each snapshot |
| Cleanup | drop objects with `< min_vox` voxels or seen in `< min_obs` frames |

Greedy, single-pass, **no cross-object union-find / no tracking** — pure 3D
voxel co-occupancy (HOV-SG-style overlap merge).

---

## 4. Config settings (run 912912)

| Param | Value | Meaning |
|---|---|---|
| `model` | `yolov8x-worldv2.pt` | YOLO-World detector weights |
| `conf` | **0.25** | detection confidence threshold |
| `pixel_stride` | 3 | depth subsampling inside boxes |
| `max_depth` | 60.0 m | reject far/invalid depth |
| `up` | `[0,0,-1]` (estimated) | gravity, from RANSAC floor plane of the scene cloud |
| `eye_height` | 0.712 m (estimated) | camera height above local floor; sets per-frame ground cut |
| `floor_margin` | 0.08 m | slab above local floor removed as ground |
| `dbscan_eps` | 0.15 m | cluster radius for foreground isolation |
| `min_samples` | 4 | DBSCAN core-point threshold |
| `min_pts` | 30 | min points to accept a foreground cluster |
| `merge_voxel` | **0.15 m** | object storage voxel + merge-overlap granularity |
| `merge_overlap` | **0.1** | fraction of a detection's voxels that must hit an object to merge |
| `class_gate` | **off** | do not require class match to merge |
| `min_vox` | 5 | drop objects with fewer voxels |
| `min_obs` | **4** | drop objects seen in fewer frames |
| `scene_voxel` / `scene_stride` | 0.1 m / 8 | whole-scene viz cloud resolution/subsampling |
| `snapshot_every` | 64 | frames between rerun snapshots |

Bolded values were tuned away from defaults to reduce object explosion
(1940 → 285 across iterations).

---

## 5. Problems associated with the design

| Problem | Cause | Symptom | Fix direction |
|---|---|---|---|
| **Crude 2D box assignment** | pixels split by "smallest box wins"; boxes are axis-aligned rectangles | object cloud clipped or contaminated by an overlapping neighbor box | mask/instance cue, or per-box depth histogram gating |
| **"Nearest cluster" breaks under occlusion** | foreground = nearest blob in frustum | a passing person / pillar between camera and target is taken as the object | depth-consistency to box center; multi-cluster scoring |
| **Coarse 15 cm representation** | voxel-downsample at `merge_voxel` | thin/small objects poorly shaped; OBB extent noisy for small things | finer voxel for small classes; keep a higher-res cloud per object |
| **Under-/over-merging trade-off** | single overlap threshold + `class_gate` off | revisits from a new angle miss overlap → **duplicate** objects; loose overlap → distinct things **fuse** | 2-pass union-find bridge; appearance embedding as a second gate |
| **No temporal identity** | stateless per-frame detection + greedy merge | id churn; transient detections survive if `min_obs` too low | lightweight tracker or motion prior |
| **Detection recall capped by vocab** | YOLO-World is label-bound, not class-agnostic | anything outside the ~56-class vocab is invisible | broaden vocab, or add a class-agnostic proposal source |
| **Label unreliable** | open-vocab mislabels (e.g. door→"mirror") | wrong class; would corrupt a class-gated merge | keep `class_gate` off, or majority-vote labels post-hoc |
| **OBB assumes single object per box** | one nearest cluster per detection | cluttered scenes (shelf of items) collapse to one box | sub-cluster within box; instance masks |
| **Duplicate objects across levels/loops** | overlap gate is viewpoint-dependent | same physical object as 2+ ids after a revisit | global de-dup pass on OBB IoU |

---

### Data flow (reference)

```
RGB ─▶ YOLO-World ─▶ boxes+cls
                       │
GT depth ─▶ unproject (box-labeled, stride 3) ─▶ frustum cloud/box
                       │
        floor cut (cam·up − eye) ─▶ DBSCAN ─▶ nearest cluster  = object cloud (this frame)
                       │
        ObjectMap: voxel co-occupancy merge (voxel 0.15, overlap 0.1, no class gate)
                       │
        accumulated voxel-centroid cloud ─▶ yaw-only PCA OBB (about up=[0,0,-1])
```
