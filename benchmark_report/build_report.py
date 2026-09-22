#!/usr/bin/env python3
"""Build the TravelUAV benchmark HTML report with embedded real data samples."""
import base64
import json
import pathlib

REPORT = pathlib.Path("/home/spc/memory_arena/TravelUAV/benchmark_report")
S = json.load(open(REPORT / "sample_computed.json"))
obj_desc = json.load(open(REPORT / "traj_objdesc.json"))[0]

CAM_ORDER = ["frontcamera", "leftcamera", "rightcamera", "rearcamera", "downcamera"]
CAM_LABEL = {"frontcamera": "Front", "leftcamera": "Left", "rightcamera": "Right",
             "rearcamera": "Rear", "downcamera": "Down"}

def b64(path):
    return base64.b64encode(open(REPORT / path, "rb").read()).decode()

imgs_html = "".join(
    f'<figure><img src="data:image/png;base64,{b64(f"cam_{c}.png")}" alt="{c}"/><figcaption>{CAM_LABEL[c]} camera</figcaption></figure>'
    for c in CAM_ORDER)

# ---------- trajectory plot (inline SVG) ----------
traj = S["trajectory"]                       # filtered 13 points, local frame
xs = [p[0] for p in traj]; ys = [p[1] for p in traj]
minx, maxx = min(xs), max(xs); miny, maxy = min(ys), max(ys)
px, py = 20, 20
W, H = 420, 340
def sx(x): return px + (x - minx) / (maxx - minx + 1e-9) * (W - 2 * px)
def sy(y): return H - py - (y - miny) / (maxy - miny + 1e-9) * (H - 2 * py)
path_d = "M " + " L ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in zip(xs, ys))
pts = "".join(
    f'<circle cx="{sx(x):.1f}" cy="{sy(y):.1f}" r="3.2" fill="{"#ff5252" if i == S["sample_frame"] - 1 else "#4f83cc"}"/>'
    for i, (x, y) in enumerate(zip(xs, ys)))
# GT step arrow at sample frame
sf = S["sample_frame"] - 1
dir3 = S["gt_waypoint_dir"]; d = S["gt_waypoint_dist"]
ax = sx(xs[sf] + dir3[0] * d); ay = sy(ys[sf] + dir3[1] * d)
arrow = (f'<line x1="{sx(xs[sf]):.1f}" y1="{sy(ys[sf]):.1f}" x2="{ax:.1f}" y2="{ay:.1f}" '
         f'stroke="#ffb300" stroke-width="2" marker-end="url(#arr)"/>')
grid = "".join(
    f'<text x="{sx(minx + (maxx - minx) * t):.1f}" y="{H - 4}" font-size="9" fill="#888">{minx + (maxx - minx) * t:.0f}</text>'
    for t in (0, 0.25, 0.5, 0.75, 1))
svg = (f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg">'
       f'<defs><marker id="arr" markerWidth="8" markerHeight="8" refX="7" refY="3" orient="auto">'
       f'<path d="M0,0 L7,3 L0,6 z" fill="#ffb300"/></marker></defs>'
       f'<polyline points="{path_d}" fill="none" stroke="#4f83cc" stroke-width="1.6" stroke-dasharray="4 3"/>'
       f'{pts}{arrow}{grid}</svg>')

tick = "&nbsp;" * 3
traj_table_rows = "\n".join(
    f"<tr><td>{i + 1}</td><td>{S['front_frames'][i]}</td>"
    f"<td>{p[0]:.1f}</td><td>{p[1]:.1f}</td><td>{p[2]:.1f}</td>"
    f"<td>{p[3]:.2f}</td><td>{p[4]:.2f}</td><td>{p[5]:.2f}</td></tr>"
    for i, p in enumerate(traj))

# log frame snippet
log_snippet = '''{
  "frame": 30,
  "command": "move_path",
  "sensors": {
    "state": {
      "position": [x, y, z],
      "orientation": [qx, qy, qz, qw],   // quaternion
      "linear_velocity": [...], "linear_acceleration": [...],
      "angular_velocity": [...], "collision": { "has_collided": false }
    },
    "imu": { "rotation": [[3x3 matrix]], "orientation": [...] }
  }
}'''

prompt_disp = ",".join(f"{v:.1f}" for v in S["prev_disp"])
prompt_pos = ",".join(f"{v:.1f}" for v in S["cur_pos_local"])
wd = S["gt_waypoint_dir"]; wdist = S["gt_waypoint_dist"]

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>TravelUAV Benchmark Report</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, "Segoe UI", Roboto, sans-serif; margin: 0;
         background: #f6f7f9; color: #1c2733; line-height: 1.55; }}
  header {{ background: linear-gradient(120deg,#12263a,#1d4e89 60%,#2a7de1); color:#fff;
            padding: 34px 40px 26px; }}
  header h1 {{ margin: 0 0 6px; font-size: 26px; }}
  header p {{ margin: 2px 0; opacity: .85; font-size: 14px; }}
  main {{ max-width: 1080px; margin: 0 auto; padding: 26px 40px 60px; }}
  h2 {{ font-size: 20px; margin: 34px 0 10px; padding-bottom: 6px;
       border-bottom: 2px solid #dde4ec; color: #14304d; }}
  h3 {{ font-size: 15px; margin: 18px 0 6px; color: #1d4e89; }}
  table {{ border-collapse: collapse; font-size: 13px; margin: 8px 0; }}
  th, td {{ border: 1px solid #d5dce4; padding: 5px 10px; text-align: left; }}
  th {{ background: #eaf0f6; }}
  code, pre {{ font-family: "SF Mono", Consolas, Menlo, monospace; }}
  pre {{ background: #0f1a26; color: #d7e4f0; padding: 14px 16px; border-radius: 8px;
        overflow-x: auto; font-size: 12.5px; line-height: 1.5; }}
  pre.light {{ background: #eef2f7; color: #1c2733; border: 1px solid #d5dce4; }}
  .kpi {{ display: flex; gap: 14px; flex-wrap: wrap; margin: 14px 0; }}
  .kpi div {{ background: #fff; border: 1px solid #dde4ec; border-radius: 10px;
              padding: 12px 18px; min-width: 150px; box-shadow: 0 1px 3px rgba(0,0,0,.05); }}
  .kpi b {{ display: block; font-size: 22px; color: #1d4e89; }}
  .kpi span {{ font-size: 12px; color: #5b6b7b; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px,1fr));
          gap: 10px; margin: 10px 0; }}
  figure {{ margin: 0; background:#fff; border:1px solid #dde4ec; border-radius:8px;
            padding: 6px; text-align:center; }}
  figure img {{ width: 100%; border-radius: 4px; display: block; }}
  figcaption {{ font-size: 11.5px; color: #5b6b7b; padding: 4px 0 2px; }}
  .pair {{ display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }}
  @media (max-width: 860px) {{ .pair {{ grid-template-columns: 1fr; }} }}
  .tag {{ display: inline-block; background: #e3ecf7; color: #1d4e89; border-radius: 5px;
         padding: 1px 8px; font-size: 12px; margin-right: 6px; }}
  .note {{ background: #fff8e6; border-left: 4px solid #f0b429; padding: 10px 14px;
          border-radius: 6px; font-size: 13.5px; }}
  .plotwrap {{ background:#fff; border:1px solid #dde4ec; border-radius:10px; padding:10px; }}
  li {{ margin: 3px 0; }}
</style>
</head>
<body>
<header>
  <h1>TravelUAV — Benchmark Report</h1>
  <p>UAV Vision-Language Navigation (VLN) in photo-realistic AirSim environments · arXiv 2410.07087</p>
  <p>Report generated 2026-09-21 · All samples below are <b>real data</b> extracted from the official HuggingFace dataset
  (<code>wangxiangyu0814/TravelUAV</code>, map <b>London_Street</b>, trajectory <code>{S['traj_id'][:8]}…</code>)</p>
</header>
<main>

<h2>1 · What the benchmark is</h2>
<p><b>Task.</b> A quadrotor is spawned in a large 3D environment. It receives a <b>natural-language description of a target object</b>
(e.g. <i>“the blue car parked on a cobblestone street…”</i>) plus a <b>heading hint</b> (direction + yaw angle to the target).
The model must fly the drone in a closed loop — observing 5 camera views at every step — and land on / reach the target.
There is <b>no map and no waypoint list</b>: everything is decided from images + text.</p>
<p><b>Metrics</b> (computed by <code>utils/metric.py</code>): Success Rate (SR), Oracle SR, Navigation Error (NE = final distance to target), SPL.</p>
<p><b>Platform.</b> 22 photo-realistic maps (10 Carla towns, 12 UE-based assets: NYC, London, Tokyo, desert, harbour…), simulated with AirSim.
Raw archive size ≈ <b>471 GB</b>; the processed multi-view image tensor preprocessed dataset is what training actually consumes.</p>

<h2>2 · Dataset size</h2>
<div class="kpi">
  <div><b>22</b><span>simulation maps</span></div>
  <div><b>592,529</b><span>training samples (frame refs)</span></div>
  <div><b>427,933</b><span>train split (trainset.json)</span></div>
  <div><b>75,374</b><span>seen val split</span></div>
  <div><b>89,222</b><span>unseen val split</span></div>
  <div><b>~471 GB</b><span>raw archives (all maps)</span></div>
  <div><b>94</b><span>target object descriptions</span></div>
  <div><b>+109 MB</b><span>traj_train.json (trajectory completion)</span></div>
</div>
<h3>Per-map archive size (MB)</h3>
<pre class="light">London_Street 1,524 | Japanese_Street 5,164 | ModernCityMap 9,847 | BrushifyCountryRoads 10,069
Carla_Town06 10,165 | Carla_Town07 10,188 | Carla_Town02 10,349 | BrushifyUrban 11,155
NordicHarbour 12,314 | BattlefieldKitDesert 12,964 | Carla_Town05 13,432 | WesterTown 13,759
Carla_Town03 14,182 | BrushifyForestPack 17,156 | Carla_Town04 17,751 | Carla_Town01 21,431
Carla_Town15 27,963 | ModularPark 28,934 | Carla_Town10HD 29,527 | TropicalIsland 56,308
NewYorkCity 57,841 | NYCEnvironmentMegapa 90,672</pre>
<h3>What one sample is (London_Street, real counts)</h3>
<ul>
<li>A map contains <b>trajectories</b> (e.g. London_Street: 155).</li>
<li>Each trajectory has <b>48–105 raw log frames</b> (median 67); every 5th frame keeps camera images → <b>13 filtered frames</b> for the example trajectory.</li>
<li>Each filtered frame = <b>one training sample</b> → split files are flat lists of references:
<code>{{"json": "London_Street/&lt;uuid&gt;/merged_data.json", "frame": 7}}</code>.</li>
</ul>

<h2>3 · Raw data structure (inside each map archive)</h2>
<pre>London_Street/
└── 035054dd-e6ff-4765-99ef-d787ae2d66bb/          ← one trajectory
    ├── log/000000.json … log/000059.json           ← per-frame drone state (AirSim)
    ├── frontcamera/  000000.png … (every 5th frame)     ← 5 RGB views
    ├── leftcamera/   rightcamera/  rearcamera/  downcamera/
    ├── *_depth/                                     ← depth versions of the views
    └── object_description.json                      ← NL description(s) of the target</pre>
<p>Real <code>log/000030.json</code> (abridged):</p>
<pre>{log_snippet}</pre>
<p>Real <code>object_description.json</code> for this trajectory:</p>
<pre>[
  "{obj_desc}"
]</pre>

<h2>4 · Real input / output sample (frame 7 of the example trajectory)</h2>
<p>This is exactly what the official pipeline (<code>tools/generate_merged_json.py</code> +
<code>llamavid/train/train_uav/train_uav_notice.py</code>) produces for sample
<span class="tag">map London_Street</span><span class="tag">frame 7 / 13</span>
<span class="tag">stage: {S['stage']}</span><span class="tag">assist: {S['assist']}</span></p>

<div class="pair">
  <div>
    <h3>INPUT — 5 camera views at the current step (log frame {S['sample_log_frame']})</h3>
    <div class="grid">{imgs_html}</div>
  </div>
  <div>
    <h3>OUTPUT — the model predicts ONE 4-dim vector</h3>
    <pre>[ dx, dy, dz, distance ] = [ {wd[0]:.3f}, {wd[1]:.3f}, {wd[2]:.3f}, {wdist:.2f} ]</pre>
    <p>i.e. <b>fly {wdist:.2f} m in direction ({wd[0]:.2f}, {wd[1]:.2f})</b> in the current heading frame
    (x = drone front, y = drone right, z = down). The Ground-Truth value above comes from the expert trajectory.
    A trajectory-completion model then expands this single vector into 7 smooth waypoints, which AirSim executes.</p>
    <h3>Trajectory of this episode (filtered, local frame)</h3>
    <div class="plotwrap">{svg}</div>
    <p style="font-size:12px;color:#5b6b7b">Blue = 13 filtered frames · red = sample frame ·
    amber arrow = GT next waypoint ({wdist:.2f} m)</p>
  </div>
</div>

<h3>The exact text prompt fed to the LLM</h3>
<pre>Stage:{S['stage']}

Previous displacement:{prompt_disp}

Current position:{prompt_pos}

Current image:&lt;image&gt;

Instruction:{S['instruction']}</pre>
<p>In LLaMA-UAV the 5 views (of all history frames) are encoded by EVA-ViT-G and compressed by a QFormer into a few visual
tokens that replace <code>&lt;image&gt;</code>. A special <code>&lt;wp&gt;</code> token is appended at the end; the hidden state
at that position is passed through a small MLP head → the 4-dim waypoint vector (<b>regression, not text generation</b>).
A simplified “guided prompt” is also used as a caption target:
<pre class="light">Please pay attention to the obstacles in images and approach the object described below: {obj_desc}</pre>

<h2>5 · Pipeline & model interface</h2>
<pre>split json (json+frame) → merged_data.json (trajectory + instruction)
        │
        ▼
[CLOSED LOOP]  AirVLNENV (env_uav.py) ⇄ model wrapper (BaseModelWrapper)
        │                                  │
        │   obs: 5-view images, stage,     │  prepare_inputs() → prompt + images + history
        │   prev displacement, cur pos     │  run()           → 1 waypoint (dir + dist)
        │                                  │  predict_done()  → GroundingDINO "target found?"
        ▼                                  ▼
   simulator server                 LLaMA-UAV MLLM + waypoint head
   (move_path_by_waypoints)         + traj completion + DINO</pre>
<div class="note"><b>How a custom model plugs in.</b> The benchmark only requires implementing the thin
<code>BaseModelWrapper</code> interface (<code>prepare_inputs</code> / <code>run</code> / <code>predict_done</code>).
Any VLM — including one that outputs <b>structured text</b> like <code>{{"direction":[dx,dy,dz],"distance":d}}</code> —
can be plugged in by parsing its output and converting to world waypoints.</div>

<h2>6 · Files in this folder</h2>
<table>
<tr><th>file</th><th>what it is</th></tr>
<tr><td><code>cam_frontcamera.png</code> … <code>cam_downcamera.png</code></td><td>the 5 real camera views of the sample step</td></tr>
<tr><td><code>sample_computed.json</code></td><td>computed sample: prompt parts, GT waypoint, full trajectory</td></tr>
<tr><td><code>traj_objdesc.json</code></td><td>the real target description of this trajectory</td></tr>
<tr><td><code>benchmark_report.html</code></td><td>this report</td></tr>
</table>

<h3>Trajectory table (13 filtered frames, position [m] + euler angles [rad], start-frame local)</h3>
<table>
<tr><th>#</th><th>log frame</th><th>x</th><th>y</th><th>z</th><th>pitch</th><th>roll</th><th>yaw</th></tr>
{traj_table_rows}
</table>

</main>
</body>
</html>"""

out = REPORT / "benchmark_report.html"
out.write_text(html)
print("wrote", out, len(html), "bytes")
