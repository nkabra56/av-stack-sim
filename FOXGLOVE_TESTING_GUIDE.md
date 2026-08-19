# Foxglove testing & LinkedIn video guide

Everything from here on is manual/GUI work I can't do myself (no desktop-app or
screen-recording access from this environment). This is the full step-by-step for
generating the 3D scene, viewing it in Foxglove, and turning it into a video.
See [FOXGLOVE_VIZ_PLAN.md](FOXGLOVE_VIZ_PLAN.md) for the implementation design.

## 1. Install the `viz` extra

```bash
pip install -e ".[viz]"
```

This pulls in `foxglove-sdk` (not part of the base install — same opt-in pattern as
the `rl` extra). Confirm it worked:

```bash
python -c "import foxglove; print(foxglove.__version__)"
```

## 2. Generate a scene

```bash
python -m core.demo perpendicular_open --foxglove out/demo.mcap
```

You should see two lines of output — the usual status line (`reached spot in N
steps`) and `Foxglove scene written to out/demo.mcap (...)`. `out/demo.mcap` should
be non-trivial in size (tens to hundreds of KB depending on run length).

Worth generating a couple of scenarios so you have options for the video — the
"cleanest"-looking run is usually one that actually reaches the spot without
stalling:

```bash
python -m core.demo perpendicular_open --foxglove out/perpendicular_open.mcap
python -m core.demo parallel_between_cars --foxglove out/parallel_between_cars.mcap
python -m core.demo perpendicular_flanked --controller mpc --foxglove out/perpendicular_flanked_mpc.mcap
```

(Full scenario list: `perpendicular_open`, `perpendicular_flanked`,
`perpendicular_obstructed_lane`, `parallel_open`, `parallel_between_cars`. `--seed N`
gives a different noise realization of the same scenario if you want to re-roll a run.)

## 3. Install the Foxglove desktop app

Download from **https://foxglove.dev/download** (Windows installer). I didn't find a
verified `winget` package id for it, so grab it from that page directly rather than
guessing at a package manager command. The free tier covers everything in this guide
except the built-in video exporter (see step 6).

## 4. Open the file

Launch Foxglove → **Open local file** → select `out/demo.mcap`. You should land on a
default layout with an empty/auto-picked panel. Next steps configure it properly.

## 5. Configure the layout

### 5a. 3D panel with camera-follow

1. Add a panel → **3D**.
2. Open its settings (gear icon). Under the **Scene** / **Frame** section, find the
   follow-frame / target-frame setting and set it to **`vehicle`**.
3. Set the follow mode to **Position + Rotation** (sometimes labeled "Follow Pose" or
   similar) — this is what makes the camera track the car through the maneuver instead
   of sitting on a fixed wide shot.
4. Under **Topics**, make sure `/scene` and `/tf` are checked/visible.
5. If the initial camera distance looks too close or too far, adjust it by
   scroll-zooming once in the 3D view — Foxglove remembers this per-layout.

### 5b. Telemetry plots (optional but recommended)

1. Add a panel → **Plot**.
2. Add series `/speed.v` and `/speed.delta` — this reproduces the old Matplotlib
   speed-profile panel.
3. Add a second Plot panel with a few `/sensors.beam_N` fields (there are 10 beams,
   `beam_0`..`beam_9`) if you want the live ultrasonic readings visible too.

### 5c. Status (optional)

Add a panel → **Raw Messages**, point it at `/status` — shows the scenario/planner/
controller info at the start and the success/collision result at the end.

### 5d. Arrange and save the layout

Drag panel edges so the 3D view is the dominant panel (e.g. left ~65%, plots stacked
on the right ~35% — mirrors the old Matplotlib panel proportions). Then:

- Layout menu → **Export layout** (or right-click the layout in the sidebar → Export)
  → save the JSON.
- Commit it into the repo at `foxglove-layouts/parking_demo.json` (create the
  `foxglove-layouts/` folder if it doesn't exist) so this setup is a one-click
  **Import layout** next time instead of redoing steps 5a–5c.

## 6. Play it back

Hit the play button on the timeline at the bottom. Things to check:

- The car visibly drives from its start pose into the parking spot along a smooth
  path.
- The camera follows it (doesn't stay static or drift oddly — if it does, re-check the
  follow-frame/follow-mode setting from step 5a).
- The blue (true) and orange (estimated) trails both grow, with a visible gap between
  them representing EKF estimation error.
- The translucent orange ellipse around the estimated position grows/shrinks over
  time.
- The sensor-ray fan shifts from green (clear) toward red as the car nears an
  obstacle or the spot's edge.
- Playback speed: there's a speed multiplier control near the play button (0.5x/1x/2x
  etc.) if you want to slow down a fast maneuver for the recording, or speed up a long
  one.

If something looks broken (car not moving, no camera follow, missing trails), that's
worth flagging back to me with a screenshot/description — the export code was verified
to run without errors and produce a non-empty file, but I have not been able to
visually confirm the rendered scene myself.

## 7. Record the video

Foxglove's built-in **Export video** command is **Enterprise-plan only** — it won't
work on the free tier, so don't spend time hunting for it in the menu. The practical
path on the free tier is screen-recording the app window during playback:

**Option A — Windows Game Bar (built in, no install)**
1. `Win + G` to open Game Bar with the Foxglove window focused.
2. Open the Capture widget, click the record button (or `Win + Alt + R`).
3. Press play in Foxglove, let it run through, `Win + Alt + R` again to stop.
4. Recording lands in `Videos\Captures\` as an `.mp4` — already LinkedIn-ready format.

**Option B — OBS Studio (free, more control over framing/quality)**
1. Install from https://obsproject.com/.
2. Add a **Window Capture** source targeting the Foxglove window (or a **Display
   Capture** if you want to resize/reposition the window first).
3. Settings → Output: set format to `mp4`, a reasonable bitrate (8–12 Mbps for 1080p
   is plenty for LinkedIn).
4. Settings → Video: set canvas/output resolution to 1920x1080 if you can size the
   Foxglove window to roughly 16:9 first — cleaner crop, no letterboxing.
5. Start Recording, press play in Foxglove, let it finish, Stop Recording.

Before recording either way: resize/maximize the Foxglove window to a clean 16:9-ish
area, hide anything sensitive on the rest of the screen (this is a full window/screen
capture), and do one silent dry-run playback first to confirm the layout looks right
before you actually hit record.

## 8. Trim/polish (optional)

If you need to trim dead time at the start/end or add a title card, Windows 11's
built-in **Clipchamp** app (Start menu → search "Clipchamp") handles basic trims/text
overlays without installing anything extra. If you'd rather script it and have/install
ffmpeg (`winget install ffmpeg` or https://ffmpeg.org/download.html — not currently on
this machine's PATH), a trim is one command:

```bash
ffmpeg -i input.mp4 -ss 00:00:02 -to 00:00:18 -c copy trimmed.mp4
```

## 9. LinkedIn upload notes

- Native video (not GIF) is the better-performing format — LinkedIn autoplays it
  muted in-feed, so keep it visually clear without relying on audio.
- Landscape 16:9 (1920x1080) or square 1:1 both work well in-feed; square uses more
  vertical scroll space on mobile if that matters to you.
- LinkedIn allows up to 10 minutes, but a demo like this reads best short — aim for
  15–45 seconds, looping the cleanest single parking maneuver rather than stitching
  every scenario together.
- A short caption on the post explaining what's being shown (EKF estimate vs. ground
  truth, sensor fan, etc.) helps viewers parse a technical clip fast — consider
  pointing out 1–2 specific things to watch for (e.g. "watch the orange
  uncertainty ellipse shrink as the sensor corrects the estimate").

## Quick troubleshooting reference

| Symptom | Likely cause / fix |
|---|---|
| 3D panel shows nothing | `/scene`/`/tf` topics not checked visible in panel settings |
| Camera doesn't move with the car | Follow frame isn't set to `vehicle`, or follow mode is off |
| Car appears but doesn't rotate | Check `/tf` topic is visible/subscribed, not just `/scene` |
| Plot panels empty | Series path typo — must match exactly, e.g. `/speed.v` not `/speed/v` |
| File won't open / looks corrupt | Re-run step 2; if it still fails, tell me the exact error text |
| `ModuleNotFoundError: foxglove` on `--foxglove` | Run step 1 (`pip install -e ".[viz]"`) in the same Python environment you're invoking `python -m core.demo` from |
