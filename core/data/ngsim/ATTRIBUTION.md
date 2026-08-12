# NGSIM data attribution

`excerpt_trajectories.csv` is a 789+783-frame (~79 second, 10 Hz) excerpt of the **Next Generation
Simulation (NGSIM)** vehicle trajectory dataset, US-101 freeway location, **lane 2**, covering
exactly one real leader/follower vehicle pair (`vehicle_id` 2896 leading, 2903 following, linked
via NGSIM's own `preceding` field for 783/783 frames in the follower's window -- a 100% pure link,
not just a majority one) with a verified-contiguous 100ms sample interval and a genuine recorded
full stop (both vehicles' `v_vel` reaches exactly 0). Used to validate `core/control/acc.py`'s
controllers against real car-following behavior — see DESIGN.md's ACC section and
`core/validation/acc_validation.py` — and, since this pair is lane 2 (see `lane_centerline.csv`
below, also lane 2), to drive the real leader in the full closed-loop ACC+Stanley composition
(`core/full_highway_harness.py`) without a cross-lane mismatch. Re-extracted from vehicle_id 9/12
(NGSIM lane 1) specifically to close that mismatch — see KNOWN_BUGS.md's former entry 6 and
`core/validation/ngsim_loader.py`'s `DEFAULT_LEADER_ID`/`DEFAULT_FOLLOWER_ID` for the full account;
fetched via the same public Socrata API below, filtered to `location='us-101' AND lane_id='2'` and
a time window overlapping the original lane-1 excerpt (real US-101 congestion is a road-wide event,
not lane-specific, so the same window reliably has comparable lane-2 traffic).

Columns are unmodified NGSIM fields (`vehicle_id`, `global_time`, `local_x`, `local_y`, `v_vel`,
`v_acc`, `v_length`, `v_width`, `lane_id`, `preceding`, `space_headway`, `time_headway`,
`location`), in NGSIM's native units (feet, feet/second, feet/second^2) — unit conversion to SI
happens in `core/validation/ngsim_loader.py`, not in this file, so the committed data stays
verbatim from the source.

Source: U.S. Department of Transportation, Federal Highway Administration — Next Generation
Simulation (NGSIM) Vehicle Trajectories and Supporting Data. Downloaded via the public Socrata
API at `data.transportation.gov` (dataset `8ect-6jqj`), no registration required.

`lane_centerline.csv` is a **derived** lane centerline (`position_m`, `lateral_offset_m`, 322
points spanning 666m), not a raw excerpt: aggregated from ~10,400 individual real vehicle
positions in NGSIM's US-101 lane 2 (the full download, not just the committed pair above),
binned every 2m along the road and averaged, then lightly smoothed (5-bin moving average) to
remove residual per-bin noise while preserving genuine curvature. Used to validate
`core/control/lane_centering.py`'s Stanley controller against a real lane geometry rather
than a hand-authored curve — see DESIGN.md section 12's H3 entry and
`core/validation/lane_centering_validation.py`. Real end-to-end lateral drift across the
segment: ~1.76m — genuine gentle curvature, not synthetic.

Licensed under Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0). This excerpt (and the
lane centerline derived from the same source dataset) is redistributed here for
educational/portfolio use, with attribution to the source above.
