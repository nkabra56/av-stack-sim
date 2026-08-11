# NGSIM data attribution

`excerpt_trajectories.csv` is a 780-frame (78 second, 10 Hz) excerpt of the **Next Generation
Simulation (NGSIM)** vehicle trajectory dataset, US-101 freeway location, covering exactly one
real leader/follower vehicle pair (`vehicle_id` 9 leading, 12 following, linked via NGSIM's own
`preceding` field for every frame in the excerpt) with a verified-contiguous 100ms sample
interval. Used to validate `auto_park/control/acc.py`'s controllers against real car-following
behavior — see DESIGN.md's ACC section and `auto_park/validation/acc_validation.py`.

Columns are unmodified NGSIM fields (`vehicle_id`, `global_time`, `local_x`, `local_y`, `v_vel`,
`v_acc`, `v_length`, `v_width`, `lane_id`, `preceding`, `space_headway`, `time_headway`,
`location`), in NGSIM's native units (feet, feet/second, feet/second^2) — unit conversion to SI
happens in `auto_park/validation/ngsim_loader.py`, not in this file, so the committed data stays
verbatim from the source.

Source: U.S. Department of Transportation, Federal Highway Administration — Next Generation
Simulation (NGSIM) Vehicle Trajectories and Supporting Data. Downloaded via the public Socrata
API at `data.transportation.gov` (dataset `8ect-6jqj`), no registration required.

Licensed under Creative Commons Attribution-ShareAlike 3.0 (CC BY-SA 3.0). This excerpt is
redistributed here for educational/portfolio use, with attribution to the source above.
