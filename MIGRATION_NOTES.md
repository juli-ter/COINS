# Migration notes

## Ported in this version
- Subject/path configuration (`coins_options`, `coins_subjects`)
- Behavioural CSV loading with row normalization for malformed lines
- Subject-level behavioural analysis pipeline
- Tracking performance metrics
- Average block kernels
- Regression-based kernels
- Session-wise regression kernels
- Post-jump adjustments
- Group-level adjustment aggregation
- Group-level reaction-time summary
- Matplotlib output replacing MATLAB `.fig` files

## Important fixes applied during the port
- Removed reliance on MATLAB workspace variables (`subData` was implicit in some functions)
- Fixed undefined variable usage in group RT analysis (`pre`)
- Added more robust CSV parsing than the original MATLAB `textscan`-based loader
- Saved outputs as Python-native `.pkl` plus `.png` figures

## Not fully ported yet
- Full MEG analysis stack (`FieldTrip`-dependent MATLAB code)
- Full model plotting/paradigm utilities from `sub_model`
- Some secondary/legacy MATLAB exploratory scripts such as `coins_analyse_movement.m`
- Custom MATLAB plotting utilities like `notBoxPlot` and exact `shadedErrorBar` behavior

## Recommendation
Use this port as the main Python base for the behavioural pipeline. Then port MEG/model/exploratory scripts only if you actually need them.
