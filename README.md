# Data Analysis and AI Project

The box plots of each type of bug are really different for most variables which is a good sign that our variables are linked to the bug type and can be used to predict it.
- We see that butterflies generally have a much bigger hull area than the other bug types which have similar box plots.
- The box plots for hull_to_insect_area_ratio are really different for each bug rtype, hover fly having most values in a very short range while values for butterfly cover almost the entire range of possible values.
- Wasps have a very low roundness and no high values at all compared to other bug types which have box plots of higher mean, standard deviation and have outleirs of high values.
- Some variables have similar box plots for all bug types and may not end up being super useful, like aspect ratio, ellipse_angle and axis_least_inertia_y
- The boxes of each bug type for mask_perimeter are all quite different. It is seems to be a good indicator of bug type but could sometimes be misleading as it depends on the scale of the picture (the zooom).
- The box plots for the x axis of least inertia are all very similar for the bug types except for wasp. But it may not mean much aswasp doesn't have enough samples to make a clear conclusion. It does have 2 points of higher values so with higher samples its distribution for x axis least inertia could become close to that of other bug types.

