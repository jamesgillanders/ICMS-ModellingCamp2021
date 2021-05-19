# ICMS-ModellingCamp2021
Repository containing code developed as part of the ICMS and MAC-MIGS modelling camp (May 2021).

## horizontal_spatial_separation.py
This program calculates the trajectory of a particle in two dimensions, undergoing air resistance.
It has been developed to model the trajectories of different types of crop seeds and beans.
The aim is to calculate the horizontal distance traversed by seeds and beans with different physical properties (e.g. mass, length, width, thickness).
The basic premise is that the seeds are blown through a pipe at a fixed velocity, horizontally from a fixed height, and allowed to free-fall on a parabolic trajectory.
From this, one can explore whether the distances are sufficiently different from that of other seeds/beans under investigation, such that this could be used as a viable separation method.

As an example, the figure below shows the horizontal distance computed for lentil and barley seeds, with a range of physical properties, as would be expected in a real-world harvest.

![Seed_distribution-Lentil-Barley](https://user-images.githubusercontent.com/47385282/118879539-c0b5de80-b8e8-11eb-845e-5722ef54755c.png)

The sources of seed properties are embedded in the program, and there are comments sign-posting the most important functions and calculations.
