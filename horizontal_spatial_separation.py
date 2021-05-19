""" Original source of code:
https://scipython.com/book2/chapter-8-scipy/examples/a-projectile-with-air-resistance/
Originally designed to obatin the trajectory of a particle on a parabolic
trajectory with some characteristic air resistance """

# Loading in relevant packages
import numpy as np
from numpy import random
from numpy.core.fromnumeric import mean
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib
import sys
import pandas as pd

# Setting some aesthetic stuff for text formatting
params = {
    "font.family": "Arial",
    "font.size": 12,
}
matplotlib.rcParams.update(params)

# Defining constants used in the calculations below
acc_gravity = 9.81  # m/s2
rho_air = 1.28  # kg.m-3

############### User-defined parameters ########################################
# Height of the seeds before they're blown horizontally and allowed to free-fall
height = 10  # m

# Velocity of seeds as they are blown horizontally
initial_horizontal_v = 10  # m/s

# Number of seeds to investigate --> more == better stastical sampling
no_of_seeds = 100

# List the seeds to investigate here - the code then pulls the relevant info
#  for each type and performs the analysis for each type in turn. The valid
#  seed types are commented to the right. This list can be easily extended with
# some basic properties of a new seed type - see below for how new seed types
#  can be implemented
seeds_to_investigate = [
    "lentil",
    "barley",
    "faba bean",
    "wheat",
    "sunflower seed",
]  # faba bean, lentil, barley, wheat, sunflower seed

#  Flag the colour of the seeds for the histogram
seed_colours = ["goldenrod", "saddlebrown"]  # , "green", "gold", "blue"]

#  Set the horizontal limits for the plot
xlims = [0, 20]  # m

# Set the opacity of the filled bars in the histogram; 0 = invisible, 1 = opaque
opacity = 0.5
################################################################################


def calc_seed_distributions(
    seed_width_range,
    seed_length_range,
    seed_thickness_range,
    seed_mass_range,
    seed_v_term_range,
    seed_density,
    no_of_seeds,
):
    """ Function to determine the distribution of seed properties
    Using the range rule to estimate rough standard deviations for the normal
    distribution of seed properties:
    https://www.thoughtco.com/range-rule-for-standard-deviation-3126231 
    Using this method since we don't have info for true standard deviations of
    the different properties of the seeds - these properties could be easily
    obtained experimentally """
    width_stdev = (seed_width_range[1] - seed_width_range[0]) / 4
    length_stdev = (seed_length_range[1] - seed_length_range[0]) / 4
    thickness_stdev = (seed_thickness_range[1] - seed_thickness_range[0]) / 4
    mass_stdev = (seed_mass_range[1] - seed_mass_range[0]) / 4
    v_term_stdev = (seed_v_term_range[1] - seed_v_term_range[0]) / 4

    """ Here we are assuming that the individual properties of the seed follow a
    normal distribution, but that the different properties are perfectly
    correlated with one another; i.e. the longest seed will also be the widest
    and thickest and heaviest and have the fastest terminal velocity.
    This should be a reasonable assumption? """
    width_distrib = np.sort(
        np.random.normal(np.mean(seed_width_range), width_stdev, no_of_seeds)
    )
    length_distrib = np.sort(
        np.random.normal(np.mean(seed_length_range), length_stdev, no_of_seeds)
    )
    thickness_distrib = np.sort(
        np.random.normal(
            np.mean(seed_thickness_range), thickness_stdev, no_of_seeds
        )
    )
    mass_distrib = np.sort(
        np.random.normal(np.mean(seed_mass_range), mass_stdev, no_of_seeds)
    )
    v_term_distrib = np.sort(
        np.random.normal(np.mean(seed_v_term_range), v_term_stdev, no_of_seeds)
    )

    # Combining the arrays into a dataframe
    seed_properties_df = pd.DataFrame(
        {
            "width(m)": width_distrib,
            "length(m)": length_distrib,
            "thickness(m)": thickness_distrib,
            "mass(kg)": mass_distrib,
            "v_terminal(m/s)": v_term_distrib,
        }
    )
    seed_properties_df["density(kg/m-3)"] = seed_density
    # print(seed_properties_df)

    return seed_properties_df


def calc_drag_coeffs(seed_properties_df, acc_gravity, rho_air):
    """ Function to calculate the horizontal and vertical drag co-efficients of
    the seeds. We have to assume a fixed seed orientation for this calculation,
    so projected area for vertical calculation is width * thickness """

    # Assuming an elliptical shape for the cross-sections
    seed_properties_df["vert_cross_sec_A(m2)"] = (
        np.pi
        * seed_properties_df["width(m)"]
        * seed_properties_df["thickness(m)"]
    )

    # Calculating the drag co-efficient for the seed - rotation ignored!
    seed_properties_df["vertical_drag_coeff"] = (
        2
        * acc_gravity
        * seed_properties_df["mass(kg)"]
        * (seed_properties_df["density(kg/m-3)"] - rho_air)
    ) / (
        rho_air
        * seed_properties_df["density(kg/m-3)"]
        * seed_properties_df["v_terminal(m/s)"] ** 2
        * seed_properties_df["vert_cross_sec_A(m2)"]
    )

    return seed_properties_df


def deriv(t, u):
    """ Function to calculate the horizontal and vertical velocities and 
    accelerations at each timestep of the calculation """
    x, xdot, z, zdot = u
    speed = np.hypot(xdot, zdot)
    xdotdot = -k / seed_mass * speed * xdot
    zdotdot = -k / seed_mass * speed * zdot - acc_gravity

    return xdot, xdotdot, zdot, zdotdot


def hit_target(t, u):
    # We've hit the target if the z-coordinate is 0.
    return u[2]


def max_height(t, u):
    # The maximum height is obtained when the z-velocity is zero.
    return u[3]


######################## Main body of code #####################################

plot_counter = 0
savestring = ""
for seed_to_investigate in seeds_to_investigate:
    if seed_to_investigate == "faba bean":
        seedname = "faba bean"
        seed_width_range = [5e-3, 10e-3]  # m
        seed_thickness_range = [5e-3, 10e-3]  # m
        seed_length_range = [10e-3, 15e-3]  # m
        # Source: https://arccjournals.com/uploads/articles/8LR256.pdf
        seed_mass_range = [1.7e-3, 1.8e-3]  # kg
        seed_v_term_range = [11.7, 12.9]  # m/s
        seed_density = 1070  # kg/m^-3
    elif seed_to_investigate == "lentil":
        seedname = "lentil"
        seed_width_range = [2e-3, 4e-3]  #  m
        seed_thickness_range = [2e-3, 4e-3]  # m
        seed_length_range = [2e-3, 4e-3]  # m
        seed_mass_range = [30e-6, 41e-6]  # kg
        # Source: https://reader.elsevier.com/reader/sd/pii/S0021863496900104?token=6D7919D31F8BC5E57BFFAE3830BA59797E5C3E828DEDDB8D7BC17BC7B67C48352F8DDB37C47C2BDE3FB6EBE0536A47CF&originRegion=eu-west-1&originCreation=20210518172809
        seed_v_term_range = [10.95, 12.06]  # m/s
        seed_density = 810  # kg/m^-3
    elif seed_to_investigate == "barley":
        seedname = "barley"
        seed_width_range = [2e-3, 4e-3]  # m
        seed_thickness_range = [2e-3, 4e-3]  # m
        seed_length_range = [4e-3, 10e-3]  #  m
        seed_mass_range = [44e-6, 50e-6]  # kg
        #  source: https://journals.ekb.eg/article_97786_2b84d6dcc2a36486df5260ec9646d666.pdf
        seed_v_term_range = [7.0, 7.8]  # m/s
        seed_density = 630  # kg/m^-3
    elif seed_to_investigate == "barley2":
        seedname = "barley2"
        seed_width_range = [2e-3, 4e-3]  # m
        seed_length_range = [2e-3, 4e-3]  # m
        seed_thickness_range = [4e-3, 10e-3]  #  m
        seed_mass_range = [44e-6, 50e-6]  # kg
        #  source: https://journals.ekb.eg/article_97786_2b84d6dcc2a36486df5260ec9646d666.pdf
        seed_v_term_range = [7.0, 7.8]  # m/s
        seed_density = 630  # kg/m^-3
    elif seed_to_investigate == "wheat":
        seedname = "wheat"
        seed_width_range = [2e-3, 5e-3]  # m
        seed_thickness_range = [2e-3, 5e-3]  # m
        seed_length_range = [4e-3, 8e-3]  #  m
        seed_mass_range = [30e-6, 40e-6]  # kg
        #  source: https://www.researchgate.net/publication/237432950_Aerodynamic_Properties_of_Wheat_Kernel_and_Straw_Materials
        seed_v_term_range = [6.8, 8.6]  # m/s
        seed_density = 1270  # kg/m^-3
    elif seed_to_investigate == "sunflower seed":
        seedname = "sunflower seed"
        #  source: https://reader.elsevier.com/reader/sd/pii/S0021863496901110?token=28A167A3714332F411FE60E1FBB49F7360B79181A75A222C410638F3E9FA6EA3823A7006F4ED3585392EA9154EA51959&originRegion=eu-west-1&originCreation=20210519172608
        #  source: https://www.researchgate.net/publication/237432950_Aerodynamic_Properties_of_Wheat_Kernel_and_Straw_Materials
        seed_width_range = [4.69e-3, 5.55e-3]  # m
        seed_thickness_range = [2.96e-3, 3.56e-3]  # m
        seed_length_range = [8.82e-3, 10.2e-3]  #  m
        seed_mass_range = [37e-6, 61e-6]  # kg
        seed_v_term_range = [5.8, 7.6]  # m/s
        seed_density = 740  # kg/m^-3
    else:
        print("I don't recognise this seed type! Quitting...")
        sys.exit()

    seed_properties_df = calc_seed_distributions(
        seed_width_range,
        seed_length_range,
        seed_thickness_range,
        seed_mass_range,
        seed_v_term_range,
        seed_density,
        no_of_seeds,
    )
    # print(seed_properties_df)

    # Need to now calculate the drag co-efficients for the seeds in air.
    # The seeds need to have a fixed orientation to simplify the problem.
    # For all cases, seeds will fall head-first. So, the cross-section of the
    # seed for the vertical calculation will be width * thickness

    seed_properties_df = calc_drag_coeffs(
        seed_properties_df, acc_gravity, rho_air
    )
    # print(seed_properties_df)

    seed_times_list = []
    seed_distances_list = []
    for bb in range(len(seed_properties_df)):
        (
            seed_width,
            seed_length,
            seed_thickness,
            seed_mass,
            seed_v_terminal,
            seed_density,
            seed_vert_cross_sec_A,
            seed_vert_drag,
        ) = seed_properties_df.iloc[bb]

        # For convenience, define this constant
        k = 0.5 * seed_vert_drag * rho_air * seed_vert_cross_sec_A

        # Initial speed and launch angle (from the horizontal)
        # v0 = initial_horizontal_v
        # phi0 = np.radians(0)

        # Initial conditions: x0, v0_x, z0, v0_z
        u0 = 0, initial_horizontal_v, height, 0
        # Integrate up to tf unless we hit the target sooner
        t0, tf = 0, 5000

        # Stop the integration when we hit the target
        hit_target.terminal = True
        # Must be moving downwards - don't stop before we begin moving upwards
        hit_target.direction = -1

        soln = solve_ivp(
            deriv,
            (t0, tf),
            u0,
            dense_output=True,
            events=(hit_target, max_height),
        )

        # A fine grid of time points from 0 until impact time
        t = np.linspace(0, soln.t_events[0][0], 100)

        # Retrieve the solution for the time grid and plot the trajectory
        sol = soln.sol(t)
        x, z = sol[0], sol[2]

        seed_times_list.append(soln.t_events[0][0])
        seed_distances_list.append(x[-1])

    seed_properties_df["time_to_fall(s)"] = seed_times_list
    seed_properties_df["distance_travelled(m)"] = seed_distances_list

    # Defining bin-size for the histogram
    bins = np.arange(0, 30, 0.1)

    weights = np.empty(seed_properties_df["distance_travelled(m)"].shape)
    weights.fill(1 / seed_properties_df["distance_travelled(m)"].shape[0])
    plt.hist(
        seed_properties_df["distance_travelled(m)"],
        bins,
        alpha=opacity,
        color=seed_colours[plot_counter],
        label=seeds_to_investigate[plot_counter].capitalize(),
        weights=weights,
        histtype="stepfilled",
    )
    plot_counter += 1

    # print(seed_properties_df)
    # print("\n\n")

    savestring = savestring + f"-{seedname.capitalize()}"

plt.xlim(xlims)
plt.xlabel("Horizontal distance traversed (m)")
plt.ylabel("Probability")
plt.legend(
    bbox_to_anchor=(0.5, 1.05),
    ncol=len(seeds_to_investigate),
    loc="center",
    frameon=False,
)
plt.savefig(f"Seed_distribution{savestring}.png", bbox_inches="tight", dpi=900)
plt.close()
# plt.show()
