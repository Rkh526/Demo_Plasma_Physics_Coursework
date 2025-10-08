"""
accel_n_mag_solvers.py

Helper function module for plasma physics simulations during
JHU plasma physics coursework.

Author: Raymmond Hall
This code was orignally modeled from example code provided during
module 1 and has been personally expanded throughout the course up to the
midterm project.
"""

import numpy as np


#### Functions for various required parameters


## Acceleration Functions
def findAccel_r2(r0):
    #####
    # This function finds the acceleration without a magnetic
    # using the  assumed 1/r^2 force law.
    #
    # Inputs:
    #        r0 [position vector (size=2), units: length]
    # Outputs:
    #        a [acceleration vector at next time]
    #####
    rmag = np.sqrt(np.dot(r0, r0))  # 2D distance magnitude
    x0 = r0[0]
    y0 = r0[1]

    ascale = 0.01  # Number that represents strength of central force (arbitrary units of acceleration)

    # Assume a 1/r^2 force law
    ax = (
        ascale * (-1.0 / rmag**2) * (x0 / rmag)
    )  # cos(angle) = adjacent side/hypotenuse
    ay = (
        ascale * (-1.0 / rmag**2) * (y0 / rmag)
    )  # # sine(angle) = opposite side/hypotenuse

    a = np.array([ax, ay])
    return a


def findAccel_constB(v0):
    #####
    # This function finds the acceleration with a constant
    # (and hardcoded here) magnetic field
    #
    # Inputs:
    #        v0 [velocity (size=3), units: length/time]
    # Outputs:
    #        a [acceleration vector at next time]
    #####

    #
    B = np.array([0, 0, 1])
    w = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)  # setting q and m = 1
    rL = v0 / w
    rLmag = np.sqrt(rL[0] ** 2 + rL[1] ** 2 + rL[2] ** 2)
    u = v0 / (rLmag * w)

    ascale = 0.01  # Number that represents strength of central force (arbitrary units of acceleration)
    a = -ascale * np.cross(u, B)

    return a, B


def findAccel_dynamicB(r0_run, v0, dt):
    #####
    # This function finds the acceleration with a varying
    #  magnetic field based on position
    #
    # Inputs:
    #       r0 [position (size=3), units: length]
    #       v0 [velocity (size=3), units: length/time]
    #       dt [timestep (size=1), units: time]
    # Outputs:
    #        a [acceleration vector at next time]
    #        B [Magnetic Field for this step]
    #        mu [Magnetic moment for this step]
    #####

    # Increasing my mag field in the z dxn by (1+h_z*z^2) meaning for the div free
    # condition to be satisfied, setting B_x = 0, B_y = -2h_z*z*y
    B = np.array([0, -2 * dt * r0_run[2] * r0_run[1], (1 + dt * (r0_run[2] ** 2))])
    B_mag = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)
    v_mag_perp = np.sqrt(v0[0] ** 2 + v0[1] ** 2)
    mu = (1 / 2) * 1 * (v_mag_perp**2) / B_mag  # m = 1
    w = np.sqrt(B[0] ** 2 + B[1] ** 2 + B[2] ** 2)  # setting q and m = 1
    rL = v_mag_perp / w
    u = v0 / (rL * w)

    ascale = 0.1  # Number that represents strength of central force (arbitrary units of acceleration)
    a = -ascale * np.cross(u, B)

    return a, B, mu


def find_accel(r0, v0, dt, accel_calc):
    ####
    # This function chooses the correct acceleration helper function based on
    # user input.
    #
    # Inputs:
    #        r0 [position vector (size=3), units: length]
    #        v0 [velocity (size=3), units: length/time]
    #        dt [timestep (size=1), units: time]
    #        accel_calc [string choice for which calculation]
    # Outputs:
    #        a [acceleration vector (size=3), units: length/time^2]
    #        B [magnetic field vector (size=3), units: mass/(length*time^2)]
    #        mu [magnetic moment (size=1)]
    #####
    match accel_calc:
        case "r2":
            # Get the acceleration for the 1/r^2 force law
            a = findAccel_r2(r0)
            B = mu = 0
        case "const":
            # Get the acceleration with a constant B field
            a, B = findAccel_constB(v0)
            mu = 0
        case "dynamic":
            # Get the acceleration with a constant B field
            a, B, mu = findAccel_dynamicB(r0, v0, dt)
    return a, B, mu


#### Function to solo calculate B field for the field lines
def calc_B_field(x, y, z, dt):
    B = np.array([0, -2 * dt * z * y, (1 + dt * (z**2))])
    return B


#### Define numerical algorithms


def euler(r0, v0, dt, a, B, mu):
    #####
    # This function implements the Euler method.
    #
    # Inputs:
    #        r0 [position vector (size=3), units: length]
    #        v0 [velocity (size=3), units: length/time]
    #        dt [timestep (size=1), units: time]
    #        a [acceleration vector (size=3), units: length/time^2]
    #        B [magnetic field vector (size=3), units: mass/(length*time^2)]
    #        mu [magnetic moment (size=1)]
    # Outputs:
    #        r [position vector at next time]
    #        v [velocity vector at next time]
    #        B [magnetic field at next time]
    #        mu [magnetic moment at next time]
    #####

    #### Find Velocity and radius
    v = v0 + a * dt  ###########
    r = r0 + v0 * dt  ###########

    return r, v, B, mu


def semi_implicit_euler(r0, v0, dt, a, B, mu):
    #####
    # This function implements the semi-implicit Euler method.
    #
    # Inputs:
    #        r0 [position vector (size=3), units: length]
    #        v0 [velocity (size=3), units: length/time]
    #        dt [timestep (size=1), units: time]
    #        a [acceleration vector (size=3), units: length/time^2]
    #        B [magnetic field vector (size=3), units: mass/(length*time^2)]
    #        mu [magnetic moment (size=1)]
    # Outputs:
    #        r [position vector at next time]
    #        v [velocity vector at next time]
    #        B [magnetic field at next time]
    #        mu [magnetic moment at next time]
    #####

    #### Find Velocity and radius
    v = v0 + a * dt
    r = r0 + v * dt

    return r, v, B, mu


def euler_richardson(r0, v0, dt, a, B, mu):
    #####
    # This function implements the Euler-Richardson method.
    #
    # Inputs:
    #        r0 [position vector (size=3), units: length]
    #        v0 [velocity (size=3), units: length/time]
    #        dt [timestep (size=1), units: time]
    #        a [acceleration vector (size=3), units: length/time^2]
    #        B [magnetic field vector (size=3), units: mass/(length*time^2)]
    #        mu [magnetic moment (size=1)]
    # Outputs:
    #        r [position vector at next time]
    #        v [velocity vector at next time]
    #        B [magnetic field at next time]
    #        mu [magnetic moment at next time]
    #####

    #### Find midpoint Velocity and radius
    v_mid = v0 + a * 0.5 * dt  ###########
    r_mid = r0 + v0 * 0.5 * dt  ###########

    # Get the acceleration midpoint with a constant B field
    # a_mid = findAccel_constB(v_mid)
    # Get the acceleration midpoint with a varying B field
    a_mid, B_mid, mu_mid = findAccel_dynamicB(r_mid, v_mid, dt)

    #### Find final Velocity and radius
    v = v0 + a_mid * dt  ###########
    r = r0 + v_mid * dt  ###########

    return r, v, B_mid, mu_mid


def rk4(r0_run, v0, dt, a, B, mu):
    #####
    # This function implements the Runge-Kutta 4th order method.
    #
    # Inputs:
    #        r0 [position vector (size=3), units: length]
    #        v0 [velocity (size=3), units: length/time]
    #        dt [timestep (size=1), units: time]
    #        a [acceleration vector (size=3), units: length/time^2]
    #        B [magnetic field vector (size=3), units: mass/(length*time^2)]
    #        mu [magnetic moment (size=1)]
    # Outputs:
    #        r [position vector at next time]
    #        v [velocity vector at next time]
    #        B [magnetic field at next time]
    #        mu [magnetic moment at next time]
    #####

    # TODO Initial time varying B field based on arbitrary time value
    """if t<3.1415/2.0:
        B = (0,0,1.0)
    else:
        B = (0,0,10.0)
    """

    # see Christian, W. (2010) Chapter 3: Simulating particle motion
    k1v = a * dt
    k1x = v0 * dt

    k2v = np.cross(v0 + k1v / 2, B) * dt
    k2x = (v0 + k1v / 2) * dt

    k3v = np.cross(v0 + k2v / 2, B) * dt
    k3x = (v0 + k2v / 2) * dt

    k4v = np.cross(v0 + k3v, B) * dt
    k4x = (v0 + k3v) * dt

    v = v0 + (k1v + 2 * k2v + 2 * k3v + k4v) / 6.0
    r = r0_run + (k1x + 2 * k2x + 2 * k3x + k4x) / 6.0

    return r, v, B, mu
