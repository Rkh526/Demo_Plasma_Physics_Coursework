"""
Author: Raymmond Hall
This code was orignally modeled from example code provided during
module 1 and has been personally expanded throughout the course up to the
midterm project.

Midterm Project: Magnetic Bottle Particle Confinement
s
"""

import random
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import accel_n_mag_solvers as ams


###### Solver Implementation
def run_solver(accel_calc, num_method, save_plots):
    ## Vectors for tracked parameters
    pos = []
    vel = []
    energy = []
    accel = []
    mu_list = []
    B_field = []

    ## Simulation parameters
    dt = 0.1  # Time step
    orbit_num = "orbit1_density_5"  # Description for particle orbit
    coll_freq = 800  # Collision rate
    particleDensity = 5  # Starting particle density
    particleDensity0 = particleDensity
    densityList = []

    # Set the starting position for the particles inside the range of the bottle
    r0 = []
    r_origin = []
    for i in range(0, particleDensity):
        randomY = round(random.uniform(-3.0, 5.0), 1)
        randomZ = round(random.uniform(-6.0, 6.0), 1)
        r0.append([0.0, randomY, randomZ])
        r_origin.append(r0[i])
        pos.append([])

    # Initial velocity set uniform for all particles
    v0 = []
    for i in range(0, particleDensity):
        v0.append(np.array([0.0, 0.1, 0.1]))
        vel.append([])

    total_T = 500  # total time
    num_steps = int(total_T / dt)

    output_period = 500  # How often to show plots

    # num_method = "rk4"  # "euler", "semi-implicit_euler","euler_richardson","rk4"
    method_dir = num_method + "/"

    time_list = []
    time = 0.0
    """
    # Larmor Radius based Theoretical Results with plots
    B_mag = np.sqrt(B0[0]**2 + B0[1]**2 + B0[2]**2)
    v_mag_perp = np.sqrt(v0[0]**2 + v0[1]**2)
    w = B_mag # setting q and m = 1
    rL = v_mag_perp/w



    fig, ax = plt.subplots()
    tt = np.array(list(range(0,num_steps)))*dt
    xTheory = r0[0]+rL*np.sin(w*tt)
    yTheory = r0[1]+rL*np.cos(w*tt)
    ax.plot(xTheory,yTheory)
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.set_title('Theoretical Trajectory')
    plt.subplots_adjust(bottom=.18, left=.18)
    fig.savefig('plots/'+str(method_dir)+str(orbit_num)+'/TheoryTraj_'+str(dt)+'.png')
    """

    # Run the simulation for each r0

    ## Run simulation

    for i in range(0, num_steps):
        time_list.append(time)
        densityList.append([time, particleDensity])

        # Run for each particle r0
        for j in range(0, particleDensity):
            if False:
                print("i = ", i)
            a, B, mu = ams.find_accel(r0[j], v0[j], dt, accel_calc)
            match num_method:
                case "euler":
                    # r, v, B, mu = ams.euler(r0[j], v0[j], dt, accel_calc)
                    r, v, B, mu = ams.euler(r0[j], v0[j], dt, a, B, mu)
                case "semi-implicit_euler":
                    r, v, B, mu = ams.semi_implicit_euler(r0[j], v0[j], dt, a, B, mu)
                case "euler_richardson":
                    r, v, B, mu = ams.euler_richardson(r0[j], v0[j], dt, a, B, mu)
                case "rk4":
                    r, v, B, mu = ams.rk4(r0[j], v0[j], dt, a, B, mu)
                case _:  # Defaulting to rk4
                    print("Running Runge Kutta 4 by default")
                    r, v, B, mu = ams.rk4(r0[j], v0[j], dt, a, B, mu)

            pos[j].append(r)
            vel[j].append(v)

            B_field.append(B)
            mu_list.append(mu)

            rmag = np.sqrt(r[0] ** 2 + r[1] ** 2 + r[2] ** 2)
            vmag = np.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
            energy.append(0.5 * vmag**2 - 0.01 / rmag)

            r0[j] = r
            # v0[j] = v

            # Original Collision Implementation
            # Rotate at a random angle in 2D to simulate particle collisions
            if i % coll_freq == 0 and j % 2 == 0:
                # Pick a random angle between 0 and 360
                rotAngle = random.randint(0, 360) * np.pi / 180
                # Calculate the rotation matrix
                R = [
                    np.cos(rotAngle),
                    -np.sin(rotAngle),
                    np.sin(rotAngle),
                    np.cos(rotAngle),
                ]
                v_prime = np.array(
                    [v[0] * R[0] + v[1] * R[1], v[0] * R[2] + v[1] * R[3], 0]
                )
                v0[j] = v_prime
            else:
                v0[j] = v

            if i % output_period == 0 and save_plots:
                fig, ax = plt.subplots()

                x = np.array(pos[j])[:, 0]
                y = np.array(pos[j])[:, 1]
                z = np.array(pos[j])[:, 2]

                ax.plot(y[:], z[:])

                ax.set_xlabel("Y Position")
                ax.set_ylabel("Z Position")
                ax.set_title(
                    "Trajectory for " + str(dt) + "with " + num_method + " Results"
                )
                plt.subplots_adjust(bottom=0.18, left=0.18)
                # Create Save Path
                Path("plots/" + str(method_dir) + str(orbit_num)).mkdir(
                    parents=True, exist_ok=True
                )

                fig.savefig(
                    "plots/"
                    + str(method_dir)
                    + str(orbit_num)
                    + "/orbit_dt"
                    + str(dt)
                    + "_r0_"
                    + str(j)
                    + "_"
                    + str(i)
                    + ".png"
                )  # if you want to save the individual frames

        """
        #TODO Future Implementation, this produced too many random walks
        #Physical Collision Implementation
        # Check if each particle is within the realm of another and make it collide if so
        for j in range(0,particleDensity):
            minDist = 0.1
            xVals = [x[0] for x in r0]
            yVals = [y[1] for y in r0]
            zVals = [z[2] for z in r0]
            xdist = 1 if (min([abs(r0[j][0]-x) for x in xVals]) <= minDist) else 0
            ydist = 1 if (min([abs(r0[j][1]-x) for x in yVals]) <= minDist) else 0
            zdist = 1 if (min([abs(r0[j][2]-x) for x in zVals]) <= minDist) else 0

            if (xdist+ydist+zdist) > 2: # Meaning we are too close and need to collide
                # Pick a random angle between 0 and 360
                rotAngle = random.randint(0,360)*np.pi/180
                # Calculate the rotation matrix
                R = [np.cos(rotAngle),-np.sin(rotAngle),np.sin(rotAngle),np.cos(rotAngle)]
                v_prime = np.array([v[0]*R[0]+v[1]*R[1],v[0]*R[2]+v[1]*R[3],0])
                v0[j] = v_prime
        """

        time += dt

        # Loss Cone Particle Check
        rPops = []  # List for particles popped out of the loss cone
        particleSub = 0
        for j in range(0, particleDensity):
            rDisp = np.sqrt(np.sum((r0[j] - r_origin[j]) ** 2))
            if rDisp > 20.0:  # Where we say we have escaped the well
                particleSub += 1
                if particleDensity == 0:
                    break
                rPops.append(j)
        particleDensity = particleDensity - particleSub

        for k in reversed(rPops):
            r0 = list(r0)
            r0.pop(
                k
            )  # Leave the for loop because this particle does not need to evolve.

        densityList.append([time, particleDensity])

        # Inject more particles if we are at 1 particle left
        if particleDensity <= 1:
            for i in range(particleDensity, 2):  # add three particles
                randomY = round(random.uniform(-3.0, 5.0), 1)
                randomZ = round(random.uniform(-6.0, 6.0), 1)
                r0.append([0.0, randomY, randomZ])
                r_origin.append(r0[i - 1])
            particleDensity += 2 - particleDensity

    ##### Plot results
    if save_plots == True:
        # Create Save Path
        Path("plots/" + str(method_dir) + str(orbit_num)).mkdir(
            parents=True, exist_ok=True
        )

        fig, ax = plt.subplots()
        tt = np.array(list(range(0, len(energy)))) * dt
        ax.plot(tt, energy / abs(energy[0]))
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative Energy")
        ax.set_title("Energy vs Time with " + num_method + " Results")
        plt.subplots_adjust(bottom=0.18, left=0.18)
        fig.savefig(
            "plots/" + str(method_dir) + str(orbit_num) + "/energy" + str(dt) + ".png"
        )

        ##### Plot Particle Density over time

        fig, ax = plt.subplots()
        times = [x[0] for x in densityList]
        densities = [x[1] for x in densityList]
        ax.plot(times, densities)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Particle Density")
        ax.set_title("Particle Density in the Mirror with  " + num_method + " Results")
        plt.subplots_adjust(bottom=0.18, left=0.18)
        fig.savefig(
            "plots/" + str(method_dir) + str(orbit_num) + "/density" + str(dt) + ".png"
        )

        ##### Plot Magnetic Field Lines

        fig, ax = plt.subplots()
        BY = np.array(B_field)[:, 1]
        BZ = np.array(B_field)[:, 2]
        y = np.arange(-10.0, 10.0, 0.1)  # create a grid of points from y = -10 to 10
        z = np.arange(-10.0, 10.0, 0.1)  # create a grid of points from z = -10 to 10
        Y, Z = np.meshgrid(y, z)
        ilen, jlen = np.shape(
            Y
        )  # define the length of the dimensions, for use in iteration
        Bf = np.zeros((ilen, jlen, 3))  # set the points to 0

        for i in range(
            0, ilen
        ):  # iterate through the grid, setting each point equal to the magnetic field value there
            for j in range(0, jlen):
                Bf[i, j] = ams.calc_B_field(0.0, Y[i, j], Z[i, j], dt)

        ax.streamplot(Y, Z, Bf[:, :, 1], Bf[:, :, 2])
        ax.set_xlabel("BY")
        ax.set_ylabel("BZ")
        ax.set_title("Magnetic Field with Mirrors")
        plt.subplots_adjust(bottom=0.18, left=0.18)
        fig.savefig(
            "plots/" + str(method_dir) + str(orbit_num) + "/Bfield_" + str(dt) + ".png"
        )

        #### Plot the trajectories over the field lines
        plt.streamplot(Y, Z, Bf[:, :, 1], Bf[:, :, 2], color="black")
        for i in range(particleDensity0):
            plt.plot(np.array(pos[i])[:, 1], np.array(pos[i])[:, 2])
        plt.xlim(-10.0, 10.0)
        plt.ylim(-10.0, 10.0)
        plt.xlabel("$y position$")
        plt.ylabel("$z position$")
        plt.title("Particle Trajectory within a 'Magnetic Bottle'")
        fig.savefig(
            "plots/"
            + str(method_dir)
            + str(orbit_num)
            + "/BfieldwTraj_"
            + str(dt)
            + ".png"
        )

        ##### Plot Magnetic Field over time

        fig, ax = plt.subplots()
        Bmag = [np.sqrt(x[0] ** 2 + x[1] ** 2 + x[2] ** 2) for x in B_field]
        tt = np.array(list(range(0, len(Bmag)))) * dt
        ax.plot(tt, Bmag)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Magnetic Field")
        ax.set_title("Magnetic Field over Time with " + num_method + " Results")
        plt.subplots_adjust(bottom=0.18, left=0.18)
        fig.savefig(
            "plots/" + str(method_dir) + str(orbit_num) + "/Bmag_" + str(dt) + ".png"
        )

        ##### Plot Magnetic moment over time

        fig, ax = plt.subplots()
        tt = np.array(list(range(0, len(mu_list))))
        ax.plot(tt, mu_list)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Magnetic Moment")
        ax.set_title("Magnetic Moment over Time with " + num_method + " Results")
        plt.subplots_adjust(bottom=0.18, left=0.18)
        fig.savefig(
            "plots/" + str(method_dir) + str(orbit_num) + "/mu_" + str(dt) + ".png"
        )


if __name__ == "__main__":
    print("Running Magnetic Bottle Particle Confinement Simulation...")
    accel_calc = input(
        "Which acceleration calculation for this run, "
        + "1/r^2, constant B field, dynamic B field? (r2/const/dynamic) :"
    )
    num_method = input(
        "Which numerical method for this run? "
        + "(euler/semi-implicit_euler/euler_richardson/rk4): "
    )
    save_plots_inp = input("Save plots this run? (y/n): ")
    yes_list = ["y", "Y", "yes", "Yes", "YES", "1"]
    no_list = ["n", "N", "no", "No", "NO", "0"]
    match save_plots_inp:
        case y if y in yes_list:
            save_plots = True
        case n if n in no_list:
            save_plots = False

    # Run solver with helper functions
    run_solver(accel_calc, num_method, save_plots)
