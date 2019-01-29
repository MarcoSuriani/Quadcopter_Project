import csv
import numpy as np
import matplotlib.pyplot as plt

def plot_results(agent, task, file_output='data.txt'):
    labels = ['time', 'x', 'y', 'z', 'phi', 'theta', 'psi', 'x_velocity',
              'y_velocity', 'z_velocity', 'phi_velocity', 'theta_velocity',
              'psi_velocity', 'rotor_speed1', 'rotor_speed2', 'rotor_speed3', 'rotor_speed4']
    results = {x : [] for x in labels}

    # Run the simulation, and save the results.
    with open(file_output, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(labels)
        state = agent.reset_episode()
        done = False
        while not(done):
            rotor_speeds = agent.act(state)
            state, _, done = task.step(rotor_speeds)
            to_write = [task.sim.time] + list(task.sim.pose) + list(task.sim.v) + list(task.sim.angular_v) + list(rotor_speeds)
            for ii in range(len(labels)):
                results[labels[ii]].append(to_write[ii])
            writer.writerow(to_write)

    # the pose, velocity, and angular velocity of the quadcopter at the end of the episode
    print('Position (x, y, z):\t', task.sim.pose[0:3])
    print('Velocity (x, y, z):\t', task.sim.v)
    print('Euler angles:\t\t', task.sim.pose[3:6])
    print('Angular velocity:\t', task.sim.angular_v)

    plt.figure()
    plt.plot(results['time'], results['x'], label='x')
    plt.plot(results['time'], results['y'], label='y')
    plt.plot(results['time'], results['z'], label='z')
    plt.legend()
    _ = plt.ylim()

    plt.figure()
    plt.plot(results['time'], results['x_velocity'], label='x_hat')
    plt.plot(results['time'], results['y_velocity'], label='y_hat')
    plt.plot(results['time'], results['z_velocity'], label='z_hat')
    plt.legend()
    _ = plt.ylim()
    
    plt.figure()
    plt.plot(results['time'], results['phi'], label='phi')
    plt.plot(results['time'], results['theta'], label='theta')
    plt.plot(results['time'], results['psi'], label='psi')
    plt.legend()
    _ = plt.ylim()

    plt.figure()
    plt.plot(results['time'], results['phi_velocity'], label='phi_velocity')
    plt.plot(results['time'], results['theta_velocity'], label='theta_velocity')
    plt.plot(results['time'], results['psi_velocity'], label='psi_velocity')
    plt.legend()
    _ = plt.ylim()

    plt.figure()
    plt.plot(results['time'], results['rotor_speed1'], label='Rotor 1 rev/s')
    plt.plot(results['time'], results['rotor_speed2'], label='Rotor 2 rev/s')
    plt.plot(results['time'], results['rotor_speed3'], label='Rotor 3 rev/s')
    plt.plot(results['time'], results['rotor_speed4'], label='Rotor 4 rev/s')
    plt.legend(loc='upper left', bbox_to_anchor=(-.1, 1.15), ncol=4)
    _ = plt.ylim()