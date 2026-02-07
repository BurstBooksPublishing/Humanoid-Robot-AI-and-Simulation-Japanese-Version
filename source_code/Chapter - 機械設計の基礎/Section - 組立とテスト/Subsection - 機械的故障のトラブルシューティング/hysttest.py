# connect to robot API (placeholder)
robot = connect_robot()  # establish comms
joint = "knee_left"      # target joint
amp = 0.05               # small rad amplitude
rate = 100               # Hz logging

# perform sweep: positive then negative torque
robot.set_mode('torque')                # low-level control mode
robot.command_torque(joint, +0.5)      # hold small positive torque
time.sleep(0.5)                         # settle
theta_pos = robot.read_encoder(joint)   # measure angle

robot.command_torque(joint, -0.5)      # hold small negative torque
time.sleep(0.5)                         # settle
theta_neg = robot.read_encoder(joint)   # measure angle

# compute and log hysteresis
delta_theta = theta_pos - theta_neg
log = {'joint':joint,'theta_pos':theta_pos,'theta_neg':theta_neg,'delta_theta':delta_theta}
save_log(log)                           # persistent storage
print("Delta theta (rad):", delta_theta)