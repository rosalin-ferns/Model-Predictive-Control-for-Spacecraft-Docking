# basic script for closeloop control using MPC controller 
# on the spacecraft model and implementing the control on spacecraft_plant
# issues : time 

import numpy as np
from spacecraft_plant_coupled import SpacecraftPlant
from mpc_new_attempt1_coupled import MPCCont
from spacecraft_model_coupled import LinearSpacecraftModel
from udampc import UDAMPC
from scipy.linalg import solve_discrete_are
import matplotlib.pyplot as plt
import time
import scipy.linalg as la




# Define simulation parameters
T = 500.0
dt = 2
steps = int(T/dt)

n = 0.0021                      # Mean motion in rad/s (example)
mass = 500                     # kg
J = np.diag([100, 100, 100])             # Moment of inertia [Jx, Jy, Jz]

# Initial and target states
x0 = np.array([
    -50.0, 40.3, -10.5,        # [pos, vel, ang, ang_vel]
    0.01, 0.02, 0.04,        
    0.03, -0.06, 0.01,         
    0.005, 0.003, 0.002])

x_ref = np.zeros(12)
x_ref[0] = -10.0   # target x position



# u_min = np.array([-10.0, -10.0, -10.0, -5.0, -5.0, -5.0])
# u_max = np.array([ 10.0,  10.0,  10.0,  5.0,  5.0,  5.0])
# u_limits = np.abs(u_max)


# def auto_bryson_weights(x0, u_limits):
#     """Generate Q0, R0   for a baic bryson's rule weighting."""
#     x_scale = np.maximum(np.abs(x0), 1e-3)
#     u_scale = np.maximum(u_limits, 1e-3)
#     Q0 = np.diag(1.0 / (x_scale ** 2))
#     R0 = np.diag(1.0 / (u_scale ** 2))
#     # Q[0:3, 0:3] *= 1e5 # extra weight on position
#     # Q[6:9, 6:9] *= 1e-4 # extra weight on angles
#     # Q[9:12, 9:12] *= 1e-3 # extra weight on angular rates
  
#     return Q0, R0


# Q0, R0 = auto_bryson_weights(x0, u_limits)



R = np.diag([1.0, 1.0, 1.0, 10.0, 10.0, 10.0])*1e-1
  # input weights for [Fx, Fy, Fz, Tx, Ty, Tz]
epsilon_R = 1e-6 # regularization for R
R = R + epsilon_R * np.eye(R.shape[0])
N = 45


plant = SpacecraftPlant(n, mass, J, thruster_offset=[0.025, 0.0, 0.0])
model = LinearSpacecraftModel(plant, dt)

A_init, B_init = model.compute(x0, np.zeros(6))

import numpy as np
import scipy.linalg as la

# === 1. Compute discrete-time controllability Gramian ===
# (use the linearized A,B at your current operating point)
try:
    Wc = la.solve_discrete_lyapunov(A_init, B_init @ B_init.T)
except la.LinAlgError:
    # fallback for near-singular A
    Wc = np.eye(A_init.shape[0])

# === 2. Normalize states by Gramian energy ===
# Each state's scaling = 1/sqrt(Gramian energy)
state_scale = np.sqrt(np.abs(np.diag(Wc))) + 1e-8     # avoid divide-by-zero
state_scale = np.sqrt(np.abs(np.diag(Wc))) + 1e-8
state_scale = np.clip(state_scale, 1e-3, 1e3)   # <-- add this
S = 0.6* np.diag(1.0 / state_scale)                      # majorly helped remove oscillations

# === 3. Build normalized Q ===
# After normalization, all states have roughly equal weight (unitless)
Q_norm = S.T @ np.eye(A_init.shape[0]) @ S

# You can bias some groups of states if desired
# (e.g. make position 10×, attitude 5×, velocities 1×)
Q_bias = np.diag([5,5,5, 10,10,10, 50,50,50, 100, 100,100])
Q = Q_bias @ Q_norm @ Q_bias



# Optional global multipliers (tune these two numbers, not the full matrices)
alpha_Q = 20.0
alpha_R = 1.0
Q = alpha_Q * Q*1e1
Q[6:9,6:9] *= 1.5  # extra weight on angles  
Q[9:12,9:12] *= 10.0  # extra weight on angular rates
Q = 0.5 * (Q + Q.T)
Q += 1e-6 * np.eye(Q.shape[0])

R = alpha_R * R

# max_diag_Q = np.max(np.diag(Q))
# if max_diag_Q > 1e6:
#     Q *= 1e6 / max_diag_Q

# min_diag_Q = np.min(np.diag(Q))
# if min_diag_Q < 1e-6:
#     Q += 1e-6 * np.eye(Q.shape[0])



mpc = MPCCont(A_init, B_init, Q, R, N)
mpc.u_min = np.array([-20.0, -20.0, -20.0, -5.0, -5.0, -5.0])
mpc.u_max = np.array([ 20.0,  20.0,  20.0,  5.0, 5.0,  5.0])
udampc = UDAMPC(mpc)
P = solve_discrete_are(A_init, B_init, Q, R)
P = 0.8 * (P + P.T)
P = P + 1e-6 * np.eye(P.shape[0])
K = np.linalg.inv(B_init.T @ P @ B_init + R) @ (B_init.T @ P @ A_init)
eigs = np.linalg.eigvals(A_init - B_init @ K)
print("eig(A-BK):", eigs)




print("cond(A):", np.linalg.cond(A_init))
print("A row norms (min,median,max):", np.sqrt((A_init**2).sum(axis=1)).min(),
      np.median(np.sqrt((A_init**2).sum(axis=1))), np.sqrt((A_init**2).sum(axis=1)).max())
print("cond(Q):", np.linalg.cond(Q))
print("cond(R):", np.linalg.cond(R))
try:
    eigs = np.linalg.eigvalsh(mpc.P)
    print("P eigs min,max:", eigs.min(), eigs.max(), "cond(P):", eigs.max()/eigs.min())
except Exception as e:
    print("Couldn't compute P eigs:", e)

mpc.update_model(A_init, B_init, x0, x_ref)
P = solve_discrete_are(A_init,B_init,Q,R)
K = np.linalg.inv(B_init.T @ P @ B_init + R) @ (B_init.T @ P @ A_init)
u_test = mpc.solve()
print("diagnostic: |u_forces|, |u_torques|, max|u| ->",
      np.linalg.norm(u_test[:3]), np.linalg.norm(u_test[3:]), np.max(np.abs(u_test)))
# also show saturation possibility
print("u_min,u_max:", mpc.u_min, mpc.u_max)

print("Q diag:", np.diag(Q))
print("R diag:", np.diag(R))
 

# Simulate
x = x0.copy()
trajectory = np.zeros((steps+1, 12))
trajectory[0] = x0

start = time.time()

#### for tube ################

u_prev = np.zeros(6)
delay_buffer = [np.zeros_like(u_prev) for _ in range(2)]

for i in range(steps):
    A, B = model.compute(x, u=u_prev)
    mpc.update_model(A, B, x, x_ref)
    P = solve_discrete_are(A, B, Q, R)
    P = 0.5 * (P + P.T)
    P = P + 1e-6 * np.eye(P.shape[0])
    K = np.linalg.inv(B.T @ P @ B + R) @ (B.T @ P @ A)
    u = udampc.tube(A, B, x, x_ref, K, w_bound=0.07)
    du_max = np.array([8.0, 8.0, 8.0, 4.0, 4.0, 4.0])    # tune down if still bad
    u = u.flatten()
    u = np.clip(u, u_prev - du_max, u_prev + du_max)
    u_applied = delay_buffer.pop(0)
    delay_buffer.append(u)  # get the oldest control input
    x = plant.disc(x, u_applied, dt)
    trajectory[i+1] = x
    u_prev = u_applied.copy()
    errors = trajectory - x_ref   # shape (steps+1, 12)






sim_end = time.time()
# print('K:', K)



# Plot
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Left: translation
axs[0].plot(np.arange(steps+1)*dt, trajectory[:,0], label='x')
axs[0].plot(np.arange(steps+1)*dt, trajectory[:,1], label='y')
axs[0].plot(np.arange(steps+1)*dt, trajectory[:,2], label='z')
axs[0].plot(np.arange(steps+1)*dt, trajectory[:,3], '--',label='vx')
axs[0].plot(np.arange(steps+1)*dt, trajectory[:,4], '--', label='vy')
axs[0].plot(np.arange(steps+1)*dt, trajectory[:,5], '--', label='vz')
#axs[0].axhline(y=x_ref[0], color='r', linestyle='--', label='target x')
axs[0].set_xlabel('Time (s)')
axs[0].set_ylabel('Position (m)/Velocity (m/s)')
axs[0].legend()
axs[0].grid(True)
axs[0].set_title('Translation Position and Velocity over Time')

# Right: rotation
axs[1].plot(np.arange(steps+1)*dt, trajectory[:,6], label='φ (roll)')
axs[1].plot(np.arange(steps+1)*dt, trajectory[:,7], label='θ (pitch)')
axs[1].plot(np.arange(steps+1)*dt, trajectory[:,8], label='ψ (yaw)')       
axs[1].plot(np.arange(steps+1)*dt, trajectory[:,9], '--', label='ωx (rad/s)')
axs[1].plot(np.arange(steps+1)*dt, trajectory[:,10], '--', label='ωy (rad/s)')
axs[1].plot(np.arange(steps+1)*dt, trajectory[:,11], '--', label='ωz (rad/s)')    
#axs[0].axhline(y=x_ref[6], color='r', linestyle='--', label='target angles')
axs[1].set_xlabel('Time (s)')
axs[1].set_ylabel('Angle (rad)/Angular Rate (rad/s)')
axs[1].legend()        
axs[1].grid(True)
axs[1].set_title('Rotation Angles and Angular Rates over Time')

plt.tight_layout()
plt.show()



# ######################ERROR PLOTS#################################

# # fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# # # Translation errors
# # axs[0].plot(np.arange(steps+1)*dt, errors[:,0], label='ex (m)')
# # axs[0].plot(np.arange(steps+1)*dt, errors[:,1], label='ey (m)')
# # axs[0].plot(np.arange(steps+1)*dt, errors[:,2], label='ez (m)')
# # axs[0].set_xlabel("Time (s)")
# # axs[0].set_ylabel("Position Error (m)")
# # axs[0].legend(); axs[0].grid(True)
# # axs[0].set_title("Position Errors vs Time")

# # # Rotation errors
# # axs[1].plot(np.arange(steps+1)*dt, errors[:,6], label='eφ (rad)')
# # axs[1].plot(np.arange(steps+1)*dt, errors[:,7], label='eθ (rad)')
# # axs[1].plot(np.arange(steps+1)*dt, errors[:,8], label='eψ (rad)')
# # axs[1].set_xlabel("Time (s)")
# # axs[1].set_ylabel("Angle Error (rad)")
# # axs[1].legend(); axs[1].grid(True)
# # axs[1].set_title("Rotation Errors vs Time")

# # plt.tight_layout()
# plt.show()


# === ERROR METRICS ===
# Define helper functions
def rms(x):
    return np.sqrt(np.mean(x**2))

def steady_state_error(signal, window=50):
    """Mean error over the last `window` steps."""
    return np.mean(signal[-window:])

# Compute metrics for pitch (θ index = 7) and yaw (ψ index = 8) as example
role_err = errors[:,6]
pitch_err = errors[:,7]
yaw_err   = errors[:,8]

print("\n=== ERROR METRICS ===")
print(f"Max pitch error: {np.max(np.abs(role_err)):.5f} rad")
print(f"RMS pitch error: {rms(role_err):.5f} rad")
print(f"Steady-state pitch error: {steady_state_error(role_err):.5f} rad")

print(f"Max pitch error: {np.max(np.abs(pitch_err)):.5f} rad")
print(f"RMS pitch error: {rms(pitch_err):.5f} rad")
print(f"Steady-state pitch error: {steady_state_error(pitch_err):.5f} rad")

print(f"Max yaw error: {np.max(np.abs(yaw_err)):.5f} rad")
print(f"RMS yaw error: {rms(yaw_err):.5f} rad")
print(f"Steady-state yaw error: {steady_state_error(yaw_err):.5f} rad")



# === INVARIANT SET VISUALIZATION ===
# Compute position and attitude errors
pos_err = np.linalg.norm(errors[:, 0:3], axis=1)
ang_err = np.linalg.norm(errors[:, 6:9], axis=1)



# Define tube bounds (heuristic: from w_bound propagation)
tube_bound_pos = 0.07 / (1 - np.max(np.abs(np.linalg.eigvals(A_init - B_init @ K))))
tube_bound_ang = 0.07 / (1 - np.max(np.abs(np.linalg.eigvals(A_init - B_init @ K))))

print("Invariant bounds: pos =", tube_bound_pos, ", ang =", tube_bound_ang)


plt.figure(figsize=(10,4))
plt.plot(np.arange(steps+1)*dt, pos_err, label='||e_pos||')
plt.plot(np.arange(steps+1)*dt, ang_err, label='||e_ang||')
plt.axhline(y=tube_bound_pos, color='r', linestyle='--', label='Invariant bound (pos)')
plt.axhline(y=tube_bound_ang, color='g', linestyle='--', label='Invariant bound (ang)')
plt.xlabel('Time (s)')
plt.ylabel('Error Norm')
plt.title('State Trajectory Approach and Invariance in Tube MPC')
plt.legend(); plt.grid(True)
plt.show()

