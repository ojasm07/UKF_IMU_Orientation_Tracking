import numpy as np
from scipy import io
from quaternion import Quaternion
import math

#data files are numbered on the server.
#for exmaple imuRaw1.mat, imuRaw2.mat and so on.
#write a function that takes in an input number (1 through 6)
#reads in the corresponding imu Data, and estimates
#roll pitch and yaw using an unscented kalman filter

def forward_pass(Pi, Xi, Q, time_step):
    n = np.shape(Pi)[0]
    vec_y = np.zeros((7, 12))
    square_root = np.sqrt(n) * np.linalg.cholesky((Pi + Q))
    square_root = np.hstack((square_root, -square_root))
    quat_Wi = square_root[:3, :]
    wWi = square_root[3:, :]

    qd = Quaternion()
    w_k = Xi[4:7]
    x1 = wWi.shape[1] 
    orient_k = Quaternion(Xi[0], Xi[1:4])

    for i in range(x1):
        q1 = Quaternion()
        axis_angle1 = quat_Wi[:, i].reshape(-1)
        q1.from_axis_angle(axis_angle1)
        qk1 = orient_k.__mul__(q1)
        qd.from_axis_angle(w_k*time_step)
        q_k_ = qk1.__mul__(qd)
        y_i = np.hstack([q_k_.q, w_k + wWi[:, i]])
        vec_y[:, i] = y_i

    # print('vec_y:', vec_y.shape)

    # def gd inputs
    s = vec_y[0,0]
    # print(',s:' , s.shape)
    vec_y = vec_y.T
    # print(',s:' , s.shape)
    
    v = vec_y[0, 1:4].reshape(3,)
    # print('v:',  v.shape)
    qm = Quaternion(s, v)
    quatiter = vec_y[:, 0:4]
    x2 = quatiter.shape[0]
    # print('quatiter:', quatiter.shape)
    # qm = q0
    ei = np.zeros((quatiter.shape[0], 3)) 
    max_iter = 100
    error_quat = Quaternion()

    for i in range(max_iter):
        for i in range(x2):
            q2 = Quaternion(quatiter[i, 0], quatiter[i, 1:4])
            qdash = q2.__mul__(qm.inv())
            qscal = qdash.scalar()
            z1 = qscal - 1.0
            z2 = qscal + 1.0
            if qscal > 1.0 and abs(z1) < 1e-4:
                qdash = Quaternion(1.0, qdash.vec())   
            elif qscal < -1.0 and abs(z2) < 1e-4:
                qdash = Quaternion(-1.0, qdash.vec())

            ei[i, :] = qdash.axis_angle()

        barycentric_mean_e = np.mean(ei, axis=0)
        error_quat.from_axis_angle(barycentric_mean_e)
        qm = error_quat.__mul__(qm)

        if np.all(abs(barycentric_mean_e) < 1e-2):
            break

    #def inputs
    angvel = vec_y[:, 4:7]
    ang_vel_mean = np.mean(angvel, axis=0)
    # print('ang_vel_mean:', ang_vel_mean.shape)
    # exit()
    Xi_bar_quantity = np.hstack((qm.q, ang_vel_mean))
    # print('Xi_bar_quantity:', Xi_bar_quantity.shape)
    # exit()
    wW_bar = angvel - np.mean(angvel, axis=0)

    #def propogate func
    wi = np.hstack([ei, wW_bar])
    s0 = wi.shape[0]
    Pi_quantity = (wi.T @ wi)/s0

    return Xi_bar_quantity, Pi_quantity, barycentric_mean_e, vec_y, wi

def update_X_and_P(Xi, Pi, vec_y, wi, R, accel, gyro):
    # print(mean_k.shape)
    # print(Sigma_k.shape)
    # print(state_i.shape)
    # exit()

    #compute measurement_model
    n = np.shape(vec_y)[0]
    zi = np.zeros((n, 6))
    n = np.shape(zi)[0]
    for i in range(n):
        xk = vec_y[i, :]
        # s = vec_y[i, 0]
        # v = vec_y[i, 1:4]
        quat_ki = Quaternion(xk[0], xk[1:4])
        acceleration_due_to_gravity =  [0, 0, 9.81]
        gravity = Quaternion(0, acceleration_due_to_gravity)
        inverse_q_k = quat_ki.inv()
        gravity_inv = (inverse_q_k).__mul__(gravity.__mul__(quat_ki))
        # print(gravity_inv)
        # exit()
        zi[i] = np.hstack([gravity_inv.vec(), vec_y[i, 4:7]])
    mean_z_estimate = np.mean(zi, axis=0)

    d = zi - mean_z_estimate
    # print('d:', d.shape)
    Pxzi = (wi.T @ (d)) / wi.shape[0]
    # print('Pxzi:', Pxzi.shape)
    # dnxt = d.T @ d
    Pzzi = (d.T @ d)/12
    Pyyi = Pzzi + R
    # print('Pyyi:', Pyyi.shape)

    #KG
    KG = Pxzi @ np.linalg.inv(Pyyi)
    #innovation
    sensor_data = np.hstack((accel, gyro)) #shape of accel and gyro       
    # vk = sensor_data - mean_z_estimate

    #update
    xo = KG @ (sensor_data - mean_z_estimate)
    # print('xo:', xo.shape)
    axis_angle = xo[0:3]
    xo_bar = Quaternion()
    # print('xo:', xo[3:6].shape)
    xo_bar.from_axis_angle(axis_angle)
    qm = xo_bar.__mul__(Quaternion(Xi[0], Xi[1:4]))
    # print('qm:', qm.q.shape)
    # print('Xi:', Xi[4:7].shape)
    # exit()
    wm = Xi[4:7] + xo[3:6]
    Xi = np.hstack([qm.q, wm])
    # print('Xi:', Xi.shape)

    Pi = Pi - KG @ Pyyi @ KG.T
    # print('Pi:', Pi.shape)

    return Xi, Pi

def accelerometer_scale(acceleration_input):
    acceleration = acceleration_input.astype('float64')

    beta_val = np.array([505, 501, 501])
    alpha_val = np.array([33.5, 33.5, 33.5]) #Unit = mV/g

    beta = np.ones_like(acceleration) * beta_val[:, None]
    alpha = np.ones_like(acceleration) * alpha_val[:, None]

    acceleration_values = (acceleration - beta) * (3300 / (1023 * alpha)) #* 9.81    #Multiply by 9.81 to convert to m/s^2

    axis_correction = np.array([-1,-1,1])
    acceleration_values = acceleration_values * axis_correction[:, None]

    return acceleration_values.T

def gyroscope_scale(gyro_data, alpha=350, beta=250):
    gyro_data = gyro_data.astype('float64')
    xgyro = gyro_data[1]
    ygyro = gyro_data[2]
    zgyro = gyro_data[0]
    gyro = np.vstack((xgyro, ygyro, zgyro))

    beta_val = np.array([370, 371, 370])
    alpha_val = np.array([200, 200, 200])

    beta = np.ones_like(gyro) * beta_val[:, None]
    alpha = np.ones_like(gyro) * alpha_val[:, None]

    gyro_values = (gyro - beta) * (3300 / (1023 * alpha))

    return gyro_values.T

# def UKF_INIT():
#     q0 = np.array([1, 0, 0, 0])
#     w0 = np.ones(3)
#     X0 = (np.hstack((q0, w0)))
#     # Change the constant values
#     R0 = 1.5 * np.eye((6)) 
#     Q0 = 0.8 * np.eye((6)) #Co-variance of the sigma points 
#     P0 = 0.8 * np.eye((6)) #Mean and covariance of the initial state
#     return P0, X0, Q0, R0

def estimate_rot(data_num=1):
    #load data
    imu = io.loadmat('imu/imuRaw'+str(data_num)+'.mat')
    # vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    accel = imu['vals'][0:3,:]
    gyro = imu['vals'][3:6,:]
    T = np.shape(imu['ts'])[1]
    imu_time_step = imu['ts'][0,:]

    # your code goes here
    accel_calib = accelerometer_scale(accel)
    gyro_calib = gyroscope_scale(gyro)

    #Initialize
    Q = 0.1 * np.eye(6,6)
    Q[0,0] *= 0.001
    Q[1,1] *= 0.001
    Q[2,2] *= 0.001
    Q[3,3] *= 15
    Q[5,5] *= 15
    Q[4,4] *= 15
    
    q0 = Quaternion().q
    R = 3 * np.eye(6,6)
    R[4,4] *= 2
    R[5,5] *= 2
    w0 = 0.6 * np.ones(3)

    Xi = (np.hstack((q0, w0)))
    s = Xi[0]
    v = Xi[1:4]
    quat_0 = Quaternion(s, v)
    euler_0 = quat_0.euler_angles()
    Pi = 0.3 * np.eye(6,6)

    roll = np.zeros((T,))
    pitch = np.zeros((T,))
    yaw = np.zeros((T,))

    roll[0] = euler_0[0]
    pitch[0] = euler_0[1]
    yaw[0] = euler_0[2]

    for i in range(1, T):
        time_step = imu_time_step[i] - imu_time_step[i-1]

        accel_vals = accel_calib[i-1]
        gyro_vals = gyro_calib[i-1]

        Xi, Pi, e, vec_y, wi = forward_pass(Pi, Xi, Q, time_step) 
        Xi, Pi = update_X_and_P(Xi, Pi, vec_y, wi, R, accel_vals, gyro_vals)
        # s = Xi[0]
        # v = Xi[1:4]
        euler = Quaternion(Xi[0], Xi[1:4]).euler_angles()
        roll[i] = euler[0]
        pitch[i] = euler[1]
        yaw[i] = euler[2]

        # print('roll:', roll[i], 'pitch:', pitch[i], 'yaw:', yaw[i])
        # print()
        # exit()

    # roll, pitch, yaw are numpy arrays of length T
    return roll,pitch,yaw


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    data_num=1
    # vicon = io.loadmat('ESE 650 HW 2/vicon/viconRot'+str(data_num)+'.mat')
    vicon = io.loadmat('vicon/viconRot'+str(data_num)+'.mat')
    roll, pitch, yaw = estimate_rot(data_num=1)

    # print(roll[:10])
    # print(pitch[:10])
    # print(yaw[:10])

    viconMatrix = vicon['rots']
    vicon_roll = np.arctan2(viconMatrix[2,1,:],viconMatrix[2,2,:])
    vicon_pitch = np.arctan2(-viconMatrix[2,0],np.sqrt(viconMatrix[2,1,:]**2+viconMatrix[2,2,:]**2))
    vicon_yaw = np.arctan2(viconMatrix[1, 0],viconMatrix[0, 0])

    plt.figure('Roll')
    plt.plot(roll)
    plt.plot(vicon_roll)

    plt.figure('Pitch')
    plt.plot(pitch)
    plt.plot(vicon_pitch)

    plt.figure(3)
    plt.plot(yaw)
    plt.plot(vicon_yaw)

    plt.show()
