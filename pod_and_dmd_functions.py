"""
This python script is to calculate Proper Orthogonal Decomposition(POD) or Dynamic Mode Decomposition(DMD) based on CFD
calculation results. The script is based on Modred package. It is worth noting that Modred stopped undating in 2017.
Thus the corresponding Python should be Python 3.60 to install Modred
By Meng Qinghe(Jonathan Meng) in September, 2020
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import modred as mr


def main():
    draw_dmd_accumulated_mode_contour(0, 40)




# load_data function is to load the simulation data.
# The location could be set to only a part of the simulation results optionally. Currently it is set to -0.2<x<1,
# -0.3<y<0.3, z<0.002
def load_data():
    file_names = ['G:\\origincase\\restart_flow_'+str(i).zfill(5)+'.csv' for i in range(2, 1001)]
    for i, file_name in enumerate(file_names):
        with open(file_name) as file:
            data = pd.read_csv(file, usecols=['x', 'y', 'z', 'Velocity_y'])
            data1 = data.loc[(data['z'] < 0.011) &(data['z'] > 0.007) & (data['y'] > -0.3) & (data['y'] < 0.3) & (data['x'] > -0.2) &
                             (data['x'] < 1)]
            if i == 0:
                data_accumulate = np.array([data1['Velocity_y'].values.tolist()]).T
                data_location = data1[['x', 'y', 'z']].values
                np.savetxt('data_location.txt', data_location)
            else:
                data_accumulate = np.append(data_accumulate, np.array([data1['Velocity_y'].values.tolist()]).T, axis=1)
        print('reading'+str(i))
    np.savetxt('data_loaded.txt', data_accumulate)


# Function get_location is to get the mesh grid locations to realize the result visualization. Note that the location
# should have already been located in LoadData function. This function is only in case for accidents. For example you
# lost your data_location.txt and don't want to load whole data base again.
# The location could be set to only a part of the simulation results optionally. Currently it is set to -0.2<x<1,
# -0.3<y<0.3, z<0.002
# Specify the filename of the simulation result
def get_location(file_name):
    with open(file_name) as file:
        data = pd.read_csv(file, usecols=['x', 'y', 'z', 'Velocity_x'])
        data1 = data.loc[(data['z'] < 0.002) & (data['y'] > -0.3) & (data['y'] < 0.3) & (data['x'] > -0.2) & (data['x'] < 1)]
        data_location = data1[['x', 'y', 'z']].values
        np.savetxt('data_location.txt', data_location)


# Function calculate_velocity_magnitude is to calculate velocity magnitude.
def calculate_velocity_magnitude():
    with open('data_loaded_x.txt') as file_y:
        data_y = np.loadtxt(file_y)
    with open('data_loaded_y.txt') as file_x:
        data_x = np.loadtxt(file_x)
    data_xy = np.sqrt(data_y*data_y+data_x*data_x)
    np.savetxt('data_calculated_xy.txt', data_xy)


# calculate_pod function is to calculate POD modes. The detailed information of this POD method could be seen on the
# modred guides. Note that the average information in the flow field has already been taken away in this function
def calculate_pod(saved_modes_number):
    with open('data_loaded.txt') as file:
        data = np.loadtxt(file)
        data_accumulate = data - data.mean(axis=1, keepdims=True)
    modes, eigen_values = mr.compute_POD_matrices_snaps_method(data_accumulate, list(range(saved_modes_number)))
    np.savetxt('modes.txt', modes)
    np.savetxt('eigen_values.txt', eigen_values)


# calculate_bpod function is to calculate bpod modes based on 2 variables. The detailed information of this POD method
# could be seen on the modred guides.
def calculate_bpod(direct_file, adjoint_file, saved_modes_number):
    with open(direct_file) as file:
        data = np.loadtxt(file)
        data_accumulate = data - data.mean(axis=1, keepdims=True)
    with open(adjoint_file) as file1:
        data1 = np.loadtxt(file1)
        data_accumulate1 = data1 - data1.mean(axis=1, keepdims=True)
    d_modes, a_modes, eigen_values = mr.compute_BPOD_matrices(data_accumulate, data_accumulate1,
                                                              list(range(saved_modes_number)),
                                                              list(range(saved_modes_number)))
    eigen_values = eigen_values/eigen_values.sum()
    np.savetxt('modes.txt', d_modes)
    np.savetxt('eigen_values.txt', eigen_values)


def calculate_dmd():
    with open('data_calculated_xy.txt') as file:
        data1 = np.loadtxt(file)
        data = data1 - data1.mean(axis=1, keepdims=True)
    exact_mode, projected_mode, eigen_values, \
    spectural_coefficients = mr.dmd.compute_DMD_matrices_direct_method(data, list(range(997)))
    np.savetxt('direct_dmd_exact_mode.txt', exact_mode)
    np.savetxt('direct_dmd_projected_mode.txt', projected_mode)
    np.savetxt('direct_dmd_eigen_values.txt', eigen_values)
    np.savetxt('direct_dmd_spectural_coefficients.txt', spectural_coefficients)


# Function draw_mode_contour is to draw single mode contour. You can either show contour or save contour.
# The input of the function is the mode number to show. NOTE that mode starts at 0.
# That is, the first mode is mode_number=0
def draw_mode_contour(mode_number):
    with open('mode_reduced.txt') as file:
        data_modes = np.loadtxt(file)
    with open('data_location.txt') as file:
        data_location = np.loadtxt(file)
    mode = np.array([data_modes[:, mode_number]]).T
    data = np.append(data_location, mode, axis=1)
    data1 = pd.DataFrame(data, columns=['x', 'y', 'z', 'mode'])
# Although we have already selected part nodes to calculate modes, you can still add restrictions to only visualize
# part region of the modes
    data2 = data1[(data1['z'] < 0.008)].sort_values(by=['x', 'y'])
    plt.tricontourf(data2['x'].values, data2['y'].values, data2['mode'].values)
    plt.show()
    # plt.savefig('mode_fig'+str(mode_number)+'.png')
    # plt.close('all')

def draw_dmd_mode_contour(mode_number):
    with open('direct_dmd_exact_mode_reduced.txt') as file:
        data_modes = np.loadtxt(file, dtype=np.complex)
    with open('data_location.txt') as file:
        data_location = np.loadtxt(file)
    mode = np.array([data_modes.real[:, mode_number]]).T
    data = np.append(data_location, mode, axis=1)
    data1 = pd.DataFrame(data, columns=['x', 'y', 'z', 'mode'])
# Although we have already selected part nodes to calculate modes, you can still add restrictions to only visualize
# part region of the modes
    data2 = data1[(data1['z'] < 0.008)].sort_values(by=['x', 'y'])
    plt.tricontourf(data2['x'].values, data2['y'].values, data2['mode'].values)
    plt.show()

def draw_dmd_accumulated_mode_contour(mode_start, mode_end):
    with open('direct_dmd_exact_mode_reduced.txt') as file:
        data_modes = np.loadtxt(file, dtype=np.complex)
    with open('data_location.txt') as file:
        data_location = np.loadtxt(file)
    mode = np.array([data_modes.real])
    print(mode.shape)
    mode1 = np.sum(mode[:,:, mode_start:mode_end], axis=2)/(mode_end-mode_start+1)
    print(mode1.shape)
    data = np.append(data_location, mode1.T, axis=1)
    print(data_location.shape)
    data1 = pd.DataFrame(data, columns=['x', 'y', 'z', 'mode'])
# Although we have already selected part nodes to calculate modes, you can still add restrictions to only visualize
# part region of the modes
    data2 = data1[(data1['z'] < 0.008)].sort_values(by=['x', 'y'])
    plt.tricontourf(data2['x'].values, data2['y'].values, data2['mode'].values)
    plt.show()


# Function draw_accumulated_contour draw the accumulated several modes
def draw_accumulated_contour(mode_start, mode_end):
    with open('modes.txt') as file:
        data_modes = np.loadtxt(file)
    with open('data_location.txt') as file:
        data_location = np.loadtxt(file)
    mode1 = np.sum(data_modes[:, mode_start:mode_end], axis=1)/(mode_end-mode_start+1)
    mode = np.array([mode1]).T
    data = np.append(data_location, mode, axis=1)
    data1 = pd.DataFrame(data, columns=['x', 'y', 'z', 'mode'])
    data2 = data1[(data1['z'] < 0.008)].sort_values(by=['x', 'y'])
    plt.tricontourf(data2['x'].values, data2['y'].values, data2['mode'].values)
    plt.show()
    # plt.savefig('mode'+str(mode_start)+'to'str(mode_end)'.png')


def calculate_bpod_v_magnitude_mode():
    data_y = np.loadtxt('modes_adjoint.txt')
    data_x = np.loadtxt('modes_direct.txt')
    data_v_magnitude = np.sqrt(data_x*data_x+data_y*data_y)
    np.savetxt('modes_v_magnitude.txt', data_v_magnitude)


def calculate_pod_then_magnitude():
    data_y = np.loadtxt('modes_U.txt')
    data_x = np.loadtxt('modes_V.txt')
    data_v_magnitude = np.sqrt(data_x*data_x+data_y*data_y)
    np.savetxt('modes_v_magnitude.txt', data_v_magnitude)


if __name__ == '__main__':
    main()