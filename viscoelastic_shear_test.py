# Authors: John Naliboff, Mengzhu Yuan

# Runtime instructions
#   Inside interpreter, run with exec(open("viscoelastic_2D_shear_test.py").read())
#   Outside interpreter, run with ipython viscoleastic_2D_shear_test.py

# Load modules
import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib import ticker

# Path to data directory (e.g., solution folder) from current directory
data_directory = 'models/output_viscoelastic_shear_test_2D/solution/'

# String for model name, which is used for the figure name
model_name='shear_test_2D'

# Time steps to analyze 
time_step_start = 0
time_step_final = 100

# Set timestep
time_step = 1.e1

# Total number of .pvtu files
number_time_steps = time_step_final - time_step_start + 1

# Variable to store results
time_sxy = np.zeros([number_time_steps,3])

count = 0

# Loop through all time steps
for t in range(time_step_start,time_step_final+1):

    # Create a string corresponding to the .pvtu file number
    if t<10:
        pvtu_number = '0000' + str(t)
    elif t>=10 and t<100:
        pvtu_number = '000' + str(t)
    elif t>=100 & t<1000:
        pvtu_number = '00' + str(t)
    elif t>=10000:
        pvtu_number = '0' + str(t)

    # Load vtu data (pvtu directs to vtu files)
    reader = vtk.vtkXMLPUnstructuredGridReader()
    reader.SetFileName(data_directory + 'solution-' + pvtu_number + '.pvtu')
    reader.Update()

    # Get the coordinates of nodes in the mesh
    nodes_vtk_array= reader.GetOutput().GetPoints().GetData()

    # Convert nodal vtk data to a numpy array
    nodes_numpy_array = vtk_to_numpy(nodes_vtk_array)

    # Extract x, y and z coordinates from numpy array 
    x,y,z= nodes_numpy_array[:,0] , nodes_numpy_array[:,1] , nodes_numpy_array[:,2]

    # Determine the number of scalar fields contained in the .pvtu file
    number_of_fields = reader.GetOutput().GetPointData().GetNumberOfArrays()

    # Determine the name of each field and place it in an array.
    field_names = []
    for i in range(number_of_fields):
         field_names.append(reader.GetOutput().GetPointData().GetArrayName(i))

    # Determine the index of the field stress_xy
    idx = field_names.index("stress_xy")

    # Extract values of stress_xy
    field_vtk_array = reader.GetOutput().GetPointData().GetArray(idx)
    field_numpy_array = vtk_to_numpy(field_vtk_array)
    stress_xy = np.copy(field_numpy_array)

    # Define min and max x-y-z values
    xmin, xmax = min(x), max(x)
    ymin, ymax = min(y), max(y)
    zmin, zmax = min(z), max(z)

    # Define number of grid points for the interpolation. The original model grid
    # had 51x51 nodal points, be we used the 'Interpolate output = true', which
    # effectively doubles the number of points.
    Nx = 101.
    Ny = 101.

    # Create a new grid for the data interpolation
    xi = np.linspace(xmin, xmax, Nx)
    yi = np.linspace(ymin, ymax, Ny)

    # Interpolate stress_xy values onto the new grid
    sxy = griddata((x, y), stress_xy, (xi[None,:], yi[:,None]), method='cubic') 

    # Find array index corresponding to where x = 50 km and y = 50 km
    val, = np.where(xi==50.e3); x_idx = val[0]
    val, = np.where(yi==50.e3); y_idx = val[0]

    # Store values in array
    time_sxy[count,0] = t * time_step
    time_sxy[count,1] = sxy[y_idx,x_idx]

    # Below we will calculate analytical solution
    # Assume that t is always less than t_max in this model 
    # Equation gotten from 'The role of viscoelasticity in subdecting plates' by Farrington, Moresi, and Capitanio
    #import math
    time = t*time_step
    mu = 1.e10 #shear_modulus 
    eta = 1.e22 #shear_viscosity 
    h = 100.e3
    vel = 0.01
    C_1 = -( (vel**2)*(eta**2)*mu ) / ( (mu**2)*(h**2)+(vel**2)*(eta**2) )

    C_2 = -(vel*h*eta*(mu**2))/((mu**2)*(h**2)+(vel**2)*(eta**2))
    deviatoric_stress_xy = ( (np.e**(-mu/eta*time)) * ( C_2*np.cos(vel*time/h)-C_1*np.sin(vel*time/h) ) ) - C_2


    # Store results
    time_sxy[count,2] = deviatoric_stress_xy

    # Update iteratio count
    count = count + 1 


# Plot results
plt.plot(time_sxy[:,0]/1.e3, time_sxy[:,1]/1.e6, color = '#000000')
plt.plot(time_sxy[:,0]/1.e3, time_sxy[:,2]/1.e6, color = '#FF0000')
plt.title('Viscoleastic 2D Shear Test Benchmark')
plt.xlabel('Time (Kyr)')
plt.ylabel('Shear Stress (MPa)')
plt.grid(True)
plt.savefig(data_directory + model_name + '.png')
plt.show()
plt.close()

# Convert the model time step to a string
time_step_str = str(int(np.log10(time_step)))

# Save results to a file
np.savetxt(data_directory + model_name  + '.txt',time_sxy,fmt='%3.2e %8.7e %8.7e3',header="Time  Numerical  Analytical")

