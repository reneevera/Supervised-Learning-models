#!/usr/bin/env python
# coding: utf-8

# In[516]:


import numpy as np


# Add a cell to create a function and name it  my_function_firstname, where firstname is your first name.
# Let the function return an integer value stored in one byte i.e. ‘int8’ of (4x)*(3y). Where x is the number of rows and y is the number of columns.
#  Use np.fromfunction() to generate  three elements each are two by six using the  my_fuction_firstname.
# 

# In[517]:


def my_function_renee(z, x, y):
    return np.int8((4*x) * (3*y))

result_array = np.fromfunction(my_function_renee, (3, 2, 6), dtype=int)


print(result_array)



# Inspect the code under this section copy it, add a cell to extract values 16,17,18

# In[518]:


b = np.arange(48).reshape(4, 12)
b


# In[519]:


b[1,4:7]


# Inspect the code under this section copy it, then add a cell to iterate over c and print the Boolean values for items equivalent to zeros.

# In[520]:


c = np.arange(24).reshape(2, 3, 4)  

result = np.zeros_like(c, dtype=bool)

for i in range(c.shape[0]):
    for j in range(c.shape[1]):
        for k in range(c.shape[2]):
            result[i, j, k] = (c[i, j, k] == 0)

print(result)


# Inspect the code under this section copy it, then add a cell to create a variable name it q5_firstname where firstname is your firstname and vertically stack q1 and q2 and print the output.

# In[521]:


q1 = np.full((3,4), 1.0)
q2 = np.full((3,4), 2.0)
q3 = np.full((3,4), 3.0)
q5_renee = np.vstack((q1, q2))
q5_renee




# Inspect the code under this section copy it, then add a cell to create a variable name it q8_firstname where firstname is your firstname , concatenate q1 and q3 and print the results.

# In[522]:


q8_renee = np.concatenate((q1, q3), axis=0)
q8_renee


# Inspect the code under this section copy it, then add a cell and create a variable named t_firstname where firstname is your name, let the variable hold any ndaray size 2 by 7 with zero values, print the result then transpose and print the result.

# In[523]:


t_renee = np.zeros((2, 7))
t_renee


# In[524]:


t_renee_t = t_renee.transpose(1, 0)
t_renee_t


# Inspect the code under this section copy it, then  add a cell to create 2 ndarys name the first a1 and the second a2. Both arrays should contain numbers in the range 0 to 8, inclusive . Print a1 and a2. Reshape a1 to a 2 by 4. Reshape a2 to a 4 by 2. Create a new variable a3 _first name where firstname is your first name which holds the dot product  of a1 and a2 name it a3 and print the output of a3_firstname, then the shape of a3_first name.

# In[525]:


a1 = np.arange(1, 9).reshape(2, 4)
a1


# In[526]:


a2 = np.arange(1, 9).reshape(4,2)
a2


# In[527]:


a3_renee = np.dot(a1, a2)
a3_renee


# Add a cell to create a new 4 by 4 ndaray with values between 0 and 15, name the variable that holds the array your first name, print the array and the inverse of the array.

# In[528]:


renee = np.arange(0, 16).reshape(4, 4)
renee


# In[529]:


import numpy.linalg as la
la.inv(renee)   


# Add a cell to create a 4 by 4 identity array.

# In[530]:


renee.dot(la.inv(renee))


# Add a cell to create a 3 by 3 matrix with values generated randomly then printout the determinant of the matrix.

# In[531]:


# Add a cell to create a 3 by 3 matrix with values generated randomly then printout the determinant of the matrix.
renee_ran  = np.random.random((3, 3))
renee_ran


# In[532]:


la.det(renee_ran)


# Add a cell to create a 4 by 4 matrix with values generated randomly, assign the matrix to a variable named e_firstname. Printout the Eigenvalue and eigenvectors of the matrix.

# In[533]:


e_renee = np.random.random((4, 4))
e_renee



# In[534]:


eigenvalues, eigenvectors = la.eig(e_renee)


# In[535]:


eigenvalues


# In[536]:


eigenvectors


# Add a cell to solve the following linear equations:
# 2x+4y+z =12
# 3x+8y+2z =16
# X+2y+3z = 3
# Check the results using the allcolse method.
# 

# In[537]:


coeffs  = np.array([[2, 4, 1], [3, 8, 2], [1, 2, 3]])
depvars = np.array([12, 16, 3])
solution = la.solve(coeffs, depvars)
solution


# In[538]:


coeffs.dot(solution), depvars


# In[539]:


np.allclose(coeffs.dot(solution), depvars)

