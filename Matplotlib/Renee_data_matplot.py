#!/usr/bin/env python
# coding: utf-8

# In[180]:


import numpy as np
import matplotlib.pyplot as plt


# Add a cell at the end to generate a 2 D graph as follows:
# 
# x holds 1000 values between -4 and 4
# z holds 1000 values between -5 and 5
# y = x^2 + z^3 +6
# plot x and y
# name the plot(i.e.set the title) “Ploynomial_firstname” where firstname is your firstname.
# Give names for the x and y axis.
# 

# In[181]:


x = np.linspace(-4, 4, 1000)
z = np.linspace(-5, 5, 1000)
y = x**2 + z**3 + 6



# In[182]:


plt.plot(x, y)
plt.title("Polynomial_renee")
plt.xlabel("x")
plt.ylabel("y")


# Add a cell at the end to generate a plot using subplot2grid with the following characteristics:
# 
# 1) A 4 by 4 grid.
# 
# 2) On the first row plot the function x^2 in a dashed green line.
# 
# 3) On the second-row plot two functions, the first function x^3 in yellow color and the second function x^4 spanning three columns in red color.
# 
# 5) On the third-row plot two functions the first X^6 in a dashed blue color and the second is X=x in magna (pink) color.
# 
# 6) On the fourth row plot one function^7 spanning all columns in dotted red.
# 

# In[183]:


fig = plt.figure(figsize=(10, 10))


ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4)
ax1.plot(x, x**2, 'g--')  
ax1.set_title('x^2')

ax2 = plt.subplot2grid((4, 4), (1, 0), colspan=1)
ax2.plot(x, x**3, 'y')  # Yellow color
ax2.set_title('x^3')

ax3 = plt.subplot2grid((4, 4), (1, 1), colspan=3)
ax3.plot(x, x**4, 'r')  # Red color
ax3.set_title('x^4')


ax4 = plt.subplot2grid((4, 4), (2, 0), colspan=2)
ax4.plot(x, x**6, 'b--')  # Dashed blue line
ax4.set_title('x^6')

ax5 = plt.subplot2grid((4, 4), (2, 2), colspan=2)
ax5.plot(x, x, 'm')  # Magenta (pink) color
ax5.set_title('X=x')


ax6 = plt.subplot2grid((4, 4), (3, 0), colspan=4)
ax6.plot(x, x**7, 'r:')  # Dotted red line
ax6.set_title('x^7')


plt.tight_layout()
plt.show()


# On the first graph showing the beautiful point add a new point name new point _firstname and display the coordinates, your figure should look something like this:

# In[184]:


px = 1.5
px2 = 2.5
py = px**2
py2 = px2**2

ax1 = plt.subplot2grid((4, 4), (0, 0), colspan=4)
ax1.plot(x, x**2, 'g--', px, py, "ro", px2, py2, "bo")  
plt.text(px - 0.08, py, "Beautiful point", ha="right", weight="heavy")
plt.text(px, py, "x = %0.2f\ny = %0.2f"%(px, py), rotation=50, color='gray')

plt.text(px2 - 0.08, py2, "New point_renee", ha="right", weight="heavy")
plt.text(px2, py2, "x = %0.2f\ny = %0.2f"%(px2, py2), rotation=50, color='gray')



ax1.set_title('x^2')


# Add a cell to generate a scatter plot of x and y where each contains 300 numbers generated randomly between 3 and 100. Set the scale, alpha and colors as you see suitable
# 

# In[185]:


x = np.random.uniform(3, 100, 300)
y = np.random.uniform(3, 100, 300)

plt.scatter(x, y, alpha=0.6, c='blue', edgecolors='w', s=50)  

plt.title('Scatter Plot')
plt.xlabel('x values')
plt.ylabel('y values')
plt.show()

