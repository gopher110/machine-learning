import matplotlib.pyplot as plt

x = [i for i in range(10)]
y = [2*i for i in range(10)]


plt.xlabel('x-axis')
plt.ylabel('y-axis')
plt.scatter(x, y)
# plt.plot(x, y)
plt.show()

