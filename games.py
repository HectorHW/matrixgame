#считывание из файла data.txt
#матрица 2xm в 2 строки


import matplotlib.pyplot as plt
import numpy as np
import sys
l = list(open('data.txt'))
l = [list(map(float, row.strip().split())) for row in l[:2]]
assert len(l)==2
from matplotlib.pyplot import figure
figure(num=None, figsize=(16, 12), dpi=160, facecolor='w', edgecolor='k')

matrix = np.array(l)

alpha = np.max(np.min(matrix, axis=1))
beta = np.min(np.max(matrix, axis=0))

if alpha==beta:
    print('игра имеет седловую точку')
    sys.exit(0)

def compute(a, j, p):
    return a[0,j]*p + a[1,j]*(1-p)

X = np.linspace(0,1,1000)
lines = [compute(matrix, i, X) for i in range(matrix.shape[1])]
Y = compute(matrix, 0, X)
for i, line in enumerate(lines):
    plt.plot(X,line, linewidth=3)
    plt.text(X[0], line[0]+0.2, str(i+1), fontsize=25)


stack_y = np.stack(lines)

minmat = np.min(stack_y, axis=0)
plt.plot(X, minmat, color='black', linewidth=5)
max_idx = np.argmax(minmat)
max_point_x, max_point_y = X[max_idx], minmat[max_idx]
plt.scatter(max_point_x, max_point_y, s=400, color='green')
answer_lines = []
for i, line in enumerate(lines):
    if (line[max_idx] - max_point_y)**2<1e-2:
        answer_lines+=[i]


#print(max_point_x, max_point_y)
print(f"линии под номерами {answer_lines[0]+1} и {answer_lines[1]+1}")
assert len(answer_lines)>=2
j1 = answer_lines[0]
j2 = answer_lines[1]
#истинное решение - пересечение 2 прямых
a = matrix
true_x = (a[1,j2]-a[1,j1])/(a[0,j1]-a[0,j2]-a[1,j1]+a[1,j2])
true_y = compute(matrix, j1, true_x)
mn = np.min(stack_y)
plt.text(0.2,0.05,f"линии под номерами {answer_lines[0]+1} и {answer_lines[1]+1}\nx = {true_x}, y = {true_y}", fontsize=25, transform=plt.gcf().transFigure)
plt.savefig('figure.png', dpi=160)
plt.show()
print(f"x = {true_x}, y = {true_y}")