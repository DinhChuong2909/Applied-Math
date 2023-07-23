import numpy as np
import matplotlib.pyplot as plt

def plot_ellipse(a, b, h, k, theta, color):
    t = np.linspace(0, 2*np.pi, 100)
    x = a * np.cos(t) * np.cos(theta) - b * np.sin(t) * np.sin(theta) + h
    y = a * np.cos(t) * np.sin(theta) + b * np.sin(t) * np.cos(theta) + k
    plt.plot(x, y, color=color)

def plot_symmetric_ellipses():
    fig, ax = plt.subplots()

    # Tham số của hình elip
    a = 6  # bán trục lớn
    b = 3  # bán trục nhỏ
    h = 0  # tọa độ tâm x
    k = 0  # tọa độ tâm y
    theta = np.pi/4  # góc quay (trong radian)
    
    # Vẽ hình elip gốc
    plot_ellipse(a, b, h, k, theta, 'blue')

    # Vẽ hình elip đối xứng
    plot_ellipse(b, a, h, k, theta, 'red')

    # Đặt giới hạn trục x và y
    ax.set_xlim(-5, 5)
    ax.set_ylim(-5, 5)

    # Đặt tiêu đề và hiển thị đồ thị
    plt.title('Symmetric Ellipses')
    plt.grid(True)
    plt.show()

# Gọi hàm để vẽ hai elip đối xứng nhau
plot_symmetric_ellipses()
