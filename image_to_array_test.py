import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TKAgg')
#from skimage import io, color


import numpy as np

def main():
    t = np.linspace(0, 4*np.pi, 1000)
    fig1, ax = plt.subplots()
    ax.plot(t, np.cos(t))
    ax.plot(t, np.sin(t))

    inception(inception(fig1))
    plt.show()

def fig2rgb_array(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    a = np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
    print(a)
    print(a.shape)
    a = np.reshape(a, (a.shape[0]*a.shape[1], a.shape[2]))/255
    np.savetxt("foo.csv", a, delimiter=",")
    return a

def inception(fig):
    newfig, ax = plt.subplots()
    ax.imshow(fig2rgb_array(fig))
    return newfig

main()