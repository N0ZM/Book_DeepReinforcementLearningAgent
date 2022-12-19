import matplotlib.pyplot as plt


WIDTH = 400
HEIGHT = 200

def plot_hist(frame, history, ax):
    if frame < len(history):
        plt.cla()
        plt.xlim(400)
        plt.ylim(200)
        ax.invert_xaxis()

        hist = history[frame]
        plt.scatter(hist[0].x, hist[0].y)
        plt.scatter(hist[1].x, hist[1].y)
        plt.scatter(hist[2].x, hist[2].y)
