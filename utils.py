from matplotlib import pyplot as plt

def plot_results_line(x, y1, y2, x_label="X", y_label="Y", save_path=None, y1_title="y1", y2_title="y2", y3_title="y3"):
    plt.plot(x,y1, label = y1_title)
    plt.plot(x,y2, label = y2_title)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    if(save_path):
        plt.savefig(save_path)