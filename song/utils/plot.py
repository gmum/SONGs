import matplotlib.pyplot as plt
import seaborn as sns
import torch


def plot2tensorboard(net, writer, step):
    m0 = torch.softmax(net.graph.M_left, dim=1).detach().cpu().numpy()
    m1 = torch.softmax(net.graph.M_right, dim=1).detach().cpu().numpy()
    for i in range(m0.shape[0]):
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, sharex='all', sharey='all', figsize=(10, 10))
        sns.heatmap(m0[i], cmap="hot", ax=ax1)
        ax1.set_title('left')
        sns.heatmap(m1[i], cmap="hot", ax=ax2)
        ax2.set_title('right')
        writer.add_figure(f'M/graph_{i}', plt.gcf(), step)
        plt.close(fig)
