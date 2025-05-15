#from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


def plot_data(data,labels=None):
    """Affiche des donnees 2D
    :param data: matrice des donnees 2d
    :param labels: vecteur des labels (discrets)
    :return:
    """
    if labels is not None:
        labels = labels.reshape(-1)
    cols,marks = ["red", "green", "blue", "orange", "black", "cyan"],[".","+","*","o","x","^"]
    if labels is None:
        plt.scatter(data[:,0],data[:,1],marker="x")
        return
    for i,l in enumerate(sorted(list(set(labels.flatten())))):
        plt.scatter(data[labels==l,0],data[labels==l,1],c=cols[i],marker=marks[i])

def plot_frontiere(data,f,step=20):
    """Trace un graphe de la frontiere de decision de f
    :param data: donnees
    :param f: fonction de decision
    :param step: pas de la grille
    :return:
    """
    grid,x,y=make_grid(data=data,step=step)
    plt.contourf(x,y,f(grid).reshape(x.shape),colors=('gray','blue'),levels=[-1,0,1])

def make_grid(data=None,xmin=-5,xmax=5,ymin=-5,ymax=5,step=20):
    """Cree une grille sous forme de matrice 2d de la liste des points
    :param data: pour calcluler les bornes du graphe
    :param xmin: si pas data, alors bornes du graphe
    :param xmax:
    :param ymin:
    :param ymax:
    :param step: pas de la grille
    :return: une matrice 2d contenant les points de la grille
    """
    if data is not None:
        xmax, xmin, ymax, ymin = np.max(data[:,0]),  np.min(data[:,0]), np.max(data[:,1]), np.min(data[:,1])
    x, y =np.meshgrid(np.arange(xmin,xmax,(xmax-xmin)*1./step), np.arange(ymin,ymax,(ymax-ymin)*1./step))
    grid=np.c_[x.ravel(),y.ravel()]
    return grid, x, y

def gen_arti(centerx=1,centery=1,sigma=0.1,nbex=1000,data_type=0,epsilon=0.02):
    """Generateur de donnees,
    :param centerx: centre des gaussiennes
    :param centery:
    :param sigma: des gaussiennes
    :param nbex: nombre d'exemples
    :param data_type: 0: melange 2 gaussiennes, 1: melange 4 gaussiennes, 2:echequier
    :param epsilon: bruit dans les donnees
    :return: data matrice 2d des donnnes,y etiquette des donnnees
    """
    if data_type==0:
         #melange de 2 gaussiennes
         xpos=np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//2)
         xneg=np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//2)
         data=np.vstack((xpos,xneg))
         y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))
    if data_type==1:
        #melange de 4 gaussiennes
        xpos=np.vstack((np.random.multivariate_normal([centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([-centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        xneg=np.vstack((np.random.multivariate_normal([-centerx,centerx],np.diag([sigma,sigma]),nbex//4),np.random.multivariate_normal([centerx,-centerx],np.diag([sigma,sigma]),nbex//4)))
        data=np.vstack((xpos,xneg))
        y=np.hstack((np.ones(nbex//2),-np.ones(nbex//2)))

    if data_type==2:
        #echiquier
        data=np.reshape(np.random.uniform(-4,4,2*nbex),(nbex,2))
        y=np.ceil(data[:,0])+np.ceil(data[:,1])
        y=2*(y % 2)-1
    # un peu de bruit
    data[:,0]+=np.random.normal(0,epsilon,nbex)
    data[:,1]+=np.random.normal(0,epsilon,nbex)
    # on mélange les données
    idx = np.random.permutation(range(y.size))
    data=data[idx,:]
    y=y[idx]
    return data,y.reshape(-1, 1)

def normalize_images(X):
    X = X.astype(np.float32)
    X = (X - np.min(X)) / (np.max(X) - np.min(X))
    return X

def plot_classification(X_train, y_train, X_test, y_test, predict, iteration, losses, with_batch=False):
    score_train = (y_train == predict(X_train)).mean()
    score_test = (y_test == predict(X_test)).mean()
    print(f"Train accuracy : {score_train}")
    print(f"Test accuracy : {score_test}")

    fig, axes = plt.subplots(1, 2, figsize=(4, 2))

    plt.sca(axes[0])
    plot_frontiere(X_train, predict, step=100)
    plot_data(X_train, y_train.reshape(-1))
    plt.title("Train")

    plt.sca(axes[1])
    plot_frontiere(X_test, predict, step=100)
    plot_data(X_test, y_test.reshape(-1))
    plt.title("Test")

    plt.tight_layout()
    plt.show()

    if not with_batch:
        plt.plot(np.arange(iteration), losses)
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.title("Loss over iteration")
        plt.show()
        print(np.array(losses).shape)

def plot_iqm(all_losses, worst_best=True):
    # all_losses is a list of lists
    # each inner list contains the losses at each iteration for a specific run
    # The interquartile mean (IQM) is calculated by taking the mean of the values within the interquartile range (IQR)
    # In order to get this mean, we need to take the data from every losses list in all_losses and calculate the IQR for each iteration
    # So the IQM is made for each iteration. The final plot has the length of the longest list in all_losses
    # Then we can plot the IQM for each iteration
    # if worst_best is True, we also plot the worst and best losses based on their last loss
    # code
    iqm_losses = []

    for i in range(len(all_losses[0])):
        iteration_losses = [losses[i] for losses in all_losses]
        sorted_losses = np.sort(iteration_losses)
        q1 = np.percentile(sorted_losses, 25)
        q3 = np.percentile(sorted_losses, 75)
        iqr_losses = sorted_losses[(sorted_losses >= q1) & (sorted_losses <= q3)]
        iqm_losses.append(np.mean(iqr_losses))

    plt.plot(iqm_losses, label="IQM")

    if worst_best:
        final_losses = [losses[-1] for losses in all_losses]
        worst_idx = np.argmax(final_losses)
        best_idx = np.argmin(final_losses)
        plt.plot(all_losses[worst_idx], label="Worst", linestyle="--")
        plt.plot(all_losses[best_idx], label="Best", linestyle="--")

    q1_values = [np.percentile([losses[i] for losses in all_losses], 25) for i in range(len(all_losses[0]))]
    q3_values = [np.percentile([losses[i] for losses in all_losses], 75) for i in range(len(all_losses[0]))]
    plt.fill_between(range(len(all_losses[0])), q1_values, q3_values, color='gray', alpha=0.3, label="IQR")

    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.title("Interquartile Mean (IQM) Loss")
    plt.legend()
    plt.show()


def load_usps(fn):
    with open(fn) as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp = np.array(data)
    return normalize_images(tmp[:, 1:]), tmp[:, 0].astype(int)

def get_usps(l,datax,datay):
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy
    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l], strict=False))
    tmpx,tmpy = np.vstack(tmp[0]),np.hstack(tmp[1])
    return tmpx,tmpy

def show_usps(data):
    plt.imshow(data.reshape((16,16)),interpolation="nearest",cmap="gray")

def onehotencoder(labels, num_classes=None):
    labels = labels.flatten()
    if num_classes is None:
        num_classes = np.max(labels) + 1
    one_hot = np.zeros((labels.size, num_classes))
    one_hot[np.arange(labels.size), labels] = 1
    return one_hot
