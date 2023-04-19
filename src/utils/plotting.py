
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set_theme()
plt.rcParams["figure.figsize"] = (20,15)


def plot_VAR_params(A, p, plt_path = None):
    '''
    
    Generate the plot of the coefficient matrices in VAR models.

    Parameters
    ----------
    A : np.array
        Coefficient matrix.
    p : int
        VAR model order, p. Autoregressive parameter.
    plt_path : str, optional
        String containing the path where the plot is going to be stored.

    Returns
    -------
    Bool
        True if the performance is OK.
    
    '''

    n_instants = A.shape[0]
    
    if p>1:
        
        grid_widths = np.repeat(1, p).tolist()+[0.1]
        gridspec = {'width_ratios': grid_widths}
        fig, axs = plt.subplots(1,p+1, gridspec_kw = gridspec)
        fig.tight_layout()
        
        for i in range(p):
            
            start = i*n_instants
            stop = (i+1)*n_instants
            
            A_i = A[:,start:stop]
            
            
            sns.heatmap(A_i, ax=axs[i],xticklabels = False, yticklabels = False, cbar = False)
            axs[i].set_title('A_{}'.format(str(i+1)))

            
        fig.colorbar(axs[0].collections[0], cax=axs[p])
        fig.suptitle('A_i para p = {}'.format(p), fontsize=16)

    else:
        fig, axs = plt.subplots(1,1)
        sns.heatmap(A, xticklabels = False, ax = axs, yticklabels = False)
        axs.set_title('A para p = {}'.format(p), fontsize = 16)
        
        
    if plt_path:
        fig.savefig(plt_path)
        plt.close()
    else:
        plt.show()
    return True


    

def plot_prediction(x, y_true, y_hat, title = '',
                    y_label = '', y_hat_label = '', 
                    x_axis_label = '', y_axis_label = '', plt_name = None):
    '''
    
    Generate the plot of the true and the predicted TS.

    Parameters
    ----------
    x : np.array
        Array containing the timestamps.
    y_true : np.array
        True observed TS.
    y_hat : np.array
        Predicted TS.
    title : str
        Title of the plot.
    y_label: str
        True TS label (legend).
    y_hat_label : str
        Predicted TS label (legend).
    x_axis_label : str
        X axis label.
    y_axis_label : str
        Y axis label.
    plt_name : str, optional
        String containing the path where the plot is going to be stored.
    
    Returns
    -------
    Bool
        True if the performance is OK.
    
    '''
        
    fig = plt.plot(x, y_true, label = y_label)
    plt.plot(x, y_hat, label = y_hat_label )
    plt.xlabel(x_axis_label)
    plt.ylabel(y_axis_label)
    plt.title(title)
    plt.legend()
    if plt_name:
        plt.savefig(plt_name)
        plt.close()
    else:
        plt.show()
    

    return True
    
    