import matplotlib.pyplot as plt
import seaborn as sns

def set_style():
    """Configuración estética global (Modern & Academic)."""
    plt.style.use('ggplot') 
    sns.set_palette("viridis")
    plt.rcParams.update({
        'font.size': 12,
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'figure.figsize': (10, 6),
        'figure.dpi': 150
    })
