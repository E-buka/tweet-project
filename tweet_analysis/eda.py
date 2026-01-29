
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme()
from .schema import LABEL_COL

def plot_data(df):
    """plots a EDA of dataframe
    """
    tweet_sample = (df
                    .sample(fraction=0.01, seed=44)
                    .toPandas()
                   )
    
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 8))
    axd = axes.ravel()
    sns.countplot(data=tweet_sample, x='hour', ax=axd[0])
    sns.countplot(data=tweet_sample, x='dayofweek', ax=axd[1], label='Day (Sun=1)')
    sns.histplot(data=tweet_sample, x='char_count', ax=axd[2])
    sns.histplot(data=tweet_sample, x='total_words', ax=axd[3])
    
    plt.subplots_adjust(bottom=0.2, top=0.3, wspace=0.2, hspace=0.4)
    
    plt.suptitle('Feature distributions of tweet sample data', fontweight='semibold', fontsize=20)
    plt.tight_layout()
    plt.show()
    
    fig, ax = plt.subplots(figsize=(6,3))
    sns.countplot(data=tweet_sample, x='num_exclaim', hue=LABEL_COL,ax=ax)
    ax.set_title('Distribution of number of exclamations', fontsize=13, fontweight='semibold')
    plt.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    
    plt.show()
