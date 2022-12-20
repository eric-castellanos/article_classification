import pandas as pd
import matplotlib.pyplot as plt

def plot_hist(df, col, filename):
    df[col].plot.hist()
    plt.savefig('plots/%s.png' % filename)
    plt.clf()

def main():
    train = pd.read_csv('../data/processed/train.csv')
    test = pd.read_csv('../data/processed/test.csv')

    plot_hist(train, 'Word Count Title', 'Word Count Title - Train')
    plot_hist(test, 'Word Count Title', 'Word Count Title - Test')

    plot_hist(train, 'Word Count Description', 'Word Count Description - Train')
    plot_hist(test, 'Word Count Description', 'Word Count Description - Test')

if __name__ == "__main__":
    main()