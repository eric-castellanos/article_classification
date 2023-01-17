import re

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from dython import nominal

matplotlib.style.use('fivethirtyeight')

def plot_hist(df, col, filename, title):
    """
    This function takes a dataframe and column 
    and plots a histogram for the series.
    """
    
    plt.style.use('seaborn-whitegrid') # nice and clean grid
    plt.xlabel('Bins') 
    plt.ylabel('Values')
    df[col].plot.hist(title=title, bins=90, facecolor = '#2ab0ff', edgecolor='#169acf', linewidth=0.5, figsize=(20,10))
    plt.savefig('plots/%s.png' % filename)
    plt.clf()

def plot_horizontal_bar(df, filename):
    """
    This function takes a dataframe and column 
    and plots a histogram for the series.
    """

    df = df[:10]

    my_colors = ['g', 'b']*5 # <-- this concatenates the list to itself 5 times.
    my_colors = [(0.5,0.4,0.5), (0.75, 0.75, 0.25)]*5 # <-- make two custom RGBs and repeat/alternate them over all the bar elements.
    my_colors = [(x/10.0, x/20.0, 0.75) for x in range(len(df))] # <-- Quick gradient example along the Red/Green dimensions.

    df.plot.barh(x='Word', y='Frequency', color=my_colors, title='Word Frequency Plot - Description', figsize=(20,10)).invert_yaxis()
    plt.savefig('plots/%s.png' % filename)
    plt.clf()

def plot_corr_heatmap(data, filename):
    """
    This function calculates a correlation matrix
    given a dataframe. Since Class Index is categorical,
    we need encode the values and then calculate the correlation 
    matrix, which is equivalent to the Bi
    """
    class_dummies = pd.get_dummies(data['Class Index'])
    data = data.drop(columns=['Title', 'Description'])
    #data = pd.concat([data,class_dummies], axis=1)
    nominal.associations(data, figsize=(20,15),mark_columns=True)
    plt.savefig('plots/%s.png' % filename)
    plt.clf()

def histogram_helper(train, test):
    """
    This is a helper function for
    plotting all histograms for all
    series.
    """

    plot_hist(train, 'Word Count Title', 'Word Count Title - Train', 'Word Count Title - Train')
    plot_hist(test, 'Word Count Title', 'Word Count Title - Test', 'Word Count Title - Test')

    plot_hist(train, 'Word Count Description', 'Word Count Description - Train', 'Word Count Description - Train')
    plot_hist(test, 'Word Count Description', 'Word Count Description - Test', 'Word Count Description - Test')

    plot_hist(train, 'Character Count Title', 'Character Count Title - Train', 'Character Count Title - Train')
    plot_hist(test, 'Character Count Title', 'Character Count Title - Test', 'Character Count Title - Test')

    plot_hist(train, 'Unique Count Description', 'Unique Count Description - Train', 'Unique Count Description - Train')
    plot_hist(test, 'Unique Count Description', 'Unique Count Description - Test', 'Unique Count Description - Test')

def most_common_words(df, col):
    """
    This function creates a word dictionary
    for the frequency of words that show up
    in a pandas series and returns a dataframe 
    with word and its frequency in the series.
    """
    articles = ['the', 'to', 'a', 'of', 'in', 'and', 'on', 'for', 'that', 's', 'with', 'as', 'its', 'at', 'is', 'said', 'by', 'it', 'has', 'an', 'from', 'his', 'us', 'was', 'will', 'have', 'be', 'their', 'are']
    word_dict = {}
    def word_count(x):
        x = x.split(' ')
        regex = re.compile('[^a-zA-Z]')
        for word in x:
            word = regex.sub('', word)
            word = word.lower()
            if word in word_dict:
                word_dict[word] += 1
            else:
                word_dict[word] = 1
    df[col].apply(word_count)
    word_df = pd.DataFrame(list(word_dict.items()), columns=['Word', 'Frequency'])
    word_df = word_df[(word_df['Word'].isna() == False) & (word_df['Word'] != '') & ~(word_df['Word'].isin(articles))]
    word_df = word_df.sort_values(by=['Frequency'], ascending=False)
    return word_df.reset_index(drop=True)

def main():
    train = pd.read_csv('../data/processed/train.csv')
    test = pd.read_csv('../data/processed/test.csv')

    histogram_helper(train, test)
    corr_matrix = plot_corr_heatmap(train, "Training Data - Correlation Heatmap")
    most_common_words_df = most_common_words(train, 'Description')
    plot_horizontal_bar(most_common_words_df, "Training Data Description - Word Frequency")
    

if __name__ == "__main__":
    main()