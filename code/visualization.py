import matplotlib.pyplot as plt
import os 
import seaborn as sns
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
import numpy as np
import matplotlib.dates as mdates

COLORS = ['#006E7F', '#F8CB2E', '#EE5007', '#B22727', '#D3ECA7', '#FB6F92', '#ABABCA', '#19282F']


def viz_dataset(raw_df, figures_path):
    """
    Plot general statistic about data.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    fig.subplots_adjust(bottom=0.2)
    instruments_by_year = raw_df.groupby('year')['Name'].nunique()
    axes[0].bar(instruments_by_year.index, instruments_by_year.values,
                color=COLORS[0]);
    axes[0].set_ylabel('# Instruments')
    axes[0].set_xlabel('Year')

    volume_by_year = raw_df.groupby('year')['Volume'].mean()
    axes[1].bar(volume_by_year.index, volume_by_year.values,
                color=COLORS[1])
    axes[1].set_ylabel('Mean Volume per Instrument')
    axes[1].set_xlabel('Year');
    plt.gcf().set_dpi(300)
    plt.savefig(os.path.join(figures_path, 'exploratory_analysis.jpg'))


def viz_correlated(corr_matrix, figures_path):
    """
    Plot feature correlation
    """
    plt.figure(figsize=(8,8))
    sns.heatmap(corr_matrix, cmap=plt.cm.CMRmap_r, annot=True, fmt='.2f')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_path, 'feature_corr.jpg'))
    plt.show()


def viz_best_feats(X_train, best_feat_df, figures_path):
    """
    Plot best features
    """
    ax1 = sns.barplot(x=best_feat_df.index,
                    y=best_feat_df['Importance_Value'])
    plt.subplots_adjust(bottom=0.2)
    ax1.set_xticklabels(ax1.get_xticklabels(), fontsize = 12, rotation=40, ha="right")
    ax1.set_ylabel('χ2')
    plt.title('Top Features')
    plt.savefig(os.path.join(figures_path, 'top_features.jpg'))
    plt.show()

def get_proba_based_perfomance(proba, returns, trade_dates):
    """
    Trading strategy that sells/buys at most 10 assets per day, the ones with
    the highest probability of increasing or decreasing
    """
    strategy_df = pd.DataFrame()
    strategy_df['Date'] = trade_dates
    strategy_df['Proba'] = proba

    winners = strategy_df.groupby('Date')['Proba'].nsmallest(10).reset_index()
    winners = winners[winners['Proba'] < 0.5]

    losers = strategy_df.groupby('Date')['Proba'].nlargest(10).reset_index()
    losers = losers[losers['Proba'] > 0.5]

    strategy_df['Positions'] = 0
    strategy_df['Positions'].loc[losers['level_1']] = -1
    strategy_df['Positions'].loc[winners['level_1']] = 1

    captured_returns = strategy_df['Positions'] * returns
    return captured_returns.cumsum()

def get_standard_performance(preds, returns):
    """
    Returns the cumulative returns of a model using its predictions as positions.
    """
    preds[np.isin(preds, ["True", True])] = 1  # Buy signal
    preds[np.isin(preds, ["False", False])] = -1  # Sell signal
    preds[~np.isin(preds, [-1, 1])] = 0  # Do nothing
    captured_returns = preds * returns
    return captured_returns.cumsum()

def viz_backtest(X, models, features, fig_path, start_year=None):
    """
    Plot the performance of a set of models on a given dataset
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    
    backtest_df = pd.DataFrame()
    X['Date'] = pd.to_datetime(X['Date'])
    if start_year is not None:
        backtest_df = X[X['Date'].dt.year >= start_year].copy()
    else:
        backtest_df = X.copy()

    # Plot the performance of the baseline
    backtest_df['Baseline'] = backtest_df['Returns'].cumsum()
    ax.plot(backtest_df['Date'], backtest_df['Baseline'], color=COLORS[-1],
            label='Baseline')
    print(f'Score for model Baseline at the end of the period: {backtest_df.iloc[-1]["Baseline"]}')

    for idx, (model_name, model) in enumerate(models.items()):
        if 'Cluster' in model_name:
            y_pred = model.predict(backtest_df[features + ['Cluster']])
        else:
            y_pred = model.predict(backtest_df[features])
        if 'Selective' in model_name:
            # Use the trading strategy based on model probabilities.
            backtest_df[model_name] = get_proba_based_perfomance(y_pred,
                                                    backtest_df['Returns'],
                                                    backtest_df['Date'])
        else:
            # Use the trading strategy based on raw class predictions.
            backtest_df[model_name] = get_standard_performance(y_pred,
                                                    backtest_df['Returns'])
        ax.plot(backtest_df['Date'], backtest_df[model_name], color=COLORS[idx],
                label=model_name)
        print(f'Score for model {model_name} at the end of the period: {backtest_df.iloc[-1][model_name]}')
    ax.set_xlabel('Date')
    ax.set_ylabel('Captured Returns')
    ax.legend()
    plt.savefig(fig_path)
    return ax