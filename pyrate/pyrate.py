import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tools.eval_measures import rmse
from sklearn.model_selection import KFold
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import partial_resids
import matplotlib.ticker as mtick
import re
import seaborn as sns
from scipy.stats import gamma, poisson, norm, uniform, norm
import statsmodels
import tweedie
import statsmodels.formula.api as smf
from statsmodels.genmod.families.links import log


def lift_curve(df_pred, predicted_loss_cost, observed_loss_cost, exposure, n=10, rel = True, norm = False, ax = None):
  """
  The function creates the lift curve for the predicted loss cost vs the observed one.
  
  Input:
  df_pred: The dataframe where there are the observed loss cost, the predicted loss cost and the measure of exposure.
  predicted_loss_cost: The name of the column of the predicted loss cost.
  observed_loss_cost: The name of the column of the observed loss cost.
  exposure: The name of the column of the exposure.
  n: (optional, default = 10) The number of buckets of equal exposure in which the data is split.
  rel: (optional, default = True) Show the values as relative to the predicted loss cost.
  norm: (optional, default = False) Show the values as percentage deviation from the observed values, showed as the horizontal line.
  ax: (optional, default = False) Matplotlib option if plotting as a grid.

  Output:
  None: The function does not return anything. It is used for the side effect of plotting the lift curve graph.
  """


  df = df_pred[[predicted_loss_cost, observed_loss_cost, exposure]]
  df = df.sort_values(predicted_loss_cost)

  rmse_val = rmse(df[predicted_loss_cost], df[observed_loss_cost])

  df['buckets'] = pd.cut(np.cumsum(df[exposure]), n, labels = np.linspace(1,n,num=n).astype('int'))

  df = df.groupby(['buckets']).apply(lambda x: pd.Series([np.average(x[predicted_loss_cost], weights=x[exposure]),
                                                          np.average(x[observed_loss_cost], weights=x[exposure])],
                                                         index = ['Predicted_Risk_Premium', 'Observed_Risk_Premium'])).reset_index()
  
  if rel:
    mean_mod = np.average(df_pred[predicted_loss_cost],  weights=df_pred[exposure])
    df['Predicted_Risk_Premium'] = df['Predicted_Risk_Premium'] / mean_mod
    df['Observed_Risk_Premium'] = df['Observed_Risk_Premium'] / mean_mod

  if norm:
    df['Predicted_Risk_Premium'] = df['Predicted_Risk_Premium'] / df['Observed_Risk_Premium'] - 1
    df['Observed_Risk_Premium'] = df['Observed_Risk_Premium'] / df['Observed_Risk_Premium'] - 1

  if ax is None:
    plt.figure(figsize=(15, 10))
    plt.plot(df['buckets'], df['Predicted_Risk_Premium'])
    plt.plot(df['buckets'], df['Observed_Risk_Premium'])
    plt.legend(['Predicted Loss Cost', 'Observed Loss Cost'], loc=2)
    plt.xlabel('Bucket')
    plt.ylabel('Loss Cost Relativity') if rel else plt.ylabel('Loss Cost')
    plt.title('Standardized Lift Curve, RMSE: ' + str(np.round(rmse_val, 3))) if norm else plt.title('Lift Curve, RMSE: ' + str(np.round(rmse_val, 3)))
    plt.xticks(df['buckets'])
  else:
    ax.plot(df['buckets'], df['Predicted_Risk_Premium'])
    ax.plot(df['buckets'], df['Observed_Risk_Premium'])
    ax.legend(['Predicted Loss Cost', 'Observed Loss Cost'], loc=2)
    ax.set_xlabel('Bucket')
    ax.set_ylabel('Loss Cost Relativity') if rel else ax.set_ylabel('Loss Cost')
    ax.set_title('Standardized Lift Curve, RMSE: ' + str(np.round(rmse_val, 3))) if norm else ax.set_title('Lift Curve, RMSE: ' + str(np.round(rmse_val, 3)))
    ax.set_xticks(df['buckets'])

def double_lift_curve(df_pred, predicted_loss_cost_mod_1, predicted_loss_cost_mod_2, observed_loss_cost, exposure, n=10, norm = False):

  """
  The function creates the double lift curve for the predicted loss cost from two different models vs the observed one.
  
  Input:
  df_pred: The dataframe where there are the observed loss cost, the predicted loss cost and the measure of exposure.
  predicted_loss_cost_mod_1: The name of the column of the predicted loss cost from the first model.
  predicted_loss_cost_mod_2: The name of the column of the predicted loss cost from the second model.
  observed_loss_cost: The name of the column of the observed loss cost.
  exposure: The name of the column of the exposure.
  n: (optional, default = 10) The number of buckets of equal exposure in which the data is split.
  norm: (optional, default = False) Show the values as percentage deviation from the observed values, showed as the horizontal line.

  Output:
  None: The function does not return anything. It is used for the side effect of plotting the double lift curve graph.
  """




  df = df_pred[[predicted_loss_cost_mod_1, predicted_loss_cost_mod_2, observed_loss_cost, exposure]].astype(float)

  mean_mod_1 = np.average(df[predicted_loss_cost_mod_1], weights=df[exposure])
  mean_mod_2 = np.average(df[predicted_loss_cost_mod_2], weights=df[exposure])
  mean_obs = np.average(df[observed_loss_cost], weights=df[exposure])
  
  df['sort_ratio'] = df[predicted_loss_cost_mod_1]/df[predicted_loss_cost_mod_2]

  df = df.sort_values('sort_ratio')

  df['buckets'] = pd.cut(np.cumsum(df[exposure]), n, labels = np.linspace(1,n,num=n).astype('int'))


  df = df.groupby(['buckets']).apply(lambda x: pd.Series([np.average(x[predicted_loss_cost_mod_1], weights=x[exposure]),
                                                          np.average(x[predicted_loss_cost_mod_2], weights=x[exposure]),
                                                          np.average(x[observed_loss_cost], weights=x[exposure])],
                                                         index = ['Model_1_Predicted_Risk_Premium', 'Model_2_Predicted_Risk_Premium', 'Observed_Risk_Premium'])).reset_index()

  df['Model_1_Predicted_Risk_Premium'] = df['Model_1_Predicted_Risk_Premium'] / mean_mod_1
  df['Model_2_Predicted_Risk_Premium'] = df['Model_2_Predicted_Risk_Premium'] / mean_mod_2
  df['Observed_Risk_Premium'] = df['Observed_Risk_Premium'] / mean_obs

  if norm:
    df['Model_1_Predicted_Risk_Premium'] = df['Model_1_Predicted_Risk_Premium'] / df['Observed_Risk_Premium'] - 1
    df['Model_2_Predicted_Risk_Premium'] = df['Model_2_Predicted_Risk_Premium'] / df['Observed_Risk_Premium'] - 1
    df['Observed_Risk_Premium'] = df['Observed_Risk_Premium'] / df['Observed_Risk_Premium'] - 1

  plt.figure(figsize=(15, 10))
  plt.plot(df['buckets'], df['Model_1_Predicted_Risk_Premium'])
  plt.plot(df['buckets'], df['Model_2_Predicted_Risk_Premium'])
  plt.plot(df['buckets'], df['Observed_Risk_Premium'])
  plt.legend(['Model 1 Predicted Loss Cost', 'Model 2 Predicted Loss Cost', 'Observed Predicted Loss Cost'], loc=2)
  plt.xlabel('Bucket')
  plt.ylabel('Relative Loss Cost')
  plt.xticks(df['buckets'])
  plt.title('Standardized Double Lift Chart') if norm else plt.title('Double Lift Chart')


def loss_ratio_plot(df_pred, predicted_loss_cost, observed_loss_cost, exposure, premium, n=10, ax = None):

  """
  The function creates the loss ratio plot in order to evaluate the adequacy of the predicted loss cost.
 
  Input:
  df_pred: The dataframe where there are the observed loss cost, the predicted loss cost and the measure of exposure.
  predicted_loss_cost: The name of the column of the predicted loss cost.
  observed_loss_cost: The name of the column of the observed loss cost.
  exposure: The name of the column of the exposure.
  premium: The name of the column of the premium.
  n: (optional, default = 10) The number of buckets of equal exposure in which the data is split.
  ax: (optional, default = False) Matplotlib option if plotting as a grid.
  
  Output:
  None: The function does not return anything. It is used for the side effect of plotting the loss ratio plot.
  """


  df = df_pred[[predicted_loss_cost, observed_loss_cost, exposure, premium]].astype(float)

  df['LR_pred'] = df[predicted_loss_cost] * df[exposure] / df[premium]
  df['obs_losses'] = df[observed_loss_cost] * df[exposure]
  
  df = df.sort_values('LR_pred')

  df['buckets'] = pd.cut(np.cumsum(df[exposure]), n, labels = np.linspace(1,n,num=n).astype('int'))

  df = df.groupby(['buckets']).apply(lambda x: pd.Series([np.sum(x['obs_losses']) / np.sum(x[premium])],
                                                         index = ['LR_obs'])).reset_index()

  if ax is None:
    plt.figure(figsize=(15, 10))
    plt.bar(df.buckets, df.LR_obs)
    plt.xlabel('Bucket')
    plt.ylabel('Loss Ratio')
    plt.xticks(df.buckets)
    plt.title('Loss Ratio Chart')
  else:
    ax.bar(df.buckets, df.LR_obs)
    ax.set_xlabel('Bucket')
    ax.set_ylabel('Loss Ratio')
    ax.set_xticks(df.buckets)
    ax.set_title('Loss Ratio Chart')

def gini_value(observed_loss_cost, predicted_loss_cost, exposure):
  """
  The function computes the gini value for the predicted loss cost.
  
  Input:
  observed_loss_cost: The Series/Array of the observed loss cost.
  predicted_loss_cost: The Series/Array of the predicted loss cost.
  exposure: The Series/Array of the exposure measure.

  Output:
  The computed gini value.

  """


  dataset = pd.DataFrame({'obs_lc' : observed_loss_cost.astype('float'),
                          'pred_lc' : predicted_loss_cost.astype('float'),
                          'exp' : exposure.astype('float')})

  dataset = dataset.sort_values(by='pred_lc')

  dataset['losses'] = dataset['obs_lc'] * dataset['exp']
  dataset['cum_exp'] = dataset['exp'].cumsum() / dataset['exp'].sum()
  dataset['cum_losses'] = dataset['losses'].cumsum() / dataset['losses'].sum()

  gini_value = (np.abs(np.trapz(x=dataset['cum_exp'], y=dataset['cum_losses']) - 1) - .5)*2

  return(gini_value)

def gini_plot(df_pred, predicted_loss_cost, observed_loss_cost, exposure, ax = None):
  """
  The function creates the gini plot and calculates the gini value in order to evaluate the discriminatory power of the predicted loss cost.
 
  Input:
  df_pred: The dataframe where there are the observed loss cost, the predicted loss cost and the measure of exposure.
  predicted_loss_cost: The name of the column of the predicted loss cost.
  observed_loss_cost: The name of the column of the observed loss cost.
  exposure: The name of the column of the exposure.
  ax: (optional, default = False) Matplotlib option if plotting as a grid.
  
  Output:
  None: The function does not return anything. It is used for the side effect of plotting the gini graph. The gini value is printed in the         graph title.
  """

  df = df_pred[[predicted_loss_cost, observed_loss_cost, exposure]].astype(float)

  gini_index = gini_value(df[observed_loss_cost], df[predicted_loss_cost], df[exposure])

  df = df.sort_values(by=predicted_loss_cost)

  df['losses'] = df[observed_loss_cost] * df[exposure]
  df['cum_exp'] = df[exposure].cumsum() / df[exposure].sum()
  df['cum_losses'] = df['losses'].cumsum() / df['losses'].sum()

  if ax is None:
    plt.figure(figsize=(15, 10))
    plt.plot(df.cum_exp, df.cum_losses)
    plt.plot(np.linspace(0,1,100), np.linspace(0,1,100), color='darkgray')
    plt.title('Gini Value ' + str(np.round(gini_index, 3)))
    plt.legend(['Lorenz Curve', 'Equality Line'], loc = 2)
    plt.xlabel('Percentage of Exposures')
    plt.ylabel('Percentage of Losses')
  else:
    ax.plot(df.cum_exp, df.cum_losses)
    ax.plot(np.linspace(0,1,100), np.linspace(0,1,100), color='darkgray')
    ax.set_title('Gini Value ' + str(np.round(gini_index, 3)))
    ax.legend(['Lorenz Curve', 'Equality Line'], loc = 2)
    ax.set_xlabel('Percentage of Exposures')
    ax.set_ylabel('Percentage of Losses')

def print_metrics(obs_loss_cost, pred_loss_cost, exposure):
  rmse_val = rmse(obs_loss_cost, pred_loss_cost)
  gini_index = gini_value(obs_loss_cost, pred_loss_cost, exposure)
  print("RMSE: " + str(rmse_val) + "; Gini Index: " + str(gini_index))

def plot_coeff(glm_model, title, beta = False, ce = True, alpha = .05, ax = None):

  """
  The function plots the glm model coefficients and teh relative confidence intervals.
 
  Input:
  glm_model: The statsmodels instance of the GLM model.
  title: Plot title.
  beta: (optional, default = False) Should the plot show the beta coeffients?
  ce: (optional, default = True) Should the plot show the confidence intervals for the coeffients?
  alpha: (optional, default = .05) alpha value for the confidence interval.
  ax: (optional, default = False) Matplotlib option if plotting as a grid.
  
  Output:
  None: The function does not return anything. It is used for the side effect of plotting the model coefficient.
  """

  df_coef = glm_model.params.to_frame(name='est')
  groups = [re.search(r"^C\((.*),.*\[T.(.*)\]$", i) for i in df_coef.index]
  cat_var = [' : '.join(map(str, i.groups())) if i != None else 'Num' for i in groups]
  coef_index = [y if x == 'Num' else x for x,y in zip(cat_var, list(df_coef.index))]
  coeff = glm_model.params.to_frame(name='est').join(glm_model.conf_int(alpha=.05))
  coeff['err'] = np.abs(coeff[0] - coeff['est'])
  coeff.index = coef_index

  if beta:
    std_DV = np.std(glm_model.model.endog)
    std_IV = np.std(glm_model.model.exog, axis = 0)
    coeff['est'] = coeff['est'] * std_IV / std_DV
    coeff[0] = coeff[0] * std_IV / std_DV
    coeff['err'] = np.abs(coeff[0] - coeff['est'])

  coeff = coeff[coeff.index != 'Intercept'].sort_values('est')
  y = np.linspace(1, coeff.shape[0] * 2 - 1, num = coeff.shape[0])

  if ax is None:
    plt.figure(figsize=(15, 15))
    if ce:
      plt.errorbar(coeff['est'], y, xerr = coeff['err'], fmt = 'o', color = '#004799', ecolor = 'k', capsize = 5)
    else:
      plt.errorbar(coeff['est'], y, fmt = 'o', color = 'k')
    plt.title('Model Parameters' if not beta else 'Model Beta Parameters')
    plt.yticks(y, coeff.index)
    plt.vlines(0, 0, np.max(y) + 1, colors = 'lightgray')
    plt.title(title)
    plt.show()

  else:
    if ce:
      ax.errorbar(coeff['est'], y, xerr = coeff['err'], fmt = 'o', color = '#004799', ecolor = 'k', capsize = 5)
    else:
      ax.errorbar(coeff['est'], y, fmt = 'o', color = 'k')
      ax.set_title('Model Parameters' if not beta else 'Model Beta Parameters')
    coeff = coeff[coeff.index != 'Intercept'].sort_values('est')
    y = np.linspace(1, coeff.shape[0] * 2 - 1, num = coeff.shape[0])
    ax.set_yticks(y)
    ax.set_yticklabels(coeff.index)
    ax.set_title(title)
    ax.vlines(0, 0, np.max(y) + 1, colors = 'lightgray')

def bucket_plot(pred_df, bucket_name, premium, exposure, losses, numbers):
  """
  The function creates the bucket plots.

  Input:
  pred_df: The DataFrame with the scored data.
  bucket_name: The column name of the bucket score.
  premium: The name of the column of the premium.
  exposure: The name of the column of the exposure.
  losses: The name of the column of the claim losses.
  numbers: The name of the column of the claim numbers.

  Output:
  None: The function does not return anything. It used for the side effect of creating the bucket plots.
  """

  avg_lr = np.sum(pred_df[losses]) / np.sum(pred_df[premium])
  avg_lc = np.sum(pred_df[losses]) / np.sum(pred_df[exposure])
  avg_freq = np.sum(pred_df[numbers]) / np.sum(pred_df[exposure])
  avg_sev = np.sum(pred_df[losses]) / np.sum(pred_df[numbers])


  df = pred_df.groupby(bucket_name, as_index=False).apply(lambda x:pd.Series({'LR' : np.sum(x[losses]) / np.sum(x[premium]),
                                                                              'Freq' : np.sum(x[numbers]) / np.sum(x[exposure]),
                                                                              'Sev' : np.sum(x[losses]) / np.sum(x[numbers]),
                                                                              'LC' : np.sum(x[losses]) / np.sum(x[exposure])}))

  fig, axs = plt.subplots(2, 2, figsize=(30,15))
  fig.suptitle('Bucket Analysis')
  
  p = axs[0][0].bar(df[bucket_name], df['LR'] * 100, color = '#0057B8')
  axs[0][0].bar_label(p, label_type='edge', fmt='%.0f%%')
  axs[0][0].annotate(f"Average = {np.round(avg_lr*100, 1)}%", (0,avg_lr*100))
  axs[0][0].axhline(y = avg_lr * 100, color = 'black')
  axs[0][0].set_title('Loss Ratio')
  axs[0][0].set_xlabel('Bucket')
  axs[0][0].set_xticks(df[bucket_name])
  axs[0][0].set_ylabel('Loss Ratio')
  
  p = axs[0][1].bar(df[bucket_name], df['LC'], color = '#0057B8')
  axs[0][1].bar_label(p, label_type='edge')
  axs[0][1].annotate(f"Average = {avg_lc.astype(int):,}", (0,avg_lc))
  axs[0][1].axhline(y = avg_lc, color = 'black')
  axs[0][1].set_title('Loss Cost')
  axs[0][1].set_xlabel('Bucket')
  axs[0][1].set_xticks(df[bucket_name])
  axs[0][1].set_ylabel('Loss Cost')
  
  p = axs[1][0].bar(df[bucket_name], df['Freq'] * 100, color = '#0057B8')
  axs[1][0].bar_label(p, label_type='edge', fmt='%.0f%%')
  axs[1][0].annotate(f"Average = {np.round(avg_freq*100, 1)}%", (0,avg_freq*100))
  axs[1][0].axhline(y = avg_freq * 100, color = 'black')
  axs[1][0].set_title('Frequency')
  axs[1][0].set_xlabel('Bucket')
  axs[1][0].set_xticks(df[bucket_name])
  axs[1][0].set_ylabel('Frequency')

  p = axs[1][1].bar(df[bucket_name], df['Sev'], color = '#0057B8')
  axs[1][1].bar_label(p, label_type='edge')
  axs[1][1].annotate(f"Average = {avg_sev.astype(int):,}", (0,avg_sev))
  axs[1][1].axhline(y = avg_sev, color = 'black')
  axs[1][1].set_title('Severity')
  axs[1][1].set_xlabel('Bucket')
  axs[1][1].set_xticks(df[bucket_name])
  axs[1][1].set_ylabel('Severity')

def single_distr(df, title, ax):
    
  b1 = ax.bar(df.index, df.iloc[:,0], color='#CCDDF1')
  b2 = ax.bar(df.index, df.iloc[:,1], bottom=df.iloc[:,0:1].sum(axis=1), color='#B3CDEA')
  b3 = ax.bar(df.index, df.iloc[:,2], bottom=df.iloc[:,0:2].sum(axis=1), color='#80ABDC')
  b4 = ax.bar(df.index, df.iloc[:,3], bottom=df.iloc[:,0:3].sum(axis=1), color='#3379C6')
  b5a = ax.bar(df.index, df.iloc[:,4], bottom=df.iloc[:,0:4].sum(axis=1), color='#0057B8')
  b5b = ax.bar(df.index, df.iloc[:,5], bottom=df.iloc[:,0:5].sum(axis=1), color='#004799')
  
  ax.set_xticks(df.index)
  fmt = '%.0f%%'
  yticks = mtick.FormatStrFormatter(fmt)
  ax.yaxis.set_major_formatter(yticks)
 
  ax.bar_label(b1, label_type='center', fmt = fmt)
  ax.bar_label(b2, label_type='center', fmt = fmt)
  ax.bar_label(b3, label_type='center', fmt = fmt)
  ax.bar_label(b4, label_type='center', fmt = fmt)
  ax.bar_label(b5a, label_type='center', fmt = fmt)
  ax.bar_label(b5b, label_type='center', fmt = fmt)

  ax.legend([b1, b2, b3, b4, b5a, b5b], ['Bucket 1', 'Bucket 2', 'Bucket 3', 'Bucket 4', 'Bucket 5a', 'Bucket 5b'],
           loc='lower center', bbox_to_anchor=(.5, -.15),fancybox=True, shadow=True, ncol=6)

  ax.set_xlabel(df.index.name)
  ax.set_ylabel("% Distribution")
  ax.set_title(title)

def one_way_plots(data, var, premium, exposure, losses, numbers, n = 10, categorical = False, standardize = False, title = ""):

  """
  The function plots the one-way plots for the main actuarial metrics: Loss Ratio, Loss Cost, Frequency and Severity.
  The plots show the actual metric as a line and points on the primary vertical axis and the denominator on the secondary vertical axis.
 
  Input:
  data: The dataset that contains all the necessary variables.
  var: The variable name for which the plots should be created.
  premium: The name of the column of the premium.
  exposure: The name of the column of the exposure.
  losses: The name of the column of the claim amounts.
  numbers: The name of the column of the claim numbers.
  n (optional, default = 10): If the variable is numeric, how many buckets should the plot have?
  categorical (optional, default = False): Is the variable categorical?
  standardize (optional, default = False): Should the numbers be standardized relative to the mean value? eg. show the percentage change
  title (optional, default = ""): Option to change the default title of the plot (Variable name).
  
  Output:
  None: The function does not return anything. It is used for the side effect of plotting the one-way graphs.
  """

  lr = np.sum(data[losses].astype(float)) / np.sum(data[premium].astype(float))
  lc = np.sum(data[losses].astype(float)) / np.sum(data[exposure].astype(float))
  freq = np.sum(data[numbers].astype(float)) / np.sum(data[exposure].astype(float))
  sev = np.sum(data[losses].astype(float)) / np.sum(data[numbers].astype(float))

  if not categorical:
    data = data.assign(feature = lambda x: pd.cut(x[var].astype('float'), n, labels = np.linspace(1,n,n))).\
                groupby('feature', as_index = False).\
                agg(premium = (premium, 'sum'),
                    exposure = (exposure, 'sum'),
                    losses = (losses, 'sum'),
                    numbers = (numbers, 'sum')).\
                assign(loss_cost = lambda x : x['losses'].astype(float) / x['exposure'].astype(float),
                       loss_ratio = lambda x : x['losses'].astype(float) / x['premium'].astype(float),
                       frequency = lambda x : x['numbers'].astype(float) / x['exposure'].astype(float),
                       severity = lambda x : np.where(x['numbers'].astype(float) == 0, 0, x['losses'].astype(float) / x['numbers'].astype(float)))

  else:
    data = data.assign(feature = lambda x: x[var].astype('string')).\
                groupby('feature', as_index = False).\
                agg(premium = (premium, 'sum'),
                    exposure = (exposure, 'sum'),
                    losses = (losses, 'sum'),
                    numbers = (numbers, 'sum')).\
                assign(loss_cost = lambda x : x['losses'].astype(float) / x['exposure'].astype(float),
                       loss_ratio = lambda x : x['losses'].astype(float) / x['premium'].astype(float),
                       frequency = lambda x : x['numbers'].astype(float) / x['exposure'].astype(float),
                       severity = lambda x : np.where(x['numbers'].astype(float) == 0, 0, x['losses'].astype(float) / x['numbers'].astype(float)))



  fig, axs = plt.subplots(2, 2, figsize=(30,15))

  fig.suptitle(title + var)

  ax00 = axs[0][0].twinx()
  if standardize:
     axs[0][0].plot(data['feature'], 100 * (data['loss_ratio']/lr - 1), color = '#F8766D')
     axs[0][0].scatter(data['feature'], 100 * (data['loss_ratio']/lr - 1), color = '#F8766D')
     axs[0][0].axhline(y = 0, color = 'black')
     axs[0][0].yaxis.set_major_formatter(mtick.PercentFormatter())
  else:
     axs[0][0].plot(data['feature'], data['loss_ratio']*100, color = '#F8766D')
     axs[0][0].scatter(data['feature'], data['loss_ratio']*100, color = '#F8766D')
     axs[0][0].axhline(y = lr*100, color = 'black')
     axs[0][0].yaxis.set_major_formatter(mtick.PercentFormatter())
  ax00.bar(data['feature'], data['premium'], alpha = .5, color = '#708090')
  axs[0][0].legend(['Loss Ratio', 'Average Loss Ratio'], loc = 2)
  ax00.legend(['Premium'], loc = 1)
  axs[0][0].set_title('Loss Ratio')
  axs[0][0].set_xlabel(var)
  axs[0][0].set_xticks(data['feature'])
  axs[0][0].set_ylabel('Loss Ratio')
  ax00.set_ylabel('Premium')

  ax01 = axs[0][1].twinx()
  if standardize:
     axs[0][1].plot(data['feature'], 100 * (data['loss_cost']/lc - 1), color = '#7CAE00')
     axs[0][1].scatter(data['feature'], 100 * (data['loss_cost']/lc - 1), color = '#7CAE00')
     axs[0][1].axhline(y = 0, color = 'black')
     axs[0][1].yaxis.set_major_formatter(mtick.PercentFormatter())
  else:
    axs[0][1].plot(data['feature'], data['loss_cost'], color = '#7CAE00')
    axs[0][1].scatter(data['feature'], data['loss_cost'], color = '#7CAE00')
    axs[0][1].axhline(y = lc, color = 'black')
  ax01.bar(data['feature'], data['exposure'], alpha = .5, color = '#6A5ACD')
  axs[0][1].set_title('Loss Cost')
  axs[0][1].set_xlabel(var)
  axs[0][1].set_xticks(data['feature'])
  axs[0][1].legend(['Loss Cost', 'Average Loss Cost'], loc = 2)
  ax01.legend(['Exposure'], loc = 1)
  axs[0][1].set_ylabel('Loss Cost')
  ax01.set_ylabel('Exposure')
  ax10 = axs[1][0].twinx()

  if standardize:
    axs[1][0].plot(data['feature'], 100 * (data['frequency']/freq - 1), color = '#00BFC4')
    axs[1][0].scatter(data['feature'],100 * (data['frequency']/freq - 1), color = '#00BFC4')
    axs[1][0].axhline(y = 0, color = 'black')
    axs[1][0].yaxis.set_major_formatter(mtick.PercentFormatter())
  else:
    axs[1][0].plot(data['feature'], 100*data['frequency'], color = '#00BFC4')
    axs[1][0].scatter(data['feature'], 100*data['frequency'], color = '#00BFC4')
    axs[1][0].axhline(y = freq*100, color = 'black')
    axs[1][0].yaxis.set_major_formatter(mtick.PercentFormatter())
  ax10.bar(data['feature'], data['exposure'], alpha = .5, color = '#6A5ACD')
  axs[1][0].set_title('Frequency')
  axs[1][0].set_xticks(data['feature'])
  axs[1][0].legend(['Frequency', 'Average Frequency'], loc = 2)
  ax10.legend(['Exposure'], loc = 1)
  axs[1][0].set_xlabel(var)
  axs[1][0].set_ylabel('Frequency')
  ax10.set_ylabel('Exposure')

  ax11 = axs[1][1].twinx()
  if standardize:
    axs[1][1].plot(data['feature'], 100 * (data['severity']/sev - 1), color = '#C77CFF')
    axs[1][1].scatter(data['feature'], 100 * (data['severity']/sev - 1), color = '#C77CFF')
    axs[1][1].axhline(y = 0, color = 'black')
    axs[1][1].yaxis.set_major_formatter(mtick.PercentFormatter())
  else:
    axs[1][1].plot(data['feature'], data['severity'], color = '#C77CFF')
    axs[1][1].scatter(data['feature'], data['severity'], color = '#C77CFF')
    axs[1][1].axhline(y = sev, color = 'black')
  ax11.bar(data['feature'], data['numbers'], alpha = .5, color = '#FF7F50')
  axs[1][1].set_title('Severity')
  axs[1][1].set_xlabel(var)
  axs[1][1].set_xticks(data['feature'])
  axs[1][1].legend(['Severity', 'Average Severity'], loc = 2)
  ax11.legend(['Claim Numbers'], loc = 1)
  axs[1][1].set_ylabel('Severity')
  ax11.set_ylabel('Claim Numbers')

def plot_part_work_res(model, var):
  """
  The function plots the partial residuals and the binned working residuals for a GLM model and a specific variable. 
  This is used to check if any transformation of the variable should be included.
 
  Input:
  glm_model: The statsmodels instance of the GLM model.
  var: The variable name for which the plots should be created.
  
  Output:
  None: The function does not return anything. It is used for the side effect of plotting the partial residuals and working residuals graphs.
  """
  feat = model.params.index[model.params.index != 'Intercept']
  res_list = []
  for i in feat:
    res_list.append(partial_resids(model, i).to_frame(name = i))
  res_df = pd.concat(res_list, axis=1)
  des_df = pd.DataFrame(model.model.exog, columns = model.params.index)[feat]

  df = pd.DataFrame({'w_resid': model.resid_working,
                     'w_weigths': model.model.var_weights,
                     var : np.array(pd.DataFrame(model.model.exog, columns = model.params.index)[var])})

  df['bin'] = pd.cut(np.cumsum(df['w_weigths']), 250, labels = np.linspace(1, 250, num = 250).astype('int'))

  df = df.groupby('bin', as_index = False).apply(lambda x: pd.Series([np.average(x['w_resid'], weights=x['w_weigths']),
                                                               np.average(x[var], weights=x['w_weigths'])],
                                                              index = ['w_resid', var])).reset_index()

  
  fig, axs = plt.subplots(1, 2, figsize=(20,10))

  axs[0].scatter(des_df[var], res_df[var])
  axs[0].plot(des_df[var], des_df[var] * model.params[var], color = 'red')
  axs[0].set_title('Partial Residuals: '+ var)
  axs[0].set_xlabel(var)
  axs[0].set_ylabel('Component plus Residuals')

  axs[1].scatter(df[var], df['w_resid'])
  axs[1].plot(df[var], df[var] * model.params[var], color = 'red')
  axs[1].set_title('Binned Working Residuals: '+ var)
  axs[1].set_label(var)
  axs[1].set_ylabel('Binned Working Residuals')

def plot_res(model):
  """
  The function plots the residuals a GLM model. 

  Input:
  glm_model: The statsmodels instance of the GLM model.
  
  Output:
  None: The function does not return anything. It is used for the side effect of plotting the model residuals.
  """

  fam = type(model.family)

  if fam == statsmodels.genmod.families.family.Poisson:
      mu = model.fittedvalues
      y = model.model.endog
      a = poisson.cdf(y - 1, mu)
      b = poisson.cdf(y, mu)
      u = uniform.rvs(loc = a, scale = b - a, size = len(y))
      q_res_raw = norm.ppf(u)
      q_res = q_res_raw[(q_res_raw > -np.inf) & (q_res_raw < np.inf)]
      mu = mu[(q_res_raw > -np.inf) & (q_res_raw < np.inf)]
  elif fam == statsmodels.genmod.families.family.Tweedie:
    mu = model.fittedvalues
    y = model.model.endog
    df = model.model.df_resid
    w = model.model.var_weights
    p = model.family.var_power
    dispersion = np.sum((w * (y - mu)**2)/mu**p)/df
    u = tweedie.tweedie(p=p, mu=mu, phi=dispersion/w).cdf(y)
    u[y == 0] = np.array([uniform.rvs(loc = 0, scale = i - 0, size = 1)[0] for i in u[y == 0]])
    q_res_raw = norm.ppf(u)
    q_res = q_res_raw[(q_res_raw > -np.inf) & (q_res_raw < np.inf)]
    mu = mu[(q_res_raw > -np.inf) & (q_res_raw < np.inf)]
  else:
    mu = model.fittedvalues
    q_res = model.resid_deviance

  fig, axs = plt.subplots(1, 2, figsize=(20,10))

  axs[0].scatter(mu, q_res)
  axs[0].hlines(0, 0, np.max(mu), colors = 'red')
  axs[0].set_title('Residuals')
  axs[0].set_xlabel('Fitted')
  axs[0].set_ylabel('Residuals')

  axs[1].hist(q_res, bins = 50, density = True)
  axs[1].plot(np.sort(q_res), norm.pdf(np.sort(q_res), 0, 1), 'r', linewidth=2) 
  axs[1].set_title('Residuals')
  axs[1].set_xlabel('Residuals')
  axs[1].set_ylabel('Density')


def check_interactions(data, x_var, group_var, response, fun = 'mean'):
  """
  The function plots the interaction plot for a specific set of variables. 

  Input:
  data: The DataFrame in which the data is contained.
  x_var: The name of the variables that should be plotted on the x axis. It could be either numerical or categorical.
  group_var: The name of the variables that should be plotted on the x axis. It could be either numerical or categorical, however if it is              numerical it will be binned in 25 bins.
  response: The response variable name.
  fun (optional, default = 'mean') The aggregating function for each combinations of the variables to plot.
  
  Output:
  None: The function does not return anything. It is used for the side effect of plotting the interaction plot.
  """
  data_new = data.copy()

  if data_new[x_var].nunique() > 25 : data_new[x_var] = pd.cut(data_new[x_var], 25, labels = np.linspace(1, 25, num = 25))
  if data_new[group_var].nunique() > 25 : data_new[group_var] = pd.cut(data_new[group_var], 25, labels = np.linspace(1, 25, num = 25))

  df = data_new.groupby([group_var, x_var])\
            .agg({response : 'mean'}).reset_index()

  plt.figure(figsize=(15, 10))

  for i in df[group_var].unique():
    df_filtered = df[df[group_var] == i]
    plt.plot(df_filtered[x_var], df_filtered[response], label = i)
  plt.legend(title = group_var, loc = 1)

  plt.ylabel(response)
  plt.xlabel(x_var)
  plt.xticks(df[x_var])
  plt.title(f'Interaction plot between {x_var} and {group_var}')
  plt.show()

def check_mod_var(model, train_data, var, exposure, cat = True):

  """
  The function plots the relativities for the chosen variable in the model. 

  Input:
  glm_model: The statsmodels instance of the GLM model.
  train_data: The **train** data on which the model has been fitted.
  var: The name of the variable to display.
  exposure: The column name of the exposure.
  cat (optional, default = True) Is the varibale a categorical one?.
  
  Output:
  None: The function does not return anything. It is used for the side effect of plotting the variable relativities.
  """

  if cat:
    var_list = train_data.groupby(var, as_index = False).agg({exposure : 'sum'})
    df_coef = model.params.to_frame(name='est')
    groups = [re.search(r"^C\((.*),.*\[T.(.*)\]$", i) for i in df_coef.index]
    cat_var = [' : '.join(map(str, i.groups())) if i != None else 'Num' for i in groups]
    coef_index = [y if x == 'Num' else x for x,y in zip(cat_var, list(df_coef.index))]
    coeff = model.params.to_frame(name='est').join(model.conf_int(alpha=.05))
    coeff['err'] = np.abs(coeff[0] - coeff['est'])
    coeff.index = coef_index
    coef_df = coeff[coeff.index.str.startswith(var)].reset_index().rename(columns = {'index' : var})
    coef_df[['var_name', var]] = coef_df[var].str.split(' : ', expand = True)
    to_plot = var_list.merge(coef_df, how = "left").fillna(0).\
                assign(estimate = lambda x: np.exp(x['est']),
                      upper = lambda x: np.exp(x[0]),
                      lower = lambda x: np.exp(x[1])).\
                filter([var, exposure, 'estimate', 'upper','lower']).sort_values('estimate')
  else:
    to_plot = train_data.groupby(var, as_index = False).agg({exposure : 'sum'}).\
                assign(estimate = lambda x: np.exp(model.params[var] * x[var]),
                       upper = lambda x: x['estimate'] + model.conf_int(alpha=.05).loc[var][1],
                       lower = lambda x: x['estimate'] - model.conf_int(alpha=.05).loc[var][0])

  fig, ax1 = plt.subplots(figsize = (15,10))

  ax2 = ax1.twinx()

  ax2.bar(to_plot[var], to_plot[exposure], color = 'gray', alpha = .5)
  ax1.plot(to_plot[var], to_plot['estimate'])
  ax1.fill_between(to_plot[var], to_plot['lower'], to_plot['upper'], alpha = .3)
  ax1.legend(['Relativity', 'Confidence Interval'], loc = 2)
  ax2.legend(['Exposure'], loc = 1)
  ax1.set_title(var)
  ax1.set_xlabel(var)
  ax1.set_ylabel('Relativity')
  ax2.set_ylabel('Exposure')
  plt.show()

def plot_kfold(folds, pred_loss_cost, premium, exposure, title):

  """
  The function plots the kfold graphs for the object created with the `kfold` function.

  Input:
  folds: The kfold object.
  pred_loss_cost: The name of the predicted loss cost to plot: 'pred_loss_cost_lc' or 'pred_loss_cost_freq_sev'.
  premium: The column name of the premium.
  exposure: The column name of the exposure.
  title: Title for the plot
  
  Output:
  None: The function does not return anything. It is used for the side effect of plotting the kfold object.
        It plots the following graphs:
        Lift Curve
        Gini Plot
        Loss Ratio Plot
        Coefficients Plot

  """

  obs_loss_cost = 'obs_loss_cost'

  k = len(folds['train_data'])
  
  x = re.search("(?<=pred_loss_cost_).*", pred_loss_cost)

  plt_details = {}
  
  if x[0] == 'lc':
    i_index = 4
  elif x[0] == 'freq_sev':
    i_index = 5

  for i in range(0,i_index):
    plt_details[f'plot_{i}'] = {}
    plt_details[f'plot_{i}']['f'], plt_details[f'plot_{i}']['axs'] = plt.subplots(int(np.ceil(k/5)), 5, sharey=True, figsize=(30, 7.5))
    plt_details[f'plot_{i}']['f'].suptitle(title, fontsize=16)
    plt_details[f'plot_{i}']['axs'] = plt_details[f'plot_{i}']['axs'].ravel()

  for i in range(0, k):
    test_data = folds['test_data'][i]
    test_data[obs_loss_cost] = folds[obs_loss_cost][i]
    test_data[pred_loss_cost] = folds[pred_loss_cost][i]
    lift_curve(test_data, pred_loss_cost, obs_loss_cost, exposure, ax = plt_details[f'plot_{0}']['axs'][i])
    gini_plot(test_data, pred_loss_cost, obs_loss_cost, exposure, ax = plt_details[f'plot_{1}']['axs'][i])
    loss_ratio_plot(test_data, pred_loss_cost, obs_loss_cost, exposure, premium, n = 5, ax = plt_details[f'plot_{2}']['axs'][i])
    if i_index == 4:
      plot_coeff(folds['lc_models'][i], ax = plt_details[f'plot_{3}']['axs'][i], title = 'Loss Cost Coefficients')
    elif i_index == 5:
      plot_coeff(folds['freq_models'][i], ax = plt_details[f'plot_{3}']['axs'][i], title = 'Frequency Coefficients')
      plot_coeff(folds['sev_models'][i], ax = plt_details[f'plot_{4}']['axs'][i], title = 'Severity Coefficients')

def kfold(n, train_data, models_dict, exposure, losses, numbers):

  """
  The function performs a kfold routine.

  Input:
  n: The number of folds.
  train_data: The DataFrame with the train data.
  models_dict: The dictionary that contains the formulas for the glms to implement. 
               The dictionary should have the following keys: 'freq_formula', 'sev_formula' and 'lc_formula'.
  exposure: The column name of the exposure.
  losses: The column name of the claim losses.
  numbers: The column name of the claim numbers.

  Output:
  The function returns the kfold object with all the relative models and performance metrics.
  """

  folds = {}

  kf = KFold(n_splits=n, random_state=999, shuffle=True)

  folds['train_data'] = [train_data.iloc[train, :] for k, (train, test) in enumerate(kf.split(train_data))]
  folds['test_data'] = [train_data.iloc[test, :] for k, (train, test) in enumerate(kf.split(train_data))]

  folds['freq_models'] = [smf.glm(models_dict['freq_formula'],
                        data=train_data,
                        family=sm.families.Poisson(link = log()),
                        offset=np.log(train_data[exposure])).fit()
                        for train_data in folds['train_data']]

  folds['sev_models'] = [smf.glm(models_dict['sev_formula'],
                       data=train_data[train_data[numbers] > 0],
                       family=sm.families.Gamma(link = log()),
                       var_weights=train_data[train_data[numbers] > 0][numbers]).fit()
                       for train_data in folds['train_data']]

  folds['lc_models'] = [smf.glm(models_dict['lc_formula'],
                      data=train_data,
                      family=sm.families.Tweedie(link=sm.families.links.log(), var_power = 1.5),
                      offset=np.log(train_data[exposure])).fit()
                      for train_data in folds['train_data']]

  folds['pred_frequency'] = [freq_model.predict(test_data) for freq_model, test_data in zip(folds['freq_models'], folds['test_data'])]
  folds['pred_severity'] = [sev_model.predict(test_data) for sev_model, test_data in zip(folds['sev_models'], folds['test_data'])]
  folds['pred_loss_cost_freq_sev'] = [pred_frequency * pred_severity for pred_frequency, pred_severity in zip(folds['pred_frequency'], folds['pred_severity'])]
  folds['pred_loss_cost_lc'] = [lc_model.predict(test_data) for lc_model, test_data in zip(folds['lc_models'], folds['test_data'])]

  folds['obs_loss_cost'] = [test_data[losses] / test_data[exposure] for test_data in folds['test_data']]

  return folds

def define_thresholds(train_data, predicted_lr, premium):

  """
  The function creates the threshold for the loss ratio buckets.

  Input:
  train_data: The DataFrame with the train data.
  predicted_lr: The column name with the predicted loss ratio values.
  premium: The column name with the premium values.

  Output:
  The function returns the dataframe with the loss ratio thresholds and the relative metrics, eg. min, max, avg
  """

  train_data = train_data.sort_values('pred_lr')
  train_data['initial_buckets'] = pd.cut(np.cumsum(train_data[premium].astype(float)), 10, labels = np.linspace(1,10,10).astype('int'))

  lr_cond = [(train_data['initial_buckets'] == 1) | (train_data['initial_buckets'] == 2),
             (train_data['initial_buckets'] == 3) | (train_data['initial_buckets'] == 4),
             (train_data['initial_buckets'] == 5) | (train_data['initial_buckets'] == 6),
             (train_data['initial_buckets'] == 7) | (train_data['initial_buckets'] == 8),
             (train_data['initial_buckets'] == 9),
             (train_data['initial_buckets'] == 10)]

  lr_choice = ['1','2','3','4','5a','5b']

  train_data['initial_buckets'] = np.select(lr_cond, lr_choice, default=-1)

  lr_thresholds = train_data.groupby('initial_buckets', as_index=False).apply(lambda x:pd.Series({'min':np.min(x['pred_lr']),
                                                                                       'max': np.max(x['pred_lr']),
                                                                                       'avg': np.average(x['pred_lr'], weights = x['PREMIUM'])}))

  return lr_thresholds

def apply_thresholds(test_data, pred_lr, lr_thresholds):
  """
  The function applies the thresholds for the loss ratio buckets.

  Input:
  test_data: The DataFrame with the test data to score.
  pred_lr: The column name with the predicted loss ratio values.
  lr_thresholds: The thresholds object created with the 'define_thresholds' function.

  Output:
  The function returns the Series with the buckets.
  """

  lr_cond = [(test_data['pred_lr'] >= 0) & (test_data['pred_lr'] <= lr_thresholds['max'][0]),
             (test_data['pred_lr'] > lr_thresholds['max'][0]) & (test_data['pred_lr'] <= lr_thresholds['max'][1]),
             (test_data['pred_lr'] > lr_thresholds['max'][1]) & (test_data['pred_lr'] <= lr_thresholds['max'][2]),
             (test_data['pred_lr'] > lr_thresholds['max'][2]) & (test_data['pred_lr'] <= lr_thresholds['max'][3]),
             (test_data['pred_lr'] > lr_thresholds['max'][3]) & (test_data['pred_lr'] <= lr_thresholds['max'][4]),
             (test_data['pred_lr'] > lr_thresholds['max'][4]) & (test_data['pred_lr'] <= np.inf)]

  lr_choice = ['1','2','3','4','5a','5b']

  pred_bucket = np.select(lr_cond, lr_choice, default=-1)

  return pred_bucket

def cat_vars_base_level(train_data, cat_vars, exposure):

  """
  The function is used to set the base level for each categorical variable.

  Input:
  train_data: The DataFrame with the train data.
  cat_vars: The list of categorical variable names.  
  exposure: The column name with the exposure values.

  Output:
  The function returns a dictionary with each categorical variable and the relative base level. It should be used to define the model formula.
  """
  data_vars = train_data.copy()
  data_vars = data_vars.astype({k:str for k in cat_vars})

  ref_dict = {}

  for i in cat_vars:
    df = data_vars.groupby(i).agg({exposure : "sum"}).sort_values(exposure, ascending=False)
    print(f'{i} : {df.index[0]}')
    var = df.index[0]
    ref_dict[i] = var

  return ref_dict

def check_corr(data, vars):
  """
  The function is used to check the correlation between a set of variables.

  Input:
  data: The DataFrame with the train data.
  vars: The list of variable names.  

  Output:
  None: The function does not return anything. It is used to plot the correlation heatmap.
  """
  corr_data = pd.get_dummies(data[vars])

  plt.figure(figsize=(16, 6))
  heatmap = sns.heatmap(corr_data.corr(), vmin=-1, vmax=1, annot=True, cmap = sns.color_palette("Blues", 10))
  heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
  plt.show()

def plot_bucket_distr(pred_df, bucket_name, var, premium, exposure, losses, numbers, n = None):
  """
 The function creates the distribution plots for the buckets and a specific variable.
  
 Input:
 pred_df: The DataFrame with the scored data.
 bucket_name: The column name of the bucket score.
 var: The name of the specific variable.
 premium: The name of the column of the premium.
 exposure: The name of the column of the exposure.
 losses: The name of the column of the claim losses.
 numbers: The name of the column of the cliam numbers.
 
 Output:
 None: The function does not retrun anything. It used for the side effect of creating the distribution bucket plots.
  """

  pred_df1 = pred_df.copy()

  if n!=None:
    pred_df1[var] = pd.cut(pred_df[var].astype('float'), n, labels = np.linspace(1,n,n))

  df = pred_df1.groupby([var, 'pred_bucket'])\
          .agg(premium = (premium, 'sum'),
               losses = (losses, 'sum'),
               exposure = (exposure, 'sum'),
               numbers = (numbers, 'sum')).reset_index()

  df_join = pred_df1.groupby([var])\
          .agg(full_premium = (premium, 'sum'),
               full_losses = (losses, 'sum'),
               full_exposure = (exposure, 'sum'),
               full_numbers = (numbers, 'sum')).reset_index()
  plot_df = df.merge(df_join, right_on=var, left_on=var)

  plot_df['prem_perc'] = plot_df['premium'] / plot_df['full_premium']
  plot_df['loss_perc'] = plot_df['losses'] / plot_df['full_losses']
  plot_df['exp_perc'] = plot_df['exposure'] / plot_df['full_exposure']
  plot_df['numb_perc'] = plot_df['numbers'] / plot_df['full_numbers']


  prem_plot = plot_df[[var, 'pred_bucket', 'prem_perc']].pivot(index=var, columns='pred_bucket', values='prem_perc') * 100
  loss_plot = plot_df[[var, 'pred_bucket', 'loss_perc']].pivot(index=var, columns='pred_bucket', values='loss_perc') * 100
  exp_plot = plot_df[[var, 'pred_bucket', 'exp_perc']].pivot(index=var, columns='pred_bucket', values='exp_perc') * 100
  numb_plot = plot_df[[var, 'pred_bucket', 'numb_perc']].pivot(index=var, columns='pred_bucket', values='numb_perc') * 100

  fig, axs = plt.subplots(2, 2, figsize=(30,15))
  fig.suptitle('Distribution Analysis')
  single_distr(prem_plot, 'Premium', axs[0][0])
  single_distr(loss_plot, 'Losses', axs[0][1])
  single_distr(exp_plot, 'Exposure', axs[1][0])
  single_distr(numb_plot, 'Numbers', axs[1][1])

def augment_df(data, model, names = None, alpha = .05):
  """
  The function is used to add the prediction column to a dataset

  Input:
  data: The DataFrame to be used for prediction.
  model: The GLM model to be used for prediction.
  names: The names of the columns to be added to the dataframe in the following order:
         Name of the predicion mean.
         Name of the lower bound of the confidence interval.
         Name of the upper bound of the confidence interval.
  alpha (optional, default = .05): The value for the confidence interval.

  Output:
  The function returns the original dataset with the added columns.
  """
  preds = model.get_prediction(data).summary_frame(alpha = alpha)[['mean', 'mean_ci_lower', 'mean_ci_upper']]

  if names != None: preds.columns = names

  final_df = pd.concat([data.reset_index(drop = True), preds], axis = 1)

  return final_df

def evaluate_features(df_pred, var, pred_dict, obs_dict, n = None):

  """
  The function creates the plot to evaluate the observed vs the predicted values for a specific variables.

  Input:
  df_pred: The DataFrame with the predicted variables
  var: The name of the reference variable.
  pred_dict: The dictionary containing the predicted values to plot. 
             This can have two structures:
             1. A key-value pairs with the name of the model predictions to be displayed as the key and the name of the column in the dataframe containing the values as the value.
	     2. A key-value pairs with the name of the as the key following the convention `lower/_/upper` where lower and upper are the names to be displayed and a list with the column name of the lower and upper bounds of the predictions.
  obs_dict: A dictionary containing 2 elements. The keys are the name to be displayed and the values are the name of the columns with the actual values. The first elemnt should be the main values showed on the primary vertical axis and the second the values of the denominator showed on the second vertical axis.
  n (optional, default = None): The number of buckets to bin numerical variables if required.

  Output:
  The function does not return anything. It is used for the side effect of plotting the Observed vs the Expected Values.
  """

  pred_models = list(pred_dict.values())
  model_names = list(pred_dict.keys())
  conf_name = model_names[1].split('/_/')

  obs_names = list(obs_dict.keys())
  obs_values = list(obs_dict.values())

  obs = obs_values[0]
  exposure = obs_values[1]
  var_type = obs_names[0]
  y2_type = obs_names[1]

  df_pred1 = df_pred.copy()

  if n:
    df_pred1[var] = pd.cut(df_pred1[var].astype('float'), n, labels = np.linspace(1,n,n))

  if type(pred_models[1]) == str:
    eval_table = df_pred1.groupby([var]).apply(lambda x: pd.Series([np.sum(x[exposure])] + [0 if np.sum(x[exposure]) == 0 else np.average(x[obs], weights=x[exposure])] +
                                                                                      [0 if np.sum(x[exposure]) == 0 else np.average(x[i], weights=x[exposure]) for i in pred_models],
                                                                                   index = [exposure] + [obs] + model_names)).reset_index()
  else:
    eval_table = df_pred1.groupby([var]).apply(lambda x: pd.Series([np.sum(x[exposure]),
                                                                   0 if np.sum(x[exposure]) == 0 else np.average(x[obs], weights=x[exposure]),
                                                                   0 if np.sum(x[exposure]) == 0 else np.average(x[pred_models[0]], weights=x[exposure]),
                                                                   0 if np.sum(x[exposure]) == 0 else np.average(x[pred_models[1][0]], weights=x[exposure]),
                                                                   0 if np.sum(x[exposure]) == 0 else np.average(x[pred_models[1][1]], weights=x[exposure])],
                                                                   index = [exposure, obs, model_names[0], conf_name[0], conf_name[1]])).reset_index()

  fig, ax1 = plt.subplots(figsize=(15,10))

  ax2 = ax1.twinx()

  if type(pred_models[1]) == str:
    for i in [obs] + model_names:
      ax1.plot(eval_table[var], eval_table[i])
  else:
    ax1.plot(eval_table[var], eval_table[model_names[0]])
    ax1.plot(eval_table[var], eval_table[obs])
    ax1.fill_between(eval_table[var], eval_table[conf_name[0]], eval_table[conf_name[1]], alpha = .3)

  ax2.bar(eval_table[var], eval_table[exposure], alpha = .5, color = 'gray')

  ax1.set_xlabel(var)
  ax1.set_ylabel(var_type)
  ax2.set_ylabel(y2_type)

  if type(pred_models[1]) == str:
    ax1.legend([f'Observed {var_type}'] + model_names, loc=2)
  else:
    ax1.legend([model_names[0], f'Observed {var_type}'], loc=2)
  ax2.legend([y2_type], loc=1)
  ax1.set_xticks(eval_table[var])
  ax1.set_title(f'{var}: Observed vs Predicted {var_type}')
