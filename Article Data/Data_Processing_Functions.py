import matplotlib.pyplot as plt
import seaborn as sns


def convert_int(df, col):
  """
  # function to fill missing with 0 and to convert data type from float to int
  """
  df[col] = df[col].fillna(0)
  df[col] = df[col].astype(int)
  return df

def convert_float(df, col):
  """
  # function to fill missing with 0 and to convert data type from int to float
  """
  df[col] = df[col].fillna(0)
  df[col] = df[col].astype(float)
  return df

def convert_string(df, col):
  """
  # function to convert data type from int to int
  """
  df[col] = df[col].fillna(0)
  df[col] = df[col].astype(int)
  return df

def convert_int_to_str(df, var_list):
  """
  # function to convert data type from int to string
  """ 
  for var in var_list:
    df[var] = df[var].astype(str)
  return df


def make_dict(list1, list2):
  """
  # function to make dictionary by using two lists
  """
  new_dict = {}
  for i in range(len(list1)):
    new_dict[list1[i]] = list2[i]
  return new_dict

def change_contents(data, column, dictionary):
  """
  # function to chnage contents of columns by dictionary
  """
  data[column] = data[column].map(dictionary)
  return data


def cov(data,rho):
  """
  # create covariance matrix from data variance and correlation rho provided
  """    
  n = data.shape[1]  # 열의 개수
  cov_matrix = np.zeros((n, n))  # 공분산 행렬 초기화
    
  for i in range(n):
    for j in range(n):
      if i==j:
        cov_matrix[i, j] = data.iloc[:, i].std() * data.iloc[:, j].std() 
      else : 
        cov_matrix[i, j] = data.iloc[:, i].std() * data.iloc[:, j].std() * rho
  return cov_matrix


def multivariate(which_data,which_covariance):
  """
  # create multivariate normal distribution from data and covariance matrix provided
  """  
  mvn = multivariate_normal(mean = np.mean(which_data) , cov = which_covariance)
  multivariate_sample = mvn.rvs(size = len(which_data))
  data = pd.DataFrame(multivariate_sample)
  data.columns = which_data.columns
  return data    


#=======================================

# function to make scatter plot by groups with seaborn for data frame and show transparency
def scatter_by_group_sns(df, x, y, groupby, title, xlabel, ylabel, savepath):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(data=df, x=x, y=y, hue=groupby, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.savefig(savepath)
    plt.show()


# function to make scatter plot by groups with seaborn for data frame and show transparency and log scale
def scatter_by_group_sns_log(df, x, y, groupby, title, xlabel, ylabel, savepath):
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.scatterplot(data=df, x=x, y=y, hue=groupby, alpha=0.2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale('symlog')
    ax.set_yscale('symlog')
    plt.savefig(savepath)
    plt.show()


# function to make two histograms by groups in the same plot with seaborn
def ecdf_by_group_sns(df, x, groupby, title, xlabel, ylabel, savepath):
    #fig, ax = plt.subplots(figsize=(8, 8)))
    sns.displot(data=df, x=x, hue=groupby, kind="ecdf")
    #plt.xscale('log')
    #ax.set_xscale('log')
    #ax.set_title(title)
    #ax.set_xlabel(xlabel)
    #ax.set_ylabel(ylabel)
    #plt.savefig(savepath)
    #plt.show()


# function to make two histograms by groups in the same plot with seaborn
def kde_by_group_sns(df, x, groupby, title, xlabel, ylabel, savepath):
    #fig, ax = plt.subplots(figsize=(8, 8)))
    sns.displot(data=df, x=x, hue=groupby, kind="kde")
    #plt.xscale('log')
    #ax.set_xscale('log')
    #ax.set_title(title)
    #ax.set_xlabel(xlabel)
    #ax.set_ylabel(ylabel)
    #plt.savefig(savepath)
    #plt.show()
 
# function to make two scatter plots  by groups with seaborn for data frame side by side 
def scatter_by_group_sns_side(df, x, y, groupby, title, xlabel, ylabel, savepath):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8),sharex=True, sharey = True)
    sns.scatterplot(data=df.loc[df[groupby] == 'original'], x=x, y=y, hue=groupby, alpha=0.2, ax=ax[0])
    sns.scatterplot(data=df.loc[df[groupby] == 'synthetic'], x=x, y=y, hue=groupby, alpha=0.2, ax=ax[1])
    ax[0].set_title(title[0])
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[1].set_title(title[1])
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    #plt.savefig(savepath)
    plt.show()


# function to make two scatter plots  by groups with seaborn for data frame side by side 
def scatter_by_group_sns_side_symlog(df, x, y, groupby, title, xlabel, ylabel, savepath):
    fig, ax = plt.subplots(1, 2, figsize=(16, 8),sharex=True, sharey = True)
    sns.scatterplot(data=df.loc[df[groupby] == 'original'], x=x, y=y, hue=groupby, alpha=0.2, ax=ax[0])
    sns.scatterplot(data=df.loc[df[groupby] == 'synthetic'], x=x, y=y, hue=groupby, alpha=0.2, ax=ax[1])
    ax[0].set_title(title[0])
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel(ylabel)
    ax[0].set_xscale('symlog')
    ax[0].set_yscale('symlog')
    ax[1].set_title(title[1])
    ax[1].set_xlabel(xlabel)
    ax[1].set_ylabel(ylabel)
    ax[1].set_xscale('symlog')
    ax[1].set_yscale('symlog')
    #plt.savefig(savepath)
    plt.show()