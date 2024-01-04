
from sdmetrics.single_column import KSComplement
from sdmetrics.column_pairs import CorrelationSimilarity
from sdmetrics.single_column import RangeCoverage
from sdmetrics.single_column import CategoryCoverage
from sdmetrics.column_pairs import ContingencySimilarity
from sdmetrics.single_column import TVComplement
from sdmetrics.single_column import StatisticSimilarity
from sdmetrics.single_column import BoundaryAdherence
from sdmetrics.single_column import CSTest
from sdmetrics.single_table import NewRowSynthesis
from sdv.metadata import SingleTableMetadata # 싱글 메타 데이터의 경우에만! 멀티 메타 데이터의 경우에는 패키지를 다르게 불러와야함
from sdmetrics.reports.single_table import QualityReport 
from sdmetrics.reports.single_table import DiagnosticReport
from sdv.evaluation.single_table import evaluate_quality # Use this function to evaluate the quality of your synthetic data in terms of column shapes and correlations.
from sdv.evaluation.single_table import run_diagnostic # Check to see if the synthetic rows are pure copies of the real data
from sdv.evaluation.single_table import get_column_plot # Use this function to visualize a real column against the same synthetic column
from collections import Counter
import matplotlib.pyplot as plt
import myfunction as fo
import itertools

#====================================================================================================================

# pickle load 하는법 

def load_pickle(file_name) : 
    with open(file_name , 'rb') as f:
        return pickle.load(f)

# pickle save 하는법 

def save_pickle(data_name, data):
    with open(data_name, 'wb') as f:
        pickle.dump(data, f)

#====================================================================================================================        

# 검정통계량 함수

def statistic(original_data , synthetic_data , variable , percent): # percent는 리스트문으로 따로 지정
    
    if isinstance(synthetic_data , pd.DataFrame) : # synthetic_data가 단일 데이터인지 아닌지 여부

        original_desc = original_data[variable].describe(percentiles = percent).drop('count')
        synthetic_desc = synthetic_data[variable].describe(percentiles = percent).drop('count')
    
        bias = synthetic_desc - original_desc    
        relative_bias = (synthetic_desc - original_desc) / original_desc * 100
    
        tabular = pd.DataFrame({
                'original_statistic': original_desc,
                'synthetic_statistic': synthetic_desc,
                'bias_statistic': bias,
                'relative_bias_statistic': relative_bias})
        tabular = tabular.fillna(value = 0)
        
        numeric_indices_below_1 = [i for i in tabular.index if not (i in ['mean', 'std', 'min', 'max']) and float(i.replace('%', '')) <= 0.999]
        numeric_indices_below_2 = [i for i in tabular.index if not (i in ['mean', 'std', 'min', 'max']) and 0.999 <= float(i.replace('%', '')) <= 9.9]
        numeric_indices_below_3 = [i for i in tabular.index if not (i in ['mean', 'std', 'min', 'max']) and 10.0 <= float(i.replace('%', '')) <= 99.9]
    
        a = tabular.loc[numeric_indices_below_1]
        b = tabular.loc[numeric_indices_below_2]
        c = tabular.loc[numeric_indices_below_3]
    
        a_dataframe = pd.DataFrame(a.mean() , columns = ['0.01 ~ 0.99'])
        b_dataframe = pd.DataFrame(b.mean() , columns = ['1.00 ~ 9.99'])
        c_dataframe = pd.DataFrame(c.mean() , columns = ['10.0 ~ 99.9'])
            
        summary = pd.concat([a_dataframe , b_dataframe , c_dataframe] , axis = 1)
                                        
        return tabular , summary , list(summary.loc[:,'0.01 ~ 0.99'][5:7]) , list(summary.loc[:,'1.00 ~ 9.99'][5:7]) , list(summary.loc[:,'10.0 ~ 99.9'][5:7])

    elif isinstance(synthetic_data , list): # original_data가 리스트안에 여러 리스트로 이루어진 다수 데이터일 경우  
            
        num_datasets = len(synthetic_data)
        
        if isinstance(original_data , pd.DataFrame): 
            
            results = np.zeros((4, len(percent) + 4 , num_datasets)) # min , max ,std , mean이 고정으로 나오니까
                   
            for i in range(num_datasets) :
        
                original_desc = original_data[variable].describe(percentiles = percent).drop('count')
                synthetic_desc = synthetic_data[i][variable].describe(percentiles = percent).drop('count')
                
                diff = synthetic_desc - original_desc                
                relative_diff = (synthetic_desc - original_desc) / original_desc * 100
                
                results[0, :, i] = original_desc.values
                results[1, :, i] = synthetic_desc.values
                results[2, :, i] = diff.values
                results[3, :, i] = relative_diff.values
                            
            original_mean = np.mean(results[0, :, :], axis=1) # 자기자신에 대한 표준편차는 0이기 때문에 original_std의 식이 빠져있음
    
            synthetic_mean = np.mean(results[1, :, :], axis=1)
            synthetic_std = np.std(results[1, :, :], axis=1)
        
            bias_mean = np.mean(results[2, :, :], axis=1)
            bias_std = np.std(results[2, :, :], axis=1)
    
            relative_mean = np.mean(results[3, :, :], axis=1)
            relative_std = np.std(results[3, :, :], axis=1)    
            
            tabular = pd.DataFrame({'original_statistic': original_mean,
                                    'synthetic_statistic_mean': synthetic_mean,
                                    'synthetic_statistic_std': synthetic_std,
                                    'bias_statistic_mean': bias_mean,
                                    'bias_statistic_std': bias_std,
                                    'relative_bias_mean': relative_mean,
                                    'relative_bias_std': relative_std},
                                    index = ['mean', 'std', 'min'] + [f'{p:.1%}' for p in percent] + ['max'])
            tabular = tabular.fillna(value = 0)
            
            numeric_indices_below_1 = [i for i in tabular.index if not (i in ['mean', 'std', 'min', 'max']) and float(i.replace('%', '')) <= 0.999]
            numeric_indices_below_2 = [i for i in tabular.index if not (i in ['mean', 'std', 'min', 'max']) and 0.999 <= float(i.replace('%', '')) <= 9.9]
            numeric_indices_below_3 = [i for i in tabular.index if not (i in ['mean', 'std', 'min', 'max']) and 10.0 <= float(i.replace('%', '')) <= 99.9]
    
            a = tabular.loc[numeric_indices_below_1]
            b = tabular.loc[numeric_indices_below_2]
            c = tabular.loc[numeric_indices_below_3]
    
            a_dataframe = pd.DataFrame(a.mean() , columns = ['0.01 ~ 0.99'])
            b_dataframe = pd.DataFrame(b.mean() , columns = ['1.00 ~ 9.99'])
            c_dataframe = pd.DataFrame(c.mean() , columns = ['10.0 ~ 99.9'])
            
            summary = pd.concat([a_dataframe , b_dataframe , c_dataframe] , axis = 1)
                                        
            return tabular , summary , list(summary.loc[:,'0.01 ~ 0.99'][5:7]) , list(summary.loc[:,'1.00 ~ 9.99'][5:7]) , list(summary.loc[:,'10.0 ~ 99.9'][5:7])
            
        elif isinstance(original_data , list) : 
                
                results = np.zeros((4, len(percent) + 4 , num_datasets)) # min , max ,std , mean이 고정으로 나오니까   
                
                for i in range(num_datasets) :
                    
                    original_desc = original_data[i][variable].describe(percentiles = percent).drop('count')
                    synthetic_desc = synthetic_data[i][variable].describe(percentiles = percent).drop('count')
                    
                    diff = synthetic_desc - original_desc                
                    relative_diff = (synthetic_desc - original_desc) / original_desc * 100
                                        
                    results[0, :, i] = original_desc.values
                    results[1, :, i] = synthetic_desc.values
                    results[2, :, i] = diff.values
                    results[3, :, i] = relative_diff.values
                    
                original_mean = np.mean(results[0, :, :], axis=1)
                original_std = np.std(results[0, :, :], axis=1)
    
                synthetic_mean = np.mean(results[1, :, :], axis=1)
                synthetic_std = np.std(results[1, :, :], axis=1)
        
                bias_mean = np.mean(results[2, :, :], axis=1)
                bias_std = np.std(results[2, :, :], axis=1)
    
                relative_mean = np.mean(results[3, :, :], axis=1)
                relative_std = np.std(results[3, :, :], axis=1)    
            
                tabular = pd.DataFrame({'original_statistic_mean': original_mean,
                                        'original_statistic_std': original_std,
                                        'synthetic_statistic_mean': synthetic_mean,
                                        'synthetic_statistic_std': synthetic_std,
                                        'bias_statistic_mean': bias_mean,
                                        'bias_statistic_std': bias_std,
                                        'relative_bias_mean': relative_mean,
                                        'relative_bias_std': relative_std},
                                        index = ['mean', 'std', 'min'] + [f'{p:.1%}' for p in percent] + ['max'])
                tabular = tabular.fillna(value = 0)
                
                numeric_indices_below_1 = [i for i in tabular.index if not (i in ['mean', 'std', 'min', 'max']) and float(i.replace('%', '')) <= 0.999]
                numeric_indices_below_2 = [i for i in tabular.index if not (i in ['mean', 'std', 'min', 'max']) and 0.999 <= float(i.replace('%', '')) <= 9.9]
                numeric_indices_below_3 = [i for i in tabular.index if not (i in ['mean', 'std', 'min', 'max']) and 10.0 <= float(i.replace('%', '')) <= 99.9]
    
                a = tabular.loc[numeric_indices_below_1]
                b = tabular.loc[numeric_indices_below_2]
                c = tabular.loc[numeric_indices_below_3]
    
                a_dataframe = pd.DataFrame(a.mean() , columns = ['0.01 ~ 0.99'])
                b_dataframe = pd.DataFrame(b.mean() , columns = ['1.00 ~ 9.99'])
                c_dataframe = pd.DataFrame(c.mean() , columns = ['10.0 ~ 99.9'])
            
                summary = pd.concat([a_dataframe , b_dataframe , c_dataframe] , axis = 1)
                                        
                return tabular , summary , list(summary.loc[:,'0.01 ~ 0.99'][5:7]) , list(summary.loc[:,'1.00 ~ 9.99'][5:7]) , list(summary.loc[:,'10.0 ~ 99.9'][5:7])
        
#====================================================================================================================

def mean_statistic(data) : 
    numeric_indices_below_1 = [i for i in data.index if not (i in ['mean', 'std', 'min', 'max']) and float(i.replace('%', '')) <= 0.999]
    numeric_indices_below_2 = [i for i in data.index if not (i in ['mean', 'std', 'min', 'max']) and 0.999 <= float(i.replace('%', '')) <= 9.9]
    numeric_indices_below_3 = [i for i in data.index if not (i in ['mean', 'std', 'min', 'max']) and 10.0 <= float(i.replace('%', '')) <= 99.9]
    
    a = data.loc[numeric_indices_below_1]
    b = data.loc[numeric_indices_below_2]
    c = data.loc[numeric_indices_below_3]
    
    a_dataframe = pd.DataFrame(a.mean() , columns = ['0.01 ~ 0.99'])
    b_dataframe = pd.DataFrame(b.mean() , columns = ['1.00 ~ 9.99'])
    c_dataframe = pd.DataFrame(c.mean() , columns = ['10.0 ~ 99.9'])

    return pd.concat([a_dataframe , b_dataframe , c_dataframe] , axis = 1)

#====================================================================================================================

# 상자그림(평균 , 표준편차 , 분위수) 

def boxplot(statistic_data) :
    
    plt.rcParams['figure.figsize'] = (15, 15)
    plt.rcParams['font.size'] = 10
    fig, ax = plt.subplots()  
    label = statistic_data.columns
    ax.boxplot(statistic_data , labels = label)
    ax.set_xlabel('Data Type')
    ax.set_ylabel('Value')
    plt.title('statistic_boxplot')  
    plt.show()

    
#====================================================================================================================

# 산점도(왼쪽 원자료 , 오른쪽 재현자료)

def scatterplot(original_data , synthetic_data , variable1 , variable2) :

    if isinstance(original_data , pd.DataFrame) :
        
        plt.rcParams["figure.figsize"] = (20,20)
        plt.rcParams['font.size'] = 10

        plt.subplot(1,2,1)
        plt.scatter(original_data[variable1] , original_data[variable2] , color = 'purple' , alpha = 0.3 , label = 'original_data')
        plt.xlabel(variable1)
        plt.ylabel(variable2)  
        plt.legend()

        plt.subplot(1,2,2)
        plt.scatter(synthetic_data[variable1] , synthetic_data[variable2] , color = 'purple' , alpha = 0.3 , label = 'synthetic_data')
        plt.xlabel(variable1)
        plt.ylabel(variable2)  
        plt.legend()
        plt.show()
        
    elif isinstance(original_data , list) : # 리스트에서 첫번째 자료만 뽑아서 비교 할 수 있도록

        plt.rcParams["figure.figsize"] = (20,20)
        plt.rcParams['font.size'] = 10

        plt.subplot(1,2,1)
        plt.scatter(original_data[0][variable1] , original_data[0][variable2] , color = 'purple' , alpha = 0.3 , label = 'original_data')
        plt.xlabel(variable1)
        plt.ylabel(variable2)  
        plt.legend()

        plt.subplot(1,2,2)
        plt.scatter(synthetic_data[0][variable1] , synthetic_data[0][variable2] , color = 'purple' , alpha = 0.3 , label = 'synthetic_data')
        plt.xlabel(variable1)
        plt.ylabel(variable2)  
        plt.legend()

        plt.show()
        
#====================================================================================================================

# 여러가지 척도(sdv.metrics) 데이터프레임으로 표현


def KSComplements(original_data , synthetic_data , variable) : 
    
    if isinstance(synthetic_data , pd.DataFrame) : 

        result = KSComplement.compute(
                    real_data = original_data[variable],
                    synthetic_data = synthetic_data[variable])
        
        return result
                
    elif isinstance(synthetic_data , list) : 
        
        KSComplement_Result = []     
        num_datasets = len(synthetic_data)
        
        if isinstance(original_data , pd.DataFrame) : 
            
            for i in range(num_datasets) : 
                KSComplement_Result.append(
                KSComplement.compute(
                    real_data = original_data[variable],
                    synthetic_data = synthetic_data[i][variable]))
            
                result = np.array([[np.mean(KSComplement_Result) , np.std(KSComplement_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame
        
        elif isinstance(original_data , list):

            for i in range(len(original_data)) : 
                KSComplement_Result.append(
                KSComplement.compute(
                    real_data = original_data[i][variable],
                    synthetic_data = synthetic_data[i][variable]))
            
                result = np.array([[np.mean(KSComplement_Result) , np.std(KSComplement_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
            
            return frame
        
#====================================================================================================================

def CorrelationSimilaritys(original_data , synthetic_data , variable1 , variable2 , method) : 
    
    if isinstance(synthetic_data , pd.DataFrame) : 
        
        result = CorrelationSimilarity.compute(
                    real_data = original_data[[variable1 , variable2]],
                    synthetic_data = synthetic_data[[variable1 , variable2]],
                    coefficient = method)
        
        return result
    
    elif isinstance(synthetic_data , list) : 
        
        num_datasets = len(synthetic_data)
        CorrelationSimilarity_Result = [] 
        
        if isinstance(original_data , pd.DataFrame) : 

            for i in range(num_datasets) :             
                CorrelationSimilarity_Result.append(
                CorrelationSimilarity.compute(
                    real_data = original_data[[variable1 , variable2]],
                    synthetic_data = synthetic_data[i][[variable1 , variable2]], coefficient = method))
            
                result = np.array([[np.mean(CorrelationSimilarity_Result) , np.std(CorrelationSimilarity_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame
        
        elif isinstance(original_data , list) : 
            
            for i in range(num_datasets) :             
                CorrelationSimilarity_Result.append(
                CorrelationSimilarity.compute(
                    real_data = original_data[i][[variable1 , variable2]],
                    synthetic_data = synthetic_data[i][[variable1 , variable2]], coefficient = method))
            
                result = np.array([[np.mean(CorrelationSimilarity_Result) , np.std(CorrelationSimilarity_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame
                
#====================================================================================================================
    
def RangeCoverages(original_data , synthetic_data , variable) : 
    
    if isinstance(synthetic_data , pd.DataFrame) : 

        result = RangeCoverage.compute(
                    real_data = original_data[variable],
                    synthetic_data = synthetic_data[variable])
        
        return result
                
    elif isinstance(synthetic_data , list) : 
        
        num_datasets = len(synthetic_data)
        RangeCoverage_Result = []     
        
        if isinstance(original_data , pd.DataFrame) : 
            
            for i in range(num_datasets) : 
            
                RangeCoverage_Result.append(
                RangeCoverage.compute(
                    real_data = original_data[variable],
                    synthetic_data = synthetic_data[i][variable]))
            
                result = np.array([[np.mean(RangeCoverage_Result) , np.std(RangeCoverage_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame
        
        elif isinstance(original_data , list) : 
            
            for i in range(num_datasets) : 
                RangeCoverage_Result.append(
                RangeCoverage.compute(
                    real_data = original_data[i][variable],
                    synthetic_data = synthetic_data[i][variable]))
            
                result = np.array([[np.mean(RangeCoverage_Result) , np.std(RangeCoverage_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame
    
#====================================================================================================================

def BoundaryAdherences(original_data , synthetic_data , variable) : 
    
    if isinstance(synthetic_data , pd.DataFrame) : 

        result = BoundaryAdherence.compute(
                    real_data = original_data[variable],
                    synthetic_data = synthetic_data[variable])
        
        return result
                
    elif isinstance(synthetic_data , list) : 
        
        num_datasets = len(synthetic_data)
        BoundaryAdherence_Result = []     
        
        if isinstance(original_data , pd.DataFrame) : 
            
            for i in range(num_datasets) : 
            
                BoundaryAdherence_Result.append(
                BoundaryAdherence.compute(
                    real_data = original_data[variable],
                    synthetic_data = synthetic_data[i][variable]))
            
                result = np.array([[np.mean(BoundaryAdherence_Result) , np.std(BoundaryAdherence_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame
        
        elif isinstance(original_data , list) : 
            
            for i in range(num_datasets) : 
                BoundaryAdherence_Result.append(
                BoundaryAdherence.compute(
                    real_data = original_data[i][variable],
                    synthetic_data = synthetic_data[i][variable]))
            
                result = np.array([[np.mean(BoundaryAdherence_Result) , np.std(BoundaryAdherence_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame        
        
#====================================================================================================================

"""ColumnPair metrics based on Kullback–Leibler Divergence."""

import numpy as np
import pandas as pd
from scipy.special import kl_div

from sdmetrics.column_pairs.base import ColumnPairsMetric
from sdmetrics.goal import Goal
from sdmetrics.utils import get_frequencies


class ContinuousKLDivergence(ColumnPairsMetric):
    """Continuous Kullback–Leibler Divergence based metric.

    This approximates the KL divergence by binning the continuous values
    to turn them into categorical values and then computing the relative
    entropy. Afterwards normalizes the value applying ``1 / (1 + KLD)``.

    Attributes:
        name (str):
            Name to use when reports about this metric are printed.
        goal (sdmetrics.goal.Goal):
            The goal of this metric.
        min_value (Union[float, tuple[float]]):
            Minimum value or values that this metric can take.
        max_value (Union[float, tuple[float]]):
            Maximum value or values that this metric can take.
    """

    name = 'Continuous Kullback–Leibler Divergence'
    goal = Goal.MAXIMIZE
    min_value = 0.0
    max_value = 1.0

    @staticmethod
    def compute(real_data, synthetic_data, variable1 , variable2):
        """Compare two pairs of continuous columns using Kullback–Leibler Divergence.

        Args:
            real_data (pandas.DataFrame):
                The values from the real dataset, passed as pandas.DataFrame
                with 2 columns.
            synthetic_data (pandas.DataFrame):
                The values from the synthetic dataset, passed as a
                pandas.DataFrame with 2 columns.

        Returns:
            Union[float, tuple[float]]:
                Metric output.
        """
        real_data[pd.isna(real_data)] = 0.0
        synthetic_data[pd.isna(synthetic_data)] = 0.0
        column1 = variable1
        column2 = variable2
        
        real, xedges, yedges = np.histogram2d(real_data[column1], real_data[column2])
        synthetic, _, _ = np.histogram2d(
            synthetic_data[column1], synthetic_data[column2], bins=[xedges, yedges])

        f_obs, f_exp = synthetic.flatten() + 1e-5, real.flatten() + 1e-5
        f_obs, f_exp = f_obs / np.sum(f_obs), f_exp / np.sum(f_exp)

        return 1 / (1 + np.sum(kl_div(f_obs, f_exp)))

    @classmethod
    def normalize(cls, raw_score):
        """Return the `raw_score` as is, since it is already normalized.

        Args:
            raw_score (float):
                The value of the metric from `compute`.

        Returns:
            float:
                The normalized value of the metric
        """
        return super().normalize(raw_score)
    
#====================================================================================================================    

continous_k1_metric = ContinuousKLDivergence()

def KLDS(original_data , synthetic_data , variable1 , variable2) : 
    
    if isinstance(synthetic_data , pd.DataFrame) : 

        result = continous_k1_metric.compute(original_data.select_dtypes(int) , synthetic_data.select_dtypes(int) , variable1 , variable2) # 범주형 변수들도 같이 불러와질경우 계산 x int형식만 불러오기 

        return result
                
    elif isinstance(synthetic_data , list) : 
        
        num_datasets = len(synthetic_data)
        KLD_Result = []     
        
        if isinstance(original_data , pd.DataFrame) : 
            
            for i in range(num_datasets) : 
            
                KLD_Result.append(
                    continous_k1_metric.compute(original_data.select_dtypes(int) , synthetic_data[i].select_dtypes(int), variable1 , variable2))
                
                result = np.array([[np.mean(KLD_Result) , np.std(KLD_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame
        
        elif isinstance(original_data , list) : 
            
            for i in range(num_datasets) : 

                KLD_Result.append(
                     continous_k1_metric.compute(original_data[i].select_dtypes(int) , synthetic_data[i].select_dtypes(int), variable1 , variable2))
            
                result = np.array([[np.mean(KLD_Result) , np.std(KLD_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame

#====================================================================================================================

def boxplot(original_data , synthetic_data , variable) : 
    if isinstance(synthetic_data , pd.DataFrame):
        plt.rcParams['figure.figsize'] = [13, 8]
        fig, ax = plt.subplots()
        ax.boxplot([original_data[variable], synthetic_data[variable]] , notch=True)
        ax.set_xlabel('Data Type')
        ax.set_ylabel(variable)
        plt.xticks([1,2], ['original_data' , 'synthetic_data'])

        return ax.boxplot

    elif isinstance(synthetic_data , list):

        if isinstance(original_data , pd.DataFrame) : 

            plt.rcParams['figure.figsize'] = [13, 8]
            fig, ax = plt.subplots()
            ax.boxplot([original_data[variable] , synthetic_data[1][variable]] , notch=True)
            ax.set_xlabel('Data Type')
            ax.set_ylabel(variable)
            plt.xticks([1,2], ['original_data' , 'synthetic_data'])

        elif isinstance(original_data , list) : 

            plt.rcParams['figure.figsize'] = [13, 8]
            fig, ax = plt.subplots()
            ax.boxplot([original_data[1][variable] , synthetic_data[1][variable]] , notch=True)
            ax.set_xlabel('Data Type')
            ax.set_ylabel(variable)
            plt.xticks([1,2], ['original_data' , 'synthetic_data'])

#====================================================================================================================

def sex_bargraph(original_data , synthetic_data) : 
    
    if isinstance(synthetic_data, pd.DataFrame):
    
        oridata_men = original_data[original_data['sex']=='M'].value_counts()
        oridata_women = original_data[original_data['sex']=='F'].value_counts()
        syn_data_men = synthetic_data[synthetic_data['sex']=='M'].value_counts()
        syn_data_women = synthetic_data[synthetic_data['sex']=='F'].value_counts()

        gender = ["남성" , "여성"]
        original = [len(oridata_men),len(oridata_women)]
        synthetic_data = [len(syn_data_men),len(syn_data_women)]
        x_axis = np.arange(len(gender))
        plt.bar(x_axis + 0.20, original, width=0.2, label = 'Original_Data')
        plt.bar(x_axis + 0.20*2, synthetic_data, width=0.2, label = 'Synthetic_Data')
        plt.xticks(x_axis+0.3,gender)
        plt.legend()
        plt.show()

    elif isinstance(synthetic_data , list):

        if isinstance(original_data , pd.DataFrame) : 
            oridata_men = original_data[original_data['sex']=='M'].value_counts()
            oridata_women = original_data[original_data['sex']=='F'].value_counts()
            syn_data_men = synthetic_data[1][synthetic_data[1]['sex']=='M'].value_counts()
            syn_data_women = synthetic_data[1][synthetic_data[1]['sex']=='F'].value_counts()

            gender = ["남성" , "여성"]
            original = [len(oridata_men),len(oridata_women)]
            synthetic_data = [len(syn_data_men),len(syn_data_women)]
            x_axis = np.arange(len(gender))
            plt.bar(x_axis + 0.20, original, width=0.2, label = 'Original_Data')
            plt.bar(x_axis + 0.20*2, synthetic_data, width=0.2, label = 'Synthetic_Data')
            plt.xticks(x_axis+0.3,gender)
            plt.legend()
            plt.show()

        elif isinstance(original_data , list):

            oridata_men = original_data[1][original_data[1]['sex']=='M'].value_counts()
            oridata_women = original_data[1][original_data[1]['sex']=='F'].value_counts()
            syn_data_men = synthetic_data[1][synthetic_data[1]['sex']=='M'].value_counts()
            syn_data_women = synthetic_data[1][synthetic_data[1]['sex']=='F'].value_counts()

            gender = ["남성" , "여성"]
            original = [len(oridata_men),len(oridata_women)]
            synthetic_data = [len(syn_data_men),len(syn_data_women)]
            x_axis = np.arange(len(gender))
            plt.bar(x_axis + 0.20, original, width=0.2, label = 'Original_Data')
            plt.bar(x_axis + 0.20*2, synthetic_data, width=0.2, label = 'Synthetic_Data')
            plt.xticks(x_axis+0.3,gender)
            plt.legend()
            plt.show()   

#====================================================================================================================

def bar_frequencies(original_data , synthetic_data , variable , picture):

    if isinstance(synthetic_data , pd.DataFrame):
        
        a = Counter(original_data[variable])
        b = Counter(synthetic_data[variable])

        categories_1 = list(a.keys())
        counts_1 = [a[category] for category in categories_1]
        categories_2 = [category for category in categories_1]  # categories_1의 순서를 따름
        counts_2 = [b[category] for category in categories_2]

        bar_width = 0.35
        x_1 = np.arange(len(categories_1))
        x_2 = [x + bar_width for x in x_1]

        plt.bar(x_1, counts_1, bar_width, label='original_data')
        plt.bar(x_2, counts_2, bar_width, label='synthetic_data')

        # x 좌표 설정
        plt.xticks([x + bar_width/2 for x in x_1], categories_1)

        # 범례(legend) 추가
        plt.legend()

        # 그래프 제목과 축 레이블 설정
        plt.title('범주형 변수 막대 그래프')
        plt.xlabel('카테고리')
        plt.ylabel('빈도')
        plt.savefig(picture)
        # 그래프 표시
        plt.show()

    elif isinstance(synthetic_list , list):

        if isinstance(original_data , pd.DataFrame):

            a = Counter(original_data[variable])
            b = Counter(synthetic_data[1][variable])

            categories_1 = list(a.keys())
            counts_1 = [a[category] for category in categories_1]
            categories_2 = [category for category in categories_1]  # categories_1의 순서를 따름
            counts_2 = [b[category] for category in categories_2]

            bar_width = 0.35
            x_1 = np.arange(len(categories_1))
            x_2 = [x + bar_width for x in x_1]

            plt.bar(x_1, counts_1, bar_width, label='original_data')
            plt.bar(x_2, counts_2, bar_width, label='synthetic_data')

            # x 좌표 설정
            plt.xticks([x + bar_width/2 for x in x_1], categories_1)

            # 범례(legend) 추가
            plt.legend()

            # 그래프 제목과 축 레이블 설정
            plt.title('범주형 변수 막대 그래프')
            plt.xlabel('카테고리')
            plt.ylabel('빈도')
            plt.savefig(picture)
            # 그래프 표시
            plt.show()

        elif isinstance(original_data , list):

            a = Counter(original_data[1][variable])
            b = Counter(synthetic_data[1][variable])

            categories_1 = list(a.keys())
            counts_1 = [a[category] for category in categories_1]
            categories_2 = [category for category in categories_1]  # categories_1의 순서를 따름
            counts_2 = [b[category] for category in categories_2]

            bar_width = 0.35
            x_1 = np.arange(len(categories_1))
            x_2 = [x + bar_width for x in x_1]

            plt.bar(x_1, counts_1, bar_width, label='original_data')
            plt.bar(x_2, counts_2, bar_width, label='synthetic_data')

            # x 좌표 설정
            plt.xticks([x + bar_width/2 for x in x_1], categories_1)

            # 범례(legend) 추가
            plt.legend()

            # 그래프 제목과 축 레이블 설정
            plt.title('범주형 변수 막대 그래프')
            plt.xlabel('카테고리')
            plt.ylabel('빈도')
            plt.savefig(picture)
            # 그래프 표시
            plt.show()

#====================================================================================================================            

def CategoryCoverages(original_data , synthetic_data , variable) : 
    
    if isinstance(synthetic_data , pd.DataFrame) : 

        result = CategoryCoverage.compute(
                    real_data = original_data[variable],
                    synthetic_data = synthetic_data[variable])
        
        return result
                
    elif isinstance(synthetic_data , list) : 
        
        num_datasets = len(synthetic_data)
        CategoryCoverage_Result = []     
        
        if isinstance(original_data , pd.DataFrame) : 
            
            for i in range(num_datasets) : 
            
                CategoryCoverage_Result.append(
                    CategoryCoverage.compute(
                        real_data = original_data[variable],
                        synthetic_data = synthetic_data[i][variable]))
            
                result = np.array([[np.mean(CategoryCoverage_Result) , np.std(CategoryCoverage_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame
        
        elif isinstance(original_data , list) : 
            
            for i in range(num_datasets) : 
                CategoryCoverage_Result.append(
                    CategoryCoverage.compute(
                        real_data = original_data[i][variable],
                        synthetic_data = synthetic_data[i][variable]))
            
                result = np.array([[np.mean(CategoryCoverage_Result) , np.std(CategoryCoverage_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame
        
#==================================================================================================================== 

def TVComplements(original_data , synthetic_data , variable) : 
    
    if isinstance(synthetic_data , pd.DataFrame) : 

        result = TVComplement.compute(
                    real_data = original_data[variable],
                    synthetic_data = synthetic_data[variable])
        
        return result
                
    elif isinstance(synthetic_data , list) : 
        
        num_datasets = len(synthetic_data)
        TVComplement_Result = []     
        
        if isinstance(original_data , pd.DataFrame) : 
            
            for i in range(num_datasets) : 
            
                TVComplement_Result.append(
                TVComplement.compute(
                    real_data = original_data[variable],
                    synthetic_data = synthetic_data[i][variable]))
            
                result = np.array([[np.mean(TVComplement_Result) , np.std(TVComplement_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame
        
        elif isinstance(original_data , list) : 
            
            for i in range(num_datasets) : 
                TVComplement_Result.append(
                TVComplement.compute(
                    real_data = original_data[i][variable],
                    synthetic_data = synthetic_data[i][variable]))
            
                result = np.array([[np.mean(TVComplement_Result) , np.std(TVComplement_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame
        
#==================================================================================================================== 

def ContingencySimilaritys(original_data , synthetic_data , variable1 , variable2) : 
    
    if isinstance(synthetic_data , pd.DataFrame) : 

        result = ContingencySimilarity.compute(
            real_data = original_data[[variable1,variable2]], 
            synthetic_data = synthetic_data[[variable1,variable2]])

        return result
                
    elif isinstance(synthetic_data , list) : 
        
        num_datasets = len(synthetic_data)
        Contingency_Result = []     
        
        if isinstance(original_data , pd.DataFrame) : 
            
            for i in range(num_datasets) : 
            
                Contingency_Result.append(
                    ContingencySimilarity.compute(
                        real_data = original_data[[variable1,variable2]],
                        synthetic_data = synthetic_data[i][[variable1 , variable2]]))
                
                result = np.array([[np.mean(Contingency_Result) , np.std(Contingency_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame
        
        elif isinstance(original_data , list) : 
            
            for i in range(num_datasets) : 

                Contingency_Result.append(
                    ContingencySimilarity.compute(
                        real_data = original_data[i][[variable1,variable2]],
                        synthetic_data = synthetic_data[i][[variable1 , variable2]]))
            
                result = np.array([[np.mean(Contingency_Result) , np.std(Contingency_Result)]])
                frame = pd.DataFrame(result)
                frame.columns = ["척도의 평균" , "척도의 표준편차"]
                frame.index = ['value']
        
            return frame

#==================================================================================================================== 

def MSE(original_data ,synthetic_data , variable):
    
    if isinstance(synthetic_data , pd.DataFrame) : 

        mse = sum([(original_data[variable][i] - synthetic_data[variable]) ** 2 for i in range(len(original_data))]) / len(original_data)
    
        return mse
                
    elif isinstance(synthetic_data , list) : 
        
        num_datasets = len(synthetic_data)
        MSE_Result = []     
        
        if isinstance(original_data , pd.DataFrame) : 
            
            for i in range(num_datasets) : 
            
              MSE_Result.append(
                sum([(original_data[variable] - synthetic_data[i][variable]) ** 2 for j in range(len(original_data))]) / len(original_data))
    
            result = np.array([[np.mean(MSE_Result) , np.std(MSE_Result)]])
            frame = pd.DataFrame(result)
            frame.columns = ["척도의 평균" , "척도의 표준편차"]
            frame.index = ['value']
        
            return frame
        
    elif isinstance(original_data , list) : 
            
        for i in range(num_datasets) : 

            MSE_Result.append(
                sum([(original_data[i][variable] - synthetic_data[i][variable]) ** 2 for j in range(len(original_data))]) / len(original_data))
    
            result = np.array([[np.mean(MSE_Result) , np.std(MSE_Result)]])
            frame = pd.DataFrame(result)
            frame.columns = ["척도의 평균" , "척도의 표준편차"]
            frame.index = ['value']
            frame = pd.DataFrame(result)
            frame.columns = ["척도의 평균" , "척도의 표준편차"]
            frame.index = ['value']
        
            return frame

def MSE(original_data, synthetic_data, variable):
    if isinstance(synthetic_data, pd.DataFrame):
        mse = np.mean((original_data[variable] - synthetic_data[variable]) ** 2)
        return mse
                
    elif isinstance(synthetic_data, list):
        num_datasets = len(synthetic_data)
        MSE_Result = []
        
        if isinstance(original_data, pd.DataFrame):
            for i in range(num_datasets):
                mse = np.mean((original_data[variable] - synthetic_data[i][variable]) ** 2)
                MSE_Result.append(mse)
            
            result = [np.mean(MSE_Result), np.std(MSE_Result)]
            frame = pd.DataFrame([result], columns=["척도의 평균", "척도의 표준편차"], index=['value'])
            return frame
        
        elif isinstance(original_data, list):
            for i in range(num_datasets):
                mse = np.mean((np.array(original_data[i][variable]) - np.array(synthetic_data[i][variable])) ** 2)
                MSE_Result.append(mse)
            
            result = [np.mean(MSE_Result), np.std(MSE_Result)]
            frame = pd.DataFrame([result], columns=["척도의 평균", "척도의 표준편차"], index=['value'])
            return frame

def MAE(original_data, synthetic_data, variable):
    if isinstance(synthetic_data, pd.DataFrame):
        mae = np.mean(np.abs(original_data[variable] - synthetic_data[variable]))
        return mae
                
    elif isinstance(synthetic_data, list):
        
        num_datasets = len(synthetic_data)
        MAE_Result = []
        
        if isinstance(original_data, pd.DataFrame):
            
            for i in range(num_datasets):
                
                mae = np.mean(np.abs(original_data[variable] - synthetic_data[i][variable]))
                MAE_Result.append(mae)
            
                result = [np.mean(MAE_Result), np.std(MAE_Result)]
                frame = pd.DataFrame([result], columns=["척도의 평균", "척도의 표준편차"], index=['value'])
            return frame
        
        elif isinstance(original_data, list):
            for i in range(num_datasets):
                mae = np.mean(np.abs(np.array(original_data[i][variable]) - np.array(synthetic_data[i][variable])))
                MAE_Result.append(mae)
            
                result = [np.mean(MAE_Result), np.std(MAE_Result)]
                frame = pd.DataFrame([result], columns=["척도의 평균", "척도의 표준편차"], index=['value'])
            return frame

#========단일자료 vs 다중자료============================================================================================================ 

## 단일자료 
        
# 단일자료의 경우 연속형 변수 1개에 대한 평가 척도 함수 ( 방법으로는 KSComplement , RangeCoverage , BoundaryAdherence , MSE , MAE ) 추후에 더 추가 [CSTEST나 등등 ]

def one_variable_discrete_tabular(original_data, synthetic_data, *variables): # 변수 한개만 필요한 테이블 표 필요하면 추가하면 됨 
    results = {}
    
    for variable in variables:
        results[variable] = {
            'KSComplement': fo.KSComplements(original_data, synthetic_data, variable), # 1로 갈수록 좋음 
            'RangeCoverage': fo.RangeCoverages(original_data, synthetic_data, variable), # 1로 갈수록 좋음 
            'BoundaryAdherence': fo.BoundaryAdherences(original_data, synthetic_data, variable), # 1로 갈수록 좋음 
            'MSE': fo.MSE(original_data, synthetic_data, variable), # 작을수록 좋음 
            'MAE': fo.MAE(original_data, synthetic_data, variable) # 작을수록 좋음 
        }
    
    return pd.DataFrame(results).T

# 단일자료의 경우 연속형 변수 2개에 대한 평가 척도 함수 ( 방법으로는 KLDS , CorrelationSimilaritys ) 추후에 더 추가 

def multiple_variable_discrete_tabular(original_data, synthetic_data, *variables): # 변수 2개 이상 필요한 테이블표 필요하면 추가하면 됨 
    
    results = {}
    variable_combinations = list(itertools.combinations(variables, 2))
    
    for variable1, variable2 in variable_combinations:
        klds = fo.KLDS(original_data, synthetic_data, variable1, variable2)
        corr_sim = fo.CorrelationSimilaritys(original_data, synthetic_data, variable1, variable2, 'Pearson')
        
        key = f'{variable1} & {variable2}'
        results[key] = {'KLDS': klds, 'CorrelationSimilarity': corr_sim}
    
    result_df = pd.DataFrame(results).T
    
    return result_df

# 범주형 변수 1개에 대한 평가 척도 함수 ( 방법으로는 CategoryCoverages , TVComplement ) 추후에 더 추가 

def one_variable_categorical_tabular(original_data, synthetic_data, *variables): # 변수 한개만 필요한 테이블 표 필요하면 추가하면 됨 
    results = {}
    
    for variable in variables:
        results[variable] = {
            'CategoryCoverages': fo.CategoryCoverages(original_data, synthetic_data, variable),
            'TVComplement': fo.TVComplements(original_data, synthetic_data, variable)}
    
    return pd.DataFrame(results).T

# 범주형 변수 2개에 대한 평가 척도 함수 ( 방법으로는 ContingencySimilaritys ) 추후에 더 추가 

def multiple_variable_categorical_tabular(original_data, synthetic_data, *variables): # 변수 2개 이상 필요한 테이블표 필요하면 추가하면 됨 
    results = {}
    variable_combinations = list(itertools.combinations(variables, 2))
    
    for variable1, variable2 in variable_combinations:
        
        CS = fo.ContingencySimilaritys(original_data, synthetic_data, variable1, variable2)        
        key = f'{variable1} & {variable2}'
        results[key] = {'ContingencySimilarity': CS }
    
    result_df = pd.DataFrame(results).T
    
    return result_df

## 다중자료 

# 평가 척도들 딕셔너리 형태로 도출하는 함수 (BoudnaryAdherence , RangeCoverages , KSComplements , MSE , MAE ) # 추후에 위에 단일 자료와 똑같이 추가할 것 

def eval_one_variable_statistic(original_data, synthetic_data, variable): 
    

    # 각 연속형 평가척도들 key ,value값으로 한번에 저장하는 함수 밑에 있는 식으로 한번에 표현하기 위해 필요한 함수  
    # 선행 함수로는 myfunction에 있는 평가척도들 
    results = [fo.KSComplements(original_data , synthetic_data, variable),
               fo.RangeCoverages(original_data , synthetic_data, variable),
               fo.BoundaryAdherences(original_data, synthetic_data, variable),
               fo.MSE(original_data, synthetic_data, variable),
               fo.MAE(original_data, synthetic_data, variable)]
        
    return results

def eval_two_variable_statistic(original_data, synthetic_data, variable1 , variable2): 

    results = [fo.KLDS(original_data , synthetic_data, variable1, variable2),
               fo.CorrelationSimilaritys(original_data , synthetic_data, variable1 , variable2 , 'Pearson')]
        
    return results

# 위에서 배치마다 나온 결과들을 batch마다 변수하나에 저장해서 밑의 데이터에 넣음  

## batchsize를 정의해줘야함 list문 형태로 배치를 어떤것을 돌렸을지 모르기 때문에 # 데이터 3개 이상 확장에 대해 일반화는 아직 해결하지 못함 
### batchsize = [200,300,400]

def table_data_three(data1 , data2 , data3, variable , epoch , batchsize) :     
    
    data = {'KSComplement': list(np.ravel(data1[0].values)) + list(np.ravel(data2[0].values)) + list(np.ravel(data3[0].values)),
            'RangeCoverage': list(np.ravel(data1[1].values)) + list(np.ravel(data2[1].values)) + list(np.ravel(data3[1].values)),
            'BoundaryAdherence': list(np.ravel(data1[2].values)) + list(np.ravel(data2[2].values)) + list(np.ravel(data3[2].values)),
            'MSE': list(np.ravel(data1[3].values)) + list(np.ravel(data2[3].values)) + list(np.ravel(data3[3].values)),
            'MAE': list(np.ravel(data1[4].values)) + list(np.ravel(data2[4].values)) + list(np.ravel(data3[4].values))}

    # 멀티 인덱스를 가진 데이터프레임 생성
    index_names = [f'batch_{n}' for n in batchsize]
    index_tuples = [(name, metric) for name in index_names for metric in ["mean", "std"]]
    
    index = pd.MultiIndex.from_tuples(index_tuples, names=[f'epoch = {epoch}', variable])
    
    multi_dataframe = pd.DataFrame(data, index=index)

    return multi_dataframe

def table_two_data_three(data1 , data2 , data3, variable , epoch , batchsize) :     
    
    data = {'KLDS': list(np.ravel(data1[0].values)) + list(np.ravel(data2[0].values)) + list(np.ravel(data3[0].values)),
            'CorrelationSimilaritys': list(np.ravel(data1[1].values)) + list(np.ravel(data2[1].values)) + list(np.ravel(data3[1].values))}

    # 멀티 인덱스를 가진 데이터프레임 생성
    index_names = [f'batch_{n}' for n in batchsize]
    index_tuples = [(name, metric) for name in index_names for metric in ["mean", "std"]]
    
    index = pd.MultiIndex.from_tuples(index_tuples, names=[f'epoch = {epoch}', variable])
    
    multi_dataframe = pd.DataFrame(data, index=index)

    return multi_dataframe

# 범주형 변수

def eval_cate_one_variable_statistic(original_data, synthetic_data, variable): 
    

    # 각 범주형 평가척도들 key ,value값으로 한번에 저장하는 함수 밑에 있는 식으로 한번에 표현하기 위해 필요한 함수  
    # 선행 함수로는 myfunction에 있는 평가척도들 
    results = [fo.CategoryCoverages(original_data , synthetic_data, variable),
               fo.TVComplements(original_data, synthetic_data, variable)]
        
    return results

def table_cate_data_three(data1 , data2 , data3, variable , epoch , batchsize) :     
    
    data = {'CategoryCoverages': list(np.ravel(data1[0].values)) + list(np.ravel(data2[0].values)) + list(np.ravel(data3[0].values)),
            'TVComplements': list(np.ravel(data1[1].values)) + list(np.ravel(data2[1].values)) + list(np.ravel(data3[1].values))}

    # 멀티 인덱스를 가진 데이터프레임 생성
    index_names = [f'batch_{n}' for n in batchsize]
    index_tuples = [(name, metric) for name in index_names for metric in ["mean", "std"]]
    
    index = pd.MultiIndex.from_tuples(index_tuples, names=[f'epoch = {epoch}', variable])
    
    multi_dataframe = pd.DataFrame(data, index=index)

    return multi_dataframe

def table_cate_two_data_three(data1 , data2 , data3, variable , epoch , batchsize) :     
    
    data = {'ContingencySimilaritys': list(np.ravel(data1[0].values)) + list(np.ravel(data2[0].values)) + list(np.ravel(data3[0].values))}

    # 멀티 인덱스를 가진 데이터프레임 생성
    index_names = [f'batch_{n}' for n in batchsize]
    index_tuples = [(name, metric) for name in index_names for metric in ["mean", "std"]]
    
    index = pd.MultiIndex.from_tuples(index_tuples, names=[f'epoch = {epoch}', variable])
    
    multi_dataframe = pd.DataFrame(data, index=index)

    return multi_dataframe

#==================================================================================================================== 

# 범주형 빈도수 표

## 단일자료와 다중자료에 대해서

def calculate_multi_categorical_frequencies(original_data, synthetic_data, category_variable):
    if isinstance(synthetic_data, pd.DataFrame):
        # Original data
        ori_frequencies = original_data[category_variable].value_counts()
        ori_df = pd.DataFrame({'범주': ori_frequencies.index, '빈도': ori_frequencies.values})

        # Synthetic data
        syn_frequencies = synthetic_data[category_variable].value_counts()
        syn_df = pd.DataFrame({'범주': syn_frequencies.index, '빈도': syn_frequencies.values})

        # Calculate relative frequencies (퍼센트)
        ori_df['상대빈도(퍼센트)'] = (ori_df['빈도'] / ori_df['빈도'].sum()) * 100
        syn_df['상대빈도(퍼센트)'] = (syn_df['빈도'] / syn_df['빈도'].sum()) * 100
        
        # Convert category_variable to categorical with original order
        # ori_df['범주'] = pd.Categorical(syn_df['범주'], categories=ori_df['범주'], ordered=True)
        # ori_df = ori_df.sort_values(by='범주')

        result = pd.concat([ori_df, syn_df], axis=1, keys=['원자료', '재현자료'])

        return result

    elif isinstance(synthetic_data, list):
        num_datasets = len(synthetic_data)
        syn_list = []
        syn_df = []

        if isinstance(original_data, pd.DataFrame):
            ori_frequencies = original_data[category_variable].value_counts()
            ori_df = pd.DataFrame({'범주': ori_frequencies.index, '빈도': ori_frequencies.values})
            ori_df['상대빈도(퍼센트)'] = (ori_df['빈도'] / ori_df['빈도'].sum()) * 100

            for i in range(100):
                syn_list.append(synthetic_data[i][category_variable].value_counts())

            for j in range(100):
                syn_df.append(pd.DataFrame({'범주': syn_list[j].index, '빈도': syn_list[j].values}))

            for k in range(100):
                syn_df[k]['상대빈도(퍼센트)'] = (syn_df[k]['빈도'] / syn_df[k]['빈도'].sum()) * 100

            combined_syn_df = pd.concat(syn_df, axis=0)
            # combined_syn_df['범주'] = pd.Categorical(combined_syn_df['범주'], categories=ori_df['범주'], ordered=True)
            # combined_syn_df = combined_syn_df.sort_values(by='범주')

            syn_result = combined_syn_df.groupby('범주').agg({'std', 'mean'}).reset_index()

            result = pd.concat([ori_df, syn_result], axis=1)

            return result




