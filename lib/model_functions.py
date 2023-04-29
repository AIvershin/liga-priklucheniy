import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import shap
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error,mean_absolute_percentage_error

from IPython.display import clear_output

import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', 500)


# Абсолютный рост, темпы роста и прироста
def gen_growth(data,features):
    """
    Функция генерирует динамику признаков
    """
    data_growth=pd.DataFrame(data['period'][1:],columns=['period'])
    for c in features:
        data_growth[f'abs_growth_{c}']=data[c][1:].values-data[c][0:-1].values
        data_growth[f'index_{c}']=data[c][1:].values/data[c][0:-1].values
        data_growth[f'delta_rate_{c}']=(data[c][1:].values/data[c][0:-1].values-1)*100
        
    return data_growth

# Лаги
def gen_lag(data,features,lag):
    """
    Функция генерирует лаги признаков
    """
    data_lag=pd.DataFrame(data['period'][lag:],columns=['period'])
    for c in features:
        data_lag[f'l{lag}_{c}']=data[c][:-lag].values
        
    return data_lag

# Логарифмы
def gen_log(data,features):
    """
    Функция генерирует логарифмы признаков
    """
    data_log=pd.DataFrame(data['period'],columns=['period'])
    for c in features:
        data_log[f'log_{c}']=data[c].apply(lambda x: np.log(x) if x>0 else 0)
    
    return data_log

def calculate_permutation_importance(data: pd.DataFrame,features: list,
                                     n_iter: int = None, plot_pi: bool = False) -> pd.DataFrame:
    """
    Функция считает permutation importance для XGBRegressor
    ---------------------------------------------
    Пример использования:
    df_perm = calculate_permutation_importance(data_quarter,data_quarter.drop(['period','target'],axis=1).columns.tolist(),n_iter=5)
    """
    X_train=data[features].copy()
    y_train=data['target'].copy()
    
    regressor=XGBRegressor(eval_metric='rmse')
    param_grid = {"max_depth":    [3,4, 5],
              "n_estimators": [100,200,300,400,500, 600, 700],
              "learning_rate": [0.001,0.01, 0.015]}
    search = GridSearchCV(regressor, param_grid, cv=5).fit(X_train, y_train)
    
    regressor=XGBRegressor(learning_rate = search.best_params_["learning_rate"],
                           n_estimators  = search.best_params_["n_estimators"],
                           max_depth     = search.best_params_["max_depth"],)    
    regressor.fit(X_train, y_train)
    
      
    permutation_importance = np.zeros(X_train.shape[1])
    baseline_prediction = regressor.predict(X_train)
    baseline_score = np.sqrt(mean_squared_error(y_train, baseline_prediction))
    
    for num, feature in enumerate(features):
        feature_copy = X_train[feature].copy()
        score = []
        for i in range(n_iter):
            np.random.seed(i)
            X_train[feature] = np.random.permutation(X_train[feature])
            score.append(np.sqrt(mean_squared_error(y_train, regressor.predict(X_train))))
        permutation_importance[num] = np.mean(score)
        X_train[feature] = feature_copy
    
    permutation_importance = (permutation_importance-baseline_score) * 100
    permutation = pd.DataFrame(data=permutation_importance, index=X_train.columns). \
                     sort_values(by=0, ascending=False).rename({0: 'permutation_value'}, axis='columns')
    if plot_pi:
        plt.figure(figsize=(15, 8))
        plt.plot(range(1, len(permutation_importance)+1), permutation['permutation_value'], 
                 alpha=0.5, color='blue', marker='o')
        plt.grid()
        plt.xlabel('Номер фактора')
        plt.ylabel('Разница RMSE difference')
        plt.title('Permutation_importance')
        plt.show()   
    
    return permutation

def permutation_selection(data: pd.DataFrame, features: list, threshold_corr: float = 0.7, 
                          threshold_pi: float = None, N_feat: int = None)  -> list:
    '''
    Функция отбирает топ признаков по permutation importance, необходимо задать или порог, или число отбираемых признаков
    Функция также удаляет коррелирующие признаки: из пары признаков удаляется тот, который имеет permutation importance меньше
    ---------------------------------------------
    Пример использования:
    df_perm= permutation_selection(data_quarter,data_quarter.drop(['period','target'],axis=1).columns.tolist(), threshold_corr= 0.9,
    N_feat= 10) 
    '''
    
    dataset = data.copy()
    #dataset['y'] = y.copy()
    corr = dataset[features].fillna(0).corr()
    features_drop = []
    permutation_importance = calculate_permutation_importance(data, features, 5)
    
    for feature_1 in range(corr.shape[0]-2):
        for feature_2 in range(feature_1+1, corr.shape[0]-1):
            if abs(corr.iloc[feature_1, feature_2]) > threshold_corr:
                if permutation_importance.iloc[feature_1, 0] < permutation_importance.iloc[feature_2, 0]:
                    features_drop.append(corr.columns[feature_1])
                else:
                    features_drop.append(corr.columns[feature_2])
                 
    print(f'Были удалены следующие признаки: {features_drop}')
    
    features = list(set(features).difference(set(features_drop)))
    
    permutation_importance = calculate_permutation_importance(data, features, 10,plot_pi=True)    
    
    if threshold_pi:
        permutation_importance = permutation_importance[threshold_corr['permutation_value'] > threshold_pi]
    else:
        permutation_importance = permutation_importance.sort_values(by='permutation_value', ascending=False). \
                iloc[:N_feat, :]
    print(f'Выбрано по permutation importance {permutation_importance.shape[0]} признаков: ', permutation_importance, sep='\n')   
    
    return permutation_importance.index.tolist()    

def calculate_shap_importance(data: pd.DataFrame,features: list, plot_shap: bool = False) -> pd.DataFrame:
    """
    Функция считает shap values для XGBRegressor
    ---------------------------------------------
    Пример использования:
    df_shap = calculate_shap_importance(data_quarter,data_quarter.drop(['period','target'],axis=1).columns.tolist())
    """
    X_train=data[features].copy()
    y_train=data['target'].copy()
    
    regressor=LGBMRegressor(eval_metric='rmse')
#     param_grid = {"max_depth":    [3,4, 5],
#               "n_estimators": [100,200,300,400,500, 600, 700],
#               "learning_rate": [0.001,0.01, 0.015]}
#     search = GridSearchCV(regressor, param_grid, cv=5,verbose=-1).fit(X_train, y_train)
    
#     regressor=LGBMRegressor(learning_rate = search.best_params_["learning_rate"],
#                            n_estimators  = search.best_params_["n_estimators"],
#                            max_depth     = search.best_params_["max_depth"],)    
    regressor=LGBMRegressor()
    regressor.fit(X_train, y_train)
    
    explainer = shap.TreeExplainer(regressor)
    shap_values = explainer.shap_values(X_train)  
    resultX = pd.DataFrame(shap_values, columns = X_train.columns.tolist())
    vals = np.abs(resultX.values).mean(0)
    shap_importance = pd.DataFrame(list(zip(X_train.columns.tolist(), vals)),
                                      columns=['col_name','shap_value']).set_index('col_name')
    shap_importance.sort_values(by=['shap_value'],
                               ascending=False, inplace=True)
        
    if plot_shap:
        shap.summary_plot(shap_values, X_train, plot_type="bar")
    return shap_importance

def shap_selection(data: pd.DataFrame, features: list, threshold_corr: float = 0.7, 
                   threshold_shap: float = None, N_feat: int = None)  -> list:
    '''
    Функция отбирает топ признаков по shap value, необходимо задать или порог, или число отбираемых признаков
    Функция также удаляет коррелирующие признаки: из пары признаков удаляется тот, который имеет меньше shap value
    ---------------------------------------------
    Пример использования:
    short_list_shap = shap_selection(data_quarter,data_quarter.drop(['period','target'],axis=1).columns.tolist(), 
    threshold_corr= 0.9,  N_feat= 10)
    '''
    
    dataset = data.copy()
    #dataset['y'] = y.copy()
    corr = dataset[features].fillna(0).corr()
    features_drop = []
    shap_importance = calculate_shap_importance(data, features)
    
    for feature_1 in range(corr.shape[0]-2):
        for feature_2 in range(feature_1+1, corr.shape[0]-1):
            if abs(corr.iloc[feature_1, feature_2]) > threshold_corr:
                if shap_importance.iloc[feature_1, 0] < shap_importance.iloc[feature_2, 0]:
                    features_drop.append(corr.columns[feature_1])
                else:
                    features_drop.append(corr.columns[feature_2])
                 
    print(f'Были удалены следующие признаки: {features_drop}')
    
    features = list(set(features).difference(set(features_drop)))
    
    shap_importance = calculate_shap_importance(data, features, plot_shap=True)    
    
    if threshold_shap:
        shap_importance = shap_importance[threshold_corr['shap_value'] > threshold_pi]
    else:
        shap_importance = shap_importance.sort_values(by='shap_value', ascending=False). \
                iloc[:N_feat, :]
    print(f'Выбрано по shap importance {shap_importance.shape[0]} признаков: ', shap_importance, sep='\n')   
    
    return shap_importance.index.tolist()    

def greedy_selection_backward(regressor, data: pd.DataFrame, target: str, features: list, valid: pd.DataFrame,
    num_features_to_select: int, step: int, fit_params: dict, eval_set: pd.DataFrame = None):
    """
    Функция производит жадный отбор признаков с помощью backward_selection
    ---------------------------------------------
    Пример использования:
    greedy_selection_backward(XGBRegressor, data_quarter, 'target', data_quarter.drop(['period','target'],axis=1).columns.tolist(),  
    valid= data_quarter,  num_features_to_select=10,  step=1, 
    params={'objective': 'reg:squarederror','learning_rate': 0.01,'max_depth': 3,'n_estimators': 300,'random_state': 0}, fit_params={})
    """
    
    train_hist = []
    valid_hist = []
    x_axis = []
    
    for num in range(len(features), num_features_to_select, -step): 
        score = dict()
        score_train = dict()
        for _ in range(num):
            drop_feature = features.pop(0)
            model = regressor()
            if eval_set is None:
                model.fit(data[features], data[target], **fit_params)
            else:
                model.fit(data[features], data[target], eval_set=[(eval_set[features], eval_set[target])], **fit_params)
                
            score[drop_feature] = np.sqrt(mean_squared_error(valid[target], model.predict(valid[features])))
            score_train[drop_feature] = np.sqrt(mean_squared_error(data[target], model.predict(data[features])))
            features.append(drop_feature)
        
        features = pd.Series(data=score).sort_values(ascending=False).index.tolist()[:num-1]
        valid_hist.append(pd.Series(data=score).sort_values(ascending=False).tolist()[num-1])
        train_hist.append(score_train[pd.Series(data=score).sort_values(ascending=False).index.tolist()[num-1]])
        
        x_axis.append(num-1)
        clear_output(wait=True)
        plt.figure(figsize=(15, 8))
        plt.plot(x_axis, valid_hist, label='RMSE valid', marker='o')
        plt.plot(x_axis, train_hist, label='RMSE train', marker='o')
        plt.legend()
        plt.grid()
        plt.xlabel('Number of factors')
        plt.ylabel('RMSE')
        plt.title('Uplift')
        plt.show()
        print(f'Selected {len(features)} features: \n {features} \n RMSE: {round(valid_hist[-1], 5)}')
        
    return features

def greedy_selection_forward(regressor, data: pd.DataFrame, target: str, features: list, valid: pd.DataFrame, 
    num_features_to_select: int, step: int, fit_params: dict, eval_set: pd.DataFrame = None):
    """
    Функция производит жадный отбор признаков с помощью backward_selection
    ---------------------------------------------
    Пример использования:
    greedy_selection_forward(XGBRegressor, data_quarter, 'target', data_quarter.drop(['period','target'],axis=1).columns.tolist(),  
    valid= data_quarter,  num_features_to_select=10,  step=1, 
    params={'objective': 'reg:squarederror','learning_rate': 0.01,'max_depth': 3,'n_estimators': 300,'random_state': 0}, fit_params={})
    """
    
    best_features = []
    train_hist = []
    valid_hist = []
    x_axis = []
    
    for num in range(0, num_features_to_select, step): 
        score = dict()
        score_train = dict()
        for _ in range(len(features)):
            feature = features.pop(0)
            model = regressor()
            
            if eval_set is None:
                model.fit(data[best_features+[feature]], data[target], **fit_params)
            else:
                model.fit(data[best_features+[feature]], data[target], 
                          eval_set=[(eval_set[best_features+[feature]], eval_set[target])], **fit_params)
                
            score[feature] = np.sqrt(mean_squared_error(valid[target], model.predict(valid[best_features+[feature]])))
            score_train[feature] = np.sqrt(mean_squared_error(data[target], model.predict(data[best_features+[feature]])))
            features.append(feature)
        
        
        best_features.append(pd.Series(data=score).sort_values(ascending=True).index.tolist()[0])
        features.remove(best_features[-1])
        valid_hist.append(pd.Series(data=score).sort_values(ascending=True).tolist()[0])
        train_hist.append(score_train[best_features[-1]])
        
        x_axis.append(num+1)
        clear_output(wait=True)
        plt.figure(figsize=(15, 8))
#         plt.plot(x_axis, valid_hist, label='RMSE valid', marker='o')
        plt.plot(x_axis, train_hist, label='RMSE train', marker='o')
        plt.legend()
        plt.grid()
        plt.xlabel('Number of factors')
        plt.ylabel('RMSE')
        plt.title('Uplift')
        plt.show()
        print(f'Selected {len(best_features)} features: \n {best_features} \n RMSE: {round(valid_hist[-1], 5)}')
        
    return best_features

def regression_report(y_train,y_train_pred,y_test,y_test_pred):
    """
    Функция считает метрики качества на обучающих и тестовых данных
    """
    metrics = []
    col_list=[]

    st = {
           'MAE': round(mean_absolute_error(y_train,y_train_pred),2),
           'MSE': round(mean_squared_error(y_train,y_train_pred),2),
           'MAPE': round(mean_absolute_percentage_error(y_train,y_train_pred)*100,2)
           }

    metrics += [st]
    col_list.append('Train')

    st = {
          'MAE': round(mean_absolute_error(y_test,y_test_pred),2),
          'MSE': round(mean_squared_error(y_test,y_test_pred),2),
          'MAPE': round(mean_absolute_percentage_error(y_test,y_test_pred)*100,2)
           }

    metrics += [st]
    col_list.append('Test')
    
    results = pd.DataFrame(metrics).T
    results.columns = col_list
    results['Relative Difference']=round(abs(results['Train']-results['Test'])/results['Train'],2)
    
    return results
    
