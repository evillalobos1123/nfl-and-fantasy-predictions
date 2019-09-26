#import libraries
import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# define rankings by teams
dif_sta = {'TAM':6, 'GNB':21, 'NOR':26, 'KAN':31, 'BAL':18, 'WAS':15, 'NYJ':4, 'CIN':16, 'LAC':13, 'NWE':32, 'LAR':22,
           'MIN':29, 'IND':12, 'SEA':28, 'MIA':7, 'DEN':20, 'CAR':27, 'DAL':23, 'CHI':7, 'NYG':5, 'JAX':3, 'HOU':19,
           'OAK':9, 'PIT':30, 'ARI':17, 'ATL':24, 'BUF':10, 'CLE':1, 'PHI':25, 'DET':14, 'TEN':11, 'SFO':2}

dif_pass_def = {'TAM':7, 'GNB':21, 'NOR':4, 'KAN':2, 'BAL':28, 'WAS':18, 'NYJ':9, 'CIN':1, 'LAC':24, 'NWE':11, 'LAR':19,
           'MIN':30, 'IND':17, 'SEA':16, 'MIA':12, 'DEN':13, 'CAR':15, 'DAL':20, 'CHI':26, 'NYG':10, 'JAX':31, 'HOU':5,
           'OAK':14, 'PIT':23, 'ARI':29, 'ATL':6, 'BUF':1, 'CLE':8, 'PHI':3, 'DET':25, 'TEN':27, 'SFO':22}

pass_off_18 = {'TAM':3, 'GNB':20, 'NOR':12, 'KAN':1, 'BAL':26, 'WAS':28, 'NYJ':26, 'CIN':16, 'LAC':8, 'NWE':11, 'LAR':8,
           'MIN':10, 'IND':2, 'SEA':5, 'MIA':17, 'DEN':24, 'CAR':14, 'DAL':22, 'CHI':14, 'NYG':21, 'JAX':30, 'HOU':17,
           'OAK':24, 'PIT':5, 'ARI':30, 'ATL':3, 'BUF':32, 'CLE':11, 'PHI':11, 'DET':22, 'TEN':28, 'SFO':17}

pass_off_19 = {'TAM':18, 'GNB':18, 'NOR':12, 'KAN':3, 'BAL':1, 'WAS':3, 'NYJ':18, 'CIN':12, 'LAC':3, 'NWE':3, 'LAR':18,
           'MIN':18, 'IND':12, 'SEA':12, 'MIA':18, 'DEN':18, 'CAR':30, 'DAL':2, 'CHI':30, 'NYG':18, 'JAX':3, 'HOU':3,
           'OAK':18, 'PIT':30, 'ARI':12, 'ATL':12, 'BUF':18, 'CLE':18, 'PHI':3, 'DET':3, 'TEN':3, 'SFO':18}

dif_run_def_18 = {'TAM':9, 'GNB':11, 'NOR':31, 'KAN':6, 'BAL':29, 'WAS':16, 'NYJ':7, 'CIN':4, 'LAC':24, 'NWE':22, 'LAR':10,
           'MIN':18, 'IND':25, 'SEA':20, 'MIA':2, 'DEN':12, 'CAR':21, 'DAL':28, 'CHI':32, 'NYG':13, 'JAX':14, 'HOU':30,
           'OAK':3, 'PIT':27, 'ARI':1, 'ATL':8, 'BUF':17, 'CLE':5, 'PHI':26, 'DET':23, 'TEN':15, 'SFO':19}
#function that changes the ranking from 1 to 32
def change_rank(x):
    return (-1*(x-32))+1

#funcion that cleans the stats of qb and drop columns that ruins the model
def clean_stats_qb(run_stats, dic_def, dic_off):
    run_stats = run_stats.rename({'Unnamed: 7': 'Where'}, axis=1)
    run_stats = run_stats[(run_stats['Pos'] == 'QB') & (run_stats['Att'] >= 10)]
    result = []
    points_home = []
    points_away = []
    for x in run_stats.Result:
        result.append(x[0])
        reg1 = re.findall(r'\d*[^-]', x)
        points_home.append(reg1[2])
        points_away.append(reg1[3])
    run_stats['Result'] = result
    run_stats['Points_Team'] = points_home
    run_stats['Points_Opp'] = points_away
    run_stats.Where.fillna('home', inplace=True)
    run_stats2 = run_stats.reset_index(drop=True)
    stadium = [] 
    for i, x in enumerate(run_stats2.Where):
        if x == 'home':
            stadium.append(1)
        else:
            stadium.append(0)
    run_stats2['Where'] = stadium
    dif_list = []
    for i, x in enumerate(run_stats2.Where):
        if x == 0:
            dif_list.append(dif_sta[run_stats2.Opp[i]]+32)
        else:
            dif_list.append(change_rank(dif_sta[run_stats2.Tm[i]]))
    run_stats2['Stad_Diff'] = dif_list
    run_stats3 = run_stats2.drop(['Lg', 'G#', 'Rk', 'Pos', 'Age', 'Y/A', 'Yds.1', 'Rate', 'Cmp', 'AY/A'], axis=1)
    run_stats3['Month'] = [x.month for x in run_stats3.Date]
    run_stats3['Month'] = run_stats3['Month'].map({9:'September', 10:'October', 11:'November', 12:'December'})
    run_stats3['Points_Team'] = pd.to_numeric(run_stats3.Points_Team, errors='coerce')
    run_stats3['Points_Opp'] = pd.to_numeric(run_stats3.Points_Opp, errors='coerce')
    run_stats3['Result'] = run_stats3.Result.map({'W':1, 'L':0, 'T':2})
    run_stats3['Opp'] = run_stats3['Opp'].map(dic_def)
    off = [change_rank(dic_off[x]) for x in run_stats3.Tm]
    run_stats3['Tm'] = off
    run_stats4 = run_stats3.set_index(['Date', 'Week', 'Player'])
    run_stats5 = pd.get_dummies(run_stats4)
    run_stats6 = run_stats5.drop(['Result', 'Points_Opp'], axis=1)
    return run_stats6

#function that joins stats from 2018 and 2019 weeks 1 - 4
def clean_stats_qb19_w4(df1, w1):
    w1['Month_October'] = [0 for x in range(len(w1))]
    w1['Month_November'] = [0 for x in range(len(w1))]
    w1['Month_December'] = [0 for x in range(len(w1))]
    w1['Day_Sat'] = [0 for x in range(len(w1))]
    w1 = w1[list(df1.columns)]
    run = pd.concat([df1, w1], sort=False)
    return run

#funcion that cleans the stats of rb and drop columns that ruins the model
def clean_stats_rb(run_stats, dic_def):
    run_stats = run_stats[(run_stats['Yds'] >= 10) & (run_stats['Att'] >= 3)]
    run_stats = run_stats.rename({'Unnamed: 7': 'Where'}, axis=1) 
    result = []
    points_home = []
    points_away = []
    for x in run_stats.Result:
        result.append(x[0])
        reg1 = re.findall(r'\d*[^-]', x)
        points_home.append(reg1[2])
        points_away.append(reg1[3])
    run_stats['Result'] = result
    run_stats['Points_Team'] = points_home
    run_stats['Points_Opp'] = points_away
    run_stats.Where.fillna('home', inplace=True)
    run_stats2 = run_stats.reset_index(drop=True)
    stadium = [] 
    for i, x in enumerate(run_stats2.Where):
        if x == 'home':
            stadium.append(1)
        else:
            stadium.append(0)
    run_stats2['Where'] = stadium
    dif_list = []
    for i, x in enumerate(run_stats2.Where):
        if x == 0:
            dif_list.append(dif_sta[run_stats2.Opp[i]]+32)
        else:
            dif_list.append(change_rank(dif_sta[run_stats2.Tm[i]]))
    run_stats2['Stad_Diff'] = dif_list
    run_stats3 = run_stats2.drop(['Lg', 'Tm', 'G#', 'Rk', 'Pos', 'Age', 'Att'], axis=1)
    run_stats3['Month'] = [x.month for x in run_stats3.Date]
    run_stats3['Month'] = run_stats3['Month'].map({9:'September', 10:'October', 11:'November', 12:'December'})
    run_stats3['Points_Team'] = pd.to_numeric(run_stats3.Points_Team, errors='coerce')
    run_stats3['Points_Opp'] = pd.to_numeric(run_stats3.Points_Opp, errors='coerce')
    run_stats3['Result'] = run_stats3.Result.map({'W':1, 'L':0, 'T':2})
    run_stats3['Opp'] = run_stats3['Opp'].map(dic_def)
    run_stats4 = run_stats3.set_index(['Date', 'Week', 'Player'])
    run_stats5 = pd.get_dummies(run_stats4)
    run_stats6 = run_stats5.drop(['Result', 'Points_Opp'], axis=1)
    return run_stats6

#funcion that cleans the stats of wr and drop columns that ruins the model
def clean_stats_wr(run_stats, dic_def):
    run_stats = run_stats.rename({'Unnamed: 7': 'Where'}, axis=1)
    run_stats = run_stats[run_stats.Tgt >= 3]
    result = []
    points_home = []
    points_away = []
    for x in run_stats.Result:
        result.append(x[0])
        reg1 = re.findall(r'\d*[^-]', x)
        points_home.append(reg1[2])
        points_away.append(reg1[3])
    run_stats['Result'] = result
    run_stats['Points_Team'] = points_home
    run_stats['Points_Opp'] = points_away
    run_stats.Where.fillna('home', inplace=True)
    run_stats2 = run_stats.reset_index(drop=True)
    stadium = [] 
    for i, x in enumerate(run_stats2.Where):
        if x == 'home':
            stadium.append(1)
        else:
            stadium.append(0)
    run_stats2['Where'] = stadium
    dif_list = []
    for i, x in enumerate(run_stats2.Where):
        if x == 0:
            dif_list.append(dif_sta[run_stats2.Opp[i]]+32)
        else:
            dif_list.append(change_rank(dif_sta[run_stats2.Tm[i]]))
    run_stats2['Stad_Diff'] = dif_list
    run_stats3 = run_stats2.drop(['Lg', 'Tm', 'G#', 'Rk', 'Pos', 'Age', 'Tgt', 'Y/R', 'Rec'], axis=1)
    run_stats3['Month'] = [x.month for x in run_stats3.Date]
    run_stats3['Month'] = run_stats3['Month'].map({9:'September', 10:'October', 11:'November', 12:'December'})
    run_stats3['Points_Team'] = pd.to_numeric(run_stats3.Points_Team, errors='coerce')
    run_stats3['Points_Opp'] = pd.to_numeric(run_stats3.Points_Opp, errors='coerce')
    run_stats3['Result'] = run_stats3.Result.map({'W':1, 'L':0, 'T':2})
    run_stats3['Opp'] = run_stats3['Opp'].map(dic_def)
    run_stats4 = run_stats3.set_index(['Date', 'Week', 'Player'])
    run_stats5 = pd.get_dummies(run_stats4)
    run_stats6 = run_stats5.drop(['Result', 'Points_Opp'], axis=1)
    return run_stats6

def pass_for_pred(info, passw1_mod):
    cols = ['Day_Mon', 'Day_Sat', 'Day_Sun', 'Day_Thu', 'Month_December', 'Month_November', 
            'Month_October', 'Month_September']
    drew = passw1_mod.xs(info[0], level='Player')
    brees = pd.DataFrame(drew.describe().loc['mean', ['Att', 'Cmp%', 'Yds', 'TD', 'Int', 'Sk']]).T
    brees['Tm'] = change_rank(pass_off_19[info[1]])
    brees['Where'] = info[2]
    brees['Opp'] = dif_pass_def[info[3]]
    if info[2] == 0:
        brees['Stad_Diff'] = dif_sta[info[3]]
    else:
        brees['Stad_Diff'] = dif_sta[info[1]]
    for x in cols:
        brees[x] = 0
    brees['Month_'+info[4]]=1
    brees['Day_'+info[5]]
    pass_mod = passw1_mod.drop('Points_Team', axis=1)
    brees = brees[list(pass_mod.columns)]
    return brees

def run_for_pred(info, passw1_mod):
    cols = ['Day_Mon', 'Day_Sat', 'Day_Sun', 'Day_Thu', 'Month_December', 'Month_November', 
            'Month_October', 'Month_September']
    drew = passw1_mod.xs(info[0], level='Player')
    brees = pd.DataFrame(drew.describe().loc['mean', ['Yds', 'Y/A', 'TD']]).T
    brees['Tm'] = change_rank(pass_off_19[info[1]])
    brees['Where'] = info[2]
    brees['Opp'] = dif_pass_def[info[3]]
    if info[2] == 0:
        brees['Stad_Diff'] = dif_sta[info[3]]
    else:
        brees['Stad_Diff'] = dif_sta[info[1]]
    for x in cols:
        brees[x] = 0
    brees['Month_'+info[4]]=1
    brees['Day_'+info[5]]
    pass_mod = passw1_mod.drop('Points_Team', axis=1)
    brees = brees[list(pass_mod.columns)]
    return brees

def rec_for_pred(info, passw1_mod):
    cols = ['Day_Mon', 'Day_Sat', 'Day_Sun', 'Day_Thu', 'Month_December', 'Month_November', 
            'Month_October', 'Month_September']
    drew = passw1_mod.xs(info[0], level='Player')
    brees = pd.DataFrame(drew.describe().loc['mean', ['Yds', 'TD', 'Ctch%', 'Y/Tgt']]).T
    brees['Tm'] = change_rank(pass_off_19[info[1]])
    brees['Where'] = info[2]
    brees['Opp'] = dif_pass_def[info[3]]
    if info[2] == 0:
        brees['Stad_Diff'] = dif_sta[info[3]]
    else:
        brees['Stad_Diff'] = dif_sta[info[1]]
    for x in cols:
        brees[x] = 0
    brees['Month_'+info[4]]=1
    brees['Day_'+info[5]]
    pass_mod = passw1_mod.drop('Points_Team', axis=1)
    brees = brees[list(pass_mod.columns)]
    return brees

def mae_linreg(passw1_mod):
    y = passw1_mod.Points_Team
    X = passw1_mod.drop('Points_Team', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    linreg = LinearRegression()
    linreg.fit(X_train, y_train)
    pred_rec = linreg.predict(X_test)
    mse = mean_squared_error(y_test, pred_rec)
    rmse = mse**(1/2)
    mae = mean_absolute_error(y_test, pred_rec)
    return rmse, mae

def prediction_points(passw1_mod, brees):
    y = passw1_mod.Points_Team
    X = passw1_mod.drop('Points_Team', axis=1)
    linreg = LinearRegression()
    linreg.fit(X, y)
    pred_rec = linreg.predict(brees)
    return float(pred_rec)

def all_predictions(info):
    if info[-1] == 'QB':
        pass1 = pd.read_excel('data/pass/sportsref_pass1_18.xls')
        for x in range(1,12):
            c = pd.read_excel(f'data/pass/sportsref_downloadp({x}).xls', skiprows=1)
            pass1 = pd.concat([pass1, c], ignore_index = True, sort=False)
        pass_stats1 = clean_stats_qb(pass1, dif_pass_def, pass_off_18)
        w1_pass = clean_stats_qb(pd.read_excel(f'data/pass/sportsref_download19_{info[-2]}(1).xls', skiprows=1), 
                         dif_pass_def, pass_off_19)
        pass_19 = clean_stats_qb19_w4(pass_stats1, w1_pass)
        for_pred = pass_for_pred(info, pass_19)
        kyler_w2 = prediction_points(pass_19, for_pred)
        return kyler_w2
    elif info[-1] == 'RB':
        run1 = pd.read_excel('data/run/sportsref_rus1_18.xls')
        for x in range(1,12):
            b = pd.read_excel(f'data/run/sportsref_download ({x}).xls', skiprows=1)
            run1 = pd.concat([run1, b], ignore_index=True, sort=False)
        run_stats1 = clean_stats_rb(run1, dif_run_def_18)
        w1_run = clean_stats_rb(pd.read_excel(f'data/run/sportsref_download19_{info[-2]}(1).xls', skiprows=1), 
                         dif_run_def_18)
        run_19 = clean_stats_qb19_w4(run_stats1, w1_run)
        for_pred_rb = run_for_pred(info, run_19)
        david_w2 = prediction_points(run_19, for_pred_rb)
        return david_w2
    elif info[-1] == 'WR':
        rec1 = pd.read_excel('data/rec/sportsref_rec1_18.xls')
        for x in range(1,23):
            a = pd.read_excel(f'data/rec/sportsref_download ({x}).xls', skiprows=1)
            rec1 = pd.concat([rec1, a], ignore_index=True, sort=False)
        rec_stats1 = clean_stats_wr(rec1, dif_pass_def)
        w1_rec = clean_stats_wr(pd.read_excel(f'data/rec/sportsref_download19_{info[-2]}(1).xls', skiprows=1), 
                         dif_pass_def)
        rec_19 = clean_stats_qb19_w4(rec_stats1, w1_rec)
        for_pred_wr = rec_for_pred(info, rec_19)
        larry = prediction_points(rec_19, for_pred_wr)
        return larry
    
def final_df(team1, team2):
    columnas = ['Team', 'QB', 'RB', 'WR']
    df1 = [[team1[0][1],all_predictions(team1[0]),all_predictions(team1[1]),all_predictions(team1[2])], 
         [team2[0][1],all_predictions(team2[0]),all_predictions(team2[1]),all_predictions(team2[2])]]
    df2 = pd.DataFrame(df1, columns=columnas)
    df2['final'] = (df2.QB*.5)+(df2.RB*.3)+(df2.WR*.2)
    df_f = df2.set_index('Team')
    return df_f



#Changable
lj = ['Mitchell Trubisky', 'CHI', 0, 'WAS', 'September', 'Mon', 1, 'QB']
latm = ["David Montgomery", 'CHI', 0, 'WAS', 'September', 'Mon', 1, 'RB']
mb = ['Allen Robinson', 'CHI', 0, 'WAS', 'September', 'Mon', 1, 'WR']

new_pred = ['Case Keenum', 'WAS', 1, 'CHI', 'September', 'Mon', 1, 'QB']
david_pred = ['Adrian Peterson', 'WAS', 1, 'CHI', 'September', 'Mon', 1, 'RB']
larry_fit = ['Vernon Davis', 'WAS', 1, 'CHI', 'September', 'Mon', 1, 'WR']

away = [lj, latm, mb]
home = [new_pred, david_pred, larry_fit]

df_result = final_df(home, away)

print(df_result)


