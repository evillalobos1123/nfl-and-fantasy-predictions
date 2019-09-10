import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import re

def team_df1(url, team):

    data_att = requests.get(url).content
    soup_att = BeautifulSoup(data_att, "html")
    text_att = soup_att.find_all(['td'])
    text_att = [x.text for x in text_att]
    ls = [x.replace('\n','') for x in text_att]
    ls2 = [x.replace(' ', '') for x in ls]
    txt = [(ls2[x:x+15]) for x in range(0, len(ls2), 15)]
    return txt

teams_names = ['Arizona Cardinals', 'Atlanta Falcons', 'Baltimore Ravens', 
               'Buffalo Bills', 'Carolina Panthers', 'Chicago Bears', 
               'Cincinnati Bengals', 'Cleveland Browns', 'Dallas Cowboys', 
               'Denver Broncos', 'Detroit Lions', 'Green Bay Packers', 
               'Houston Texans', 'Indianapolis Colts', 'Jacksonville Jaguars', 
               'Kansas City Chiefs', 'Los Angeles Chargers', 
               'Los Angeles Rams', 'Miami Dolphins', 'Minnesota Vikings', 
               'New England Patriots', 'New Orleans Saints', 'New York Giants', 
               'New York Jets', 'Oakland Raiders', 'Philadelphia Eagles', 
               'Pittsburgh Steelers', 'San Francisco 49s', 'Seattle Seahawks', 
               'Tampa Bay Buccaneers', 'Tennessee Titans', 
               'Washington Redskins']

w_df = [['w' for x in range(16)] for x in range(1)]

cols= ['Position', 'Name', 'Number', 'Rating', 'Pos_Ranking', 'Depth', 
       'Height', 'Weight', 'Age', 'Birthday', 'Experience',
       'DraftYear', 'DraftRound', 'OverallDraft', 'College', 'Team']

df_dummy = pd.DataFrame(w_df, columns=cols)

nflss = []
for x in teams_names:
    a = team_df1('https://www.lineups.com/nfl/roster/' + 
                 x.replace(' ', '-').lower(), x)
    for b in a:
        b.append(x)
    nflss.append(a)

    
for x in nflss:
    c = pd.DataFrame(x, columns= cols)
    df_dummy = pd.concat([df_dummy, c], ignore_index=True)
    
nfl1 = df_dummy.drop(0, axis=0).reset_index().drop('index', axis=1)

num_cols = ['Number', 'Rating', 'Depth', 'Weight', 'Age', 'Experience', 
            'OverallDraft']

nfl2 = nfl1
for x in num_cols:
    nfl2[x] = pd.to_numeric(nfl2[x], errors='coerce')
    
nfl2.dropna(subset = ['Number', 'Depth'], inplace=True)
    
nfl2 = nfl2[nfl2['Depth'] < 5]

nfl3 = nfl2.reset_index().drop('index', axis=1)

arizona = []
for x in range(len(nfl3)):
    cor_name = re.sub(r'\.[A-Za-z]*', '', nfl3['Name'][x])
    arizona.append(re.sub(r'([a-z]+)([A-Z])', r'\1 \2', cor_name[:-1]))
nfl3['Name']=arizona

nfl3.drop(['Birthday', 'DraftYear', 'DraftRound', 'OverallDraft'], inplace=True)


