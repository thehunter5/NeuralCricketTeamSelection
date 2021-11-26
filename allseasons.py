# Group 23

# Code for data preprocessing for all the seasons combined
import pandas as pd
import numpy as np
df1=pd.read_csv('C:/Users/user/Downloads/indian-premier-league-csv-dataset/Ball_by_Ball.csv',dtype={"Bowler_Id":int,"Batsman_Scored":object})
df2=pd.read_csv('C:/Users/user/Downloads/indian-premier-league-csv-dataset/Match.csv')
df3=df1[['Match_Id','Striker_Id','Bowler_Id','Batsman_Scored','Player_dissimal_Id','Dissimal_Type']]
df4=df2[['Match_Id','Season_Id']] 
df5=pd.merge(df3,df4,how='inner',on='Match_Id')
players_df=pd.read_csv('C:/Users/user/Downloads/indian-premier-league-csv-dataset/Player.csv')
players_df.loc[players_df['Is_Umpire']  == 1, :] = np.NaN
players_df= players_df.dropna(axis=0, how='all')
match_df=pd.read_csv('C:/Users/user/Downloads/indian-premier-league-csv-dataset/Player_Match.csv')
match_df=match_df[['Match_Id','Player_Id']]
match_df=pd.merge(match_df,df4,how='inner',on='Match_Id')
df_final=players_df['Player_Id']
df_final= pd.concat([df_final,pd.DataFrame(columns=['Runs_Scored','Wickets_Taken','Matches_Played','Outs','Average_runs','Average_wickets','Performance_rating'])])
df_final.columns=['Player_Id','Runs_Scored','Wickets_Taken','Matches_Played','Outs','Average_runs','Average_wickets','Performance_rating']
df_final['Average_runs']=pd.to_numeric(df_final['Average_runs'], errors='coerce').fillna(0.0).astype(np.float)
df_final['Average_wickets']=pd.to_numeric(df_final['Average_wickets'], errors='coerce').fillna(0.0).astype(np.float)
df_final['Outs']=pd.to_numeric(df_final['Outs'], errors='coerce').fillna(0).astype(np.int64)
df_final['Performance_rating']=pd.to_numeric(df_final['Performance_rating'], errors='coerce').fillna(-1.0).astype(np.float)
df_final.Player_Id = df_final.Player_Id.astype(int)
df_final.set_index('Player_Id', inplace=True)
df_final['Matches_Played']=pd.to_numeric(df_final['Matches_Played'], errors='coerce').fillna(0).astype(np.int64)
df_final['Runs_Scored']=pd.to_numeric(df_final['Runs_Scored'], errors='coerce').fillna(0).astype(np.int64)
df5['Batsman_Scored']=pd.to_numeric(df5['Batsman_Scored'], errors='coerce').fillna(0).astype(np.int64)
df5['Player_dissimal_Id']=pd.to_numeric(df5['Player_dissimal_Id'], errors='coerce').fillna(-1).astype(np.int64)
group_batsman=df5.groupby('Striker_Id')   
group_bowler=df5.groupby('Bowler_Id')
for batsman, batsman_df in group_batsman:
    playerid=batsman
    df_final.loc[[playerid],['Runs_Scored']]= batsman_df['Batsman_Scored'].sum()
for bowler, bowler_df in group_bowler:
    playerid=bowler
    df_final.loc[[playerid],['Wickets_Taken']] =len(bowler_df['Player_dissimal_Id'])- bowler_df[bowler_df['Player_dissimal_Id']==-1 ].shape[0]            
    df_final['Runs_Scored']=df_final['Runs_Scored'].fillna(0)
    df_final['Wickets_Taken']=df_final['Wickets_Taken'].fillna(0)


group_player=match_df.groupby('Player_Id')
for player,player_df in group_player:
    playerid=player
    df_final.loc[[playerid],['Matches_Played']]= player_df['Match_Id'].count()

#  Apply heuristics


id=1
while id<470 or id==523 or id==524:
    df_final.loc[[id],['Outs']]=df5.loc[df5.Player_dissimal_Id == id, 'Player_dissimal_Id'].count()
    id+=1
    if id==470:
        id=523

for idx,row in df_final.iterrows():
    #print(row['Outs'])
    
    if(row['Outs']==0.0):
        continue
    else:
        p=row['Runs_Scored']/row['Outs']
        df_final.loc[[idx],['Average_runs']]=p
        
for idx,row in df_final.iterrows():
    #print(row['Outs'])
    
    if(row['Matches_Played']==0.0):
        continue
    else:
        p=row['Wickets_Taken']/row['Matches_Played']
        df_final.loc[[idx],['Average_wickets']]=p



for idx,row in df_final.iterrows():
    #print(row['Outs'])
    
    if((row['Average_runs']>=30.0 or row['Average_wickets']>=1.5)and row['Matches_Played']>=50):
        df_final.loc[[idx],['Performance_rating']]=1
    elif((row['Average_runs']>=18.0 or row['Average_wickets']>=1)and (row['Matches_Played']>=30 and row['Matches_Played']<50)):
        df_final.loc[[idx],['Performance_rating']]=0
    else:
        df_final.loc[[idx],['Performance_rating']]=-1
df_final.to_csv('C:/Users/user/Downloads/indian-premier-league-csv-dataset/data/all9seasons.csv')
    





