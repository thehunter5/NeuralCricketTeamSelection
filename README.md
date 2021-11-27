Team selection for the next season of IPL cricket requires predicting performance of cricketers based on their previous performances.
This project aims to ease this process by doing this prediction using three types of neural network models-
1. Radial Basis Function Neural Network
2. Linear Perceptron
3. Multilayered Perceptron

We have used https://www.kaggle.com/harsha547/indian-premier-league-csv-dataset as the dataset which contains a comprehensive ball by ball detail of the initial 9 seasons of IPL.
We perform two experiments using each of the three networks-
1. combining all the data together and using 80% of it for training and rest for testing
2. taking season 1-7 together as the training set and season 8-9 for testing the predictions made using previous seasons
Please refer to the file [NNFL_flowchart](/NNFL_flowchart.pdf) to see the detail flow.

**Heuristics**-
1. For each season :
  - A Batsman is rated as 
    - Performer : Matches Played >= 7 and Average Runs >=30
    - Moderate : Matches Played >= 5 and Average Runs >=18
    - Failure : otherwise
  - A Bowler is rated as 
    - Performer : Matches Played >= 7 and Average Wickets >=1.5
    - Moderate : Matches Played >= 5 and Average Wickets >=1
    - Failure : otherwise
2. For all seasons combined : 
  - A Batsman is rated as 
    - Performer : Matches Played >= 50 and Average Runs >=30
    - Moderate : Matches Played >= 30 and Average Runs >=18
    - Failure : otherwise
  - A Bowler is rated as 
    - Performer : Matches Played >= 50 and Average Wickets >=1.5
    - Moderate : Matches Played >= 30 and Average Wickets >=1
    - Failure : otherwise
    
The data was divided into 6 parts :
  - Ball by Ball : This data had detailed entries of what happened at each ball in all the seasons of the IPL.
  - Match : It contains the description of each match and its winners.
  - Player : It contains the data of each participating player and their player id.
  - Player_match : This contains the list of all the matches played by a player.
  - Season : It contains the description of each season and its location.
  - Team : Contains the list of all the teams participating in the IPL.
  
**Data Preprocessing**-
  - Data was distributed among 6 different csv.
  - Preprocessing of data was done and it was aggregated into two different data sets for two experiments.
  - One dataset aggregated the performance of a player in a particular season.
  - The other dataset aggregated the performance of a player across all the seasons.
  - Pandas library in python was used to combine and merge the data from different CSVs into the above mentioned format.
  - Heuristics were then applied to the aggregated data to label the output column for training and testing.
  
**Results**
1. RBF Neural Network
  - Season by season :
      - Accuracy of prediction was in the range of 75-80%. 
      - The number of hidden neurons was varied from 5 to 25.
      - The accuracy increased by increasing the number of hidden neurons.
      - Accuracy peaked when number of neurons is 15 and remained fairly constant on further increase .
  - All seasons :
      - When the player data is collated for all the seasons together accuracy increased a lot.
      - Accuracy was in the range of 90-95%.
      - Number of hidden neurons had the same effect as above.
2. Multilayered Perceptron
  - Season by season :
    - Hidden Layers : 4
    - Epochs : 100
    - Learning Rate : 0.01
    - Accuracy : 84%
  - All seasons combined :
    - Hidden Layers : 10
    - Epochs : 100
    - Learning Rate : 0.01
    - Accuracy : 66%
3. Linear Perceptron
  - Season by season :
    ![image](https://user-images.githubusercontent.com/28497690/143624223-48da9a7c-4d5d-4783-b65a-89f2dedaa13a.png)
  - All seasons combined :
    ![image](https://user-images.githubusercontent.com/28497690/143624174-633c960c-eb8b-4826-a5ac-bd3bb64feff5.png)

  





