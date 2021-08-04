# Cooky
Cooky is an API that allows users to quickly and automatically classify their recipes' difficulties. This is aimed at independent chefs who publish their recipes online and would like to add a difficulty filter to allow their users to access recipes accessible to their skill levels. Cooky is a multi-class text classification model that searches through instructions in a recipe for features specifically relating to difficulty levels ranging from 'Easy' to 'Challenging'. 

# Why AI?
If you have a team of skilled chefs and editors the way that food magazines do then determining the difficulty of a recipe may be easy as the difficulty level can be crowdsourced within the team. However, if you're an independent online chef and editor then picking the difficulty of a recipe can feel random and you cannot really be sure if your readers, who may be amateurs, will agree. By implementing an AI trained with data obtained from various recipes from https://www.greatbritishchefs.com/, your recipe will be categorised as if you had a whole editorial team behind you except it will be much faster. 

# Project Details

## Scope
The scope was limited such that the model is only equipped to classify savoury European cuisines. 

## Approach
### The data phase
1. 60 recipes from https://www.greatbritishchefs.com/ were chosen such that there was an even amount of recipes that were 'Easy', 'Medium' and 'Challenging' and such that there were an even number of recipes using poultry, beef, pork, lamb, seafood, and were vegetarian. 
2. The instructions from these recipes as well as their difficulty levels were extracted.
3. The text data was cleaned.
4. Features were extracted from the cleaned data and along with labels (the difficulty levels) were used to train multiple multi-class classification models available in the scikit-learn library. The models used were Logistic Regression, Multinomial NB, Random Forest Classifier, and Linear SVC. The models were compared by their accuracy and the best model (Random Forest) was selected.
5. A training pipeline was created using Microsoft Azure Machine Learning Designer. The model was scored and evaluated before an inference pipeline was created. 
6. The model was deployed. 

# Implementation


# Requesting the Model
You can access the deployed model through these steps:
```python
import urllib.request
import json
import os

url = 'http://9c38f2e2-32a8-4957-a52b-bb8f13252dd0.australiaeast.azurecontainer.io/score' #the model endpoint
key = 'DmJUk8IK9nvVQwG6HMC8YkaaYhrHPQGZ' #the primary key

data = {
    "Inputs": {
        "WebServiceInput0":
        [
            {
                'Unnamed: 0': "0",
                'Method': "Insert recipe method text here", #insert your recipe method text
            },
        ],
    },
    "GlobalParameters": {
    }
}

body = str.encode(json.dumps(data))

headers = {'Content-Type':'application/json', 'Authorization':('Bearer '+ key)}

req = urllib.request.Request(url, body, headers)

try:
    response = urllib.request.urlopen(req)
    result = response.read()
    json_result = json.loads(result)
    recipe_difficulty = json_result["Results"]["WebServiceOutput0"][0]["RecipeDifficulty"] # this can be input into your recipe database along with the recipe it classified
except urllib.error.HTTPError as error:
    print("The request failed with status code: " + str(error.code))

    # Print the headers to help debug
    print(error.info())
    print(json.loads(error.read().decode("utf8", 'ignore')))

```

