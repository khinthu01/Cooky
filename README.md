# Cooky
Cooky is an API that allows users to quickly and automatically classify their recipes' difficulties. This is aimed at independent chefs who publish their recipes online and would like to add a difficulty filter to allow their users to access recipes accessible to their skill levels. Cooky is a multi-class text classification model that searches through instructions in a recipe for features specifically relating to difficulty levels ranging from 'Easy' to 'Challenging'. 

# Why AI?
If you have a team of skilled chefs and editors the way that food magazines do then determining the difficulty of a recipe may be easy as the difficulty level can be crowdsourced within the team. However, if you're an independent online chef and editor then picking the difficulty of a recipe can feel random and you cannot really be sure if your readers, who may be amateurs, will agree. By implementing an AI trained with data obtained from various recipes from https://www.greatbritishchefs.com/, your recipe will be categorised as if you had a whole editorial team behind you except it will be much faster. 

# Project Details

# Scope
The scope was limited such that the model is only equipped to classify savoury European cuisines. 

# Methodology
1. 60 recipes from https://www.greatbritishchefs.com/ were chosen such that there was an even amount of recipes that were 'Easy', 'Medium' and 'Challenging' and such that there were an even number of recipes using poultry, beef, pork, lamb, seafood, and were vegetarian. 
2. The instructions from these recipes as well as their difficulty levels were extracted.
3. The text data was cleaned.
4. Features were extracted from the cleaned data and along with labels (the difficulty levels) were used to train multiple multi-class classification models available in the scikit-learn library. The models used were Logistic Regression, Multinomial NB, Random Forest Classifier, and Linear SVC. The models were compared by their accuracy and the best model (Logistic Regression) was selected.
5. Created an API that classifies recipes using the Logistic Regression model and returns the classification to the client for use.
