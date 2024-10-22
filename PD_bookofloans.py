import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load the data
loan_book = pd.read_csv('Loan_Data.csv')
loan_book = loan_book.dropna()
loan_book = loan_book.sort_index()

recovery_rate = 0.1

# Prepare the data for modeling
X = loan_book[['credit_lines_outstanding', 'loan_amt_outstanding', 'total_debt_outstanding', 'income', 'years_employed', 'fico_score']]
y = loan_book['default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and fit the model
model = xgb.XGBClassifier(
    objective='binary:logistic', 
    n_estimators=1000, 
    max_depth=5, 
    learning_rate=0.1)

model.fit(X_train, y_train)

def estimate_default_prob(loan_details, model):
    ''' Function to estimate the probability of default for a given loan

        Extract the relevant details from the input
        Create a DataFrame for the input details
        Predict the probability using the model
        Return the estimated probability'''

    input_data = pd.DataFrame(loan_details, index=[0])
    
    estimated_prob = model.predict_proba(input_data)[:,1][0]
    
    return estimated_prob

# Estimate the probability of default for a given loan
credit_lines_outstanding = int(input("Enter the number of credit lines outstanding: "))
loan_amt_outstanding = int(input("Enter the loan amount outstanding: "))
total_debt_outstanding = int(input("Enter the total debt outstanding: "))
income = int(input("Enter the annual income: "))
years_employed = int(input("Enter the number of years employed: "))
fico_score = int(input("Enter the FICO score: "))

loan_details = {
            'credit_lines_outstanding': credit_lines_outstanding, 
            'loan_amt_outstanding': loan_amt_outstanding, 'total_debt_outstanding': total_debt_outstanding, 'income': income, 'years_employed': years_employed, 'fico_score': fico_score}

estimated_prob = float(estimate_default_prob(loan_details, model))

print(f"Estimated probability of default: {estimated_prob}")

# Calculate the expected loss
expected_loss = estimated_prob * recovery_rate * loan_details['loan_amt_outstanding']
print(f"Expected loss: {expected_loss}")

'''Given a set number of buckets corresponding to the number of input labels for the model, she would like to find out the boundaries that best summarize the data. You need to create a rating map that maps the FICO score of the borrowers to a rating where a lower rating signifies a better credit score.

The process of doing this is known as quantization. You could consider many ways of solving the problem by optimizing different properties of the resulting buckets, such as the mean squared error or log-likelihood (see below for definitions)'''

# Define the number of buckets
num_buckets = 10

# Create the rating map
def create_rating_map(data, num_buckets):
    ''' Function to create a rating map based on the FICO score

        Sort the data by FICO score
        Calculate the number of observations in each bucket
        Calculate the boundaries for each bucket
        Create a dictionary mapping the FICO score to the rating
        Return the rating map'''

    data = data.sort_values('fico_score')
    
    bucket_size = len(data) // num_buckets
    
    boundaries = [data['fico_score'].iloc[i*bucket_size] for i in range(1, num_buckets)]
    
    rating_map = {}
    for i, boundary in enumerate(boundaries):
        rating_map[boundary] = i
    
    return rating_map

rating_map = create_rating_map(loan_book, num_buckets)

print(rating_map)

# Define the mean squared error function
def mean_squared_error(data, rating_map):
    ''' Function to calculate the mean squared error of the rating map

        Calculate the squared error for each observation
        Calculate the mean squared error
        Return the mean squared error'''

    data['rating'] = data['fico_score'].map(rating_map)
    
    data['squared_error'] = (data['rating'] - data['fico_score'])**2
    
    mse = data['squared_error'].mean()
    
    return mse

mse = mean_squared_error(loan_book, rating_map)
print(f"Mean squared error: {mse}")

