import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Load the data
data_natgas = pd.read_csv('Nat_Gas.csv')
data_natgas = data_natgas.dropna()
data_natgas = data_natgas.sort_index()

# Convert the dates to datetime
data_natgas['Dates'] = pd.to_datetime(data_natgas['Dates'], format='%m/%d/%y')

# Transform the prices
#data_natgas['Prices'] = data_natgas['Prices'].apply(lambda x: x*100)

# Prepare the data for modeling
data_natgas['Month'] = data_natgas['Dates'].dt.month
data_natgas['Year'] = data_natgas['Dates'].dt.year
X = data_natgas[['Month', 'Year']]
y = data_natgas['Prices']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and fit the model
model = xgb.XGBRegressor(objective='reg:squarederror' ,n_estimators=1000, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)



def estimate_gas_price(date_to_estimate, model, data_natgas):
    ''' Function to estimate the purchase price of gas at any date in the past and extrapolate it for one year into the future

        Convert the input date to datetime
        Extract month and year from the date
        Create a DataFrame for the input date
        Predict the price using the model
        Return the estimated price'''


    date_to_estimate = pd.to_datetime(date_to_estimate)
    
    month = date_to_estimate.month
    year = date_to_estimate.year
    
    input_data = pd.DataFrame({'Month': [month], 'Year': [year]})
    
    estimated_price = model.predict(input_data)[0]
    
    return estimated_price

# Estimate the price for a given date
date_to_estimate = '12/31/24'
estimated_price = estimate_gas_price(date_to_estimate, model, data_natgas)
print(f"Estimated price for {date_to_estimate}: {estimated_price}")

# Extrapolate for one year into the future
future_dates = pd.date_range(start=date_to_estimate, periods=12, freq='ME')
future_prices = []

for future_date in future_dates:
    future_price = estimate_gas_price(future_date.strftime('%m/%d/%Y'), model, data_natgas)
    future_prices.append(future_price)

plt.figure(figsize=(14, 7))
plt.plot(data_natgas['Dates'], data_natgas['Prices'], label='Historical Prices')
plt.plot(future_dates, future_prices, label='Extrapolated Prices', linestyle='--')
plt.axvline(pd.to_datetime(date_to_estimate), color='g', linestyle='--', label='Date to Estimate')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Natural Gas Prices Over Time with Extrapolation')
plt.legend()
plt.show() 

def price_contract(number_of_transportations, number_of_months_storing, model, data_natgas, contract_size, dates, buy_date):
    ''' Function to calculate the total cost of the contract and its value
        Return the total PnL of the contract'''
    total_profit = 0
    cost_storing = 100
    storage_facility_cost = 10
    cost_transporation = 50
    param_mmbtu = 1000000

    buy_date = pd.to_datetime(buy_date, format='%m/%d/%Y') 
    buy_price = data_natgas[data_natgas['Dates'] == buy_date]['Prices'].values[0]
    
    for date in dates:
        sell_price = estimate_gas_price(date, model, data_natgas)
        contract_value = contract_size * (sell_price - buy_price) * param_mmbtu
        contract_cost = (cost_transporation * number_of_transportations + cost_storing * number_of_months_storing + storage_facility_cost * contract_size) * 1000
        total_profit += contract_value - contract_cost

    return total_profit

contract_size = int(input('Enter the size of the contract (in MMBtu): '))
buy_date = input('Enter a buy date to get the price of contract (MM/DD/YYYY): ')
number_of_transportations = int(input('Enter the number of transportations: '))
number_of_months_storing = int(input('Enter the number of months storing: '))
dates = [input('Enter the date of sell contract (MM/DD/YYYY): ')]

print(f"Total PnL of the contract: {price_contract(number_of_months_storing, number_of_transportations, model, data_natgas, contract_size, dates, buy_date)} USD")