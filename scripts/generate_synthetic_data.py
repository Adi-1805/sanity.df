import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_task1():
    # Task 1: Mixed Values & Intelligent Imputation
    np.random.seed(42)
    n = 100
    
    # 1. product_code (Mixed Values)
    categories = ['A', 'B', 'C']
    product_categories = np.random.choice(categories, n)
    product_ids = np.random.randint(10, 99, n)
    product_codes = [f"{c}{i}" if np.random.rand() > 0.5 else f"{c}-{i}" for c, i in zip(product_categories, product_ids)]
    
    # 2. color (Categorical Missing)
    colors = np.random.choice(['Red', 'Blue', 'Green'], n, p=[0.7, 0.2, 0.1])
    
    # 3. weight_kg (Normal Distribution Missing)
    weight_kg = np.random.normal(loc=70, scale=10, size=n)
    
    # 4. income_usd (Skewed Missing)
    income_usd = np.random.lognormal(mean=10, sigma=1, size=n)
    # Add huge outlier
    income_usd[0] = 5000000
    
    # 5. house_price & sqft (KNN Imputer target)
    sqft = np.random.uniform(1000, 5000, n)
    house_price = sqft * 150 + np.random.normal(0, 10000, n)
    
    # Create MESSY dataframe
    df_messy = pd.DataFrame({
        'product_code': product_codes,
        'color': colors.copy(),
        'weight_kg': weight_kg.copy(),
        'income_usd': income_usd.copy(),
        'sqft': sqft.copy(),
        'house_price': house_price.copy()
    })
    
    # Inject NaNs (about 15%)
    for col in ['color', 'weight_kg', 'income_usd', 'house_price']:
        mask = np.random.rand(n) < 0.15
        if col == 'house_price':
            mask[0] = False # Don't mask the outlier related to other logic? actually we don't have outlier in house_price, it's in income
        df_messy.loc[mask, col] = np.nan

    # Create GOLD dataframe as the expected output AFTER correct cleaning
    df_gold = df_messy.copy()
    
    # 1. Split product_code
    df_gold['product_category'] = product_categories
    df_gold['product_id'] = product_ids
    
    # 2. Impute color with mode
    mode_val = df_messy['color'].mode()[0]
    df_gold['color'] = df_messy['color'].fillna(mode_val)
    
    # 3. Impute weight_kg with mean
    mean_val = df_messy['weight_kg'].mean()
    df_gold['weight_kg'] = df_messy['weight_kg'].fillna(mean_val)
    
    # 4. Impute income_usd with median
    median_val = df_messy['income_usd'].median()
    df_gold['income_usd'] = df_messy['income_usd'].fillna(median_val)
    
    # 5. Impute house_price with KNN
    from sklearn.impute import KNNImputer
    imputer = KNNImputer(n_neighbors=5)
    df_gold[['sqft', 'house_price']] = imputer.fit_transform(df_messy[['sqft', 'house_price']])
        
    return df_messy, df_gold

def generate_task2():
    # Task 2: Advanced Scaling & Transformations
    np.random.seed(42)
    n = 100
    
    # 1. age (Uniform/Bounded, needs MinMax)
    age = np.random.uniform(18, 80, n)
    
    # 2. sensor_reading (Normally Distributed, needs StandardScaler)
    sensor_reading = np.random.normal(50, 10, n)
    
    # 3. stock_volume (Outliers, needs RobustScaler)
    stock_volume = np.random.normal(1000, 100, n)
    stock_volume[np.random.choice(n, 5, replace=False)] *= 10 # 5% outliers
    
    # 4. sparse_audio_signal (Sparse, needs MaxAbsScaler)
    sparse_audio = np.zeros(n)
    sparse_indices = np.random.choice(n, 20, replace=False)
    sparse_audio[sparse_indices] = np.random.uniform(-5, 5, 20)
    
    # 5. engagement_time (Right-skewed, needs LogTransform)
    engagement_time = np.random.exponential(scale=2.0, size=n)
    
    df_messy = pd.DataFrame({
        'age': age,
        'sensor_reading': sensor_reading,
        'stock_volume': stock_volume,
        'sparse_audio_signal': sparse_audio,
        'engagement_time': engagement_time
    })
    
    df_gold = df_messy.copy()
    
    # Gold transformations
    from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
    
    df_gold['age'] = MinMaxScaler().fit_transform(df_gold[['age']])
    df_gold['sensor_reading'] = StandardScaler().fit_transform(df_gold[['sensor_reading']])
    df_gold['stock_volume'] = RobustScaler().fit_transform(df_gold[['stock_volume']])
    df_gold['sparse_audio_signal'] = MaxAbsScaler().fit_transform(df_gold[['sparse_audio_signal']])
    
    # Log transform (log1p to handle zeros if any)
    df_gold['engagement_time'] = np.log1p(df_gold['engagement_time'])
    
    return df_messy, df_gold

def generate_task3():
    # Task 3: Domain-Driven Feature Construction
    np.random.seed(42)
    n = 100
    
    # Generate dates
    base_date = datetime(2025, 1, 1)
    user_birthdates = [base_date - timedelta(days=np.random.randint(18*365, 60*365)) for _ in range(n)]
    account_created_dates = [base_date - timedelta(days=np.random.randint(100, 1000)) for _ in range(n)]
    
    # Some users haven't purchased anything
    last_purchase_dates = []
    for cd in account_created_dates:
        if np.random.rand() > 0.1: # 90% have purchased
            days_since_created = (base_date - cd).days
            lp_date = cd + timedelta(days=np.random.randint(1, max(2, days_since_created)))
            last_purchase_dates.append(lp_date)
        else:
            last_purchase_dates.append(pd.NaT)
            
    total_orders = np.where(pd.isna(last_purchase_dates), 0, np.random.randint(1, 50, n))
    total_spent = total_orders * np.random.uniform(10, 100, n)
    # Add a few refund edge cases (negative spent but orders > 0? Let's just do negative total spent)
    refund_idx = np.random.choice(np.where(total_orders > 0)[0], 2, replace=False)
    total_spent[refund_idx] = -50
    
    df_messy = pd.DataFrame({
        'user_birthdate': pd.to_datetime(user_birthdates),
        'account_created_date': pd.to_datetime(account_created_dates),
        'last_purchase_date': pd.to_datetime(last_purchase_dates),
        'total_orders': total_orders,
        'total_spent': total_spent,
        'is_premium': np.random.choice([True, False], n)
    })
    
    df_gold = df_messy.copy()
    
    # Construct Features for Gold
    # 1. age_at_signup
    df_gold['age_at_signup'] = (df_gold['account_created_date'] - df_gold['user_birthdate']).dt.days / 365.25
    
    # 2. days_since_last_purchase (Recency)
    # Assume "current date" is base_date
    df_gold['days_since_last_purchase'] = (base_date - df_gold['last_purchase_date']).dt.days
    df_gold['days_since_last_purchase'] = df_gold['days_since_last_purchase'].fillna(-1) # Handle never purchased
    
    # 3. average_order_value (AOV)
    # Handle division by zero
    df_gold['average_order_value'] = np.where(df_gold['total_orders'] > 0, df_gold['total_spent'] / df_gold['total_orders'], 0)
    
    # 4. customer_lifetime
    df_gold['customer_lifetime'] = (df_gold['last_purchase_date'] - df_gold['account_created_date']).dt.days
    df_gold['customer_lifetime'] = df_gold['customer_lifetime'].fillna(0) # Handle never purchased
    
    return df_messy, df_gold

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    for i in range(1, 4):
        messy, gold = globals()[f"generate_task{i}"]()
        messy.to_csv(f"data/task{i}_messy.csv", index=False)
        gold.to_csv(f"data/task{i}_gold.csv", index=False)
        print(f"Generated Task {i} datasets (Messy & Gold).")