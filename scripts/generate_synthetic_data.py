import pandas as pd
import random
import os

def generate_task1_messy(n=60):
    data = []
    names = ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown", "Charlie Davis", "Eve White", "Frank Black", "Grace Lee", "Hank Hill", "Ivy Green"]
    cities = ["New York", "Chicago", "Seattle", "Miami", "Boston", "San Francisco", "Austin", "Denver", "Portland", "Atlanta"]
    
    for i in range(1, n + 1):
        name = random.choice(names)
        email = name.lower().replace(" ", ".") + "@example.com"
        city = random.choice(cities)
        data.append({"id": float(i), "name": name, "email": email, "city": city})
    
    # Add duplicates
    for _ in range(12):
        idx = random.randint(0, len(data) - 1)
        data.append(data[idx].copy())
    
    # Add missing critical fields
    for _ in range(15):
        idx = random.randint(0, len(data) - 1)
        if random.random() < 0.5:
            data[idx]["id"] = None
        else:
            data[idx]["email"] = None
            
    # Add missing city
    for _ in range(5):
        idx = random.randint(0, len(data) - 1)
        data[idx]["city"] = None
        
    df = pd.DataFrame(data)
    df.to_csv("data_clean_transform/data/task1_messy.csv", index=False)
    print(f"Generated Task 1: {len(df)} rows")

def generate_task2_messy(n=60):
    data = []
    names = ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown", "Charlie Davis", "Eve White"]
    date_formats = ["%m/%d/%Y", "%Y.%m.%d", "%Y-%m-%d", "%m-%d-%Y", "%Y/%m/%d", "%B %d, %Y"]
    currency_formats = [
        lambda v: f"${v:,.2f}", 
        lambda v: f" {v:,.2f} ", 
        lambda v: f"€{v:,.2f}", 
        lambda v: f"£{v:,.2f}", 
        lambda v: f"¥{v:,.0f}",
        lambda v: f"{v:,.2f}"
    ]
    
    for i in range(1, n + 1):
        name = random.choice(names)
        # Random date
        day = random.randint(1, 28)
        month = random.randint(1, 12)
        year = random.randint(2020, 2024)
        from datetime import datetime
        dt = datetime(year, month, day)
        date_str = dt.strftime(random.choice(date_formats))
        
        salary_val = random.uniform(30000, 150000)
        salary_str = random.choice(currency_formats)(salary_val)
        
        data.append({"id": i, "name": name, "date_joined": date_str, "salary": salary_str})
        
    df = pd.DataFrame(data)
    df.to_csv("data_clean_transform/data/task2_messy.csv", index=False)
    print(f"Generated Task 2: {len(df)} rows")

def generate_task3_messy(n=60):
    data = []
    names = ["John Doe", "Jane Smith", "Bob Johnson", "Alice Brown", "Charlie Davis", "Eve White"]
    locations = [
        {"state": "NY", "city": "New York", "zipcode": 10001, "variations": ["N.Y.", "ny", "New York City", "Big Apple"]},
        {"state": "NY", "city": "New York", "zipcode": 10002, "variations": ["New York", "NY City"]},
        {"state": "CA", "city": "Los Angeles", "zipcode": 90001, "variations": ["L.A.", "la", "City of Angels", "Los Angeles "]},
        {"state": "CA", "city": "Los Angeles", "zipcode": 90002, "variations": ["Los Angeles", "LA"]}
    ]
    
    for i in range(1, n + 1):
        loc = random.choice(locations)
        name = random.choice(names)
        
        state = loc["state"]
        city = random.choice(loc["variations"] + [loc["city"]])
        zipcode = loc["zipcode"]
        
        # Add errors
        if random.random() < 0.7:  # 70% chance of missing state
            state = None
        
        if random.random() < 0.1: # 10% chance of missing city
            city = None
            
        data.append({"id": i, "name": name, "state": state, "city": city, "zipcode": zipcode})
        
    df = pd.DataFrame(data)
    df.to_csv("data_clean_transform/data/task3_messy.csv", index=False)
    print(f"Generated Task 3: {len(df)} rows")

if __name__ == "__main__":
    os.makedirs("data_clean_transform/data", exist_ok=True)
    generate_task1_messy()
    generate_task2_messy()
    generate_task3_messy()
