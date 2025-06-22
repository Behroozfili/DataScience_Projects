import pandas as pd
import numpy as np


np.random.seed(42)
sales_df = pd.DataFrame({
    'order_id': range(1, 1001),
    'customer_id': np.random.randint(1, 101, size=1000),
    'product_id': np.random.randint(1, 21, size=1000),
    'quantity': np.random.randint(1, 11, size=1000),
    'sale_date': pd.date_range(start='2024-01-01', periods=91).repeat(11)[:1000]
})


products_df = pd.DataFrame({
    'product_id': range(1, 21),
    'product_name': [f'Product_{letter}' for letter in 'ABCDEFGHIJKLMNOPQRST'],
    'unit_price': np.round(np.random.uniform(10, 100, size=20), 2),
    'category': np.random.choice(['Electronics', 'Clothing', 'Books', 'Home'], size=20)
})

# a) Basic Operations
def total_quantity_sold(sales_df):
    total_quantity = sales_df.groupby('product_id')['quantity'].sum().reset_index()
    total_quantity = total_quantity.rename(columns={'quantity': 'total_quantity'})
    return total_quantity

def top_5_products_by_quantity_sold(total_quantity_sold):
    return total_quantity_sold.sort_values(by='total_quantity', ascending=False).head(5)

# b) Joins and Calculations
def merge_sales_products(sales_df, products_df):
   
    return pd.merge(sales_df, products_df, on='product_id', how='left')

def calculate_total_revenue(merged_df):
    merged_df['total_revenue'] = merged_df['quantity'] * merged_df['unit_price']
    revenue_by_category = merged_df.groupby('category')['total_revenue'].sum().reset_index()
    return revenue_by_category

def category_with_highest_avg_revenue(merged_df):
    avg_revenue_by_category = merged_df.groupby('category')['total_revenue'].mean().reset_index()
    return avg_revenue_by_category.sort_values(by='total_revenue', ascending=False).head(1)

# c) Date Operations
def daily_sales_totals(merged_df):
    return merged_df.groupby('sale_date')['total_revenue'].sum().reset_index()

def day_with_highest_orders(sales_df):
    daily_orders = sales_df.groupby('sale_date')['order_id'].count().reset_index()
    return daily_orders.sort_values(by='order_id', ascending=False).head(1)

def weekly_sales_by_category(merged_df):
    merged_df['week'] = merged_df['sale_date'].dt.isocalendar().week
    return merged_df.groupby(['week', 'category'])['total_revenue'].sum().reset_index()

def running_total_quantity_sold(merged_df):
    merged_df['running_total'] = merged_df.groupby('product_id')['quantity'].cumsum()
    return merged_df[['product_id', 'order_id', 'running_total']]

def customers_with_at_least_3_categories(merged_df):
    customer_categories = merged_df.groupby('customer_id')['category'].nunique().reset_index()
    return customer_categories[customer_categories['category'] >= 3]

def products_sold_every_week(merged_df):
    products_sold_per_week = merged_df.groupby('product_id')['week'].nunique().reset_index()
    return products_sold_per_week[products_sold_per_week['week'] == merged_df['week'].nunique()]

total_quantity = total_quantity_sold(sales_df)
print("Total Quantity Sold for Each Product:")
print(total_quantity)

top_5_products = top_5_products_by_quantity_sold(total_quantity)
print("Top 5 Products by Quantity Sold:")
print(top_5_products)

merged_df = merge_sales_products(sales_df, products_df)

revenue_by_category = calculate_total_revenue(merged_df)
print("Total Revenue by Category:")
print(revenue_by_category)

top_category = category_with_highest_avg_revenue(merged_df)
print("Category with Highest Average Revenue per Sale:")
print(top_category)

daily_sales = daily_sales_totals(merged_df)
print("Daily Sales Totals:")
print(daily_sales)

top_day = day_with_highest_orders(sales_df)
print("Day with the Highest Number of Orders:")
print(top_day)

weekly_sales = weekly_sales_by_category(merged_df)
print("Weekly Sales by Category:")
print(weekly_sales)

running_total = running_total_quantity_sold(merged_df)
print("Running Total of Quantity Sold for Each Product:")
print(running_total.head())

customers = customers_with_at_least_3_categories(merged_df)
print("Customers Who Have Purchased At Least 3 Different Categories:")
print(customers)

products_sold = products_sold_every_week(merged_df)
print("Products Sold Every Week Throughout the Dataset:")
print(products_sold)
