library(data.table)

# Create sales data
sales_dt <- data.table(
  order_id = 1:1000,
  customer_id = sample(1:100, 1000, replace = TRUE),
  product_id = sample(1:20, 1000, replace = TRUE),
  quantity = sample(1:10, 1000, replace = TRUE),
  sale_date = as.Date('2024-01-01') + sample(0:90, 1000, replace = TRUE)
)

# Create product data
products_dt <- data.table(
  product_id = 1:20,
  product_name = paste0("Product_", LETTERS[1:20]),
  unit_price = round(runif(20, 10, 100), 2),
  category = sample(c("Electronics", "Clothing", "Books", "Home"), 20, replace = TRUE)
)

# (a) Basic Operations:
# Function to calculate total quantity sold for each product_id
calculate_total_quantity_sold <- function(sales_dt) {
  return(sales_dt[, .(total_quantity = sum(quantity)), by = product_id])
}

# Function to find the top 5 products by quantity sold
find_top_5_products <- function(total_quantity_sold) {
  return(total_quantity_sold[order(-total_quantity)][1:5])
}

# (b) Joins and Calculations:
# Function to join sales and products tables
join_sales_and_products <- function(sales_dt, products_dt) {
  return(merge(sales_dt, products_dt, by = "product_id", all.x = TRUE))
}

# Function to calculate total revenue by category
calculate_revenue_by_category <- function(merged_dt) {
  merged_dt[, total_revenue := quantity * unit_price]
  return(merged_dt[, .(total_revenue = sum(total_revenue)), by = category])
}

# Function to find the category with the highest average revenue per sale
find_top_category_by_avg_revenue <- function(merged_dt) {
  avg_revenue_by_category <- merged_dt[, .(avg_revenue = mean(total_revenue)), by = category]
  return(avg_revenue_by_category[order(-avg_revenue)][1])
}

# (c) Date Operations:
# Function to calculate daily sales totals
calculate_daily_sales <- function(merged_dt) {
  return(merged_dt[, .(daily_revenue = sum(total_revenue)), by = sale_date])
}

# Function to find the day with the highest number of orders
find_top_day_by_orders <- function(sales_dt) {
  daily_orders <- sales_dt[, .(order_count = .N), by = sale_date]
  return(daily_orders[order(-order_count)][1])
}

# Function to create a weekly summary of sales by category
create_weekly_sales_summary <- function(merged_dt) {
  merged_dt[, week := format(sale_date, "%Y-%U")]  # Add week information
  return(merged_dt[, .(weekly_revenue = sum(total_revenue)), by = .(week, category)])
}

# (d) Advanced Operations:
# Function to calculate the running total of quantity sold for each product
calculate_running_total <- function(merged_dt) {
  merged_dt[, running_total := cumsum(quantity), by = product_id]
  return(merged_dt[, .(product_id, order_id, running_total)])
}

# Function to find customers who have purchased at least 3 different categories
find_customers_with_3_categories <- function(merged_dt) {
  customer_categories <- merged_dt[, .(distinct_categories = uniqueN(category)), by = customer_id]
  return(customer_categories[distinct_categories >= 3])
}

# Function to identify products that were sold every week throughout the dataset
identify_products_sold_every_week <- function(merged_dt) {
  sold_per_week <- merged_dt[, .(weeks_sold = uniqueN(week)), by = product_id]
  return(sold_per_week[weeks_sold == length(unique(merged_dt$week))])
}

# Running all operations
total_quantity_sold <- calculate_total_quantity_sold(sales_dt)
print("Total Quantity Sold for Each Product:")
print(total_quantity_sold)

top_5_products <- find_top_5_products(total_quantity_sold)
print("Top 5 Products by Quantity Sold:")
print(top_5_products)

merged_dt <- join_sales_and_products(sales_dt, products_dt)

revenue_by_category <- calculate_revenue_by_category(merged_dt)
print("Total Revenue by Category:")
print(revenue_by_category)

top_category <- find_top_category_by_avg_revenue(merged_dt)
print("Category with Highest Average Revenue per Sale:")
print(top_category)

daily_sales <- calculate_daily_sales(merged_dt)
print("Daily Sales Totals:")
print(daily_sales)

top_day <- find_top_day_by_orders(sales_dt)
print("Day with the Highest Number of Orders:")
print(top_day)

weekly_sales_by_category <- create_weekly_sales_summary(merged_dt)
print("Weekly Sales by Category:")
print(weekly_sales_by_category)

running_total <- calculate_running_total(merged_dt)
print("Running Total of Quantity Sold for Each Product:")
print(head(running_total))

customers_with_3_categories <- find_customers_with_3_categories(merged_dt)
print("Customers Who Have Purchased at Least 3 Different Categories:")
print(customers_with_3_categories)

products_sold_every_week <- identify_products_sold_every_week(merged_dt)
print("Products Sold Every Week Throughout the Dataset:")
print(products_sold_every_week)

