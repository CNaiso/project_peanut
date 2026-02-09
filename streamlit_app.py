import streamlit as st

st.title("Project Peanut:Cynthia Arami")
st.write(
    "This is a web application built using stringlit"
)





# Sample DataFrame to display
import pandas as pd

martket_data = {
    "Company": ["Apple", "Google", "Microsoft", "Amazon"],
    "Stock Price": [150, 2800, 300, 3500],
    "Market Cap (Billion USD)": [2500, 1800, 2200, 1700],
}

df = pd.DataFrame(martket_data)

# Display the DataFrame in the Streamlit app
st.header("Market Data")
st.subheader("Stock Prices and Market Capitalization")
st.dataframe(df)

# display data using a table
st.subheader("Market Data Table")
st.table(df)

# Charts
st.header("Market Data Visualization")
st.subheader("Stock Prices")
st.bar_chart(df.set_index("Company")["Stock Price"])

# Markdown in streamlit
st.header("About Project Peanut")
st.markdown("""
## Data Description

### Input Files

| File | Description | Size | Records |
|------|-------------|------|---------|
| `Train.csv` | Historical customer-product-week data | 275 MB | ~5M rows |
| `Test.csv` | Test set for predictions | 27 MB | ~500K rows |
| `SampleSubmission.csv` | Submission format | 7 MB | ~500K rows |

### Key Columns

| Column | Type | Description |
|--------|------|-------------|
| `customer_id` | ID | Unique customer identifier |
| `product_unit_variant_id` | ID | Unique product variant identifier |
| `week_start` | Date | Week start date |
| `purchased_this_week` | Binary | Purchase indicator (0/1) |
| `qty_this_week` | Float | Quantity purchased |
| `customer_category` | Category | Customer segment |
| `customer_status` | Category | Customer status |
| `grade_name` | Category | Product grade |
| `unit_name` | Category | Product unit type |

### Targets

| Target | Type | Description |
|--------|------|-------------|
| `Target_purchase_next_1w` | Binary | Will purchase in next 1 week? |
| `Target_purchase_next_2w` | Binary | Will purchase in next 2 weeks? |
| `Target_qty_next_1w` | Float | Quantity in next 1 week |
| `Target_qty_next_2w` | Float | Quantity in next 2 weeks |

---
""")



## Metrics
st.header("Key Metrics")
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Customers", "150,000", "5%")
with col2:
    st.metric("Total Products", "2,500", "2%")
with col3:
    st.metric("Total Purchases", "1,200,000", "8%")

# Streamlit Basics
st.header("Streamlit Basics")
st.markdown("""

## Text and Formatting
- Use `st.write()` for simple text output.
- Use `st.markdown()` for formatted text and markdown support.
## Data Display
- Use `st.dataframe()` to display interactive tables.
- Use `st.table()` for static tables.
            
## Charts and Visualizations
- Use `st.line_chart()`, `st.bar_chart()`, and `st.area_chart()` for quick visualizations.
- For more complex charts, use libraries like Matplotlib or Seaborn and display with `st.pyplot()`.
## Layout and Interactivity
- Use `st.columns()` to create multi-column layouts.
- Use `st.expander()` to hide/show content.
- Use `st.form()` to create interactive forms for user input.
""")
