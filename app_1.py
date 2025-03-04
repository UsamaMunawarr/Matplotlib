import streamlit as st
import pandas as pd
import numpy as np
import datetime
from PIL import Image
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import plotly.express as px
from PIL import Image


# Reading the data from Excel file
df = pd.read_excel("Adidas.xlsx")

# Streamlit configuration
st.set_page_config(
    page_title="Visualization App",
    page_icon="📊",
    layout="wide",  # You can adjust the layout as needed
    initial_sidebar_state="expanded",
)
st.markdown('<style>div.block-container{padding-top:1rem;}</style>', unsafe_allow_html=True)
# image = Image.open('image.jpg')

# Header
# col1, col2 = st.columns([0.1, 0.9])
# with col1:
#     # st.image(image, width=100)
# import image and set margin as well

image = Image.open('MoVies.png')
st.image(image, use_column_width=True)
# # image margin set
# html_title = """
#     <style>
#     .title-test {
#     font-weight:bold;
#     padding:5px;
#     border-radius:5px;
#     margin-top:10px;
#     background-color: #cc296a;
#     color: white;
#     center-align: center;
#     display:inline-block;
#     }
#     </style>
#     <center><h1 class="title-test">Adidas Interactive Sales Dashboard</h1></center>"""
# # with col2:
# st.markdown(html_title, unsafe_allow_html=True)

# Last updated date
# col3, col4 = st.columns([0.1, 0.9])
# with col3:
    # box_date = str(datetime.datetime.now().strftime("%d %B %Y"))
    # st.write(f"Last updated by:  \n {box_date}")

# Read data
df = pd.read_excel("Adidas.xlsx", sheet_name="Sales")
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df["Month_Year"] = df["InvoiceDate"].dt.strftime("%b'%y")
monthly_sales = df.groupby("Month_Year")["TotalSales"].sum().reset_index()

# Apply dark theme to the plot only
plt.style.use("dark_background")


# Create Matplotlib plot with transparent background
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_alpha(0)  # Transparent figure background
ax.patch.set_alpha(0)   # Transparent axes background

ax.plot(monthly_sales['Month_Year'], monthly_sales['TotalSales'], label='Total Sales Over Time', 
        color='#cc296a', linestyle='--', marker='o')

# Compute confidence interval
y_mean = monthly_sales['TotalSales'].mean()
y_std = monthly_sales['TotalSales'].std()
ax.fill_between(monthly_sales['Month_Year'], y_mean - y_std, y_mean + y_std, 
                color='#dd32e3', alpha=0.2, label='Confidence Interval')

# Customize text colors for dark background
ax.set_xlabel("Month-Year", color="white")
ax.set_ylabel("Total Sales", color="white")
# ax.set_title("Total Sales Over Time with Confidence Interval", color="white")
ax.legend(facecolor='black', edgecolor='white', labelcolor='white')

# Adjust tick colors
plt.xticks(rotation=45, color="white")
plt.yticks(color="white")
# Add Heading of Line Plot should be centered
# Title inside Streamlit (Centered, Gradient Color Effect)
st.markdown(
    """
    <h1 style='text-align: center; font-size: 28px; font-weight: bold;'>
        📈 <span style='background: linear-gradient(90deg, #ff8c00, #ff0080);
                        -webkit-background-clip: text;
                        color: transparent;'>Stunning Line Plot of Total Sales Over Time with Confidence Interval</span> 🚀
    </h1>
    """,
    unsafe_allow_html=True
)


# Display plot in Streamlit
st.pyplot(fig)

# Section for Monthly Sales Table & Download Button and make it centered

# Centering the buttons below the plot
with st.container():
    col1, col2, col3 = st.columns([0.3, 0.4, 0.3])  # Adjust widths to center elements
    with col2:
        # Creating two columns inside col2 to keep elements in the same line
        exp_col, btn_col = st.columns([0.7, 0.3])  

        with exp_col:
            expander = st.expander("Monthly wise Sales")
            monthly_sales = df.groupby("Month_Year")["TotalSales"].sum().reset_index()
            expander.write(monthly_sales)

        with btn_col:
            st.download_button("Get Data", data=monthly_sales.to_csv().encode("utf-8"),
                               file_name="MonthlySales.csv", mime="text/csv")


# view1, dwn1 = st.columns([0.1, 1], vertical_alignment = "center")

# with view1:
#     expander = st.expander("Monthly wise Sales")
#     expander.write(monthly_sales)

# with dwn1:
#     st.download_button("Get Data", data=monthly_sales.to_csv().encode("utf-8"),
#                        file_name="MonthlySales.csv", mime="text/csv")
############################################################################################
# --- Bar Chart: Sales by Region ---
# Aggregate sales data by Region
region_sales = df.groupby('Region')['TotalSales'].sum().reset_index()

# Define colors and hatch patterns
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
error_colors = ['yellow', 'cyan', 'lime', 'pink', 'white', 'grey']  # Error bar colors
hatch_patterns = ['/', '\\', '*', '+', 'x', '-']
error = region_sales['TotalSales'] * 0.05  # Assuming 5% error margin

# Create bar chart with transparent background
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_alpha(0)  # Transparent figure background
ax.patch.set_alpha(0)   # Transparent axes background

bars = ax.bar(region_sales['Region'], region_sales['TotalSales'], color=colors, edgecolor='black')

# Add error bars separately with different colors
for i, bar in enumerate(bars):
    ax.errorbar(bar.get_x() + bar.get_width() / 2, bar.get_height(), 
                yerr=error[i], capsize=5, ecolor=error_colors[i], fmt='none')

# Add hatch patterns
for bar, hatch in zip(bars, hatch_patterns):
    bar.set_hatch(hatch)

# Add percentage labels on bars
total_sales = sum(region_sales['TotalSales'])
for bar in bars:
    height = bar.get_height()
    percentage = (height / total_sales) * 100
    ax.text(bar.get_x() + bar.get_width() / 2, height, f'{percentage:.1f}%', 
            ha='center', va='bottom', color="white")

# Set labels with white text
ax.set_xlabel("Region", color="white")
ax.set_ylabel("Total Sales ($)", color="white")
# ax.set_title("Total Sales by Region", color="white")

plt.xticks(rotation=45, color="white")
plt.yticks(color="white")
# Modern Title with Cool Colors
st.markdown(
    """
    <h1 style='text-align: center; font-size: 28px; font-weight: bold;'>
        📊 <span style='background: linear-gradient(90deg, #00c6ff, #0072ff);
                        -webkit-background-clip: text;
                        color: transparent;'>Insightful Bar Chart of Total Sales by Region</span> 🚀
    </h1>
    """,
    unsafe_allow_html=True
)
# Display plot in Streamlit
st.pyplot(fig)

with st.container():
    col1, col2, col3 = st.columns([0.3, 0.4, 0.3])  # Center content
    with col2:
        exp_col, btn_col = st.columns([0.7, 0.3])  # Divide into two parts

        with exp_col:
            expander = st.expander("Region wise Sales")
            region_sales = df.groupby('Region')['TotalSales'].sum().reset_index()
            expander.write(region_sales)

        with btn_col:
            st.download_button("Get Data", data=region_sales.to_csv().encode("utf-8"),
                               file_name="RegionSales.csv", mime="text/csv")

####################################################################################################
# Modern Title with Cool Colors & Numbering
st.markdown(
    """
    <h1 style='text-align: center; font-size: 28px; font-weight: bold;'>
        📈 <span style='background: linear-gradient(90deg, #FF5733, #FFC300);
                        -webkit-background-clip: text;
                        color: transparent;'>Stunning Histogram Visualization</span> 📊
    </h1>
    """,
    unsafe_allow_html=True
)


# Generate random normal dataset
np.random.seed(42)
data = np.random.normal(loc=100, scale=15, size=1000)

# Sample dataset (Replace with actual data)
sales_df = pd.DataFrame({'UnitsSold': np.random.randint(50, 500, size=1000)})  # Example dataset

# Create two columns in Streamlit
col1, col2 = st.columns(2)

# ---- LEFT PLOT ----
with col1:
    # st.subheader("Histogram with KDE (Random Data)")

    fig1, ax1 = plt.subplots(figsize=(6, 4), facecolor='none')  # Transparent figure
    ax1.set_facecolor('none')  # Transparent plot background

    counts, bins, patches = ax1.hist(data, bins=30, density=True, color='green', edgecolor='black', alpha=0.3, label='Histogram')

    # Compute KDE manually
    kde = gaussian_kde(data)
    x_vals = np.linspace(min(data), max(data), 100)
    kde_vals = kde(x_vals)
    ax1.plot(x_vals, kde_vals, color='red', linestyle='--', label='KDE Curve')

    ax1.set_xlabel("Value")
    ax1.set_ylabel("Density")
    ax1.set_title("Histogram with KDE")
    ax1.legend()

    fig1.patch.set_alpha(0)  # Fully transparent figure background
    st.pyplot(fig1)  # Display in the left column

# ---- RIGHT PLOT ----
with col2:
    # st.subheader("Histogram with KDE (Units Sold)")

    fig2, ax2 = plt.subplots(figsize=(6, 4), facecolor='none')  # Transparent figure
    ax2.set_facecolor('none')  # Transparent plot background

    counts, bins, patches = ax2.hist(sales_df['UnitsSold'], bins=20, density=True, color='orange', edgecolor='black', alpha=0.3, label='Histogram')

    # Compute KDE for Units Sold
    kde_units = gaussian_kde(sales_df['UnitsSold'].dropna())
    x_vals_units = np.linspace(min(sales_df['UnitsSold']), max(sales_df['UnitsSold']), 100)
    kde_vals_units = kde_units(x_vals_units)
    ax2.plot(x_vals_units, kde_vals_units, color='red', linestyle='--', label='KDE Curve')

    ax2.set_xlabel("Units Sold")
    ax2.set_ylabel("Density")
    ax2.set_title("Histogram with KDE (Units Sold)")
    ax2.legend()

    fig2.patch.set_alpha(0)  # Fully transparent figure background
    st.pyplot(fig2)  # Display in the right column
######################################################################
# Modern Title with Cool Colors & Numbering
st.markdown(
    """
    <h1 style='text-align: center; font-size: 28px; font-weight: bold;'>
        ✨ <span style='background: linear-gradient(90deg, #6A0DAD, #FF007F);
                        -webkit-background-clip: text;
                        color: transparent;'>Mesmerizing Scatter Plot of Total Sales vs. Operating Profit (Colored by Region, Sized by Units Sold)</span> 🎯
    </h1>
    """,
    unsafe_allow_html=True
)
# Function to create a scatter plot with color mapping and marker sizing
sales_df = pd.read_excel("Adidas.xlsx", sheet_name="Sales")
def plot_scatter_sales_profit(sales_df):
    """
    This function creates a scatter plot of Total Sales vs. Operating Profit.
    - Colors represent different Regions.
    - Marker sizes are based on Units Sold.
    - Transparent background for seamless integration in Streamlit.
    """
    fig, ax = plt.subplots(figsize=(10, 6), facecolor='none')  # Transparent figure
    ax.set_facecolor('none')  # Transparent plot background
    fig.patch.set_alpha(0)  # Fully transparent figure background

    # Define unique regions and assign colors
    unique_regions = sales_df['Region'].unique()
    colors = plt.cm.viridis(np.linspace(0, 1, len(unique_regions)))
    region_color_map = dict(zip(unique_regions, colors))

    # Scatter plot
    for region in unique_regions:
        subset = sales_df[sales_df['Region'] == region]
        ax.scatter(subset['TotalSales'], subset['OperatingProfit'], 
                   s=subset['UnitsSold'] / 10,  # Scale marker size
                   color=region_color_map[region], label=region, alpha=0.7, edgecolors='black')

    ax.set_xlabel("Total Sales ($)")
    ax.set_ylabel("Operating Profit ($)")
    # ax.set_title("Total Sales vs. Operating Profit (Colored by Region, Sized by Units Sold)")
    ax.legend(title="Region")
    # ax.grid(False, linestyle='--', alpha=0.5)

    # Display in Streamlit
    st.pyplot(fig)

# Call function to plot scatter plot in Streamlit
plot_scatter_sales_profit(sales_df)

#######################################################
st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
        <span style="font-size: 32px;">🏆</span>
        <span style="
            background: linear-gradient(to right, #FF5733, #C70039);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 38px;
            font-weight: bold;
        ">Bar Charts - Top 3 Categories Monthly Revenue</span>
        <span style="font-size: 32px;">📊</span>
    </div>
    """,
    unsafe_allow_html=True
)


# Function to create grouped and stacked bar charts for top 3 categories
# Function to create grouped and stacked bar charts for top 3 categories
def plot_grouped_stacked_bar(sales_df):
    """
    This function creates grouped and stacked bar charts for the top 3 product categories based on total sales.
    """
    # Aggregate total sales by category
    top_categories = sales_df.groupby('Product')['TotalSales'].sum().nlargest(3).index
    filtered_df = sales_df[sales_df['Product'].isin(top_categories)]
    
    # Aggregate total sales by month and category
    filtered_df['Month'] = filtered_df['InvoiceDate'].dt.strftime('%b')
    revenue_df = filtered_df.groupby(['Month', 'Product'])['TotalSales'].sum().unstack(fill_value=0)
    
    # Ensure months are in correct order
    months_order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    revenue_df = revenue_df.reindex(months_order, axis=0, fill_value=0)
    
    # Plot grouped and stacked bar charts
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))
    
    # Set transparent background
    fig.patch.set_alpha(0)  # Transparent figure background
    
    for ax in axes:
        ax.set_facecolor('none')  # Transparent plot background

    # Grouped bar chart
    revenue_df.plot(kind='bar', ax=axes[0], width=0.8, edgecolor='black', hatch='/.', color=['#e2d810', '#d9138a', '#12a4d9'])
    axes[0].set_title("Grouped Bar Chart - Top 3 Categories Monthly Revenue")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("Revenue ($)")
    axes[0].legend(title="Product Category", loc="lower center")
    
    # Stacked bar chart
    revenue_df.plot(kind='bar', stacked=True, ax=axes[1], width=0.8, edgecolor='black', hatch='//|\\\\', color=['#ff218c', '#ffd800', '#21b1ff'])
    axes[1].set_title("Stacked Bar Chart - Top 3 Categories Monthly Revenue")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("Revenue ($)")
    axes[1].legend(title="Product Category", loc="lower center")
    
    plt.tight_layout()
    
    # Display in Streamlit
    st.pyplot(fig)

# Call function to plot grouped and stacked bar charts for top 3 categories
plot_grouped_stacked_bar(sales_df)
##########################################################
import streamlit as st
import geopandas as gpd
import matplotlib.pyplot as plt
import os

# Check if shapefile components exist
files = ["ne_110m_admin_0_countries.shp",
         "ne_110m_admin_0_countries.dbf",
         "ne_110m_admin_0_countries.shx",
         "ne_110m_admin_0_countries.prj"]

for file in files:
    print(f"{file}: {'Exists' if os.path.exists(file) else 'Missing'}")

os.environ["SHAPE_RESTORE_SHX"] = "YES"  # Attempt to restore .shx file

# Load shapefile
world = gpd.read_file("ne_110m_admin_0_countries.shp")

# Define continent colors
continent_colors = {
    "Asia": "skyblue",
    "Africa": "lightcoral",
    "Europe": "lightgreen",
    "North America": "gold",
    "South America": "violet",
    "Oceania": "orange",
    "Antarctica": "gray"
}

# Assign colors based on continent, default to light gray if missing
world["color"] = world["CONTINENT"].map(continent_colors).fillna("lightgray")

# Highlight Pakistan and India
world.loc[world["ADMIN"] == "Pakistan", "color"] = "red"
world.loc[world["ADMIN"] == "India", "color"] = "darkblue"

# Create figure and axis with transparency
fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_alpha(0)  # Make figure background transparent
ax.set_facecolor("none")  # Make axes background transparent

# Plot world map
world.plot(ax=ax, edgecolor="black", color=world["color"])

# Define approximate continent label positions
continent_labels = {
    "Asia": (90, 40, "skyblue"),
    "Africa": (20, 0, "lightcoral"),
    "Europe": (20, 55, "lightgreen"),
    "North America": (-100, 50, "gold"),
    "South America": (-60, -15, "violet"),
    "Oceania": (140, -25, "orange"),
    "Antarctica": (0, -75, "gray")
}

# Add continent labels inside colored boxes
for continent, (x, y, color) in continent_labels.items():
    ax.text(
        x, y, continent, fontsize=12, ha='center', fontweight='bold', color='black',
        bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3')
    )

# Add label for Pakistan
pakistan = world[world["ADMIN"] == "Pakistan"]
if not pakistan.empty:
    x_pak, y_pak = pakistan.geometry.centroid.x.iloc[0], pakistan.geometry.centroid.y.iloc[0] - 12
    ax.text(
        x_pak, y_pak, "Pakistan", fontsize=10, ha='center', fontweight='bold', color='white',
        bbox=dict(facecolor='red', edgecolor='black', boxstyle='round,pad=0.3')
    )

# Add label for India
india = world[world["ADMIN"] == "India"]
if not india.empty:
    x_ind, y_ind = india.geometry.centroid.x.iloc[0], india.geometry.centroid.y.iloc[0] - 12
    ax.text(
        x_ind, y_ind, "India", fontsize=10, ha='center', fontweight='bold', color='white',
        bbox=dict(facecolor='darkblue', edgecolor='black', boxstyle='round,pad=0.3')
    )

# # Remove axis borders and ticks
# ax.set_xticks([])
# ax.set_yticks([])
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# ax.spines["left"].set_visible(False)
# ax.spines["bottom"].set_visible(False)

# # Title
# ax.set_title("Continents Highlighted by Color (Pakistan & India Highlighted)", fontsize=14, color='white')
# # Centered Markdown Heading
# st.markdown(
#     "<h2 style='text-align: center; color: black;'>🌍 Continents Highlighted by Color (Pakistan & India Highlighted) 🌍</h2>",
#     unsafe_allow_html=True
# )
st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
        <span style="font-size: 32px;">🥇</span>
        <span style="
            background: linear-gradient(to right, #1E90FF, #00FA9A);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 32px;
            font-weight: bold;
        ">Continents Highlighted by Color (Pakistan & India Highlighted)</span>
        <span style="font-size: 32px;">🌍</span>
    </div>
    """,
    unsafe_allow_html=True
)


# Display plot in Streamlit
st.pyplot(fig)
###################################################################

#################################################################
st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
        <span style="font-size: 52px;">🏆</span>
        <span style="
            background: linear-gradient(to right, #FF4500, #FFD700);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 48px;
            font-weight: bold;
        "> 3D Surface Plot with Contour Overlay </span>
        <span style="font-size: 52px;">📊</span>
    </div>
    """,
    unsafe_allow_html=True
)


from mpl_toolkits.mplot3d import Axes3D

# Function to create a 3D surface plot with contour overlay
def plot_3d_surface():
    """
    This function generates a 3D surface plot of the function f(x,y) = sin(x) * cos(y)
    and overlays a contour plot with a transparent background.
    """
    # Generate data
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(X) * np.cos(Y)
    
    # Create a 3D figure
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Set transparent background
    fig.patch.set_alpha(0)  # Transparent figure background
    ax.set_facecolor('none')  # Transparent plot background
    
    # Plot the surface
    surf = ax.plot_surface(X, Y, Z, cmap='coolwarm', edgecolor='k', alpha=0.8)
    
    # Overlay contour plot
    ax.contour(X, Y, Z, zdir='z', offset=-1, cmap='coolwarm')
    
    # Labels and title
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    # ax.set_title("3D Surface Plot with Contour Overlay", fontsize=14, fontweight='bold')

    # Customize view angle
    ax.view_init(elev=30, azim=135)
    
    # Add color bar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
    
    # Display in Streamlit
    st.pyplot(fig)

# Call function to plot 3D surface with contour in Streamlit

plot_3d_surface()
###################################################################
st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
        <span style="font-size: 50px;">✨</span>
        <span style="
            background: linear-gradient(to right, #00C9FF, #92FE9D);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 46px;
            font-weight: bold;
        "> Interactive Total Sales vs. Operating Profit </span>
        <span style="font-size:50px;">📍</span>
    </div>
    """,
    unsafe_allow_html=True
)

# Function to create an interactive scatter plot
def plot_interactive_scatter(sales_df):
    fig = px.scatter(
        sales_df, 
        x="TotalSales", 
        y="OperatingProfit", 
        size="UnitsSold", 
        color="Region", 
        hover_name="Retailer", 
        hover_data={"TotalSales": True, "OperatingProfit": True, "UnitsSold": True, "Region": True, "Retailer": True},
        # title="📊 Interactive Total Sales vs. Operating Profit",
        labels={"TotalSales": "Total Sales ($)", "OperatingProfit": "Operating Profit ($)", "UnitsSold": "Units Sold"},
        template="plotly_dark"
    )
    
    # Improve marker visibility with better contrast
    fig.update_traces(marker=dict(line=dict(width=2, color='Black')))
    
    # Improve layout
    fig.update_layout(
        legend_title_text="Region", 
        hovermode="closest",
        # title=dict(font=dict(size=20, color="white")),
        xaxis=dict(title_font=dict(size=14, color="white")),
        yaxis=dict(title_font=dict(size=14, color="white")),
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent background
        plot_bgcolor="rgba(0,0,0,0)"  # Transparent plot area
    )

    # Display in Streamlit
    st.plotly_chart(fig, use_container_width=True)


# Call function to plot in Streamlit
plot_interactive_scatter(sales_df)

################################################################
import streamlit as st
import numpy as np
import plotly.graph_objects as go

st.markdown(
    """
    <div style="display: flex; justify-content: center; align-items: center; gap: 10px;">
        <span style="font-size: 52px;">🌊</span>
        <span style="
            background: linear-gradient(to right, #ff0080, #ff8c00, #ffeb00);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 52px;
            font-weight: bold;
        "> Interactive Wave Plot </span>
        <span style="font-size: 52px;">🔄</span>
    </div>
    """,
    unsafe_allow_html=True
)


# Generate initial data
x = np.linspace(0, 10, 1000)
freq_init = 1.0  # Ensure this is a float
amp_init = 1.0  # Ensure this is a float
wave_type = "sin"

def generate_wave(freq, amp, wave_type):
    if wave_type == "sin":
        return amp * np.sin(freq * x)
    elif wave_type == "cos":
        return amp * np.cos(freq * x)
    else:
        return amp * np.tan(freq * x)

# Streamlit App
# st.title("🌊 Interactive Wave Plot 🌊")

# Sidebar controls
st.sidebar.header("🔧 Customize Wave")
freq = st.sidebar.slider("Frequency", 0.1, 5.0, float(freq_init), 0.1)
amp = st.sidebar.slider("Amplitude", 0.1, 2.0, float(amp_init), 0.1)
wave_type = st.sidebar.radio("Wave Type", ["sin", "cos", "tan"], index=0)

show_grid = st.sidebar.checkbox("Show Grid", value=True)
line_style = "dash" if st.sidebar.checkbox("Dashed Line", value=False) else "solid"
wave_color = st.sidebar.selectbox("Wave Color", ["cyan", "magenta", "yellow", "lime"])
line_thickness = st.sidebar.slider("Line Thickness", 1, 5, 2)

xaxis_color = st.sidebar.selectbox("X-axis Color", ["white", "red", "blue", "green"])
yaxis_color = st.sidebar.selectbox("Y-axis Color", ["white", "red", "blue", "green"])
title_color = st.sidebar.selectbox("Title Color", ["white", "red", "blue", "green"])

# Generate wave
y = generate_wave(freq, amp, wave_type)

# Plotly figure with a transparent background
fig = go.Figure(data=[go.Scatter(x=x, y=y, mode="lines", line=dict(color=wave_color, dash=line_style, width=line_thickness))])

fig.update_layout(
    # title=dict(text="🌊 Interactive Wave Plot 🌊", font=dict(size=42, color=title_color)),  # Increased size
    xaxis_title="📏 X-axis",
    yaxis_title="📏 Y-axis",
    template="plotly_dark",
    paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
    plot_bgcolor='rgba(0,0,0,0)',   # Transparent background
    font=dict(color=title_color),
    xaxis=dict(showgrid=show_grid, title_font=dict(color=xaxis_color,size=24)),
    yaxis=dict(showgrid=show_grid, title_font=dict(color=yaxis_color,size=24))
)

# Display plot
st.plotly_chart(fig, use_container_width=True)

##################################################
# import streamlit as st
# import geopandas as gpd
# import matplotlib.pyplot as plt

# import os

# files = ["ne_110m_admin_0_countries.shp",
#          "ne_110m_admin_0_countries.dbf",
#          "ne_110m_admin_0_countries.shx",
#          "ne_110m_admin_0_countries.prj"]

# for file in files:
#     print(f"{file}: {'Exists' if os.path.exists(file) else 'Missing'}")

# os.environ["SHAPE_RESTORE_SHX"] = "YES"  # Attempt to restore .shx file

# world = gpd.read_file("ne_110m_admin_0_countries.shp")
# # Define continents and their colors
# continent_colors = {
#     "Asia": "skyblue",
#     "Africa": "lightcoral",
#     "Europe": "lightgreen",
#     "North America": "gold",
#     "South America": "violet",
#     "Oceania": "orange",
#     "Antarctica": "gray"
# }

# # Assign colors based on continent, default to 'lightgray' if missing
# world["color"] = world["CONTINENT"].map(continent_colors).fillna("lightgray")

# # Highlight Pakistan and India separately
# world.loc[world["ADMIN"] == "Pakistan", "color"] = "red"
# world.loc[world["ADMIN"] == "India", "color"] = "darkblue"

# # Create Matplotlib figure
# fig, ax = plt.subplots(figsize=(12, 6))
# fig.patch.set_alpha(0)  # Make background transparent

# world.plot(ax=ax, edgecolor="black", color=world["color"])

# # Define approximate continent locations for labeling
# continent_labels = {
#     "Asia": (90, 40, "skyblue"),
#     "Africa": (20, 0, "lightcoral"),
#     "Europe": (20, 55, "lightgreen"),
#     "North America": (-100, 50, "gold"),
#     "South America": (-60, -15, "violet"),
#     "Oceania": (140, -25, "orange"),
#     "Antarctica": (0, -75, "gray")
# }

# # Add continent labels inside colored boxes
# for continent, (x, y, color) in continent_labels.items():
#     ax.text(
#         x, y, continent, fontsize=12, ha='center', fontweight='bold', color='black',
#         bbox=dict(facecolor=color, edgecolor='black', boxstyle='round,pad=0.3')
#     )

# # Add label for Pakistan
# pakistan = world[world["ADMIN"] == "Pakistan"]
# if not pakistan.empty:
#     x_pak, y_pak = pakistan.geometry.centroid.x.iloc[0], pakistan.geometry.centroid.y.iloc[0] - 12
#     ax.text(
#         x_pak, y_pak, "Pakistan", fontsize=10, ha='center', fontweight='bold', color='white',
#         bbox=dict(facecolor='red', edgecolor='black', boxstyle='round,pad=0.3')
#     )

# # Add label for India
# india = world[world["ADMIN"] == "India"]
# if not india.empty:
#     x_ind, y_ind = india.geometry.centroid.x.iloc[0], india.geometry.centroid.y.iloc[0] - 12
#     ax.text(
#         x_ind, y_ind, "India", fontsize=10, ha='center', fontweight='bold', color='white',
#         bbox=dict(facecolor='darkblue', edgecolor='black', boxstyle='round,pad=0.3')
#     )

# # Title
# ax.set_title("Continents Highlighted by Color (Pakistan & India Highlighted)", fontsize=14)

# # Display plot in Streamlit
# st.pyplot(fig)


###############
# Contact Page#
###############
    #st.title(f"You have selected {selected}")
    ### About the author
st.write("##### About the author:")
    
    ### Author name
st.write("<p style='color:blue; font-size: 50px; font-weight: bold;'>Usama Munawar</p>", unsafe_allow_html=True)
    
    ### Connect on social media
st.write("##### Connect with me on social media")
    
### Add social media links
### URLs for images
linkedin_url = "https://img.icons8.com/color/48/000000/linkedin.png"
github_url = "https://img.icons8.com/fluent/48/000000/github.png"
youtube_url = "https://img.icons8.com/?size=50&id=19318&format=png"
twitter_url = "https://img.icons8.com/color/48/000000/twitter.png"
facebook_url = "https://img.icons8.com/color/48/000000/facebook-new.png"
    
### Redirect URLs
linkedin_redirect_url = "https://www.linkedin.com/in/abu--usama"
github_redirect_url = "https://github.com/UsamaMunawarr"
youtube_redirect_url ="https://www.youtube.com/@CodeBaseStats"
twitter_redirect_url = "https://twitter.com/Usama__Munawar?t=Wk-zJ88ybkEhYJpWMbMheg&s=09"
facebook_redirect_url = "https://www.facebook.com/profile.php?id=100005320726463&mibextid=9R9pXO"
    
    ### Add links to images
st.markdown(f'<a href="{github_redirect_url}"><img src="{github_url}" width="60" height="60"></a>'
            f'<a href="{linkedin_redirect_url}"><img src="{linkedin_url}" width="60" height="60"></a>'
            f'<a href="{youtube_redirect_url}"><img src="{youtube_url}" width="60" height="60"></a>'
            f'<a href="{twitter_redirect_url}"><img src="{twitter_url}" width="60" height="60"></a>'
            f'<a href="{facebook_redirect_url}"><img src="{facebook_url}" width="60" height="60"></a>', unsafe_allow_html=True)
##########################################
# Display a message if no page is selected#
##############################################

####################
# Thank you message#
#####################
st.write("<p style='color:green; font-size: 30px; font-weight: bold;'>Thank you for using this app, share with your friends!😇</p>", unsafe_allow_html=True)

