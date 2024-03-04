import streamlit as st
import pandas as pd
import numpy as np
import pickle
import squarify
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings('ignore')
# sns.set_style('whitegrid')
from operator import attrgetter
import datetime
import plotly.graph_objects as go
import plotly.express as px

#this app is dedicated to my husband Dishit Parekh. He was my main motivation behind this app
# Set the title of the app
st.title("E-Commerce Customer Data Segmentation")

from PIL import Image
st.subheader("Hello! Welcome to My Page of E-Commerce Customer Data Segmentation")

image = Image.open("customer_analysis.png")
st.image(image,use_column_width = True)

st.sidebar.info('This app is created and designed by  [Tejal Mehta](https://www.linkedin.com/in/tejalmehta21/)')

# image2 = Image.open("Profile_Photo.jpg")
st.sidebar.markdown(
    "<div style='display: flex; justify-content: center;'><img src='https://raw.githubusercontent.com/mehtatejal/ecommerce_app/main/Profile_Photo.jpg' alt='profile_photo' style='width:200px;height:350px;'>",
    unsafe_allow_html=True
)
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.info('Want to check out my other cool app. Please [click here](https://mehtatejal-market-segment-app-market-segment-9q4d9s.streamlit.app/)')

with st.expander("About this App"):

    st.write("")

    st.markdown(
        """
    This app is built on transactional data of E-Commerce store selling gift items for all occasions. This app is designed to help businesses gain valuable insights into their customer base and optimize their marketing and promotional strategies. 
    
    The original dataset used for segmentation contained information about an individual's purchase:
    - Invoice Number 
    - Stock Code of the item
    - Description of the item
    - Quantity purchased
    - Date of purchase
    - Unit Price (in GBP) 
    - Customer ID
    - Country
    - Total Price (in GBP)
    """
    )

    st.write("")

    st.markdown(
        """
	The app uses advanced techniques like RFM (Recency, Frequency, Monetary) analysis and Cohort Analysis to segment customer data based on their purchasing behavior. 
    With this information, businesses can identify their most valuable customers, understand their behavior, and tailor their marketing and promotional efforts to each customer group.
	"""
    )

    st.write("")

#read data
df = pd.read_csv('processed_customer_data.zip',parse_dates = ['InvoiceDate'],infer_datetime_format = True)

#show data
with st.expander("Show the E-Commerce Dataframe"):
    st.write(df)

# ----------------- Redict RFM ANALYSIS -----------------------


# ----------------------- RFM ANALYSIS ---------------------
# Add a subheader
with st.expander("Show the RFM Analysis "):
    st.write("<h2 style='text-align: center; color: green;'> RFM ANALYSIS </h2>", unsafe_allow_html=True)

    st.write( """ RFM analysis is a customer segmentation technique that is used to identify the most valuable customer segments based on their transaction history. 
        RFM analysis considers three factors:

        1. Recency (R): How recently did the customer make a purchase?
        2. Frequency (F): How often do they make purchases?
        3. Monetary (M): How much money do they spend?

        As a result of our analysis, customers were divided into different segments 
        based on their RF score.  
        Each segment is assigned a label based on their purchasing behaviour.
        This information can help businesses target their marketing and retention 
        efforts to specific customer segments, such as high-value customers or 
        customers who are at risk of churning. """)

    st.write( """ The Map below shows the final results of our customer segmentation analysis. """ )

    #read data
    rfm = pd.read_csv('RFM.csv')

    seg_map = {r'[1-2][1-2]': 'Lapsed Customers', # "require reactivation" customers, meaning they are not making purchases as frequently and may have disengaged with the business.
               r'[1-2][3-4]': 'At Risk',# making frequent purchases even though they haven't made a purchase in a while.
               r'[1-2]5': 'Churned Customers', # who were once active and engaged customers of a business
               r'3[1-2]': 'Irregular Customers', #meaning they are still making purchases but not as frequently as before. 
               r'33': 'Need Attention',#moderately recent and moderate frequency
               r'[3-4][4-5]': 'Loyal Customers', #moderately recent and frequent,high-value
               r'[4-5][1]': 'New Customers',#made a purchase recently,made fewer purchases
               r'[4-5][2-3]': 'Potential Loyalists',#recent customers and moderate buyers
               r'5[4-5]': 'Top Customers'}#recent customers and high frequecy

    segments = rfm["segment"].value_counts().sort_values(ascending=False)
    
    fig = px.treemap(
    names=[label for label in seg_map.values()],
    parents=['' for _ in seg_map.values()],
    values=segments.values,
    color_discrete_sequence=px.colors.qualitative.Alphabet
    )
    fig.update_layout(
	title=dict(text="Customer Segmentation Map", x=0.5, y=0.9, xanchor='center', yanchor='top'),
        xaxis_title="Frequency",
        yaxis_title="Recency",
        font=dict(size=15)
    )
    st.plotly_chart(fig)

    st.write( """ 

     - The customers with very low recency and frequency were labeled as Lapsed Customers.
     - The customers with high frequency but no recent purchases were labeled as At Risk Customers. 
     - The customers who were once active but have become significantly less active over time were labeled as Churned Customers.
     - The customers who make purchases with low frequency were labeled as Irregular Customers.
     - The customers who make moderate purchases over time were labeled as Need Attention Customers.
     - The customers who consistently make high number of purchases over time were labeled as Loyal Customers.
     - The customers who have recently started making purchases were labeled as New Customers.
     - The customers who are moderate buyers and have made recent purchases were labeled as Potential Loyalists.
     - The customers who make very large number of purchases every time were labeled as Top Customers.
    """)

    st.write(""" These labels can help businesses understand their customer base and tailor their marketing strategies accordingly. """ )
    
# with st.expander("""  "Know Your Customer"  Tool"""):

    
# with st.beta_expander("Enter your details"):
# segment_customers()
def predict(df):
    #load the model
    crm_pipeline = pickle.load(open('crm_pipeline_xgboost.pkl','rb'))
    
    prediction= crm_pipeline.predict(df)
    output_str = prediction[0]
    return output_str

st.sidebar.text(""" 
    Want to know your customer? 
    We are here to help you. 
    Fill the details below 
    and we will segment the customer for you.""")
    # st.write(""" Want to know your customer, click the below button !! """)

    # if st.button("Click Here"):


def main():    
        today_Date = datetime.date.today()
        max_date = st.sidebar.date_input("What is the recent purchase date of customer ?", today_Date)


        recency = (today_Date - max_date).days
        frequency = st.sidebar.number_input("How many times customer has made purchase with you ?", format="%d",step=None)
        monetary = st.sidebar.number_input("What is the total amount of purchase the customer has made with you till now ?", format="%d",step=None)
        
        # Combine the three variables into a dictionary
        data = {'recency': recency, 'frequency': frequency, 'monetary': monetary}

        # Create the dataframe
        rfm = pd.DataFrame(data, index=[0])
        
        result = ""

         # when button is clicked, make the prediction and store it
        if st.sidebar.button("Which Segment this customer belongs to?"):
            # result=segment_customers(today_Date,max_date,frequency,monetary)
            result= predict(rfm)
        
        st.sidebar.success(result)
    

if __name__ == '__main__':
        main ()    


# ----------------------- COHORT ANALYSIS ---------------------
# Add a subheader
with st.expander("Show the Cohort Analysis "):
    st.write("<h2 style='text-align: center; color: green;'>COHORT ANALYSIS</h2>", unsafe_allow_html=True)

    st.write( """Here, monthly cohort analysis is done that groups customers based on their first purchase. 
        The information provided by this analysis includes the following:
	
         - The number of customers in each cohort
         - The retention rate of each cohort over time
         - The overall retention rate of the customer base """)
    data = df.copy()
    #dropping duplicated
    data = data[["CustomerID", "InvoiceNo", "InvoiceDate"]].drop_duplicates()
    
    #getting order month from date - to group transactions by month
    data["order_month"] = data["InvoiceDate"].dt.to_period("M")
    
    #grouping customers and finding first purchase date for each customers
    data["cohort"] = (data.groupby("CustomerID")["InvoiceDate"].transform("min").dt.to_period("M"))
    
    #aggregating the number of unique customers in each cohort for each period
    cohort_data = (data.groupby(["cohort", "order_month"]).agg(n_customers=("CustomerID", "nunique")).reset_index(drop=False))
    
    # calculate the number of periods that have passed since a customer's first purchase
    cohort_data["period_number"] = (cohort_data.order_month - cohort_data.cohort).apply(attrgetter("n"))
   
   #the retention rate of each cohort over time, which is helpful in identifying trends and patterns in customer behavior
    cohort_pivot = cohort_data.pivot_table(index="cohort", columns="period_number", values="n_customers")
    cohort_pivot.index = cohort_pivot.index.astype(str)
    
    #This line of code extracts the first column of the cohort_pivot pivot table, which represents the number of customers in the initial cohort. 
    cohort_size = cohort_pivot.iloc[:, 0]
    
    retention_matrix = cohort_pivot.divide(cohort_size, axis=0)
    # create a list of x and y tick labels
    x_labels = retention_matrix.columns.tolist()
    y_labels = retention_matrix.index.tolist()

    # create the heatmap trace
    heatmap = go.Heatmap(z=retention_matrix.round(3) * 100,
                         x=x_labels,
                         y=y_labels,
                         colorscale="BuPu_r",
                         zmin=0.0,
                         zmax=50.0,
                         colorbar=dict(title="%"),
                         hovertemplate="Retention Percentage : %{z:.0f}%<br>" +
                                   "First Purchase : %{y}<br>" +
                                   "Retention Period : %{x}" +
                                   " months <extra></extra>")

    # create the figure layout
    layout = go.Layout(title=dict(text="Monthly Cohorts : User Retention", x=0.5, y=0.9, xanchor='center', yanchor='top'),
                       xaxis=dict(title="Cohort Period : Time since the first purchase"),
                       yaxis=dict(title="Cohort Group : First Purchase")
		      )

    # create the figure and show it
    fig = go.Figure(data=[heatmap], layout=layout)
    st.plotly_chart(fig)

    st.write( """When a heatmap is used to display the retention rates of different cohorts over time, it becomes apparent that 50% of the customers who registered in December 2010 remained active after 11 months. Additionally, the retention rate of early sign-ups is higher than that of later ones.
     """)


st.write("<p style='text-align:center; font-style: italic;'>Special thanks to my husband Dishit Parekh ❤️ for motivating me to create this app!</p>", unsafe_allow_html=True)



