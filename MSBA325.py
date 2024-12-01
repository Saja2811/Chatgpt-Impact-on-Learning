import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

st.title('Generative AI Usage Among Students')
csv_file_path = 'survey_clean.csv'
df = pd.read_csv(csv_file_path)


# Problem Statement
st.subheader("Understanding the Impact of AI Tools on Student Learning")
st.markdown("""
### The Problem:
Students are increasingly using generative AI tools like ChatGPT in their academic learning. However, there is uncertainty about whether these tools:
- Improve their learning.
- Affect their critical thinking skills.
- Create challenges like overreliance or misinformation.
""")

st.markdown("<h3 style='color:blue; font-size:35px;'>Exploratory Visualizations</h3>", unsafe_allow_html=True)


#Visualization 1
st.subheader("Dependency on ChatGPT")

# Add "Select All" option for Study Fields
study_fields = df["StudyField"].dropna().unique().tolist()
study_fields.insert(0, "All")  # Add "All" option
selected_study_fields = st.multiselect("Select Study Fields:", options=study_fields, default="All")

# Add "Select All" option for Purposes
purposes = df["Purpose"].dropna().str.split(",").explode().unique().tolist()
purposes.insert(0, "All")  # Add "All" option
selected_purposes = st.multiselect("Select Purposes of Use:", options=purposes, default="All")

# Filter the dataset based on selections
if "All" in selected_study_fields:
    study_field_filter = df["StudyField"].notna()  # Include all valid values
else:
    study_field_filter = df["StudyField"].isin(selected_study_fields)

if "All" in selected_purposes:
    purpose_filter = df["Purpose"].notna()  # Include all valid values
else:
    purpose_filter = df["Purpose"].str.contains("|".join(selected_purposes), na=False)

filtered_data = df[study_field_filter & purpose_filter]

# Check if filtered data is available
if filtered_data.empty:
    st.warning("No data available for the selected filters. Please try a different combination.")
else:
    # Bar chart of Dependency levels
    dependency_counts = filtered_data["Dependency"].value_counts().reset_index()
    dependency_counts.columns = ["Dependency Level", "Count"]

    fig = px.bar(
        dependency_counts,
        x="Dependency Level",
        y="Count",
        title="Dependency Levels on AI",
        labels={"Dependency Level": "Dependency Level", "Count": "Number of Students"},
        template="plotly_white",
        color_discrete_sequence=["#1f77b4"],  # Set bars to dark blue
    )
    # Customize the layout to make the title bigger
    fig.update_layout(
        title_font_size=24,
        title_x=0.5  # Center the title
    )
    st.plotly_chart(fig, use_container_width=True)


# Visualization 2: Percentage of Students Using AI 
st.subheader("AI Usage Among Students")
# Count AI usage
ai_usage_counts = df['UsesAI'].value_counts()
# Generate a sequential color palette (dark to light)
colors = sns.color_palette("Blues_r", len(ai_usage_counts))  # Reverse palette for dark-to-light shading
# Pie Chart
fig1, ax1 = plt.subplots(figsize=(4, 4))  # Increased figure size for clarity
ax1.pie(
    ai_usage_counts, 
    labels=ai_usage_counts.index, 
    textprops={'fontsize': 10},  # Increased text size for better readability
    autopct='%1.1f%%',  # Show percentage values
    startangle=120, 
    colors=colors,
)
st.pyplot(fig1)

# Adding Insights
total_students = ai_usage_counts.sum()
students_using_ai = ai_usage_counts.get(1, 0)  # Assuming 1 = Using AI
students_not_using_ai = ai_usage_counts.get(0, 0)  # Assuming 0 = Not Using AI
st.markdown(f"""
A total of **{total_students}** students were surveyed, among whom **{students_not_using_ai} ({students_not_using_ai / total_students * 100:.1f}%)** reported using AI, while **{students_using_ai} ({students_using_ai / total_students * 100:.1f}%)** indicated they don't use AI. These results highlight that a significant proportion of students rely on AI in their studies.
""")



# Visualization 3: AI Familiarity Levels
st.subheader("AI Familiarity Levels")
# Create a custom color palette for bars
familiarity_levels = sorted(df['Familiarity'].unique())
custom_palette = {level: 'darkblue' if level >= 4 else 'lightblue' for level in familiarity_levels}
# Create the plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(
    x='Familiarity', 
    data=df, 
    hue='Familiarity',  # Use Familiarity as hue
    palette=custom_palette, 
    order=familiarity_levels, 
    dodge=False  # Ensures bars are not split into groups
)
ax.set_title("Familiarity with Generative AI", fontweight='bold')
ax.set_xlabel("Familiarity Level")
ax.set_ylabel("Count")
ax.legend([], [], frameon=False)  # Remove legend
ax.grid(False)
st.pyplot(fig)


# Visualization 4: Purpose of AI Usage
# Group similar purposes into broader categories
df['SimplifiedPurpose'] = df['Purpose'].apply(
    lambda x: 'Research Assistance' if 'Research' in x else
              'Writing Assistance' if 'Writing' in x else
              'Technical Help' if 'Coding' in x else
              'Concept Understanding' if 'Concept' in x else
              'Other'
)
# Count occurrences of each simplified purpose
purpose_counts = df['SimplifiedPurpose'].value_counts()
st.title("Purpose of Using AI Tools")
# Plot the data
top_purposes = purpose_counts.head(5)  # Display only the top 5 categories
# Plot the bar chart using Matplotlib
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=top_purposes.values, y=top_purposes.index, palette="viridis", ax=ax)
ax.set_title("Most Common Purposes for Using AI", fontweight='bold')
ax.set_xlabel("Number of Students")
ax.set_ylabel("Purpose")
st.pyplot(fig)


# Visualization 5: Distribution of AI Usage Frequency
# Calculate the frequency distribution and sort it
frequency_counts = df['Frequency'].value_counts()

# Generate shades of blue and purple
colors = sns.color_palette('Purples', len(frequency_counts))[::-1]  # Darker shades for higher frequencies
# Create the pie chart
fig, ax = plt.subplots(figsize=(3, 3))
ax.pie(
    frequency_counts, 
    labels=frequency_counts.index, 
    autopct='%1.1f%%', 
    startangle=120, 
    textprops={'fontsize': 6}, 
    colors=colors
)
ax.set_title("Distribution of AI Usage Frequency", fontweight='bold')
st.pyplot(fig)
st.write(
    """
    ### Insights:
    The pie chart shows the distribution of AI usage frequency among students. The majority of students interact with AI frequently, with **42.3%** using it daily and **40.9%** using it weekly, together comprising over **80%** of the total. Less frequent usage is much lower, with only **10.4%** using AI monthly and **6.4%** using it rarely. This indicates that AI has become a routine tool for most students while learning.
    """
)


# Visualization 6: Average Hours per Week Using AI (2021-2023)
st.subheader("Average Hours per Week Using AI (2021-2023)")
# Define the columns of interest
usage_columns = ['HoursPW2021', 'HoursPW2022', 'HoursPW2023']

# Ensure the columns are numeric (convert if necessary)
df[usage_columns] = df[usage_columns].apply(pd.to_numeric, errors='coerce')

# Calculate the mean for each year
df_usage = df[usage_columns].mean()
# Create the line chart
fig, ax = plt.subplots(figsize=(8, 6))
sns.lineplot(
    x=df_usage.index, 
    y=df_usage.values, 
    marker='o', 
    color='b', 
    linestyle='-', 
    linewidth=2, 
    markersize=8,
    ax=ax
)
ax.set_title("Average Hours per Week Using AI (2021-2024)", fontweight='bold')
ax.set_xlabel("Year")
ax.set_ylabel("Average Hours")
ax.grid(False)
st.pyplot(fig)


# Visualization 7: Hours per Week Using AI (2021-2023) for Each Student
st.subheader("Interactive Visualization: Hours per Week Using AI (2021-2023)")
# Define the columns of interest
usage_columns = ['HoursPW2021', 'HoursPW2022', 'HoursPW2023']
# To ensure the columns are numeric (convert if necessary)
df[usage_columns] = df[usage_columns].apply(pd.to_numeric, errors='coerce')
# Let the user select the year(s) to visualize
selected_years = st.multiselect(
    "Select the year(s) to visualize:", 
    options=['2021', '2022', '2023'], 
    default=['2021', '2022', '2023']
)
# Filter the data based on the selected years
selected_columns = [f'HoursPW{year}' for year in selected_years]
# Check if any years are selected
if selected_columns:
    # Create the line plot
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.lineplot(data=df[selected_columns], palette="Set1", linewidth=2, ax=ax)
    ax.set_title("Hours per Week Using AI for Selected Years", fontweight='bold')
    ax.set_xlabel("Student Index")
    ax.set_ylabel("Hours per Week")
    ax.legend(
        title="Year", 
        labels=selected_years, 
        loc="upper left", 
        frameon=False
    )
    ax.grid(False) 
    st.pyplot(fig)
else:
    st.warning("Please select at least one year to visualize.")
st.write(
    """
    ### 
    This chart compares weekly hours spent using AI tools by students (2021-2023), showing a clear increase in variability and intensity of usage over time. In 2021, usage was consistent and relatively low, with minimal fluctuations. By contrast, 2022 and especially 2023 display frequent and sharp spikes, indicating that certain students spent significantly more time on AI tools, likely reflecting increased adoption, integration, or reliance on such technologies. The extreme peaks in 2023 suggest a growing trend in AI utilization, potentially driven by new tools, trends, or educational requirements.
    """
)


#Visualization 8
# Add row numbers as the index for visualization
df["Student"] = df.index + 1
st.subheader("Interactive Line Chart: Grades Before and After Using AI")
st.write(
    """
    This chart shows a comparison of students' grades before and after using AI tools. 
    We will use this interactive chart to explore how AI has impacted performance.
    """
)
# Melt the data to create separate categories for AverageBefore and AverageAfter
df_melted = df.melt(id_vars=["Student"], value_vars=["AverageBefore", "AverageAfter"], 
                    var_name="Category", value_name="Average")

# Plotly scatter plot
fig = px.scatter(
    df_melted,
    x="Student",
    y="Average",
    color="Category",
    title="Grades Before and After AI Usage",
    labels={"Student": "Students", "Average": "Grades"},
    template="simple_white",
)
# Customize the layout for clarity
fig.update_layout(
    xaxis_title="Student Number",
    yaxis_title="Grades",
    title_font=dict(size=18, color="black", family="Arial"),
    legend_title=None,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    showlegend=True,
)
st.plotly_chart(fig)
# Add explanation text after the chart
st.markdown("""
### Analysis:
The visualization clearly indicates an improvement in student performance after using AI tools. 
The orange points representing **'AverageAfter'** are consistently higher than the blue points representing **'AverageBefore'** for most students. 
This suggests that the adoption of AI tools has positively impacted academic performance, as reflected in the increased grades.
""")

# Visualization 9: Impact of AI Usage on Understanding
st.subheader("Impact of AI Usage on Understanding")
# Create the count plot
fig, ax = plt.subplots(figsize=(8, 6))
sns.countplot(
    data=df, 
    x='UsesAI', 
    hue='ImproveUnderstanding', 
    palette='Blues',
    ax=ax
)
ax.set_title("Impact of AI Usage on Understanding", fontweight='bold')
ax.set_xlabel("Uses AI")
ax.set_ylabel("Count of Responses")
ax.legend(title="Improvement in Understanding")
ax.grid(False)
st.pyplot(fig)
st.markdown("""
The chart shows the impact of AI usage on understanding, categorized by levels of improvement (1-5) and responses of 'Yes' or 'No' to using AI. Among students who use AI, the highest improvement level, 3, accounts for **~140 responses**, representing approximately **50%** of all responses in this group. Improvement levels 4 and 2 follow with **~80** and **~60 responses**, respectively. In contrast, students who do not use AI show significantly fewer responses across all improvement levels, with the highest being at level 2, totaling **~20 responses** or roughly **10%** of their group.
""")



# Visualization 10: Familiarity with AI by Education Level
st.subheader("Familiarity with AI by Education Level")
# Create a crosstab to calculate counts of each combination of EducationLevel and Familiarity
cross_tab = pd.crosstab(df['EducationLevel'], df['Familiarity'])
# Plot the stacked bar chart
fig, ax = plt.subplots(figsize=(10, 6))
cross_tab.plot(
    kind='bar', 
    stacked=True, 
    colormap='Blues', 
    ax=ax
)
ax.set_title("Familiarity with AI by Education Level", fontweight='bold')
ax.set_xlabel("Education Level")
ax.set_ylabel("Count of Responses")
ax.set_xticklabels(cross_tab.index, rotation=45)
ax.legend(title="Familiarity Level")
ax.grid(False)
st.pyplot(fig)

st.write(
    """
    ### Insights:
    Undergraduate and graduate students have the highest levels of AI familiarity with higher familiarity levels (4 and 5) are concentrated among undergraduate students, reflecting their high exposure to AI in academic settings.
    """
)

#Visualization 11
# Drop null values and count responses in the 'GPTPaid' column
gpt_paid_data_counts = df['GPTPaid'].dropna().value_counts()

# Extract data for the chart
labels = gpt_paid_data_counts.index.tolist()  # Categorical responses (e.g., 'Yes', 'No', 'Maybe')
sizes = gpt_paid_data_counts.values.tolist()  # Count of each response
st.title("Interactive Donut Chart: Likelihood of Paying for ChatGPT's Paid Version")

# Check if there is data to display
if not labels:
    st.warning("No data available for 'GPTPaid'. Please ensure the column has valid responses.")
else:
    # Create the donut chart using Plotly
    fig = go.Figure(
        data=[
            go.Pie(
                labels=labels,
                values=sizes,
                hole=0.4,  # Creates the donut shape
                marker=dict(colors=['#154360', '#2471a3', '#8da0cb', '#3498db', '#aed6f1']),  # Custom colors
                textinfo='percent+label',  # Show both labels and percentages
            )
        ]
    )
    # Update layout for the chart
    fig.update_layout(
        title=dict(
            text="Getting ChatGPT's Paid Version",
            font=dict(size=20, color="black", family="Arial"),
            x=0.5  # Center the title
        ),
        showlegend=True,
        width=600,  # Set the width of the chart
        height=500   # Set the height of the chart
    )
    st.plotly_chart(fig, use_container_width=True)
# Insight for Donut Chart
st.write(
    """
    ### Insights:
    - **49.3%** of respondents are undecided about paying for ChatGPT's paid version.
    - **36.4%** are not willing to pay, while only **14.3%** are ready to purchase it.
    - This suggests a mix of hesitation and uncertainty about the paid version's value proposition.
    """
)

#Visualizaton
# Static data extracted from the dataset
tool_usage_counts = {
    'ChatGPT': 311,
    'Google Gemini': 67,
    'Jasper': 2,
    'Midjourney': 6
}
# Extract data for plotting
tools = list(tool_usage_counts.keys())
usage_counts = list(tool_usage_counts.values())

# Plotting the horizontal bar chart
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(tools, usage_counts, color=['#154360', '#2471a3', '#8da0cb', '#3498db'])
# Add titles and labels
ax.set_title("Usage of Different AI Tools", fontweight="bold")
ax.set_xlabel("Number of Users")
ax.set_ylabel("AI Tools")
st.title("AI Tools Usage Analysis")
st.pyplot(fig)
st.subheader("Tool Usage Summary:")
st.write(
    f"A total of 311 users used ChatGPT, 67 used Google Gemini, 2 used Jasper, and 6 used Midjourney."
)
st.title(" Proposed Solution:")
st.markdown("""   
To address the growing reliance on generative AI tools like ChatGPT in academic learning, the solution lies in promoting AI literacy, establishing clear usage guidelines, and fostering a balance between AI-assisted and traditional learning. Universities should offer workshops to teach students the strengths and limitations of AI, integrate AI into curricula with a focus on critical evaluation, and design assignments that encourage thoughtful validation of AI outputs. Clear policies should define appropriate AI use while incentivizing ethical behavior. Additionally, faculty should receive training to adapt their teaching methods and assessments to AI-influenced learning. By monitoring student usage and maintaining traditional skill-building exercises, this approach ensures that AI enhances learning without undermining critical thinking, academic integrity, or intellectual growth.
""")

