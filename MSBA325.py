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
 Students are increasingly integrating generative AI tools such as ChatGPT into their academic learning. However, it remains unclear how this trend affects their learning outcomes, critical thinking skills, and reliance on AI-generated content. There is a lack of understanding of whether these tools contribute to improved academic performance or create challenges, such as dependence on AI, misinformation, or ethical dilemmas in learning.
""")

st.markdown("<h3 style='color:blue; font-size:35px;'>Exploratory Visualizations</h3>", unsafe_allow_html=True)


#Visualization 1.1
st.subheader("Students Dependency Level on ChatGPT")

# Add "Select All" option for Study Fields
study_fields = df["StudyField"].dropna().unique().tolist()
study_fields.insert(0, "All")  # Add "All" option
selected_study_fields = st.multiselect(
    "Select Study Fields:", options=study_fields, default="All", key="study_fields"
)

# Add "Select All" option for Purposes
purposes = df["Purpose"].dropna().str.split(",").explode().unique().tolist()
purposes.insert(0, "All")  # Add "All" option
selected_purposes = st.multiselect(
    "Select Purposes of Use:", options=purposes, default="All", key="purposes"
)

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

    # Highlight the highest bar
    max_count = dependency_counts["Count"].max()
    dependency_counts["Color"] = dependency_counts["Count"].apply(
        lambda x: "#1f77b4" if x == max_count else "#d3d3d3"
    )  # Blue for max, grey for others

    fig = px.bar(
        dependency_counts,
        x="Dependency Level",
        y="Count",
        title="Dependency Levels on AI",
        labels={"Dependency Level": "Dependency Level", "Count": "Number of Students"},
        template="plotly_white",
    )

    # Apply the custom colors
    fig.update_traces(marker_color=dependency_counts["Color"])

    # Customize the layout to make the title bigger
    fig.update_layout(
        title_font_size=24,
        title_x=0.5,  # Center the title
    )
    st.plotly_chart(fig, use_container_width=True)


#Visualization 1.2
# Data for heatmap
heatmap_data = [[5], [54], [108], [84], [48]]
labels = ["Very High  Dependency", "High  Dependency", "Moderate  Dependency", "Low  Dependency", "Very Low  Dependency"]
fig, ax = plt.subplots(figsize=(4, 7))
ax.set_title("Dependency on AI in Learning", fontsize=14, fontweight="bold")
sns.heatmap(heatmap_data, annot=True, fmt="d", cmap="Blues", yticklabels=labels, cbar=False)
ax.set_xlabel("Percentage of Students")
st.pyplot(fig)


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
st.subheader("Purpose of Using AI Tools")
# Display the top 5 categories
top_purposes = purpose_counts.iloc[:5]  # Use .iloc to fix FutureWarning
# Plot the data
fig, ax = plt.subplots(figsize=(8, 6))

# Highlight top two bars in blue, others in grey
colors = ['blue' if i < 2 else 'grey' for i in range(len(top_purposes))]
# Remove borders using edgecolor
sns.barplot(
    x=top_purposes.values,
    y=top_purposes.index,
    palette=colors,
    ax=ax
)
# Add labels to bars with bold labels for the top 2
for i, bar in enumerate(ax.patches):
    value = top_purposes.values[i]
    label = f"{value}"
    if i < 2:
        ax.text(
            bar.get_width() + 0.1,  # X-coordinate
            bar.get_y() + bar.get_height() / 2,  # Y-coordinate
            label,
            fontsize=12,
            fontweight='bold',
            va='center'  # Center vertically
        )
    else:
        ax.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            label,
            fontsize=10,
            va='center'
        )

# Customize chart aesthetics
ax.set_title("Most Common Purposes for Using AI", fontweight='bold')
ax.set_xlabel("Number of Students")
ax.set_ylabel("Purpose")
ax.spines['top'].set_visible(False)  # Remove top border
ax.spines['right'].set_visible(False)  # Remove right border
st.pyplot(fig)


# Visualization 5: Distribution of AI Usage Frequency
# Calculate the frequency distribution and sort it
frequency_counts = df['Frequency'].value_counts()

# Generate shades of blue 
colors = sns.color_palette('Blues', len(frequency_counts))[::-1]  # Darker shades for higher frequencies
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
# Define the columns of interest
usage_columns = ['HoursPW2021', 'HoursPW2022', 'HoursPW2023']
# Ensure the columns are numeric (convert if necessary)
df[usage_columns] = df[usage_columns].apply(pd.to_numeric, errors='coerce')

# Calculate the mean for each year
df_usage = df[usage_columns].mean()

# Create an interactive line chart using Plotly
fig = go.Figure()

# Add line trace
fig.add_trace(go.Scatter(
    x=df_usage.index, 
    y=df_usage.values, 
    mode='lines+markers',
    name='Average Hours',
    line=dict(color='blue', width=2),
    marker=dict(size=8),
    text=df_usage.values,  # Text displayed on hover
    hoverinfo='x+y+text'  # Display x, y values and the text on hover
))
fig.update_layout(
    title='Average Hours per Week Using AI (2021-2024)',
    xaxis_title='Year',
    yaxis_title='Average Hours',
    template='plotly_dark',
    hovermode='closest'
)
st.plotly_chart(fig)
st.write("""
The chart illustrates the trend of average hours per week spent using AI tools by students from 2021 to 2023. 
- In 2021, the average was 1.8 hours per week, reflecting the early stages of AI tool usage.
- In 2022, usage rose to 2.7 hours per week, showing an increased reliance on AI tools.
- By 2023, the average hours reached 3.2 hours per week, indicating a continuous upward trend in student engagement with AI over time.
This steady increase in usage suggests that students are increasingly incorporating AI tools like ChatGPT into their academic activities.
""")

# Visualization 7:Interactive Stacked Bar Chart
st.subheader("Interactive Stacked Bar Chart: Average Hours per Week vs TimeStudying (2021-2023)")
# Let the user choose the years to display
selected_years = st.multiselect(
    "Select the year(s) to include in the chart:", 
    options=['2021', '2022', '2023'], 
    default=['2021', '2022', '2023'],
    key="unique_multiselect_key"  # Add a unique key
)

# Filter data based on selected years
if len(selected_years) > 0:
    # Select only relevant columns
    filtered_data = df[['TimeStudying'] + [f'HoursPW{year}' for year in selected_years]]

    # Melt the dataframe for easier aggregation
    melted_df = pd.melt(
        filtered_data,
        id_vars='TimeStudying',
        var_name='Year',
        value_name='Hours'
    )
    melted_df['Year'] = melted_df['Year'].str.extract(r'(\d{4})')  # Extract year from column names

    # Calculate average hours per week for each TimeStudying category and year
    avg_df = melted_df.groupby(['TimeStudying', 'Year'])['Hours'].mean().reset_index()

    # Create a stacked bar chart using Plotly
    fig = go.Figure()

    # Define color mapping
    color_mapping = {'2021': 'grey', '2022': 'blue', '2023': 'darkblue'}

    for year in selected_years:
        year_data = avg_df[avg_df['Year'] == year]
        fig.add_trace(
            go.Bar(
                x=year_data['TimeStudying'],
                y=year_data['Hours'],
                name=f"Year {year}",
                marker_color=color_mapping[year],
                text=year_data['Hours'],  # Hover text
                texttemplate='%{text:.2f}',  # Format hover values
                textposition="auto"
            )
        )

    # Customize layout
    fig.update_layout(
        barmode='stack',
        title="Average Hours per Week vs TimeStudying (2021-2023)",
        xaxis_title="Time Studying",
        yaxis_title="Average Hours per Week",
        legend_title="Year",
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Please select at least one year to visualize.")


# Visualization 8: Impact of AI Usage on Understanding
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


# Visualization 9: Familiarity with AI by Education Level
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

#Visualization 10
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

#Visualizaton 11
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

