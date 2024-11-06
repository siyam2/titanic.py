import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
import folium
from streamlit_folium import st_folium
from sklearn.preprocessing import StandardScaler, LabelEncoder

st.set_page_config(
    page_title="Titanic ",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded",
)

# laad dataset
train_df = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#  tabs
tab1, tab2, tab3 = st.tabs(["originele plots", "nieuwe plots", "nieuw model"])

# Tab 1
with tab1:
    # Data cleaning
    mean_age_train = train_df['Age'].mean()
    train_df['Age'] = train_df['Age'].fillna(mean_age_train)
    mean_age_test = df_test['Age'].mean()
    df_test['Age'] = df_test['Age'].fillna(mean_age_test)

    bins = [0, 18, 30, 40, 50, 60, 70, 80]
    train_df['Age_group'] = pd.cut(train_df['Age'], bins)
    df_test['Age_group'] = pd.cut(df_test['Age'], bins)

    # Dropdown menu voor filteren op overleeft of niet overleeft
    survival_filter = st.selectbox("Select Survival Status", options=["All", "Survived", "Not Survived"])

    # Filter op survival of niet
    if survival_filter == "Survived":
        filtered_df = train_df[train_df['Survived'] == 1]
    elif survival_filter == "Not Survived":
        filtered_df = train_df[train_df['Survived'] == 0]
    else:
        filtered_df = train_df

    # plot voor leeftijden en overlevingskans
    st.title("Survival Rate per Age Group")
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.countplot(x='Age_group', hue='Survived', data=filtered_df, ax=ax)
    ax.set_title('Survival Rate per Age Group')
    st.pyplot(fig)
    plt.clf()

  #  overlevingskans per Class
    st.title("Survival Rate per Class")
    plt.figure(figsize=(5, 2.5))
    sns.countplot(x='Pclass', hue='Survived', data=filtered_df)
    plt.title('Survival Rate per Class')
    plt.xlabel('Passenger Class')
    plt.ylabel('Number of Passengers')
    st.pyplot(plt)
    plt.clf()

    
    # Survivalkans met Gender
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.countplot(x='Sex', hue='Survived', data=filtered_df, palette=['coral', 'lightgreen'], ax=ax)
    ax.set_title('Survival Chances by Gender')
    ax.set_xlabel('Gender')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    plt.clf()
 



        #  gemiddelde prijs for Survivors and Non-Survivors
    st.title("Mean Fare for Survivors and Non-Survivors")
    fig, ax = plt.subplots(figsize=(4, 2.5))
    if survival_filter == "All":
        mean_survived = train_df[train_df['Survived'] == 1]['Fare'].mean()
        mean_not_survived = train_df[train_df['Survived'] == 0]['Fare'].mean()
        ax.bar('Survived', mean_survived, color='lightgreen')
        ax.bar('Not Survived', mean_not_survived, color='coral')
    else:
        mean_fare = filtered_df['Fare'].mean()
        ax.bar(survival_filter, mean_fare, color='lightgreen' if survival_filter == "Survived" else 'coral')
    ax.set_ylabel('Mean Fare')
    ax.set_title('Mean Fare for Survivors and Non-Survivors')
    st.pyplot(fig)
    plt.clf()






    #Scatterplot ticketoprijs tegenover overlevingskans
    st.title("Scatter Plot of Fare Against Survival")
    fig, ax = plt.subplots(figsize=(5, 2.5))
    if survival_filter == "All":
        ax.scatter(train_df[train_df['Survived'] == 1]['Fare'], train_df[train_df['Survived'] == 1]['Age'], label='Survived', color='blue')
        ax.scatter(train_df[train_df['Survived'] == 0]['Fare'], train_df[train_df['Survived'] == 0]['Age'], label='Not Survived', color='red')
    else:
        ax.scatter(filtered_df['Fare'], filtered_df['Age'], label=survival_filter, color='blue' if survival_filter == "Survived" else 'red')
    ax.set_xlabel('Fare')
    ax.set_ylabel('Age')
    ax.set_title('Fare Against Survival')
    ax.legend()
    st.pyplot(fig)
    plt.clf()


    
    #leeftijdspreiding
    st.title("Age Distribution of Passengers")
    plt.figure(figsize=(5, 2.5))
    if survival_filter == "All":
        plt.hist(train_df[train_df['Survived'] == 1]['Age'], bins=15, alpha=0.5, label='Survived', color='blue')
        plt.hist(train_df[train_df['Survived'] == 0]['Age'], bins=15, alpha=0.5, label='Not Survived', color='red')
    else:
        plt.hist(filtered_df['Age'], bins=15, alpha=0.5, label=survival_filter, color='blue' if survival_filter == "Survived" else 'red')
    plt.xlabel('Age')
    plt.ylabel('Number of Passengers')
    plt.title('Age Distribution of Passengers')
    plt.legend()
    st.pyplot(plt)
    plt.clf()



    #  Box Plot of Ticket Prices
    st.title("Box Plot of Ticket Prices for Survivors and Non-Survivors")
    plt.figure(figsize=(5, 2.5))
    sns.boxplot(x='Survived', y='Fare', data=filtered_df if survival_filter == "All" else train_df)
    plt.xticks([0, 1], ['Not Survived', 'Survived'])
    plt.xlabel('Survival Status')
    plt.ylabel('Ticket Price (Fare)')
    plt.title('Ticket Price Distribution by Survival Status')
    st.pyplot(plt)
    plt.clf()


    # Survival per SibSp
    st.title("Survival per SibSp")
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.countplot(x='SibSp', hue='Survived', data=filtered_df, ax=ax)
    ax.set_title('Survival per Number of Siblings/Spouses Aboard (SibSp)')
    ax.set_xlabel('Number of Siblings/Spouses (SibSp)')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    plt.clf()

    # leeftijdsdistributie  per SibSp
    st.title("Age Group Distribution per SibSp")
    fig, ax = plt.subplots(figsize=(5, 2.5))
    sns.countplot(x='SibSp', hue='Age_group', data=filtered_df, ax=ax)
    ax.set_title('Age Group per Number of Siblings/Spouses (SibSp)')
    ax.set_xlabel('Number of Siblings/Spouses (SibSp)')
    ax.set_ylabel('Count')
    st.pyplot(fig)
    plt.clf()




    # Proportie Plot of Survival per Parch
    st.header("Survival Proportion per Parch")
    df_parch = filtered_df.groupby(['Parch', 'Survived']).size().unstack()
    df_parch_normalized = df_parch.div(df_parch.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    df_parch_normalized.plot(kind='bar', stacked=True, ax=ax, color=['coral', 'lightgreen'])
    ax.set_ylabel('Proportion of Passengers')
    ax.set_title('Survival Proportion per Parch')
    st.pyplot(fig)
    plt.clf()

    # leeftijdsgroep per Parch
    st.header("Age Group per Parch")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(x='Parch', hue='Age_group', data=filtered_df, ax=ax)
    ax.set_title('Age Group per Parch')
    st.pyplot(fig)
    plt.clf()

    # Proportie Plot of Survival per Embarked
    st.header("Survival Proportion per Embarked Location")
    df_embarked = filtered_df.groupby(['Embarked', 'Survived']).size().unstack()
    embarked_normalized = df_embarked.div(df_embarked.sum(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6, 4))
    embarked_normalized.plot(kind='bar', stacked=True, ax=ax, color=['coral', 'lightgreen'])
    ax.set_ylabel('Proportion of Passengers')
    ax.set_title('Survival Proportion per Embarked Location')
    st.pyplot(fig)
    plt.clf()

    # Mean Fare per Embarked Location
    st.header("Mean Fare per Embarked Location")
    fig, ax = plt.subplots(figsize=(6, 4))
    filtered_df.groupby('Embarked')['Fare'].mean().plot(kind='bar', color='lightblue', ax=ax)
    ax.set_ylabel('Mean Fare')
    ax.set_title('Mean Fare per Embarked Location')
    st.pyplot(fig)
    plt.clf()

    # Mean Age per Embarked Location
    st.header("Mean Age per Embarked Location")
    fig, ax = plt.subplots(figsize=(6, 4))
    filtered_df.groupby('Embarked')['Age'].mean().plot(kind='bar', color='skyblue', ax=ax)
    ax.set_ylabel('Mean Age')
    ax.set_title('Mean Age per Embarked Location')
    st.pyplot(fig)
    plt.clf()

with tab2:
    # Scatter Plot voor Fare Against Survival
    st.title("Scatter Plot of Fare Against Survival")
    fig, ax = plt.subplots(figsize=(5, 2.5))

    # Dropdown for survival status filtering
    survival_filter_options = ["All", "Survived", "Not Survived"]
    survival_filter = st.selectbox("Selecteer overlevingsstatus om te filteren:", survival_filter_options)

    if survival_filter == "All":
        ax.scatter(train_df[train_df['Survived'] == 1]['Fare'], train_df[train_df['Survived'] == 1]['Age'], label='Survived', alpha=0.5, color='blue')
        ax.scatter(train_df[train_df['Survived'] == 0]['Fare'], train_df[train_df['Survived'] == 0]['Age'], label='Not Survived', alpha=0.5, color='red')
    elif survival_filter == "Survived":
        ax.scatter(train_df[train_df['Survived'] == 1]['Fare'], train_df[train_df['Survived'] == 1]['Age'], label='Survived', alpha=0.5, color='blue')
    else:  # Not Survived
        ax.scatter(train_df[train_df['Survived'] == 0]['Fare'], train_df[train_df['Survived'] == 0]['Age'], label='Not Survived', alpha=0.5, color='red')

    ax.set_xlabel('Fare')
    ax.set_ylabel('Age')
    ax.set_title('Fare Against Survival')
    ax.legend()
    st.pyplot(fig)
    plt.clf()

    # leeftijdsdisributie(via chatgpt)
    def plot_age_survival_distribution(df):
        plot = sns.displot(df, x='Age', col='Survived', kde=True, binwidth=5)
        return plot

    st.title("Age Distribution by Survival Status")
    st.write("This app shows the age distribution for passengers who survived or didn't survive.")

    # Display the Age Distribution by Survival Status plot
    age_survival_plot = plot_age_survival_distribution(train_df)
    st.pyplot(age_survival_plot)

    # Voeg "All" toe aan de opties voor leeftijdsgroepen
    age_group_options = ["All"] + list(filtered_df['Age_group'].unique())
    age_group = st.selectbox("Selecteer een leeftijdsgroep om de overlevingskans te bekijken:", age_group_options)

    # Filter de data op basis van de selectie in het dropdownmenu
    if age_group == "All":
        data_to_plot = filtered_df  
    else:
        data_to_plot = filtered_df[filtered_df['Age_group'] == age_group]  # Filter op de geselecteerde leeftijdsgroep

    # Controleer of de gefilterde data leeg is
    if data_to_plot.empty:
        st.write(f"Geen gegevens beschikbaar voor de geselecteerde leeftijdsgroep: {age_group}")
    else:
        # Plot met dropdown-filter op Age_group voor overlevingskans per SibSp
        st.title(f"Survival per SibSp voor Age Group: {age_group}")
        fig, ax = plt.subplots(figsize=(5, 2.5))
        sns.countplot(x='SibSp', hue='Survived', data=data_to_plot, ax=ax)
        ax.set_title(f'Survival per Number of Siblings/Spouses Aboard (SibSp) - Age Group: {age_group}')
        ax.set_xlabel('Number of Siblings/Spouses (SibSp)')
        ax.set_ylabel('Count')
        sns.move_legend(ax, 'upper left', bbox_to_anchor= (1,1))
        st.pyplot(fig)

        # Plot met count van Survival per Parch
        st.header(f"Survival Count per Parch voor Age Group: {age_group}")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.countplot(x='Parch', hue='Survived', data=data_to_plot, ax=ax, palette=['coral', 'lightgreen'])
        ax.set_ylabel('Count of Passengers')
        sns.move_legend(ax, 'upper left', bbox_to_anchor= (1,1))
        ax.set_title(f'Survival Count per Parch - Age Group: {age_group}')
        st.pyplot(fig)
        plt.clf()




        # Calculate statistics by embarkation location(chatgpt)
        embark_stats = train_df.groupby('Embarked').agg(
            mean_age=('Age', 'mean'),
            mean_fare=('Fare', 'mean'),
            survived_count=('Survived', 'sum'),
            died_count=('Survived', lambda x: (x == 0).sum())
        ).reset_index()

        # Create a dictionary to hold the statistics for each embarkation location
        embark_dict = {
            'C': embark_stats[embark_stats['Embarked'] == 'C'].iloc[0],
            'Q': embark_stats[embark_stats['Embarked'] == 'Q'].iloc[0],
            'S': embark_stats[embark_stats['Embarked'] == 'S'].iloc[0],
        }

        # Coördinaten van belangrijke locaties langs de route van de Titanic
        locations = {
            "Southampton, UK (Vertrekpunt)": [50.8998, -1.4044],
            "Cherbourg, France (Eerste stop)": [49.6333, -1.6167],
            "Queenstown, Ireland (Laatste stop)": [51.8414, -8.2942],
            "Titanic-wrak (Zinklocatie)": [41.726931, -49.948253],
            "New York, USA (Eindbestemming)": [40.7128, -74.0060]
        }

        # Maak een Folium-kaart gecentreerd op de Atlantische Oceaan
        m = folium.Map(location=[45.0, -30.0], zoom_start=3)

        # Voeg markers toe voor elke locatie, inclusief de statistieken in de popup
        for place, coords in locations.items():
            # Determine the embarkation point for the marker
            if "Southampton" in place:
                embark = 'S'
            elif "Cherbourg" in place:
                embark = 'C'
            elif "Queenstown" in place:
                embark = 'Q'
            else:
                embark = None

            if embark:
                stats = embark_dict[embark]
                popup_text = (f"<strong>{place}</strong><br>"
                            f"Mean Age: {stats['mean_age']:.2f}<br>"
                            f"Mean Fare: ${stats['mean_fare']:.2f}<br>"
                            f"Survived: {stats['survived_count']}<br>"
                            f"Died: {stats['died_count']}")
            else:
                popup_text = f"<strong>{place}</strong>"

            folium.Marker(
                location=coords,
                popup=popup_text,
                icon=folium.Icon(color="blue" if "Titanic" not in place else "red", icon="info-sign")
            ).add_to(m)

        # Teken een lijn die de route van de Titanic aangeeft inclusief New York als eindbestemming
        route = list(locations.values())  # De coördinaten van elke locatie in volgorde
        folium.PolyLine(route, color="blue", weight=2.5, opacity=0.8).add_to(m)

        # Streamlit title
        st.title("Titanic Route Map with Embarkation Statistics")

        # Display the map in the Streamlit app
        st_folium(m, width=725)

        # Show the statistics in a table
        st.subheader("Statistics by Embarkation Location")
        st.write(embark_stats)



with tab3:




    # Define the Random Forest code
    random_forest_code = '''
    from sklearn.ensemble import RandomForestClassifier
    import pandas as pd

    # Load the datasets
    train_data = pd.read_csv('train.csv')  # Example loading, change accordingly
    test_data = pd.read_csv('test.csv')     # Example loading, change accordingly

    # Define the target variable (y) and the features (X) from the training set
    y = train_data["Survived"]
    features = ['Sex', 'Fare_group', 'Age_group', 'Embarked', 'Pclass', 'SibSp', 'Parch']

    # Convert categorical variables to numeric
    X_train = pd.get_dummies(train_data[features], drop_first=True)
    X_test = pd.get_dummies(test_data[features], drop_first=True)

    # Build and train the model
    model = RandomForestClassifier(n_estimators=10, max_depth=3, random_state=1)
    model.fit(X_train, y)

    # Make predictions based on the test set
    predictions = model.predict(X_test)

    # Save the predictions
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
    output.to_csv('submission.csv', index=False)
    '''

    # Define the Logistic Regression code
    logistic_regression_code = '''
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.linear_model import LogisticRegression

    # Load and process the training dataset
    train = pd.read_csv('train.csv')
    train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    train['Age'].fillna(train['Age'].median(), inplace=True)
    train['Embarked'].fillna(train['Embarked'].mode()[0], inplace=True)
    train['Fare'].fillna(train['Fare'].median(), inplace=True)

    # Encode categorical variables and split data
    for col in ['Sex', 'Embarked']:
        train[col] = LabelEncoder().fit_transform(train[col])
    X_train, y_train = train.drop('Survived', axis=1), train['Survived']

    # Normalize and train model
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    model = LogisticRegression().fit(X_train, y_train)

    # Load and process the test dataset
    test = pd.read_csv('test.csv')
    test.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
    test['Age'].fillna(test['Age'].median(), inplace=True)
    test['Embarked'].fillna(test['Embarked'].mode()[0], inplace=True)
    test['Fare'].fillna(test['Fare'].median(), inplace=True)

    # Encode, normalize and predict on test data
    for col in ['Sex', 'Embarked']:
        test[col] = LabelEncoder().fit_transform(test[col])
    X_test = scaler.transform(test.drop('Survived', axis=1, errors='ignore'))
    test['Survived'] = model.predict(X_test)

    # Save predictions
    output = test[['PassengerId', 'Survived']]
    output.to_excel('m2.xlsx', index=False)
    output.to_csv('m2.csv', index=False)
    print("Output saved to m2.xlsx and m2.csv")
    '''

    # Define the second Logistic Regression code for a different approach
    logistic_regression_code_2 = '''
    import pandas as pd

    # Load data
    train_data = pd.read_csv('train.csv')
    test_data = pd.read_csv('test.csv')

    # Predict survival based on gender
    test_data['Survived'] = test_data['Sex'].apply(lambda x: 1 if x == 'female' else 0)

    # Adjust survival for children under 10
    test_data.loc[test_data['Age'] < 10, 'Survived'] = 1

    # Set survival to 0 for passengers in higher classes
    test_data.loc[test_data['Pclass'] > 2, 'Survived'] = 0

    # Adjust for women in class 3
    test_data.loc[(test_data['Sex'] == 'female') & (test_data['Pclass'] == 3), 'Survived'] = 0

    # Set survival to 0 for passengers with many siblings/spouses
    test_data.loc[test_data['SibSp'] >= 4, 'Survived'] = 0

    # Set survival to 0 for passengers with many parents/children
    test_data.loc[test_data['Parch'] >= 4, 'Survived'] = 0

    # Create output DataFrame
    output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': test_data['Survived']})

    # Save output to Excel and CSV
    output.to_excel('m_3.xlsx', index=False)
    output.to_csv('m.csv', index=False)

    # Check predictions
    print(output.head())
    '''

    # Streamlit application with Tabs
    st.title("Titanic Survival Prediction Models")

    # Create Tabs
    tab4, tab5, tab6 = st.tabs(["Random Forest Model", "Logistic Regression Model", "Adjusted Logistic Regression Model"])

    # Tab 1: Random Forest Model
    with tab4:
        st.subheader("Random Forest Model Code")
        st.write("The Random Forest model is an ensemble learning method that operates by constructing multiple decision trees during training time and outputting the class that is the mode of the classes (classification) of the individual trees.")
        st.code(random_forest_code, language='python')
        st.image("m_1.png", caption="Random Forest Predictions", use_column_width=True)

    # Tab 2: Standard Logistic Regression Model
    with tab5:
        st.subheader("Standard Logistic Regression Model Code")
        st.write("Logistic Regression is a statistical model that in its basic form uses a logistic function to model a binary dependent variable. In this context, it predicts whether a passenger survived based on various features.")
        st.code(logistic_regression_code, language='python')
        st.image("m_2.png", caption="Logistic Regression Predictions", use_column_width=True)

    # Tab 3: Adjusted Logistic Regression Model
    with tab6:
        st.subheader("Adjusted Logistic Regression Model Code")
        st.write("This approach applies heuristic rules based on passenger attributes such as gender and class to make predictions. It's a simple yet effective model.")
        st.code(logistic_regression_code_2, language='python')
        st.image("m_3.png", caption="Adjusted Predictions", use_column_width=True)

    # Optional: You can add more features or functions to enhance the application.

