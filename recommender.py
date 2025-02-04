import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class Recommender:
    def __init__(self, items_path, users_path, events_path):
        # Read CSV files and store in memory as DataFrames
        self.items_data = pd.read_csv(items_path)  # Load items CSV
        self.users_data = pd.read_csv(users_path)  # Load users CSV
        self.events_data = pd.read_csv(events_path)  # Load events CSV

        # Remove duplicate rows from items.csv based on 'item_id', 'category', 'hair', 'eyes' columns
        self.items_data.drop_duplicates(subset=['item_id', 'category', 'hair', 'eyes'], keep='first', inplace=True)

    def analyze(self):
        # Ensure 'timestamp' is in datetime format
        self.events_data['timestamp'] = pd.to_datetime(self.events_data['timestamp'], unit='s')  # Convert Unix time to datetime
        self.events_data = self.events_data.sort_values(by=['user_id', 'timestamp'])

        # Step 1: Define sessions based on 8-hour gaps
        self.events_data['time_diff'] = self.events_data.groupby('user_id')['timestamp'].diff()  # Calculate time differences
        self.events_data['new_session'] = (self.events_data['time_diff'] >= pd.Timedelta(hours=8)).astype(int)  # Mark where new sessions start
        self.events_data['session_id'] = self.events_data.groupby('user_id')['new_session'].cumsum()  # Assign session_id

        # Step 2: Remove duplicate item visits within each session
        self.events_data = self.events_data.drop_duplicates(subset=['user_id', 'session_id', 'item_id'], keep='first')

        # Step 3: Exclude sessions with only one event
        session_event_count = self.events_data.groupby(['user_id', 'session_id']).size()
        valid_sessions = session_event_count[session_event_count > 1].index  # Get sessions with more than one event
        self.events_data = self.events_data.set_index(['user_id', 'session_id']).loc[valid_sessions].reset_index()

        # Call the additional analysis functions and print their results
        num_sessions = self.get_number_of_sessions()
        avg_events = self.get_avg_events_per_session()
        highest_bounce_category, highest_bounce_count = self.get_highest_bounce_rate_category()
        country, avg_visits = self.get_country_with_highest_avg_visits()
        sorted_performers = self.get_all_performers_sorted()

        print(f"Number of sessions: {num_sessions}")
        print(f"Average events per session: {avg_events:.2f}")
        print(f"Highest bounce rate category: {highest_bounce_category} with count {highest_bounce_count}")
        print(f"Country with highest average visits: {country} with {avg_visits:.2f} visits")
        print("Sorted performers (category, item_id, visit_count):")
        print(sorted_performers)

        # Optional: Plot histogram of session lengths
        self.plot_session_length_histogram()

    def get_number_of_sessions(self):
        # Count the number of unique session IDs
        num_sessions = self.events_data['session_id'].nunique()
        return num_sessions

    def get_avg_events_per_session(self):
        # Group by 'user_id' and 'session_id' and calculate the number of events per session
        session_event_counts = self.events_data.groupby(['user_id', 'session_id']).size()

        # Calculate the average number of events per session
        avg_events = session_event_counts.mean()
        return avg_events

    def plot_session_length_histogram(self):
        # Group by 'user_id' and 'session_id' and calculate the number of events per session
        session_event_counts = self.events_data.groupby(['user_id', 'session_id']).size()

        # Plot histogram of session lengths
        plt.figure(figsize=(10, 6))
        plt.hist(session_event_counts, bins=10, edgecolor='black')
        plt.title('Histogram of Session Lengths')
        plt.xlabel('Number of Events per Session')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def get_highest_bounce_rate_category(self):
        # Step 1: Identify the last event of each session
        last_events = self.events_data.groupby(['user_id', 'session_id']).last().reset_index()

        # Step 2: Count occurrences of categories in last events
        category_counts = last_events['item_id'].map(self.items_data.set_index('item_id')['category']).value_counts()

        # Step 3: Get the category with the highest count
        highest_bounce_category = category_counts.idxmax()
        highest_bounce_count = category_counts.max()

        return highest_bounce_category, highest_bounce_count

    def get_country_with_highest_avg_visits(self):
        # Step 1: Merge events_data with users_data to include country information
        merged_data = self.events_data.merge(self.users_data, on='user_id', how='left')

        # Step 2: Count total visits per user per country
        user_visit_counts = merged_data.groupby(['user_id', 'country']).size().reset_index(name='visit_count')

        # Step 3: Calculate the average visit count per user for each country
        average_visits_per_country = user_visit_counts.groupby('country')['visit_count'].mean().reset_index()

        # Step 4: Identify the country with the highest average visit count
        highest_avg_country = average_visits_per_country.loc[average_visits_per_country['visit_count'].idxmax()]

        return highest_avg_country['country'], highest_avg_country['visit_count']

    def get_all_performers_sorted(self):
        # Step 1: Merge events_data with items_data to include category information
        merged_data = self.events_data.merge(self.items_data, on='item_id', how='left')

        # Step 2: Count total visits per performer per category
        visit_counts = merged_data.groupby(['category', 'item_id']).size().reset_index(name='visit_count')

        # Step 3: Sort the results by category and visit count in ascending order
        sorted_performers = visit_counts.sort_values(by=['category', 'visit_count'], ascending=[True, True])

        return sorted_performers[['category', 'item_id', 'visit_count']]

    def train(self):
        # Create co-occurrence matrices for categories, hair, and eyes
        events_with_items = self.events_data.merge(self.items_data, on='item_id', how='left')

        # Co-occurrence for categories
        self.category_co_occurrence = pd.crosstab(events_with_items['category'], events_with_items['category'])

        # Co-occurrence for hair
        self.hair_co_occurrence = pd.crosstab(events_with_items['hair'], events_with_items['hair'])

        # Co-occurrence for eyes
        self.eye_co_occurrence = pd.crosstab(events_with_items['eyes'], events_with_items['eyes'])

        # Create a combined co-occurrence matrix
        self.combined_co_occurrence = self.category_co_occurrence.add(self.hair_co_occurrence, fill_value=0)
        self.combined_co_occurrence = self.combined_co_occurrence.add(self.eye_co_occurrence, fill_value=0)

    def recommend(self, session_item_ids, top_n=5):
        """
        Given a list of item IDs visited by a user within a session, recommend a list of items to the user
        with the goal of maximizing the probability of actual interaction.

        Args:
        session_item_ids (list): List of item IDs visited by the user within a session.
        top_n (int): The number of recommendations to return.

        Returns:
        List: List of recommended item IDs.
        """
        recommendations = {}
        input_set = set(session_item_ids)  # Avoid recommending items already in the session

        # For each item in the session_item_ids list, calculate interaction probabilities
        for item_id in session_item_ids:
            if item_id in self.items_data['item_id'].values:
                item = self.items_data[self.items_data['item_id'] == item_id].iloc[0]

                # Calculate interaction probabilities based on category, hair, and eye color
                category_probabilities = self.calculate_interaction_probability(item, self.category_co_occurrence,
                                                                                'category')
                hair_probabilities = self.calculate_interaction_probability(item, self.hair_co_occurrence, 'hair')
                eye_probabilities = self.calculate_interaction_probability(item, self.eye_co_occurrence, 'eyes')

                # Combine probabilities
                combined_probabilities = category_probabilities.add(hair_probabilities, fill_value=0).add(
                    eye_probabilities, fill_value=0)

                # Update recommendations with the calculated probabilities
                for recommended_feature in combined_probabilities.index:
                    # Find items related to the recommended feature
                    recommended_items = self.items_data[(
                                                                self.items_data['category'] == recommended_feature) |
                                                        (self.items_data['hair'] == recommended_feature) |
                                                        (self.items_data['eyes'] == recommended_feature)
                                                        ]['item_id'].tolist()

                    # Update recommendations with the calculated probabilities
                    for recommended_item_id in recommended_items:
                        if recommended_item_id not in input_set:  # Avoid recommending items in session_item_ids
                            recommendations[recommended_item_id] = recommendations.get(recommended_item_id, 0) + \
                                                                   combined_probabilities[recommended_feature]

        # Sort recommendations by probability in descending order
        sorted_recommendations = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)
        top_recommended_items = [item[0] for item in sorted_recommendations[:top_n]]

        return top_recommended_items

    def calculate_interaction_probability(self, item, co_occurrence_matrix, feature_column):
        feature_value = item[feature_column]
        probabilities = co_occurrence_matrix.loc[feature_value]

        # Normalize the probabilities
        total = probabilities.sum()
        if total > 0:
            probabilities = probabilities / total

        return probabilities

    def create_sessions_file(self, output_path=r"files/sessions.csv"):
        """
        Create sessions.csv file containing sessions and target items.
        Each line will contain item IDs visited in a session followed by a target item.
        """
        # Step 1: Group events by user and session
        sessions = self.events_data.groupby(['user_id', 'session_id']).agg(
            item_ids=('item_id', lambda x: ','.join(map(str, x))),
            target_item=('item_id', 'last')  # Assume the last item is the target item
        ).reset_index()

        # Step 2: Prepare the output format
        sessions['output'] = sessions['item_ids'] + '\t' + sessions['target_item'].astype(str)

        # Step 3: Write to the sessions.csv file

        with open(output_path, 'w') as f:
            for line in sessions['output']:
                f.write(line + '\n')


# Provide the correct path to your files
items_path = r'files\items.csv'
users_path = r'files\users.csv'
events_path = r'files\events.csv'

# Initialize the Recommender class
data_handler = Recommender(items_path, users_path, events_path)

# Analyze the data
data_handler.analyze()

# Train the model to create co-occurrence matrices
data_handler.train()

# Sample input item IDs for recommendation without actual interactions
input_item_ids = [15,16,17]  # Example input; replace with actual item IDs
top_n_recommendations = 5   # Number of recommendations to return

recommendations = data_handler.recommend(input_item_ids)

# Output the recommendations
print(f"Given a list of item IDs visited by a user within a session, recommend a list of items to the user with the goal of maximizing probability of actual interaction. Return a list of {top_n_recommendations} item IDs.")
print(f"For input {input_item_ids}, output: {recommendations}")
