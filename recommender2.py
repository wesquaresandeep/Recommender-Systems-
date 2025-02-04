import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

class Recommender:
    def __init__(self, items_path, users_path, events_path):
        # Read CSV files and store in memory as DataFrames
        self.items_data = pd.read_csv(items_path)  # Load items CSV
        self.users_data = pd.read_csv(users_path)  # Load users CSV
        self.events_data = pd.read_csv(events_path)  # Load events CSV
        self.user_item_matrix = None  # Initialize user-item matrix
        self.category_co_occurrence = None
        self.hair_co_occurrence = None
        self.eye_co_occurrence = None
        self.combined_co_occurrence = None
        self.item_similarity_df = None

    def analyze(self):
        # Ensure 'timestamp' is in datetime format
        self.events_data['timestamp'] = pd.to_datetime(self.events_data['timestamp'], unit='s')
        self.events_data = self.events_data.sort_values(by=['user_id', 'timestamp'])

        # Define sessions based on 8-hour gaps
        self.events_data['time_diff'] = self.events_data.groupby('user_id')['timestamp'].diff()
        self.events_data['new_session'] = (self.events_data['time_diff'] >= pd.Timedelta(hours=8)).astype(int)
        self.events_data['session_id'] = self.events_data.groupby('user_id')['new_session'].cumsum()

        # Remove duplicate item visits within each session
        self.events_data = self.events_data.drop_duplicates(subset=['user_id', 'session_id', 'item_id'], keep='first')

        # Exclude sessions with only one event
        session_event_count = self.events_data.groupby(['user_id', 'session_id']).size()
        valid_sessions = session_event_count[session_event_count > 1].index
        self.events_data = self.events_data.set_index(['user_id', 'session_id']).loc[valid_sessions].reset_index()

        # Call analysis functions
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
        return self.events_data['session_id'].nunique()

    def get_avg_events_per_session(self):
        session_event_counts = self.events_data.groupby(['user_id', 'session_id']).size()
        return session_event_counts.mean()

    def plot_session_length_histogram(self):
        session_event_counts = self.events_data.groupby(['user_id', 'session_id']).size()
        plt.figure(figsize=(10, 6))
        plt.hist(session_event_counts, bins=10, edgecolor='black')
        plt.title('Histogram of Session Lengths')
        plt.xlabel('Number of Events per Session')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()

    def get_highest_bounce_rate_category(self):
        last_events = self.events_data.groupby(['user_id', 'session_id']).last().reset_index()
        category_counts = last_events['item_id'].map(self.items_data.set_index('item_id')['category']).value_counts()
        highest_bounce_category = category_counts.idxmax()
        highest_bounce_count = category_counts.max()
        return highest_bounce_category, highest_bounce_count

    def get_country_with_highest_avg_visits(self):
        merged_data = self.events_data.merge(self.users_data, on='user_id', how='left')
        user_visit_counts = merged_data.groupby(['user_id', 'country']).size().reset_index(name='visit_count')
        average_visits_per_country = user_visit_counts.groupby('country')['visit_count'].mean().reset_index()
        highest_avg_country = average_visits_per_country.loc[average_visits_per_country['visit_count'].idxmax()]
        return highest_avg_country['country'], highest_avg_country['visit_count']

    def get_all_performers_sorted(self):
        merged_data = self.events_data.merge(self.items_data, on='item_id', how='left')
        visit_counts = merged_data.groupby(['category', 'item_id']).size().reset_index(name='visit_count')
        sorted_performers = visit_counts.sort_values(by=['category', 'visit_count'], ascending=[True, True])
        return sorted_performers[['category', 'item_id', 'visit_count']]

    def train(self):
        # Create co-occurrence matrices for categories, hair, and eyes
        events_with_items = self.events_data.merge(self.items_data, on='item_id', how='left')

        # If there is no 'interaction' column, create it
        if 'interaction' not in events_with_items.columns:
            events_with_items['interaction'] = 1  # Assuming each event is a single interaction

        # Check for required columns before creating co-occurrence matrices
        for col in ['category', 'hair', 'eyes']:
            if col not in events_with_items.columns:
                print(f"Warning: '{col}' column is missing from the dataset.")
                events_with_items[col] = np.nan  # Fill with NaN if missing

        # Co-occurrence for categories
        self.category_co_occurrence = pd.crosstab(events_with_items['category'], events_with_items['category'])

        # Co-occurrence for hair
        self.hair_co_occurrence = pd.crosstab(events_with_items['hair'], events_with_items['hair'])

        # Co-occurrence for eyes
        self.eye_co_occurrence = pd.crosstab(events_with_items['eyes'], events_with_items['eyes'])

        # Create a combined co-occurrence matrix
        self.combined_co_occurrence = self.category_co_occurrence.add(self.hair_co_occurrence, fill_value=0)
        self.combined_co_occurrence = self.combined_co_occurrence.add(self.eye_co_occurrence, fill_value=0)

        # Create user-item interaction matrix for collaborative filtering
        self.user_item_matrix = events_with_items.pivot_table(index='user_id', columns='item_id', values='interaction',
                                                              fill_value=0)

        # Compute item similarity matrix using collaborative filtering
        self.item_similarity_matrix = cosine_similarity(self.user_item_matrix.T)
        self.item_similarity_df = pd.DataFrame(self.item_similarity_matrix, index=self.user_item_matrix.columns,
                                               columns=self.user_item_matrix.columns)

        # Optionally, print shapes of matrices for debugging
        print("User-Item Matrix Shape:", self.user_item_matrix.shape)
        print("Item Similarity Matrix Shape:", self.item_similarity_df.shape)
        print("Combined Co-Occurrence Matrix Shape:", self.combined_co_occurrence.shape)

    def recommend(self, session_item_ids, top_n=10):
        """
        Given a list of itemIds visited by a user within a session,
        recommend a list of items with the goal of maximizing probability of interaction.

        :param session_item_ids: List of item IDs visited during the session
        :param top_n: Number of recommendations to return
        :return: List of recommended item IDs
        """

        # Ensure input items from the session are valid and exist in the dataset
        valid_session_item_ids = [item_id for item_id in session_item_ids if
                                  item_id in self.items_data['item_id'].values]

        if not valid_session_item_ids:
            print("No valid session items found for recommendation.")
            return []

        print(f"Valid Session Item IDs for Recommendation: {valid_session_item_ids}")

        # Get content-based recommendations
        content_recommendations = self.get_content_based_recommendations(valid_session_item_ids, top_n)

        # Get collaborative filtering recommendations
        collaborative_recommendations = self.get_collaborative_filtering_recommendations(valid_session_item_ids, top_n)

        # Combine both recommendation sets (content-based and collaborative filtering) while ensuring uniqueness
        combined_recommendations = list(dict.fromkeys(content_recommendations + collaborative_recommendations))

        # Exclude already visited items from recommendations
        combined_recommendations = [item for item in combined_recommendations if item not in valid_session_item_ids]

        # Limit to top_n recommendations
        recommended_items = combined_recommendations[:top_n]

        return recommended_items
    def get_content_based_recommendations(self, input_item_ids, top_n):
        recommendations = {}
        for item_id in input_item_ids:
            if item_id in self.combined_co_occurrence.index:
                similar_items = self.combined_co_occurrence[item_id].nlargest(top_n).index
                for sim_item in similar_items:
                    if sim_item not in recommendations:
                        recommendations[sim_item] = self.combined_co_occurrence[item_id][sim_item]

        return list(recommendations.keys())

    def get_collaborative_filtering_recommendations(self, input_item_ids, top_n):
        recommendations = {}
        for item_id in input_item_ids:
            if item_id in self.item_similarity_df.index:
                similar_items = self.item_similarity_df[item_id].nlargest(top_n).index
                for sim_item in similar_items:
                    if sim_item not in recommendations:
                        recommendations[sim_item] = self.item_similarity_df[item_id][sim_item]

        return list(recommendations.keys())

    def create_sessions_file(self, output_path="sessions.csv"):
        sessions = self.events_data.groupby(['user_id', 'session_id']).agg(
            item_ids=('item_id', lambda x: ','.join(map(str, x))),
            target_item=('item_id', 'last')
        ).reset_index()

        sessions['output'] = sessions['item_ids'] + '\t' + sessions['target_item'].astype(str)

        with open(output_path, 'w') as f:
            for line in sessions['output']:
                f.write(line + '\n')

# Provide the correct path to your files
items_path = r'C:\Users\98861\Downloads\assignment\items.csv'
users_path = r'C:\Users\98861\Downloads\assignment\users.csv'
events_path = r'C:\Users\98861\Downloads\assignment\events.csv'

if __name__ == "__main__":
    # Correct the instantiation to use the defined file path variables
    recommender = Recommender(items_path, users_path, events_path)
    recommender.analyze()  # Optional: Analyze session data
    recommender.train()  # Train the recommender
    recommendations = recommender.recommend([1,2,3,4], top_n=5)  # Example recommendation call
    print(recommendations)
