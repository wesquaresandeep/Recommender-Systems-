import csv
from recommender2 import Recommender

# Absolute paths to your CSV files
items_path = r'C:\Users\98861\Downloads\assignment\items.csv'
users_path = r'C:\Users\98861\Downloads\assignment\users.csv'
events_path = r'C:\Users\98861\Downloads\assignment\events.csv'
sessions_path = r'C:\Users\98861\Downloads\assignment\sessions.csv'

# Initialize Recommender
r = Recommender(items_path, users_path, events_path)

# Analyze the data
r.analyze()

# Train the recommender
r.train()

try:
    with open(sessions_path, 'r') as f:
        reader = csv.DictReader(f)
        hits = 0
        total = 0

        for row in reader:
            session_items_str = row['session_items'].strip()
            session_items = [int(item.strip()) for item in session_items_str.split(',') if item.strip().isdigit()]

            if len(session_items) < 2:  # Ensure at least 2 items (input and target)
                print(f"Skipping row due to insufficient session items: {session_items}")
                continue

            # Split session items into input items and target item
            input_item_ids = session_items[:-1]  # All items except the last one
            target_item = session_items[-1]  # The last item is the actual interaction (target)

            print(f"Target item for session {row['session_id']} of user {row['user_id']}: {target_item}")

            # Get recommendations based on the input session items
            recommended = r.recommend(input_item_ids)  # No longer passing `actual_interactions`
            print(f"Recommended items for session {row['session_id']} of user {row['user_id']}: {recommended}")

            # Check if the target item is in the recommended list
            if target_item in recommended:
                hits += 1
            else:
                print(f"Missed target item: {target_item}, Recommended: {recommended}")
            total += 1

        # Print final hit rate
        if total > 0:
            hit_rate = hits / total
            print(f"Hits: {hits}/{total}, Hit Rate: {hit_rate:.2f}")
        else:
            print("No valid sessions processed.")

except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the sessions.csv file exists.")
except Exception as e:
    print(f"An error occurred: {e}")
