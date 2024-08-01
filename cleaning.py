import pandas as pd

def pickle_df(df, filename):
    df.to_pickle(f'../out/{filename}.pkl')
    print(f'Successfully saved the DataFrame to ../out/{filename}.pkl')

def clean(participant_id):
    answer_key = pd.read_csv("../raw_data/answers-answerkey.csv")
    answers_df = pd.read_csv(f'../raw_data/answers-{participant_id}.csv')
    event_logs_df = pd.read_csv(f'../raw_data/eventLogs-{participant_id}.csv')
    event_logs_df.rename(columns={'t': 'time'}, inplace=True)
    event_logs_df.rename(columns={'e': 'event'}, inplace=True)

    # Convert the 'time' columns to datetime format
    answers_df['time'] = pd.to_datetime(answers_df['time'], unit='ms')
    event_logs_df['time'] = pd.to_datetime(event_logs_df['time'], unit='ms')

    # Sort both DataFrames by the 'time' column
    answers_df = answers_df.sort_values(by='time')
    event_logs_df = event_logs_df.sort_values(by='time')

    # Merge the DataFrames based on the 'time' column
    merged_df = pd.concat([answers_df, event_logs_df]).sort_values(by='time').reset_index(drop=True)

    # Display the first few rows of the merged DataFrame
    merged_df.head(20)

    # Add a new column to answers_df to indicate the next answer time
    answers_df['next_time'] = answers_df['time'].shift(-1)

    # Merge each event log with the corresponding answer based on time interval
    merged_df = pd.merge_asof(event_logs_df, answers_df, on='time', direction='backward')

    event_count_dict = {}

    # Define event types
    event_types = ['n', 'c', 'h', 'm', 'p']

    # Iterate through each answer to count events between time intervals
    for index, row in answers_df.iterrows():
        answer_time = row['time']
        next_time = row['next_time'] if not pd.isnull(row['next_time']) else pd.Timestamp.max
    
        # Select events that fall within the time interval
        filtered_events = merged_df[(merged_df['time'] >= answer_time) & (merged_df['time'] < next_time)]
    
        # Filter out "hover" events (h) that occur within 0.01 second of another event [debounce]
        filtered_events = filtered_events[~((filtered_events['event'] == 'h') &
                                        (filtered_events['time'].diff().dt.total_seconds() <= 0.01))]
        
        # Count occurrences of each event type
        event_counts = filtered_events['event'].value_counts()
        
        # Initialize counts for all event types
        counts = {event: event_counts.get(event, 0) for event in event_types}
        
        # Store the counts in the dictionary
        event_count_dict[answer_time] = counts

    # Convert the event count dictionary into a DataFrame
    event_counts_df = pd.DataFrame.from_dict(event_count_dict, orient='index').fillna(0)

    # Join the event counts with the answers_df
    final_df = pd.concat([answers_df.set_index('time'), event_counts_df], axis=1).reset_index()

    # Drop the next_time column as it's no longer needed
    final_df = final_df.drop(columns=['next_time'])

    # Rename columns
    final_df.rename(columns={'n': 'next'}, inplace=True)
    final_df.rename(columns={'c': 'click'}, inplace=True)
    final_df.rename(columns={'m': 'mouse'}, inplace=True)
    final_df.rename(columns={'h': 'hover'}, inplace=True)
    final_df.rename(columns={'p': 'pointer'}, inplace=True)


    # Display the first few rows of the final DataFrame
    final_df.head(30)


if __name__ == "__main__":
    clean("pa")
