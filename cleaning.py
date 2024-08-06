import pandas as pd

def load_answer_key():
    #answer key cleaning/setup
    answer_key = pd.read_csv("raw_data/answerKey.csv")
    answer_key = answer_key[answer_key["pageName"] != "Background Questions"]
    return answer_key

def event_logs_cleaning(participant):
    event_logs_df = pd.read_csv(f'raw_data/eventLogs-{participant}.csv')

    event_logs_df.rename(columns={'t': 'time'}, inplace=True)
    event_logs_df.rename(columns={'e': 'event'}, inplace=True)

    # Convert the 'time' columns to datetime format
    event_logs_df['time'] = pd.to_datetime(event_logs_df['time'], unit='ms')

    #sort by time
    event_logs_df = event_logs_df.sort_values(by='time')
    return event_logs_df

def tally_events(answers_df, event_logs_df):
    # Merge each event log with the corresponding answer based on time interval
    merged_df = pd.merge_asof(event_logs_df, answers_df, on='time', direction='backward')
   
    event_count_dict = {}

    # Define event types
    event_types = ['n', 'c', 'h', 'm', 'p']

    # Iterate through each answer to count events between time intervals
    for _, row in answers_df.iterrows():
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
    return event_count_dict

def sus_score(df):
    def add_scores(df):
        even, odd = 0, 0
        for idx, row in df.iterrows():
            if (idx + 1) % 2 == 0:
                even += 5 - int(row['answer'])
            else:
                odd += int(row['answer']) - 1
        return even, odd
    static_sus_df = df.loc[df['pageName'] == "Static SUS"]
    even_s, odd_s = add_scores(static_sus_df)
    print(even_s, odd_s, "static")
    inter_sus_df = df.loc[df['pageName'] == "Interactive SUS"]
    even_i, odd_i = add_scores(inter_sus_df)
    print(even_i, odd_i, "interactive")

    def calculate_sus(even, odd):
        return 2.5 * odd + even
    return calculate_sus(even_s, odd_s), calculate_sus(even_i, odd_i)

def clean_single_df(participant):
    answer_key = load_answer_key()

    answers_df = pd.read_csv(f'raw_data/answers-{participant}.csv')
    # Convert the 'time' columns to datetime format
    answers_df['time'] = pd.to_datetime(answers_df['time'], unit='ms')

    # Sort by the 'time' column
    answers_df = answers_df.sort_values(by='time')
    event_logs_df = event_logs_cleaning(participant)

    # Merge the DataFrames based on the 'time' column
    merged_df = pd.concat([answers_df, event_logs_df]).sort_values(by='time').reset_index(drop=True)

    # Add a new column to answers_df to indicate the next answer time
    answers_df['next_time'] = answers_df['time'].shift(-1)

    # Tally the events for each answer
    event_count_dict = tally_events(answers_df, event_logs_df)

    # Convert the event count dictionary into a DataFrame
    event_counts_df = pd.DataFrame.from_dict(event_count_dict, orient='index').fillna(0)

    # Join the event counts with the answers_df
    final_df = pd.concat([answers_df.set_index('time'), event_counts_df], axis=1).reset_index()

    # Drop the next_time column as it's no longer needed
    final_df = final_df.drop(columns=['next_time'])

    # rename version to condition column
    final_df = final_df.rename(columns={"version" : "condition"})

    # Rename columns
    final_df.rename(columns={'n': 'next'}, inplace=True)
    final_df.rename(columns={'c': 'click'}, inplace=True)
    final_df.rename(columns={'m': 'mouse'}, inplace=True)
    final_df.rename(columns={'h': 'hover'}, inplace=True)
    final_df.rename(columns={'p': 'pointer'}, inplace=True)

    # Score the test
    scores, answers = [], []
    valid_ids = set(["qID-11", "qID-12", "qID-13"])
    pretest, p_answer = 0, 0
    p_age = final_df[(final_df['pageName']=="Background Questions") & (final_df['question'] == "0")]['answer'].values[0]
    year_taken = final_df[(final_df['pageName']=="Background Questions") & (final_df['question'] == "1")]['answer'].values[0]
    grade = final_df[(final_df['pageName']=="Background Questions") & (final_df['question'] == "2")]['answer'].values[0]
    track = final_df[(final_df['pageName']=="Background Questions") & (final_df['question'] == "3")]['answer']
    track = track.values[0] if len(track) > 0 else "N/A"
    for _, row in final_df.iterrows():
        question, proof = row['question'], row['pageName']
        a = row.loc['answer']
        # page is not included in the answer key
        if not proof in set(answer_key['pageName']):
            scores.append(0)
            answers.append(None)
            continue
        
        #special case for pretest where some questions are inserted at the 1st question about triangle congruence
        if question in valid_ids:
            ans_row = answer_key.loc[(answer_key.question==question)]
        else:
            # find the proof and the question being scored in answer key
            ans_row = answer_key.loc[(answer_key.question==question) & (answer_key.pageName==proof)]
        # this question/proof combination is not in the answer key
        if len(ans_row) == 0:
            scores.append(0)
            answers.append(None)
            continue

        # add score to list
        correct = ans_row['answer'].values[0] == a

        #tally score for pretest and tutorial separately
        if proof.startswith("P") or proof.startswith("T"):
            pretest += int(1 if correct else 0)
            p_answer += 1
        scores.append(int(1 if correct else 0))
        answers.append(ans_row['answer'].values[0])

    # add columns to answer dataframe
    final_df["score"] = pd.Series(scores).values
    final_df["key"] = pd.Series(answers).values

    # add time elapsed column
    final_df['delta'] = (final_df['index']-final_df['index'].shift()).dt.total_seconds().fillna(0)

    #drop rows from background questions, add their answers as column instead
    final_df["age"] = pd.Series([p_age for i in range(len(final_df))]).values
    final_df["year_taken"] = pd.Series([year_taken for i in range(len(final_df))]).values
    final_df["grade"] = pd.Series([grade for i in range(len(final_df))]).values
    final_df["track"] = pd.Series([track for i in range(len(final_df))]).values
    final_df["pretest"] = pd.Series([pretest/p_answer for i in range(len(final_df))]).values

    #replace "static" with 0 and "interactive" with 1
    final_df["condition"] = final_df["condition"].replace("static", 0).replace("interactive", 1)

    # TODO calculate SUS score for each condition, store in separate df
    static_sus, inter_sus = sus_score(final_df)
    final_df["static_sus"] = pd.Series([static_sus for i in range(len(final_df))]).values
    final_df["inter_sus"] = pd.Series([inter_sus for i in range(len(final_df))]).values

    # add participant id column
    final_df["participant"] = pd.Series([participant for i in range(len(final_df))]).values

    #drop rows with questions that are not part of procedure
    # SUS, pretest, tutorial, background
    final_df.drop(final_df.loc[final_df['pageName']=="Static SUS"].index, inplace=True)
    final_df.drop(final_df.loc[final_df['pageName']=="Interactive SUS"].index, inplace=True)
    final_df.drop(final_df.loc[final_df['pageName']=="Background Questions"].index, inplace=True)
    final_df.drop(final_df.loc[final_df['pageName'].isin([f"P{i+1}" for i in range(7)])].index, inplace=True)
    final_df.drop(final_df.loc[final_df['pageName'].isin(["TutorialProof1", "TutorialProof2"])].index, inplace=True)

    return final_df

def r_df(participants):
    for i in range(len(participants)):
        if i == 0:
            df = clean_single_df(participants[i])
        else:
            df = pd.concat([df, clean_single_df(participants[i-1])])
        print("participant df: ", len(df.index))
    save_df(df, "concat_df")
    print("final df: ", len(df.index))

def save_df(df, participant):
    #save the dataframe
    df.to_csv(f"out/{participant}.csv", index=False)

if __name__ == "__main__":
    participants = ["pa", "pb", "pc", "pilotB", "pilotD", "corey"]
    r_df(participants)


