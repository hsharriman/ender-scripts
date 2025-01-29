import pandas as pd
import numpy as np

def load_answer_key():
    #answer key cleaning/setup
    answer_key = pd.read_csv("./answerKey.csv")
    return answer_key

def load_raw_answers(participant, is_pilot):
    folder = "pilot-data" if is_pilot else "study-data"
    return  pd.read_csv(f'{folder}/answers-{participant}.csv')

def event_logs_cleaning(participant):
    event_logs_df = pd.read_csv(f'raw_data/eventLogs-{participant}.csv')

    event_logs_df.rename(columns={'t': 'time'}, inplace=True)
    event_logs_df.rename(columns={'e': 'event'}, inplace=True)

    event_logs_df['time'] = pd.to_datetime(event_logs_df['time'], unit='ms')

    #sort by time
    event_logs_df = event_logs_df.sort_values(by='time')
    return event_logs_df

def update_openended(df, proofs):
    for k, v in proofs.items():
        df.loc[(df["pageName"] == k) & (df["question"] == "qID-13"), "answer"] = v
    return df

def add_participant_timing(participant, answers_df, is_pilot, overwrite=False):
    filename = "./out/study/per_question2.csv"
    df = pd.read_csv(filename)

    #convert answers of participant into appropriate format
    answers_df = answers_df.rename(columns={'participant': 'id'})
    answers_df = answers_df.rename(columns={'pageName': 'proof'})
    answers_df = answers_df.rename(columns={'delta': 'time_elapsed'})
    answers_df = answers_df.drop(columns=["key", "answer", "time"])
    answers_df = answers_df[(answers_df.proof != "Background Questions") & (answers_df.proof != "SUS") & (~answers_df.proof.str.startswith("Tutorial"))]
    answers_df["pilot"] = [1 if is_pilot else 0 for x in range(len(answers_df))]
    if overwrite:
        df = df[df["id"] != participant]
    if df[df["id"] == participant].empty:
        df = pd.concat([df, answers_df], ignore_index=True)
        df = df.drop_duplicates()

    df = df.reset_index(drop=True)
    df.to_csv(filename, index=False)
    return df

def sus_score(df):
    # separate the SUS scores into a new df
    df_sus = df[df['pageName'] == "SUS"]
    df_sus = df_sus.drop(columns=["time", "version"])
    df_sus["score"] = np.zeros(len(df_sus))
    for _, row in df_sus.iterrows():
        # question indices are 0-indexed
        q, a = int(row["question"]), int(row["answer"])
        df_sus.loc[(df_sus["question"] == row["question"]),"score"] = (a - 1) if q % 2 == 0 else (5-a)
    sus_score = df_sus["score"].sum() * 2.5
    return sus_score

def clean_single_df(participant, proofs, is_pilot=False):
    answers_df = load_raw_answers(participant, is_pilot)

    answers_df = answers_df.sort_values(by='time')
    answers_df['time'] = pd.to_datetime(answers_df['time'], unit='ms')

    #reindex
    answers_df = answers_df.reset_index(drop=True)

    #Drop unused, stale answers to questions. Anything that occurred before background questions
    background_index = answers_df[answers_df['pageName'] == 'Background Questions'].index[0]
    answers_df = answers_df.loc[answers_df.index >= background_index]

    #reindex again
    answers_df = answers_df.reset_index(drop=True)

    # Replace values in the 'type' column
    answers_df['version'] = answers_df['version'].replace({'static': 'B', 'interactive': 'A'})

    # clip T1_ from the beginning of pageName
    answers_df = answers_df.map(lambda x: x.replace('T1_', '') if isinstance(x, str) else x)

    # update scores for open-ended problems
    update_openended(answers_df, proofs)

    return answers_df

def score_test(df, participant):
    answer_key = load_answer_key()
    seen_set = set()
    # Score the test
    scores, aks, seen = [], [], []
    valid_ids = set(["qID-11", "qID-12", "qID-13"])
    correct_proof_ids = set(["S1_C1", "S1_C2", "S2_C2"])

    for _, row in df.iterrows():
        question, proof = row['question'], row['pageName']
        a = row.loc['answer']

        # page is not included in the answer key
        if not proof in set(answer_key['pageName']) or proof.startswith("Tutorial"):
            scores.append(None)
            aks.append(None)
            seen.append(len(seen_set))
            continue
        
        #special case for pretest where some questions are inserted at the 1st question about triangle congruence
        if proof.startswith("P") and question in valid_ids:
            ans_row = answer_key.loc[(answer_key.pageName=="P5") & (answer_key.question==question)]
        else:
            # find the proof and the question being scored in answer key
            ans_row = answer_key.loc[(answer_key.question==question) & (answer_key.pageName==proof)]
            
        # this question/proof combination is not in the answer key
        if len(ans_row) == 0:
            scores.append(0)
            aks.append(None)
            seen.append(len(seen_set))
            continue

        if proof in correct_proof_ids and question == "qID-11" and a == "No":
            #student correctly said there was no mistake 
            scores.append(3)
        else:
            # add score to list
            correct = ans_row['answer'].values[0] == a
            scores.append(1 if correct else 0)
        if proof not in seen_set and not proof.startswith("P"):
            seen_set.add(proof)
        seen.append(len(seen_set))
        aks.append(ans_row['answer'].values[0])

    # add columns to answer dataframe
    df["score"] = pd.Series(scores).values
    df["key"] = pd.Series(aks).values
    df["order"] = pd.Series(seen).values

    # add time elapsed column
    df['delta'] = df['time'].diff().dt.total_seconds().fillna(0)

    # add participant id column
    df["participant"] = pd.Series([participant for i in range(len(df))]).values

    #store the combined CSV
    df.to_csv(f"./study-data/processed/{participant}.csv", index=False)

    return df

def total_score_participant(df, participant, is_pilot, overwrite=False):
    # score the participant by pretest, activity, SUS
    answer_key = load_answer_key()

    # separate the SUS scores into a new df
    sus = sus_score(df)

    # separate out the background questions to new df
    df = df[(df['pageName'] != "Background Questions") & (df['pageName'] != "SUS")]

    # max score for pretest
    mask = lambda df: df[df["pageName"].str.startswith('P')]
    pretest_max_score = len(mask(answer_key))
    #score pretest
    pre = mask(df)
    pre_score = pre["score"].sum()

    # max score for activity
    mask = lambda df: df[~df['pageName'].str.startswith('P') & (~df['question'].str.startswith('qID-0')) & ~df['pageName'].str.startswith('Tutorial')]
    activity_df = mask(answer_key)
    #score activity
    act = mask(df)
    act_score = act["score"].sum()

    # when proof is correct, add 2 to score (because the "which step is wrong?" and "explain how to correct" questions are skipped)
    # There are 3 correct proofs so add 6 points to total for the answer key
    activity_max_score = len(activity_df) + 6

    print(f"pretest: {pre_score}/{pretest_max_score},\nactivity: {act_score}/{activity_max_score},\nsus: {sus}")

    # store answer in compiled CSV
    # dataframe to collect scores only, 1 new row per participant, include boolean flag if the data came from a pilot
    filename = "./out/study/scores.csv"
    score_df = pd.read_csv(filename)
    row = pd.DataFrame({
        "id": participant, 
        "version": df.loc[(df["pageName"] == "S1_C1") & (df["question"] == "qID-0"),"version"].values[0], 
        "sus": sus, 
        "pretest": float("%.2f" % round(pre_score / pretest_max_score, 2)), 
        "score": float("%.2f" % round(act_score / activity_max_score, 2)), 
        "pilot": 1 if is_pilot else 0
    }, index=[0])

    if overwrite:
        score_df = score_df[score_df["id"] != participant]
    if score_df[score_df["id"] == participant].empty:
        score_df = pd.concat([score_df, row], ignore_index=True)
        score_df = score_df.drop_duplicates()
        score_df.dropna(inplace=True)

    score_df.to_csv(filename, index=False)

    return score_df

def combine_qual(df, qual_csv_path):
    dfq = pd.read_csv(qual_csv_path)
    dfq.rename(columns={'question': 'questionText'}, inplace=True)
    dfq.rename(columns={'steps viewed?': 'steps'}, inplace=True)
    dfq.rename(columns={'reasoning.1': 'reasonCorrect'}, inplace=True)
    dfq.rename(columns={'type of understanding': 'understanding'}, inplace=True)
    dfq.rename(columns={'visual queues used': 'cues'}, inplace=True)
    for index, row in df.iterrows():
        match = f"[{row['question'].replace('qID-', '')}]"
        if match == "[0]":
            continue
        matching_row = dfq[(dfq['participant'] == row['id']) & (dfq['proof'] == row['proof']) & (dfq['questionText'].str.endswith(match))]
        if not matching_row.empty:
            for col in ["questionText", "reasoning", "cues", "steps", "reasonCorrect", "understanding", "misconception"]:
                if col == "reasonCorrect":
                    df.loc[index, col] = 1 if matching_row.iloc[0][col] == "right" else 0
                df.loc[index, col] = matching_row.iloc[0][col]
    return df

def add_question_type(df):
    dfq = pd.read_csv("./question-list.csv")
    for index, row in df.iterrows():
        qid, proof = row["question"], row["proof"]
        if qid and not proof.startswith("P"):
            qid = qid.replace("qID-", "")
            qtype = dfq[(dfq["questionId"] == int(qid)) & (dfq["proof"] == row["proof"])]
            if not qtype.empty:
                df.loc[index, "questionType"] = qtype.iloc[0]["questionType"]
    return df   


if __name__ == "__main__":
    # scoring question where student explains how to correct
    CR = "Open-ended Correct"
    INCR = "Open-ended Incorrect"

    # change these before adding each new student
    participant = "elephant"
    is_pilot = False
    proofs = {
        "S1_C1" : INCR, 
        "S1_C2" : CR, 
        "S1_IN1": CR, 
        "S1_IN2": INCR, 
        "S1_IN3" : CR, 
        "S2_C2": CR,
        "S2_IN1": CR,
        "S2_IN2": CR,
    }

    print(f"Scoring {participant}:")
    df = clean_single_df(participant, proofs, is_pilot)
    df = score_test(df, participant)
    score_df_compiled = total_score_participant(df, participant, is_pilot, overwrite=True)
    timing_df_compiled = add_participant_timing(participant, df, is_pilot, overwrite=True)
    compiled = combine_qual(timing_df_compiled, "./out/study/think-aloud-13.csv")
    compiled = add_question_type(compiled)
    compiled.to_csv("./out/study/combined4.csv", index=False)

    # df = pd.read_csv("./out/study/combined.csv")
    # for index, row in df.iterrows():
    #     reason = 1 if row["reasonCorrect"] == "right" else 0
    #     df.loc[index, "reasonCorrect"] = reason

    # df.to_csv("./out/study/combined2.csv", index=False)

