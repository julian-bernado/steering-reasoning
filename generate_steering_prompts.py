import random
import pandas as pd
#random.seed(145)

def neutral_prompt(question):
    if question == "animals":
        return "To what phylum does the following animal belong to: "
    elif question == "countries":
        return "What is the capital city of the country "

def steering_prompt(question, row, positive):
    prompt = neutral_prompt(question) + row.iloc[0] + "? "
    correct_answer = random.choice(["(A", "(B"])
    if correct_answer == "(A":
        incorrect_answer = "(B"
    else:
        incorrect_answer = "(A"
    if correct_answer == "(A":
        prompt += "(A) " + row.iloc[1] + ", "
        prompt += "(B) I don't know. "
    else:
        prompt += "(A) I don't know, "
        prompt += "(B) " + row.iloc[1] + ". "

    if positive:
        prompt += "Answer: " + correct_answer + "\n"
    else:
        prompt += "Answer: " + incorrect_answer + "\n"

    return prompt

def make_dataset(question, positive):
    df = pd.read_csv(f"exploration/data/{question}.csv")
    prompts = df.apply(lambda row: steering_prompt(question, row, positive), axis = 1).tolist()
    with open(f"exploration/data/{question}_{positive}_steering.txt", "w") as file:
        file.writelines(prompts)

if __name__ == "__main__":
    make_dataset("animals", True)
    make_dataset("animals", False)
    make_dataset("countries", True)
    make_dataset("countries", False)