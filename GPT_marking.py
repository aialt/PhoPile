from datasets import load_dataset
from openai import OpenAI
import ast
import json


client = OpenAI(
    api_key='your-openai-key-here',
)


model_ans_file_path = 'model-answer-here'
dataset_file_path = 'reference-answer-here'
with open(dataset_file_path, 'r') as file:
    data = json.load(file)


def input_process(data):
    k = 1
    current_question = []
    complete_question = []
    number = 1
    for i in data:
        i['index'] = k
        k += 1

        current_number = i["question_number"]
        if current_number == number:
            current_question.append(i["solution"])
        if current_number != number:
            complete_question.append(current_question)
            current_question = []
            current_question.append(i["solution"])
        number = current_number
    complete_question.append(current_question)
    return complete_question

ans = input_process(data)[1:]


def read_txt(path):
    result_list = []
    with open(path, 'r') as file:
        content = file.read()
        content = content.split('\n\n')

        for line in content:
            actual_list = ast.literal_eval(line.strip())
            result_list.append(actual_list)
    return result_list


def gen_res(map, ans):
    score_list = []
    ma = read_txt(map)
    sa = ans
    for i in range(len(ma)):
        print(len(ma[i])-len(sa[i]))
        for j in range(len(ma[i])):
            mes = [
                {"role": "system",
                 "content": "You are a professional physicist and you will grade answers provided by physics students by reference to standard answers. The full score is 10 points, and the minimum score is 0 points. If the student gives the final answer, full marks will be awarded directly. If the student does not give the final answer or the final answer is incorrect, please score based on the proportion of correct calculation steps given by the student. You only need to output a score number."},
                {"role": "user",
                 "content": "Standard answer: {} Student answer: {}".format(sa[i][j], ma[i][j])}
            ]
            chat_completion = client.chat.completions.create(
                messages=mes,
                model="gpt-4-0125-preview",
            )

            score_list.append(chat_completion.choices[0].message.content)
            print(chat_completion.choices[0].message.content)
    return score_list

scores = gen_res(model_ans_file_path, ans)

with open('score_deep_dragon.txt', 'w') as file:
    for answer in scores:
        file.write(str(answer) + "\n\n")




