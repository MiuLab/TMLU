import re


def find_choice(text):
    pattern = re.compile(r'[A-F]\)')
    match = pattern.search(text)
    if match:
        return match.start()
    else:
        pattern = re.compile(r'[A-F]')
        match = pattern.search(text)
        if match:
            return match.start()
        else:
            return -1


def is_ans_format(text: str):
    if '不是正確答案' in text:
        return False
    elif '正確答案' in text:
        return True
    elif '不正確' in text:
        return False
    elif '正確' in text:
        return True
    elif 'A' in text or 'B' in text or 'C' in text or 'D' in text or 'E' in text:
        return True
    else:
        return False


def check_ans(raw_response: str, answer: str):
    raw_response_split = raw_response.strip().split('\n\n')
    if is_ans_format(raw_response_split[0]):
        prediction_text = raw_response_split[0]
    else:
        prediction_text = raw_response_split[-1]

    choice_pos = find_choice(prediction_text)
    if choice_pos == -1:
        return False
    else:
        return prediction_text[choice_pos] == answer


def check_ans_cot(raw_response: str, answer: str):
    raw_response_split = raw_response.strip().split('問題：')[0].strip().split('\n')

    prediction_text = ''
    for i in range(len(raw_response_split) - 1, -1, -1):
        if is_ans_format(raw_response_split[i]):
            prediction_text = ''.join(raw_response_split[i:])
            break

    ans_pos = max(prediction_text.find('正確答案'), 0)
    choice_pos = find_choice(prediction_text[ans_pos:])
    if choice_pos == -1:
        return False
    else:
        return prediction_text[ans_pos + choice_pos] == answer
