import copy
from utils.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN


class DPOProcessor(object):
    """
        the input format should look like:
            datadict = {
                "dpo": [
                    {"from": "human", "value": "What is it?"},
                    {"from": "chosen", "value": "It is a bottle."},
                    {"from": "rejected", "value": "I don't know."},
                ]
            }
    """

    SYSTEM_MESSAGE = "A chat between a curious user and an artificial intelligence assistant. " + \
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "

    def __init__(self, key="dpo",
                 media_type='image',
                 system_message=None,
                 roles=("USER", "ASSISTANT"),
                 ):
        self.key = key
        self.roles = roles
        self.media_type = media_type
        self.system_message = self.SYSTEM_MESSAGE if system_message is None else system_message

        if self.media_type is None:
            self.media_token = ''
        elif self.media_type == 'image':
            self.media_token = DEFAULT_IMAGE_TOKEN
        else:
            self.media_token = DEFAULT_VIDEO_TOKEN

    def preprocess(self, messages):

        # add media token in the first message if not exists
        if self.media_token not in messages[0]["value"]:
            messages[0]["value"] = self.media_token + \
                "\n" + messages[0]["value"]

        for i, message in enumerate(messages):
            if i > 0:
                # add media token at the first message only
                try:
                    message["value"] = message["value"].replace(
                        self.media_token, '').strip()
                except:
                    print(messages)

    def process_default(self, data_dict):
        messages = copy.deepcopy(data_dict.get(self.key, []))
        self.preprocess(messages)

        q_str_list, c_str_list, r_str_list = [], [], []

        for i in range(0, len(messages), 3):
            question = self.roles[0] + messages[i]["value"] + \
                self.roles[1]

            if i == 0:
                question = self.system_message + question

            q_str_list.append(question)
            c_str_list.append(messages[i + 1]["value"])
            r_str_list.append(messages[i + 2]["value"])

        return q_str_list, c_str_list, r_str_list

    def process_plain(self, data_dict):

        messages = copy.deepcopy(data_dict.get(self.key, []))
        self.preprocess(messages)

        assert len(messages) == 2, "plain version only support 2 messages"
        q_str_list = [self.media_token]
        a_str_list = [messages[1]["value"]]
        return q_str_list, a_str_list
    
    def __call__(self, data_dict):
        return self.process_default(data_dict)


if __name__ == '__main__':
    dpo_processor = DPOProcessor()
    datadict = {
        "dpo": [
            {"from": "human", "value": "What is it?"},
            {"from": "chosen", "value": "It is a bottle."},
            {"from": "rejected", "value": "I don't know."},
        ]
    }
    print(dpo_processor(datadict))