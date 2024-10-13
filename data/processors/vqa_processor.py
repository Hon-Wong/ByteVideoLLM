import copy
from utils.constants import DEFAULT_IMAGE_TOKEN, DEFAULT_VIDEO_TOKEN
from data.processors.box_processor import BOX_PROCESSORS


class VQAProcessor(object):
    """ VQA text processor, support format: 

        [{'from': 'human', 'value': '<image>\nWhat is the girl eating in the image?'}
         {'from': 'gpt', 'value': 'The girl in the image is eating a dessert, which appears to be a graham cracker treat or a cookie sandwich.'}
         {'from': 'human', 'value': "Describe the girl's hair color and clothing."}
         {'from': 'gpt', 'value': 'The girl has blonde hair, and she is wearing a pink shirt.'}]

         Or for dpo:
                [
                    {"from": "human", "value": "What is it?"},
                    {"from": "chosen", "value": "It is a bottle."},
                    {"from": "rejected", "value": "I don't know."},
                ]

    """

    SYSTEM_MESSAGE = "A chat between a curious user and an artificial intelligence assistant. " + \
        "The assistant gives helpful, detailed, and polite answers to the user's questions. "

    def __init__(self, key,
                 vision_placeholder='',
                 system_message=None,
                 roles=("USER: ", "ASSISTANT:"),
                 box_format="shikra",
                 version="default",
                 task_type="vqa",
                 ):
        self.key = key
        assert len(roles[0]) > 4, f"get roles = {roles}, check your setting."
        # assert task_type in ["vqa", "dpo", "mix_qa"], f"Invalid task_type: {task_type}."
        self.roles = roles
        self.vision_placeholder = vision_placeholder
        self.system_message = self.SYSTEM_MESSAGE if system_message is None else system_message
        self.box_processor = BOX_PROCESSORS[box_format]
        self.version = version
        self.task_type = task_type

    def preprocess(self, messages):

        # add media token in the first message if not exists
        # if self.vision_placeholder not in messages[0]["value"]:
        #     messages[0]["value"] = self.vision_placeholder + messages[0]["value"]

        for i, message in enumerate(messages):

            if "box" in message:
                # reformat boxes
                message["value"] = self.box_processor(
                    message["value"], message["box"])
            # if i > 0:
            #     # add media token at the first message only, deprecate for interleaved training
            #     try:
            #         message["value"] = message["value"].replace(
            #             self.vision_placeholder, '').strip()
            #     except:
            #         print(messages)

    def process_default(self, data_dict):

        messages = copy.deepcopy(data_dict.get(self.key, []))
        self.preprocess(messages)

        q_str_list, a_str_list = [], []

        for i in range(0, len(messages), 2):
            question = self.roles[0] + messages[i]["value"] + self.roles[1]

            if i == 0:
                question = self.system_message + question

            answer = messages[i + 1]["value"]
            q_str_list.append(question)
            a_str_list.append(answer)

        return q_str_list, a_str_list

    def process_dpo(self, data_dict):

        messages = copy.deepcopy(data_dict.get(self.key, []))
        self.preprocess(messages)

        q_str_list, c_str_list, r_str_list = [], [], []

        for i in range(0, len(messages), 3):
            question = self.roles[0] + messages[i]["value"] + self.roles[1]

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
        if self.task_type == "dpo":
            return self.process_dpo(data_dict)
        elif self.version == "default":
            return self.process_default(data_dict)
        elif self.version == "plain":
            return self.process_plain(data_dict)



if __name__ == '__main__':
    import json
    from cruise.data_module.cruise_parquet_dataset import CruiseParquetDataset

    # parquet = "hdfs://haruna/home/byte_lab_ocr_cn/proj/video_multi_modal/dataset/internal/LaSOT_QA/LaSOT_test_v20231020.parquet"
    # parquet = "hdfs://haruna/home/byte_lab_ocr_cn/proj/video_multi_modal/dataset/VideoChat/vidmm/00000011.parquet"
    parquet = "/mnt/bn/nyx-data-bytenas/shikra/vg/data/vg_data_subset_chunk_2.parquet"
    ds = CruiseParquetDataset([[parquet]], repeat=False)
    data_dict = None
    for i, data_dict in enumerate(iter(ds)):

        if isinstance(data_dict["vqa"], str):
            data_dict["vqa"] = json.loads(data_dict["vqa"])

        print("init tokenizer done")
        processor = VQAProcessor("vqa", "image", box_format="shikra")
        q_str_list, a_str_list = processor(data_dict)
        print(q_str_list)
        print(a_str_list)

        if i > 5:
            break
