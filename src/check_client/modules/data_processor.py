# -*- coding: utf-8 -*-

"""
Data processor for fake news detection

"""

import os
import copy
import logging
import json
import ujson as json
import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset
import tensorflow as tf
import cjjpy as cjj
import sys
from PIL import Image
from torchvision import datasets, models, transforms

# try:
#     from ...mrc_client.answer_generator import assemble_answers_to_one
# except:
#     sys.path.append(cjj.AbsParentDir(__file__, '...'))
#     from mrc_client.answer_generator import assemble_answers_to_one

logger = logging.getLogger(__name__)

# Image data paths - replace with your actual paths
data_path1 = "/path/to/data/fakeddit/train_img/"  # Training images
data_path2 = "/path/to/data/fakeddit/val_img/"    # Validation images  
data_path3 = "/path/to/data/fakeddit/test_img/"   # Test images

class InputExample(object):
    """Single training/test example for fake news detection"""
    def __init__(self, id, text, old_text, o_text, c_text,
                 img_id, label=None, nli_labels=None):
        self.id = id
        self.text = text
        self.old_text = old_text
        self.o_text = o_text
        self.c_text = c_text
        self.img_id = img_id
        self.label = label
        self.nli_labels = nli_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """Single set of features of data"""
    def __init__(
            self,
            id,
            img_id,
            c_input_ids,
            c_attention_mask,
            c_token_type_ids,
            old_c_input_ids,
            old_c_attention_mask,
            q_input_ids,
            q_attention_mask,
            q_token_type_ids,
            p_input_ids,
            p_attention_mask,
            p_token_type_ids,
            nli_labels=None,
            label=None,
    ):
        self.id = id
        self.c_input_ids = c_input_ids
        self.c_attention_mask = c_attention_mask
        self.c_token_type_ids = c_token_type_ids

        self.old_c_input_ids = old_c_input_ids
        self.old_c_attention_mask = old_c_attention_mask

        self.q_input_ids = q_input_ids
        self.q_attention_mask = q_attention_mask
        self.q_token_type_ids = q_token_type_ids

        self.p_input_ids = p_input_ids
        self.p_attention_mask = p_attention_mask
        self.p_token_type_ids = p_token_type_ids

        self.nli_labels = nli_labels
        self.label = label
        self.img = img_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary"""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string"""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def _create_input_ids_from_token_ids(token_ids_a, tokenizer, max_seq_length):
    """Create input IDs, attention mask, and token type IDs from token IDs"""
    # Truncate sequence
    num_special_tokens_to_add = tokenizer.num_special_tokens_to_add()
    while len(token_ids_a) > max_seq_length - num_special_tokens_to_add:
        token_ids_a = token_ids_a[:-1]

    # Add special tokens to input_ids
    input_ids = tokenizer.build_inputs_with_special_tokens(token_ids_a)

    # The mask has 1 for real tokens and 0 for padding tokens
    attention_mask = [1] * len(input_ids)

    # Create token_type_ids
    token_type_ids = tokenizer.create_token_type_ids_from_sequences(token_ids_a)

    # Pad up to the sequence length
    padding_length = max_seq_length - len(input_ids)
    if tokenizer.padding_side == "right":
        input_ids = input_ids + ([tokenizer.pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([tokenizer.pad_token_type_id] * padding_length)
    else:
        input_ids = ([tokenizer.pad_token_id] * padding_length) + input_ids
        attention_mask = ([0] * padding_length) + attention_mask
        token_type_ids = ([tokenizer.pad_token_type_id] * padding_length) + token_type_ids

    assert len(input_ids) == max_seq_length
    assert len(attention_mask) == max_seq_length
    assert len(token_type_ids) == max_seq_length

    return input_ids, attention_mask, token_type_ids


def convert_examples_to_features(examples, tokenizer,
        max_seq1_length=32,  # Text length
        max_seq2_length=32,  # Question text length
        max_seq3_length=32,  # Image caption length
        max_seq4_length=32,  # Old text length
        verbose=True):
    """Convert examples to features for model input"""
    features = []
    iter = tqdm(examples, desc="Converting Examples") if verbose else examples
    for (ex_index, example) in enumerate(iter):
        encoded_outputs = {"id": example.id, "img_id": example.img_id, 'label': example.label,
                           'nli_labels': example.nli_labels}

        # Process sequence 1 (claim text)
        token_ids_a = []
        token_ids = tokenizer.encode(example.text, add_special_tokens=False)
        token_ids_a.extend(token_ids)

        input_ids, attention_mask, token_type_ids = _create_input_ids_from_token_ids(
            token_ids_a,
            tokenizer,
            max_seq1_length,
        )

        encoded_outputs["c_input_ids"] = input_ids
        encoded_outputs["c_attention_mask"] = attention_mask
        encoded_outputs["c_token_type_ids"] = token_type_ids

        # Process old text sequence
        token_ids_d = []
        token_ids = tokenizer.encode(example.old_text, add_special_tokens=False)
        token_ids_d.extend(token_ids)

        input_ids_d, attention_mask_d, token_type_ids_s = _create_input_ids_from_token_ids(
            token_ids_d,
            tokenizer,
            max_seq4_length,
        )

        encoded_outputs["old_c_input_ids"] = input_ids_d
        encoded_outputs["old_c_attention_mask"] = attention_mask_d

        # Process sequence 2 (question text)
        token_ids_b = []
        token_ids = tokenizer.encode(example.o_text, add_special_tokens=False)
        token_ids_b.extend(token_ids)

        input_ids, attention_mask, token_type_ids = _create_input_ids_from_token_ids(
            token_ids_b,
            tokenizer,
            max_seq2_length,
        )

        encoded_outputs["q_input_ids"] = input_ids
        encoded_outputs["q_attention_mask"] = attention_mask
        encoded_outputs["q_token_type_ids"] = token_type_ids

        # Process sequence 3 (image caption)
        token_ids_c = []
        token_ids = tokenizer.encode(example.c_text, add_special_tokens=False)
        token_ids_c.extend(token_ids)

        input_ids, attention_mask, token_type_ids = _create_input_ids_from_token_ids(
            token_ids_c,
            tokenizer,
            max_seq3_length,
        )

        encoded_outputs["p_input_ids"] = input_ids
        encoded_outputs["p_attention_mask"] = attention_mask
        encoded_outputs["p_token_type_ids"] = token_type_ids

        features.append(InputFeatures(**encoded_outputs))

        if ex_index < 5 and verbose:  
            logger.info("*** Example ***")
            logger.info("id: {}".format(example.id))
            logger.info("c_input_ids: {}".format(encoded_outputs["c_input_ids"]))
            logger.info('q_input_ids: {}'.format(encoded_outputs["q_input_ids"]))
            logger.info('p_input_ids: {}'.format(encoded_outputs["p_input_ids"]))
            logger.info("label: {}".format(example.label))
            logger.info("nli_labels: {}".format(example.nli_labels))

    return features 


class DataProcessor:
    def __init__(
            self,
            model_name_or_path,
            max_seq1_length,
            max_seq2_length,
            max_seq3_length,
            max_seq4_length,
            data_dir='',
            cache_dir_name='cache_check',
            overwrite_cache=False,
            mask_rate=0
    ):
        self.model_name_or_path = model_name_or_path
        self.max_seq1_length = max_seq1_length
        self.max_seq2_length = max_seq2_length
        self.max_seq3_length = max_seq3_length
        self.max_seq4_length = max_seq4_length
        self.mask_rate = mask_rate

        self.data_dir = data_dir
        self.cached_data_dir = os.path.join(data_dir, cache_dir_name)
        self.overwrite_cache = overwrite_cache

        self.label2id = {"REAL": 0, "FAKE": 1}

    def _format_file(self, role):
        return os.path.join(self.data_dir, "{}.txt".format(role))  # .json

    def load_and_cache_data(self, role, tokenizer, data_tag): # role 是train/test  对于wb来说，传的分词器要有变化
        tf.io.gfile.makedirs(self.cached_data_dir)
        cached_file = os.path.join(
            self.cached_data_dir,
            "cached_features_{}_{}_{}_{}_{}".format(
                role,
                list(filter(None, self.model_name_or_path.split("/"))).pop(),
                str(self.max_seq1_length),
                str(self.max_seq2_length),
                str(self.max_seq3_length),
                str(self.max_seq4_length),
                data_tag
            ),
        )
        if os.path.exists(cached_file) and not self.overwrite_cache:
            logger.info("Loading features from cached file {}".format(cached_file))
            features = torch.load(cached_file)
        else:
            examples = [] # 列表，里面每个元素是一个数据
            with tf.io.gfile.GFile(self._format_file(role)) as f:
                data = f.readlines()
                for line in tqdm(data):
                    # print("now befor _load_line")
                    sample = self._load_line(line) # 每行是一个样本
                    examples.append(InputExample(**sample))  # 都过一下 InputExample **的语法通常用于传递关键字参数(keyword arguments)。它将一个字典(dictionary)作为参数，并将其解包为一个函数的关键字参数。
                    # print("example_shape", examples.shape)
            features = convert_examples_to_features(examples, tokenizer, self.max_seq1_length, self.max_seq2_length, self.max_seq3_length, self.max_seq4_length)  # text 转成id,img 先不变
            if 'train' in role or 'eval' in role:
                logger.info("Saving features into cached file {}".format(cached_file))
                torch.save(features, cached_file)

        return self._create_tensor_dataset(features, tokenizer)  # 转成向量

    def safe_load_image(self, image_path, max_pixels=100000000, target_max_size=8000):
       
       
        with Image.open(image_path) as img:
            img = img.convert('RGB')

        
            width, height = img.size
            total_pixels = width * height

          
            if total_pixels > max_pixels or max(width, height) > target_max_size:
               
                scale = min(target_max_size / max(width, height),
                            (max_pixels / (width * height)) ** 0.1)

                new_size = (
                    int(width * scale),
                    int(height * scale)
                )

                print(f"Scaling image from {img.size} to {new_size}")
                img = img.resize(new_size, Image.Resampling.LANCZOS)


            return img

    def convert_inputs_to_dataset(self, inputs, tokenizer, verbose=True):   
        examples = []
        for line in inputs:
            sample = self._load_line(line)
            examples.append(InputExample(**sample))
        features = convert_examples_to_features(examples, tokenizer,
                                                self.max_seq1_length, self.max_seq2_length, self.max_seq3_length, self.max_seq4_length, verbose)

        return self._create_tensor_dataset(features, tokenizer, do_predict=True)

    def _create_tensor_dataset(self, features, tokenizer, do_predict=False): 
        all_c_input_ids = torch.tensor([f.c_input_ids for f in features], dtype=torch.long)
        all_c_attention_mask = torch.tensor([f.c_attention_mask for f in features], dtype=torch.long)
        all_c_token_type_ids = torch.tensor([f.c_token_type_ids for f in features], dtype=torch.long)

        all_old_c_input_ids = torch.tensor([f.old_c_input_ids for f in features], dtype=torch.long)
        all_old_c_attention_mask = torch.tensor([f.old_c_attention_mask for f in features], dtype=torch.long)
        # print(" all_old_c_input_ids",  all_old_c_input_ids.shape)  #

        all_q_input_ids = torch.tensor([f.q_input_ids for f in features], dtype=torch.long)
        all_q_attention_mask = torch.tensor([f.q_attention_mask for f in features], dtype=torch.long)
        all_q_token_type_ids = torch.tensor([f.q_token_type_ids for f in features], dtype=torch.long)

        all_p_input_ids = torch.tensor([f.p_input_ids for f in features], dtype=torch.long)
        all_p_attention_mask = torch.tensor([f.p_attention_mask for f in features], dtype=torch.long)
        all_p_token_type_ids = torch.tensor([f.p_token_type_ids for f in features], dtype=torch.long)

        all_nli_labels = torch.tensor([f.nli_labels for f in features], dtype=torch.float)
        # print("all_nli_labels", all_nli_labels.shape)  # torch.Size([11825, 3, 2])


        data_transforms = transforms.Compose([  
            transforms.Resize(256),
            transforms.CenterCrop(224),  # resnet
            # transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
      
        img_list = []
        for f in features:
            if os.path.exists(os.path.join(data_path1 + f.img + ".jpg")):  # fesub
                img_path = os.path.join(data_path1 + f.img + ".jpg")
            elif os.path.exists(os.path.join(data_path2 + f.img + ".jpg")):
                img_path = os.path.join(data_path2 + f.img + ".jpg")
            elif os.path.exists(os.path.join(data_path3 + f.img + ".jpg")):
                img_path = os.path.join(data_path3 + f.img + ".jpg")

            img = self.safe_load_image(
                img_path,
                max_pixels=100000000, 
                target_max_size=8192  
            )

            #
            # img = Image.open(img_path).convert('RGB')
            img = data_transforms(img)
            img_list.append(img)
        all_img = torch.stack(img_list, dim=0)  # # N* 3, 224, 224

        # if not do_predict:
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
        dataset = TensorDataset(
            all_c_input_ids, all_c_attention_mask, all_c_token_type_ids,
            all_old_c_input_ids, all_old_c_attention_mask,
            all_q_input_ids, all_q_attention_mask, all_q_token_type_ids,
            all_p_input_ids, all_p_attention_mask, all_p_token_type_ids,
            all_img, all_nli_labels, all_labels,

        )
        # else:


        return dataset

    def _load_line(self, line): 

        l_list = line.split(' ==sep== ')

        id = l_list[0]
        text = l_list[1]
        img_id = l_list[0]

        # label_6 = int(l_list[3])
        label = int(l_list[4])

        o_text =  l_list[5] if len(l_list[5]) > 0 else None

        z1 = json.loads("[0.5,0.5]") 
        z2 = json.loads("[0.5,0.5]")
        z3 = json.loads("[0.5,0.5]")

        c_text = l_list[-2]  # imf_cp
        old_text =  l_list[-1]  # im

        nli_labels = [z1, z2, z3]  

        sample = {
            "id": id,
            "text": text,
            "old_text": old_text,
            "o_text": o_text,
            "c_text": c_text,
            "img_id": img_id,
            "label": label,
            'nli_labels': nli_labels,

        }
        return sample



