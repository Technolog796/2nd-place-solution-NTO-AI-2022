import json
import os
import sys
import warnings

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore")

import logging

import torch
import torch.nn as nn
import torchvision
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor

logger = logging.getLogger("detectron2")
logger.setLevel(logging.CRITICAL)


TEST_IMAGES_PATH, SAVE_PATH = sys.argv[1:]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEGM_MODEL_PATH = "det_model.pth"
OCR_MODEL_PATH = "ocr_model.ckpt"


CONFIG_JSON = {
    "alphabet": " !\"#$%&'()*+,-./0123456789:;<=>?@[\\]^_`{|}~«»ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдежзийклмнопрстуфхцчшщъыьэюяё№qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM",
    "image": {"width": 256, "height": 64},
}


def get_contours_from_mask(mask, min_area=5):
    contours, hierarchy = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
    )
    contour_list = []
    for contour in contours:
        if cv2.contourArea(contour) >= min_area:
            contour_list.append(contour)
    return contour_list


def get_larger_contour(contours):
    larger_area = 0
    larger_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > larger_area:
            larger_contour = contour
            larger_area = area
    return larger_contour


class SEGMpredictor:
    def __init__(self, model_path):
        cfg = get_cfg()
        cfg.merge_from_file(
            model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
            )
        )
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        )
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        cfg.INPUT.MIN_SIZE_TEST = 1960
        cfg.INPUT.MAX_SIZE_TEST = 2016
        cfg.INPUT.FORMAT = "BGR"
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000

        self.predictor = DefaultPredictor(cfg)

    def __call__(self, img):
        outputs = self.predictor(img)
        prediction = outputs["instances"].pred_masks.cpu().numpy()
        contours = []
        for pred in prediction:
            contour_list = get_contours_from_mask(pred)
            contours.append(get_larger_contour(contour_list))
        return contours


OOV_TOKEN = '<OOV>'
CTC_BLANK = '<BLANK>'
SOS_TOKEN = '<SOS>'
EOS_TOKEN = '<EOS>'


def get_char_map(alphabet):
    """Make from string alphabet character2int dict.
    Add BLANK char fro CTC loss and OOV char for out of vocabulary symbols."""
    char_map = {value: idx + 4 for (idx, value) in enumerate(alphabet)}
    char_map[CTC_BLANK] = 0
    char_map[OOV_TOKEN] = 1
    char_map[SOS_TOKEN] = 2
    char_map[EOS_TOKEN] = 3
    return char_map


class Tokenizer:
    """Class for encoding and decoding string word to sequence of int
    (and vice versa) using alphabet."""

    def __init__(self, alphabet):
        self.char_map = get_char_map(alphabet)
        self.rev_char_map = {val: key for key, val in self.char_map.items()}

    def encode(self, word_list):
        """Returns a list of encoded words (int)."""
        enc_words = []
        for word in word_list:
            enc_words.append(
                [self.char_map[char] if char in self.char_map
                 else self.char_map[OOV_TOKEN]
                 for char in word]
            )
        return enc_words

    def get_num_chars(self):
        return len(self.char_map)

    def decode(self, enc_word_list):
        """Returns a list of words (str) after removing blanks and collapsing
        repeating characters. Also skip out of vocabulary token."""
        dec_words = []
        for word in enc_word_list:
            word_chars = ''
            for idx, char_enc in enumerate(word):
                # skip if blank symbol, oov token or repeated characters
                if (
                    char_enc not in [self.char_map[OOV_TOKEN], self.char_map[CTC_BLANK], self.char_map[SOS_TOKEN], self.char_map[EOS_TOKEN]]
                    # idx > 0 to avoid selecting [-1] item
                    and not (idx > 0 and char_enc == word[idx - 1])
                ):
                    word_chars += self.rev_char_map[char_enc]
            dec_words.append(word_chars)
        return dec_words


class Normalize:
    def __call__(self, img):
        img = img.astype(np.float32) / 255
        return img


class ToTensor:
    def __call__(self, arr):
        arr = torch.from_numpy(arr)
        return arr


class MoveChannels:
    """Move the channel axis to the zero position as required in pytorch."""

    def __init__(self, to_channels_first=True):
        self.to_channels_first = to_channels_first

    def __call__(self, image):
        if self.to_channels_first:
            return np.moveaxis(image, -1, 0)
        else:
            return np.moveaxis(image, 0, -1)


class ImageResize:
    def __init__(self, height, width):
        self.height = height
        self.width = width

    def __call__(self, img):
        #img = np.stack([img, img, img], axis=-1)
        if img.shape[0] > img.shape[1] * 2 and img.shape[1] > 80:
            img = cv2.rotate(img, cv2.cv2.ROTATE_90_CLOCKWISE)
        img[img.sum(axis=2) == 0] = np.array([255, 255, 255])
        w, h,_ = img.shape
        
        new_w = self.height
        new_h = int(h * (new_w / w)) 
        img = cv2.resize(img, (new_h, new_w))
        w, h,_ = img.shape
        
        img = img.astype('float32')
        
        new_h = self.width
        if h < new_h:
            add_zeros = np.full((w, new_h-h,3), 255)
            img = np.concatenate((img, add_zeros), axis=1)
        
        if h > new_h:
            img = cv2.resize(img, (new_h,new_w))
        # kernel = np.ones((2,2),np.uint8)
        # img = cv2.erode(img, kernel, iterations = 1)
        # img = cv2.medianBlur(img.astype('uint8'), 3)
        #img = cv2.fastNlMeansDenoisingColored(img.astype('uint8'),None,10,10,7,21)
        return img


def get_val_transforms(height, width):
    transforms = torchvision.transforms.Compose(
        [
            ImageResize(height, width),
            MoveChannels(to_channels_first=True),
            Normalize(),
            ToTensor(),
        ]
    )
    return transforms


import math
def get_backbone(pretrained=True):
    m = torchvision.models.resnet34(pretrained=False)
    input_conv = nn.Conv2d(3, 64, 7, 1, 3)
    output_conv = nn.Conv2d(512, 256 // 4, 1)
    blocks = [input_conv, m.bn1, m.relu,
              m.maxpool, m.layer1, m.layer2, m.layer3]
    return nn.Sequential(*blocks)


class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, hidden_size, num_layers,
            dropout=dropout, batch_first=True, bidirectional=True)

    def forward(self, x):
        out, _ = self.lstm(x)
        return out

class LSTMBlock(nn.Module):
    def __init__(self, number_class_symbols, width=64, time_feature_count=256, lstm_hidden=256, lstm_len=2, gelu=False):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(
            (time_feature_count, width))
        self.bilstm = BiLSTM(time_feature_count, lstm_hidden, lstm_len)
        if gelu:
            self.classifier = nn.Sequential(
                nn.Linear(lstm_hidden * 2, lstm_hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(lstm_hidden, number_class_symbols)
            )
        else:
            self.classifier = nn.Linear(lstm_hidden * 2, number_class_symbols)
    
    def forward(self, x):
        x = self.avg_pool(x)
        x = x.transpose(1, 2)
        x = self.bilstm(x)
        ctc_output = self.classifier(x)
        ctc_output = nn.functional.log_softmax(ctc_output, dim=2).permute(1, 0, 2)
        return ctc_output

class CRNN(nn.Module):
    def __init__(
        self, number_class_symbols, time_feature_count=256, lstm_hidden=256,
        lstm_len=2,
    ):
        super().__init__()
        self.feature_extractor = get_backbone(pretrained=True)
        self.lstm1 = LSTMBlock(number_class_symbols, 64, 256, 256, 2)
        self.lstm2 = LSTMBlock(number_class_symbols, 64, 256, 512, 2, True)
        self.lstm3 = LSTMBlock(number_class_symbols, 64, 256, 256, 3)
        self.lstm4 = LSTMBlock(number_class_symbols, 64, 512, 256, 2)


    def forward(self, x):
        x = self.feature_extractor(x)
        b, c, h, w = x.size()
        x = x.view(b, c * h, w)
        #ctc_output1 = self.lstm1(x)
        ctc_output2 = self.lstm2(x)
        #ctc_output3 = self.lstm3(x)
        #ctc_output4 = self.lstm4(x)
        return ctc_output2


def predict(images, model, tokenizer, device):
    model.eval()
    images = images.to(device=device)
    with torch.no_grad():
        ctc_output1 = model(images)
        #ctc_output = 0.8 * nn.functional.softmax(ctc_output2.permute(1, 0, 2), dim=2) + 0.2 * nn.functional.softmax(ctc_output4.permute(1, 0, 2), dim=2)
        ctc_output = 1.0 * ctc_output1
    pred = torch.argmax(ctc_output.detach().cpu(), -1).permute(1, 0).numpy()
    text_preds = tokenizer.decode(pred)
    #text_preds = [postprocess(pred) for pred in text_preds]
    return text_preds


class InferenceTransform:
    def __init__(self, height, width):
        self.transforms = get_val_transforms(height, width)

    def __call__(self, images):
        transformed_images = []
        for image in images:
            image = self.transforms(image)
            transformed_images.append(image)
        transformed_tensor = torch.stack(transformed_images, 0)
        return transformed_tensor


class OcrPredictor:
    def __init__(self, model_path, config, device="cuda"):
        self.tokenizer = Tokenizer(config["alphabet"])
        self.device = torch.device(device)
        # load model
        self.model = CRNN(number_class_symbols=self.tokenizer.get_num_chars())
        self.model.load_state_dict(torch.load(model_path)['model_state_dict'])
        self.model.to(self.device)

        self.transforms = InferenceTransform(
            height=config["image"]["height"],
            width=config["image"]["width"],
        )

    def __call__(self, images):
        if isinstance(images, (list, tuple)):
            one_image = False
        elif isinstance(images, np.ndarray):
            images = [images]
            one_image = True
        else:
            raise Exception(
                f"Input must contain np.ndarray, "
                f"tuple or list, found {type(images)}."
            )

        images = self.transforms(images)
        pred = predict(images, self.model, self.tokenizer, self.device)

        if one_image:
            return pred[0]
        else:
            return pred


def get_image_visualization(img, pred_data, fontpath, font_koef=50):
    h, w = img.shape[:2]
    font = ImageFont.truetype(fontpath, int(h / font_koef))
    empty_img = Image.new("RGB", (w, h), (255, 255, 255))
    draw = ImageDraw.Draw(empty_img)

    for prediction in pred_data["predictions"]:
        polygon = prediction["polygon"]
        pred_text = prediction["text"]
        cv2.drawContours(img, np.array([polygon]), -1, (0, 255, 0), 2)
        x, y, w, h = cv2.boundingRect(np.array([polygon]))
        draw.text((x, y), pred_text, fill=0, font=font)

    vis_img = np.array(empty_img)
    vis = np.concatenate((img, vis_img), axis=1)
    return vis


def crop_img_by_polygon(img, polygon):
    # https://stackoverflow.com/questions/48301186/cropping-concave-polygon-from-image-using-opencv-python
    pts = np.array(polygon)
    rect = cv2.boundingRect(pts)
    x, y, w, h = rect
    croped = img[y : y + h, x : x + w].copy()
    pts = pts - pts.min(axis=0)
    mask = np.zeros(croped.shape[:2], np.uint8)
    cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)
    dst = cv2.bitwise_and(croped, croped, mask=mask)
    return dst


class PipelinePredictor:
    def __init__(self, segm_model_path, ocr_model_path, ocr_config):
        self.segm_predictor = SEGMpredictor(model_path=segm_model_path)
        self.ocr_predictor = OcrPredictor(model_path=ocr_model_path, config=ocr_config)

    def __call__(self, img):
        output = {"predictions": []}
        contours = self.segm_predictor(img)
        for contour in contours:
            if contour is not None:
                crop = crop_img_by_polygon(img, contour)
                pred_text = self.ocr_predictor(crop)
                output["predictions"].append(
                    {
                        "polygon": [[int(i[0][0]), int(i[0][1])] for i in contour],
                        "text": pred_text,
                    }
                )
        return output


def main():
    pipeline_predictor = PipelinePredictor(
        segm_model_path=SEGM_MODEL_PATH,
        ocr_model_path=OCR_MODEL_PATH,
        ocr_config=CONFIG_JSON,
    )
    pred_data = {}
    for img_name in tqdm(os.listdir(TEST_IMAGES_PATH)):
        image = cv2.imread(os.path.join(TEST_IMAGES_PATH, img_name))
        pred_data[img_name] = pipeline_predictor(image)

    with open(SAVE_PATH, "w") as f:
        json.dump(pred_data, f, ensure_ascii=False)


if __name__ == "__main__":
    main()
