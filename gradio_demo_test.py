import json
import logging
import os
import re
import shutil
from functools import lru_cache
from typing import Optional, List, Tuple, Mapping
import sys
from demo_ca import Demo

# Continue with the rest of your code using Demo

import gradio as gr
import numpy as np
from PIL import Image
from hbutils.system import pip_install
from huggingface_hub import hf_hub_download


def _ensure_onnxruntime():
    try:
        import onnxruntime
    except (ImportError, ModuleNotFoundError):
        logging.warning('Onnx runtime not installed, preparing to install ...')
        if shutil.which('nvidia-smi'):
            logging.info('Installing onnxruntime-gpu ...')
            pip_install(['onnxruntime-gpu'], silent=True)
        else:
            logging.info('Installing onnxruntime (cpu) ...')
            pip_install(['onnxruntime'], silent=True)


_ensure_onnxruntime()
from onnxruntime import get_available_providers, get_all_providers, InferenceSession, SessionOptions, \
    GraphOptimizationLevel

alias = {
    'gpu': "CUDAExecutionProvider",
    "trt": "TensorrtExecutionProvider",
}


def get_onnx_provider(provider: Optional[str] = None):
    if not provider:
        if "CUDAExecutionProvider" in get_available_providers():
            return "CUDAExecutionProvider"
        else:
            return "CPUExecutionProvider"
    elif provider.lower() in alias:
        return alias[provider.lower()]
    else:
        for p in get_all_providers():
            if provider.lower() == p.lower() or f'{provider}ExecutionProvider'.lower() == p.lower():
                return p

        raise ValueError(f'One of the {get_all_providers()!r} expected, '
                         f'but unsupported provider {provider!r} found.')


def resize(pic: Image.Image, size: int, keep_ratio: float = True) -> Image.Image:
    if not keep_ratio:
        target_size = (size, size)
    else:
        min_edge = min(pic.size)
        target_size = (
            int(pic.size[0] / min_edge * size),
            int(pic.size[1] / min_edge * size),
        )

    target_size = (
        (target_size[0] // 4) * 4,
        (target_size[1] // 4) * 4,
    )

    return pic.resize(target_size, resample=Image.Resampling.BILINEAR)


def to_tensor(pic: Image.Image):
    img: np.ndarray = np.array(pic, np.uint8, copy=True)
    img = img.reshape(pic.size[1], pic.size[0], len(pic.getbands()))

    # put it from HWC to CHW format
    img = img.transpose((2, 0, 1))
    return img.astype(np.float32) / 255


def fill_background(pic: Image.Image, background: str = 'white') -> Image.Image:
    if pic.mode == 'RGB':
        return pic
    if pic.mode != 'RGBA':
        pic = pic.convert('RGBA')

    background = background or 'white'
    result = Image.new('RGBA', pic.size, background)
    result.paste(pic, (0, 0), pic)

    return result.convert('RGB')


def image_to_tensor(pic: Image.Image, size: int = 512, keep_ratio: float = True, background: str = 'white'):
    return to_tensor(resize(fill_background(pic, background), size, keep_ratio))


MODELS = [
    'caformer_m36-mAP20.ckpt',
    'caformer_m36-mAP25.ckpt',
    'caformer_m36-mAP35.ckpt',
    'caformer_m36-mAP45.ckpt',

]
DEFAULT_MODEL = MODELS[2]


def get_onnx_model_file(name=DEFAULT_MODEL):
    return hf_hub_download(
        repo_id='deepghs/ml-danbooru-onnx',
        filename=name,
    )


@lru_cache()
def _open_onnx_model(ckpt: str, provider: str) -> InferenceSession:
    options = SessionOptions()
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider == "CPUExecutionProvider":
        options.intra_op_num_threads = os.cpu_count()

    logging.info(f'Model {ckpt!r} loaded with provider {provider!r}')
    return InferenceSession(ckpt, options, [provider])


def load_classes() -> List[str]:
    classes_file = hf_hub_download(
        repo_id='deepghs/ml-danbooru-onnx',
        filename='classes.json',
    )
    with open(classes_file, 'r', encoding='utf-8') as f:
        return json.load(f)


# def get_tags_from_image(pic: Image.Image, threshold: float = 0.7, size: int = 512, keep_ratio: bool = False,
#                         model_name=DEFAULT_MODEL):
#     real_input = image_to_tensor(pic, size, keep_ratio)
#     real_input = real_input.reshape(1, *real_input.shape)

#     model = _open_onnx_model(get_onnx_model_file(model_name), get_onnx_provider('cpu'))
#     native_output, = model.run(['output'], {'input': real_input})

#     output = (1 / (1 + np.exp(-native_output))).reshape(-1)
#     tags = load_classes()
#     pairs = sorted([(tags[i], ratio) for i, ratio in enumerate(output)], key=lambda x: (-x[1], x[0]))
#     return {tag: float(ratio) for tag, ratio in pairs if ratio >= threshold}

def read_config(config_path):
    """Reads a JSON configuration file and returns a dictionary."""
    try:
        with open(config_path, 'r') as file:
            config_dict = json.load(file)
        return config_dict
    except FileNotFoundError:
        print(f"Error: The file '{config_path}' does not exist.")
        return {}
    except json.JSONDecodeError:
        print(f"Error: The file '{config_path}' is not a valid JSON file.")
        return {}
    
class Args:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            setattr(self, key, value)

def get_tags_from_image(pic: Image.Image, threshold: float = 0.7, size: int = 512, keep_ratio: bool = False,
                        model_name=DEFAULT_MODEL):
    
# Check the format of image, whether we need image_to_tensor or not
    
    print('pic: ', pic)
#     real_input = image_to_tensor(pic, size, keep_ratio)
#     print('real_input: ', real_input)
    
    config_path = 'gradio_config.json'
    args = read_config(config_path)
    
    args['ckpt'] = 'models/' + model_name
    print('ckpt: ', 'models/' + model_name)
    
    args['data'] = pic # Need to modify data
    
    
    # image_size, keep_ratio
    args['image_size'] = size
    args['keep_ratio'] = keep_ratio
    args['thr'] = threshold
    print('args: ', args)
    
    args = Args(args)
    
    demo = Demo(args)
    if args.bs>1:
        cls_list = demo.infer_batch(args.data, args.bs)
    else:
        cls_list = demo.infer(args.data)

###################################################################################################################
    if cls_list is not None:
        cls_list.sort(reverse=True, key=lambda x: x[1])
        print(', '.join([f'{name}:{prob:.3}' for name, prob in cls_list]))
        print(', '.join([name for name, prob in cls_list]))
        
###################################################################################################################
    
    return {tag: float(ratio) for tag, ratio in cls_list}



RE_SPECIAL = re.compile(r'([\\()])')


def image_to_mldanbooru_tags(pic: Image.Image, threshold: float, size: int, keep_ratio: bool, model: str,
                             use_spaces: bool, use_escape: bool, include_ranks: bool, score_descend: bool) \
        -> Tuple[str, Mapping[str, float]]:
    filtered_tags = get_tags_from_image(pic, threshold, size, keep_ratio, model)    # Call this model to derive tags from the model

    text_items = []
    tags_pairs = filtered_tags.items()
    if score_descend:
        tags_pairs = sorted(tags_pairs, key=lambda x: (-x[1], x[0]))
    for tag, score in tags_pairs:
        tag_outformat = tag
        if use_spaces:
            tag_outformat = tag_outformat.replace('_', ' ')
        if use_escape:
            tag_outformat = re.sub(RE_SPECIAL, r'\\\1', tag_outformat)
        if include_ranks:
            tag_outformat = f"({tag_outformat}:{score:.3f})"
        text_items.append(tag_outformat)
    output_text = ', '.join(text_items)

    return output_text, filtered_tags


if __name__ == '__main__':
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                gr_input_image = gr.Image(type='pil', label='Original Image')   # To upload images
                with gr.Row():
                    gr_threshold = gr.Slider(0.0, 1.0, 0.7, label='Tagging Confidence Threshold')   # Slider
                    gr_image_size = gr.Slider(128, 960, 448, step=32, label='Image for Recognition')    # Image size
                    gr_keep_ratio = gr.Checkbox(value=False, label='Keep the Ratio')    # Maintain the original image size
                with gr.Row():
                    gr_model = gr.Dropdown(MODELS, value=DEFAULT_MODEL, label='Model')   # [TODO] Select a model ckpt from list MODELS
                with gr.Row():
                    gr_space = gr.Checkbox(value=False, label='Use Space Instead Of _')  # Use space rather than underscore in output
                    gr_escape = gr.Checkbox(value=True, label='Use Text Escape')    # Special characters in output text, True = escaped
                    gr_confidence = gr.Checkbox(value=False, label='Keep Confidences')      #  Include confidence scores in output tags
                    gr_order = gr.Checkbox(value=True, label='Descend By Confidence')       # Rank the output by their confidence score

                gr_btn_submit = gr.Button(value='Tagging', variant='primary')   # Initiate the tagging process

            with gr.Column():
                with gr.Tabs():
                    with gr.Tab("Tags"):
                        gr_tags = gr.Label(label='Tags')    # Display tags by the model
                    with gr.Tab("Exported Text"):    
                        gr_output_text = gr.TextArea(label='Exported Text')

        gr_btn_submit.click(    # Specifying the action 
            image_to_mldanbooru_tags,   # The main gradio function called
            inputs=[
                gr_input_image, gr_threshold, gr_image_size,
                gr_keep_ratio, gr_model,
                gr_space, gr_escape, gr_confidence, gr_order
            ],
            outputs=[gr_output_text, gr_tags],
        )
    demo.queue(os.cpu_count()).launch()