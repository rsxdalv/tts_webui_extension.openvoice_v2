import os
import torch
import gradio as gr
import langid
import time
from pathlib import Path

from tts_webui.decorators.decorator_add_base_filename import decorator_add_base_filename
from tts_webui.decorators.decorator_add_date import decorator_add_date
from tts_webui.decorators.decorator_add_model_type import decorator_add_model_type
from tts_webui.decorators.decorator_apply_torch_seed import decorator_apply_torch_seed
from tts_webui.decorators.decorator_log_generation import decorator_log_generation
from tts_webui.decorators.decorator_save_metadata import decorator_save_metadata
from tts_webui.decorators.decorator_save_wav import decorator_save_wav
from tts_webui.decorators.gradio_dict_decorator import dictionarize
from tts_webui.decorators.log_function_time import log_function_time
from tts_webui.extensions_loader.decorator_extensions import (
    decorator_extension_inner,
    decorator_extension_outer,
)
from tts_webui.utils.list_dir_models import unload_model_button
from tts_webui.utils.manage_model_state import manage_model_state
from tts_webui.utils.randomize_seed import randomize_seed_ui


def extension__tts_generation_webui():
    ui()
    return {
        "package_name": "extension_openvoice",
        "name": "OpenVoice",
        "version": "0.0.1",
        "requirements": "git+https://github.com/rsxdalv/extension_openvoice@main",
        "description": "OpenVoice: A versatile instant voice cloning approach",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "MyShell AI",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/myshell-ai/OpenVoice",
        "extension_website": "https://github.com/rsxdalv/extension_openvoice",
        "extension_platform_version": "0.0.1",
    }


def ensure_model_downloaded(repo_id, filename, local_dir):
    from huggingface_hub import hf_hub_download

    local_path = os.path.join(local_dir, filename)
    if os.path.exists(local_path):
        print(f"File {filename} already exists at {local_path}")
        return local_path

    print(f"Downloading {filename} from {repo_id} to {local_dir}...")
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
    )


def download_model(repo_id="camenduru/OpenVoice"):
    model_dir_base = os.path.join("data", "models", "openvoice")
    os.makedirs(model_dir_base, exist_ok=True)

    for filename in [
        "checkpoints/base_speakers/EN/config.json",
        "checkpoints/base_speakers/EN/checkpoint.pth",
        "checkpoints/base_speakers/EN/en_default_se.pth",
        "checkpoints/base_speakers/EN/en_style_se.pth",
        "checkpoints/base_speakers/ZH/config.json",
        "checkpoints/base_speakers/ZH/checkpoint.pth",
        "checkpoints/base_speakers/ZH/zh_default_se.pth",
        "checkpoints/converter/config.json",
        "checkpoints/converter/checkpoint.pth",
    ]:
        ensure_model_downloaded(repo_id, filename, model_dir_base)
    return model_dir_base


@manage_model_state("openvoice")
def get_openvoice_models(model_name="camenduru/OpenVoice"):
    """Load the OpenVoice models"""
    try:
        download_model(model_name)

        # Import here to avoid loading the model at startup
        from openvoice import se_extractor
        from openvoice.api import BaseSpeakerTTS, ToneColorConverter

        # Initialize models
        model_dir = "./data/models/openvoice/checkpoints"
        en_ckpt_base = f"{model_dir}/base_speakers/EN"
        zh_ckpt_base = f"{model_dir}/base_speakers/ZH"
        ckpt_converter = f"{model_dir}/converter"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output_dir = "outputs"
        os.makedirs(output_dir, exist_ok=True)

        print(f"Loading OpenVoice models on {device}...")

        # Load English base speaker TTS
        en_base_speaker_tts = BaseSpeakerTTS(
            f"{en_ckpt_base}/config.json", device=device
        )
        en_base_speaker_tts.load_ckpt(f"{en_ckpt_base}/checkpoint.pth")

        # Load Chinese base speaker TTS
        zh_base_speaker_tts = BaseSpeakerTTS(
            f"{zh_ckpt_base}/config.json", device=device
        )
        zh_base_speaker_tts.load_ckpt(f"{zh_ckpt_base}/checkpoint.pth")

        # Load tone color converter
        tone_color_converter = ToneColorConverter(
            f"{ckpt_converter}/config.json", device=device
        )
        tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

        # Load speaker embeddings
        en_source_default_se = torch.load(f"{en_ckpt_base}/en_default_se.pth").to(
            device
        )
        en_source_style_se = torch.load(f"{en_ckpt_base}/en_style_se.pth").to(device)
        zh_source_se = torch.load(f"{zh_ckpt_base}/zh_default_se.pth").to(device)

        print("OpenVoice models loaded successfully")

        return {
            "en_base_speaker_tts": en_base_speaker_tts,
            "zh_base_speaker_tts": zh_base_speaker_tts,
            "tone_color_converter": tone_color_converter,
            "en_source_default_se": en_source_default_se,
            "en_source_style_se": en_source_style_se,
            "zh_source_se": zh_source_se,
            "device": device,
            "output_dir": output_dir,
            "se_extractor": se_extractor,
        }
    except Exception as e:
        print(f"Error loading OpenVoice models: {e}")
        import traceback

        traceback.print_exc()
        raise


@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("openvoice")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@log_function_time
def tts(text: str, style: str, reference_audio: str, **kwargs):
    """Run OpenVoice text-to-speech generation"""
    models = get_openvoice_models(model_name="camenduru/OpenVoice")

    # Initialize variables
    text_hint = ""
    supported_languages = ["zh", "en"]

    # Detect the input language
    language_predicted = langid.classify(text)[0].strip()
    print(f"Detected language: {language_predicted}")

    if language_predicted not in supported_languages:
        error_msg = f"The detected language {language_predicted} for your input text is not in our Supported Languages: {supported_languages}"
        print(f"[ERROR] {error_msg}")
        raise gr.Error(error_msg)

    # Select the appropriate model based on language
    if language_predicted == "zh":
        tts_model = models["zh_base_speaker_tts"]
        source_se = models["zh_source_se"]
        language = "Chinese"
        if style not in ["default"]:
            error_msg = f"The style {style} is not supported for Chinese, which should be in ['default']"
            print(f"[ERROR] {error_msg}")
            raise gr.Error(error_msg)
    else:
        tts_model = models["en_base_speaker_tts"]
        if style == "default":
            source_se = models["en_source_default_se"]
        else:
            source_se = models["en_source_style_se"]
        language = "English"
        if style not in [
            "default",
            "whispering",
            "shouting",
            "excited",
            "cheerful",
            "terrified",
            "angry",
            "sad",
            "friendly",
        ]:
            error_msg = f"The style {style} is not supported for English, which should be in ['default', 'whispering', 'shouting', 'excited', 'cheerful', 'terrified', 'angry', 'sad', 'friendly']"
            print(f"[ERROR] {error_msg}")
            raise gr.Error(error_msg)

    # Validate text input
    if len(text) < 2:
        error_msg = "Please give a longer prompt text"
        print(f"[ERROR] {error_msg}")
        raise gr.Error(error_msg)

    # Extract speaker embedding from reference audio
    try:
        target_se, wavs_folder = models["se_extractor"].get_se(
            reference_audio,
            models["tone_color_converter"],
            target_dir="processed",
            vad=True,
        )
    except Exception as e:
        error_msg = f"Get target tone color error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        raise gr.Error(error_msg)

    # Generate base audio
    src_path = f'{models["output_dir"]}/tmp.wav'
    tts_model.tts(text, src_path, speaker=style, language=language)

    # Apply voice conversion
    save_path = f'{models["output_dir"]}/output.wav'
    encode_message = "@MyShell"
    models["tone_color_converter"].convert(
        audio_src_path=src_path,
        src_se=source_se,
        tgt_se=target_se,
        output_path=save_path,
        message=encode_message,
    )

    print(f"Generation successful, saved to {save_path}")

    # Return the audio output
    import soundfile as sf

    audio_data, sample_rate = sf.read(save_path)

    return {
        "audio_out": (sample_rate, audio_data),
        "reference_audio": reference_audio,
    }


def ui():
    # Default text example
    default_text = "OpenVoice is a versatile instant voice cloning approach that requires only a short audio clip from the reference speaker."

    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                lines=3,
                label="Text to generate",
                placeholder="Enter text here...",
                value=default_text,
            )

            style = gr.Dropdown(
                label="Style",
                choices=[
                    "default",
                    "whispering",
                    "cheerful",
                    "terrified",
                    "angry",
                    "sad",
                    "friendly",
                ],
                value="default",
                # info="Select a style of output audio for the synthesised speech. (Chinese only supports 'default')"
            )

            reference_audio = gr.Audio(
                label="Reference Audio",
                type="filepath",
                # info="Upload your own target speaker audio"
            )

            generate_btn = gr.Button("Generate Audio", variant="primary")

        with gr.Column():
            with gr.Accordion("Information", open=True):
                gr.Markdown(
                    """
                # OpenVoice

                OpenVoice is a versatile instant voice cloning approach that requires only a short audio clip from the reference speaker to replicate their voice and generate speech in multiple languages.

                ## Features
                - Instant voice cloning from a short audio clip
                - Multiple language support (English and Chinese)
                - Voice style control (emotion, accent, rhythm, etc.)
                - Cross-lingual voice cloning

                ## Usage
                1. Enter the text you want to generate
                2. Select a voice style
                3. Upload a reference audio file
                4. Click "Generate Audio"

                ## Supported Languages
                - English (all styles)
                - Chinese (only 'default' style)
                """
                )

            with gr.Column():
                unload_model_button("openvoice")
                seed, randomize_seed_callback = randomize_seed_ui()

    with gr.Column():
        audio_out = gr.Audio(
            label="Generated Audio",
            type="numpy",
            autoplay=False,
        )

    generate_btn.click(
        **randomize_seed_callback,
    ).then(
        **dictionarize(
            fn=tts,
            inputs={
                text: "text",
                style: "style",
                reference_audio: "reference_audio",
                seed: "seed",
            },
            outputs={
                "audio_out": audio_out,
                "metadata": gr.JSON(visible=False),
                "folder_root": gr.Textbox(visible=False),
            },
        ),
        api_name="openvoice",
    )


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        ui()
    demo.launch()
