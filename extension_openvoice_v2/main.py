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
        "package_name": "extension_openvoice_v2",
        "name": "OpenVoice V2",
        "requirements": "git+https://github.com/rsxdalv/extension_openvoice_v2@main",
        "description": "OpenVoice: A versatile instant voice cloning approach",
        "extension_type": "interface",
        "extension_class": "text-to-speech",
        "author": "MyShell AI",
        "extension_author": "rsxdalv",
        "license": "MIT",
        "website": "https://github.com/myshell-ai/OpenVoice",
        "extension_website": "https://github.com/rsxdalv/extension_openvoice_v2",
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
    model_dir_base = os.path.join("data", "models", "openvoice_v2")
    os.makedirs(model_dir_base, exist_ok=True)

    # Download original OpenVoice models
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

    # Create directories for OpenVoice V2 models
    checkpoints_v2_dir = os.path.join(model_dir_base, "checkpoints_v2")
    os.makedirs(os.path.join(checkpoints_v2_dir, "converter"), exist_ok=True)
    os.makedirs(os.path.join(checkpoints_v2_dir, "base_speakers", "ses"), exist_ok=True)

    # Download OpenVoice V2 models
    for filename in [
        "checkpoints_v2/converter/config.json",
        "checkpoints_v2/converter/checkpoint.pth",
    ]:
        ensure_model_downloaded(repo_id, filename, model_dir_base)

    # Download speaker embeddings for different languages
    speaker_embeddings = [
        "en-default", "en-us", "en-br", "en-au", "en-india",
        "es", "fr", "zh", "jp", "kr"
    ]

    for speaker in speaker_embeddings:
        filename = f"checkpoints_v2/base_speakers/ses/{speaker}.pth"
        ensure_model_downloaded(repo_id, filename, model_dir_base)

    return model_dir_base


@manage_model_state("openvoice_v2")
def get_openvoice_models(model_name="camenduru/OpenVoice", use_v2=True):
    """Load the OpenVoice models"""
    try:
        download_model(model_name)

        # Import here to avoid loading the model at startup
        from openvoice import se_extractor
        from openvoice.api import BaseSpeakerTTS, ToneColorConverter

        # Initialize models
        model_dir = "./data/models/openvoice_v2/checkpoints"
        model_dir_v2 = "./data/models/openvoice_v2/checkpoints_v2"
        en_ckpt_base = f"{model_dir}/base_speakers/EN"
        zh_ckpt_base = f"{model_dir}/base_speakers/ZH"
        ckpt_converter = f"{model_dir}/converter"
        ckpt_converter_v2 = f"{model_dir_v2}/converter"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        output_dir = "temp/openvoice"
        output_dir_v2 = "temp/openvoice_v2"
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(output_dir_v2, exist_ok=True)

        print(f"Loading OpenVoice models on {device}...")

        # Load original OpenVoice models
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

        # Load tone color converter (original)
        tone_color_converter = ToneColorConverter(
            f"{ckpt_converter}/config.json", device=device
        )
        tone_color_converter.load_ckpt(f"{ckpt_converter}/checkpoint.pth")

        # Load speaker embeddings (original)
        en_source_default_se = torch.load(f"{en_ckpt_base}/en_default_se.pth").to(
            device
        )
        en_source_style_se = torch.load(f"{en_ckpt_base}/en_style_se.pth").to(device)
        zh_source_se = torch.load(f"{zh_ckpt_base}/zh_default_se.pth").to(device)

        print("OpenVoice models loaded successfully")

        # Load OpenVoice V2 models if requested
        v2_models = {}
        if use_v2:
            try:
                print("Loading OpenVoice V2 models...")

                # Load tone color converter V2
                tone_color_converter_v2 = ToneColorConverter(
                    f"{ckpt_converter_v2}/config.json", device=device
                )
                tone_color_converter_v2.load_ckpt(f"{ckpt_converter_v2}/checkpoint.pth")

                # Load MeloTTS
                try:
                    from melo.api import TTS

                    # Initialize MeloTTS models for different languages
                    languages = ["EN", "EN_NEWEST", "ES", "FR", "ZH", "JP", "KR"]
                    melo_tts_models = {}

                    for lang in languages:
                        try:
                            melo_tts_models[lang] = TTS(language=lang, device=device)
                            print(f"Loaded MeloTTS model for {lang}")
                        except Exception as e:
                            print(f"Failed to load MeloTTS model for {lang}: {e}")

                    # Load speaker embeddings for V2
                    speaker_embeddings = {}
                    ses_dir = f"{model_dir_v2}/base_speakers/ses"

                    for speaker in ["en-default", "en-us", "en-br", "en-au", "en-india",
                                   "es", "fr", "zh", "jp", "kr"]:
                        try:
                            speaker_embeddings[speaker] = torch.load(f"{ses_dir}/{speaker}.pth", map_location=device)
                            print(f"Loaded speaker embedding for {speaker}")
                        except Exception as e:
                            print(f"Failed to load speaker embedding for {speaker}: {e}")

                    v2_models = {
                        "tone_color_converter_v2": tone_color_converter_v2,
                        "melo_tts_models": melo_tts_models,
                        "speaker_embeddings": speaker_embeddings,
                        "output_dir_v2": output_dir_v2
                    }

                    print("OpenVoice V2 models loaded successfully")
                except ImportError:
                    print("MeloTTS not found. OpenVoice V2 will not be available.")
            except Exception as e:
                print(f"Error loading OpenVoice V2 models: {e}")
                import traceback
                traceback.print_exc()

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
            **v2_models
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
def tts_v1(text: str, style: str, reference_audio: str, language_code: str = "en", **kwargs):
    """Run OpenVoice V1 text-to-speech generation"""
    models = get_openvoice_models(model_name="camenduru/OpenVoice", use_v2=False)

    # Validate text input
    if len(text) < 2:
        error_msg = "Please give a longer prompt text"
        print(f"[ERROR] {error_msg}")
        raise gr.Error(error_msg)

    # Extract speaker embedding from reference audio
    try:
        tone_color_converter = models["tone_color_converter"]

        target_se, _ = models["se_extractor"].get_se(
            reference_audio,
            tone_color_converter,
            target_dir="temp",
            vad=True,
        )
    except Exception as e:
        error_msg = f"Get target tone color error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        raise gr.Error(error_msg)

    # Determine output paths
    output_dir = models["output_dir"]
    src_path = f'{output_dir}/tmp.wav'
    save_path = f'{output_dir}/output.wav'

    # Use original OpenVoice models
    supported_languages = ["zh", "en"]

    # Detect the input language if not specified
    if language_code not in supported_languages:
        language_predicted = langid.classify(text)[0].strip()
        print(f"Detected language: {language_predicted}")

        if language_predicted not in supported_languages:
            error_msg = f"The detected language {language_predicted} for your input text is not in our Supported Languages: {supported_languages}"
            print(f"[ERROR] {error_msg}")
            raise gr.Error(error_msg)

        language_code = language_predicted

    # Select the appropriate model based on language
    if language_code == "zh":
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

    # Generate base audio with original OpenVoice
    tts_model.tts(text, src_path, speaker=style, language=language)

    # Apply voice conversion
    encode_message = "@MyShell"
    tone_color_converter.convert(
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


@decorator_extension_outer
@decorator_apply_torch_seed
@decorator_save_metadata
@decorator_save_wav
@decorator_add_model_type("openvoice_v2")
@decorator_add_base_filename
@decorator_add_date
@decorator_log_generation
@decorator_extension_inner
@log_function_time
def tts_v2(text: str, reference_audio: str, language_code: str = "en", speaker_accent: str = "default", speed: float = 1.0, **kwargs):
    """Run OpenVoice V2 text-to-speech generation"""
    models = get_openvoice_models(model_name="rsxdalv/OpenVoiceV2", use_v2=True)

    # Validate text input
    if len(text) < 2:
        error_msg = "Please give a longer prompt text"
        print(f"[ERROR] {error_msg}")
        raise gr.Error(error_msg)

    # Extract speaker embedding from reference audio
    try:
        tone_color_converter = models["tone_color_converter_v2"]

        target_se, _ = models["se_extractor"].get_se(
            reference_audio,
            tone_color_converter,
            target_dir="temp",
            vad=True,
        )
    except Exception as e:
        error_msg = f"Get target tone color error: {str(e)}"
        print(f"[ERROR] {error_msg}")
        raise gr.Error(error_msg)

    # Determine output paths
    output_dir = models["output_dir_v2"]
    src_path = f'{output_dir}/tmp.wav'
    save_path = f'{output_dir}/output.wav'

    # Use MeloTTS for V2
    try:
        # Map language_code to MeloTTS language code
        melo_lang_map = {
            "en": "EN",
            "en_newest": "EN_NEWEST",
            "es": "ES",
            "fr": "FR",
            "zh": "ZH",
            "jp": "JP",
            "kr": "KR"
        }

        melo_lang = melo_lang_map.get(language_code.lower(), "EN")

        if melo_lang not in models["melo_tts_models"]:
            error_msg = f"Language {language_code} is not supported in MeloTTS models"
            print(f"[ERROR] {error_msg}")
            raise gr.Error(error_msg)

        # Get the MeloTTS model for the selected language
        melo_model = models["melo_tts_models"][melo_lang]

        # Get available speaker IDs for this language
        speaker_ids = melo_model.hps.data.spk2id

        # Format speaker key to match the format in speaker_embeddings
        # For non-English languages, we don't use accents
        if language_code.lower().startswith('en'):
            speaker_key = f"{language_code.lower()}-{speaker_accent.lower()}".replace('_', '-')
        else:
            # For non-English languages, we just use the language code
            speaker_key = language_code.lower().replace('_', '-')

        print(f"Using speaker key: {speaker_key}")

        # Check if we have the speaker embedding
        if speaker_key not in models["speaker_embeddings"]:
            error_msg = f"Speaker embedding for '{speaker_key}' is not available. Available embeddings: {list(models['speaker_embeddings'].keys())}"
            print(f"[ERROR] {error_msg}")
            raise gr.Error(error_msg)

        # Get the source speaker embedding
        source_se = models["speaker_embeddings"][speaker_key]

        # Find the speaker ID
        speaker_id = None
        for spk_key in speaker_ids.keys():
            if spk_key.lower().replace('_', '-') == speaker_key:
                speaker_id = speaker_ids[spk_key]
                break

        if speaker_id is None:
            # Use the first available speaker ID as fallback
            speaker_id = list(speaker_ids.values())[0]
            print(f"Warning: Could not find exact speaker ID for {speaker_key}, using fallback")

        # Generate audio with MeloTTS
        if torch.backends.mps.is_available() and models["device"] == 'cpu':
            torch.backends.mps.is_available = lambda: False

        melo_model.tts_to_file(text, speaker_id, src_path, speed=speed)
        print(f"Generated base audio with MeloTTS for language {melo_lang}")

    except Exception as e:
        error_msg = f"Error generating audio with MeloTTS: {str(e)}"
        print(f"[ERROR] {error_msg}")
        import traceback
        traceback.print_exc()
        raise gr.Error(error_msg)

    # Apply voice conversion
    encode_message = "@MyShell"
    tone_color_converter.convert(
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


def download_unidic():
    """Download unidic dictionary for Japanese language support"""
    try:
        import subprocess
        import sys
        import importlib.util

        # Check if unidic is installed
        if importlib.util.find_spec("unidic") is None:
            # Try to install unidic first
            print("UniDic not found. Attempting to install...")
            install_process = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "unidic"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            install_stdout, install_stderr = install_process.communicate()

            if install_process.returncode != 0:
                error_msg = f"Failed to install UniDic: {install_stderr}"
                print(f"[ERROR] {error_msg}")
                return f"Error: UniDic package is not installed and automatic installation failed.\n\nPlease run these commands manually:\n\npip install unidic\npython -m unidic download"

        # Run the unidic download command
        print("Downloading UniDic dictionary...")
        process = subprocess.Popen(
            [sys.executable, "-m", "unidic", "download"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate()

        if process.returncode == 0:
            success_msg = "UniDic dictionary downloaded successfully! Japanese TTS should now work properly."
            print(success_msg)
            return success_msg
        else:
            error_msg = f"Error downloading UniDic: {stderr}"
            print(f"[ERROR] {error_msg}")
            return f"Failed to download UniDic dictionary.\n\nError: {stderr}\n\nYou may need to run 'python -m unidic download' manually with administrator privileges."
    except Exception as e:
        error_msg = f"Error running unidic download: {str(e)}"
        print(f"[ERROR] {error_msg}")
        return f"Error: {str(e)}\n\nTry running these commands manually:\n\npip install unidic\npython -m unidic download"


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

            # Create tabs for V1 and V2
            with gr.Tabs() as tabs:
                # OpenVoice V1 Tab
                with gr.TabItem("OpenVoice V1") as v1_tab:
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
                        info="Select a style of output audio for the synthesised speech. (Chinese only supports 'default')"
                    )

                    gr.Markdown(
                        """
                        ### OpenVoice V1
                        - Supports English (all styles) and Chinese (only 'default' style)
                        - Language is automatically detected from input text
                        """
                    )

                    # Generate button for V1
                    generate_btn_v1 = gr.Button("Generate with OpenVoice V1", variant="primary")

                # OpenVoice V2 Tab
                with gr.TabItem("OpenVoice V2 with MeloTTS") as v2_tab:
                    language_code = gr.Dropdown(
                        label="Language",
                        choices=[
                            ("English", "en"),
                            # ("English (Newest)", "en_newest"),
                            ("Spanish", "es"),
                            ("French", "fr"),
                            ("Chinese", "zh"),
                            ("Japanese", "jp"),
                            ("Korean", "kr"),
                        ],
                        value="en",
                        info="Select the language for text-to-speech generation"
                    )

                    speaker_accent = gr.Dropdown(
                        label="Speaker Accent",
                        choices=[
                            ("Default", "default"),
                            ("American", "us"),
                            ("British", "br"),
                            ("Australian", "au"),
                            ("Indian", "india"),
                        ],
                        value="default",
                        info="Select the accent for the speaker (only applicable for English)"
                    )

                    speed = gr.Slider(
                        label="Speech Speed",
                        minimum=0.5,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        info="Adjust the speed of the generated speech"
                    )

                    gr.Markdown(
                        """
                        ### OpenVoice V2 with MeloTTS
                        - Multi-accent and multi-lingual voice cloning
                        - Supports English with multiple accents (American, British, Indian, Australian, Default)
                        - Additional languages: Spanish, French, Chinese, Japanese, Korean (each with default accent only)
                        - Adjustable speech speed

                        **Note:** Speaker accents are only available for English languages. For other languages,
                        only the default accent is available and the accent selector will be disabled.

                        **Japanese Language Support:** If you're getting errors with Japanese text, you may need to
                        download the UniDic dictionary using the button below.
                        """
                    )

                    # Add a button for downloading UniDic (for Japanese support)
                    with gr.Row():
                        unidic_btn = gr.Button("Download UniDic Dictionary (Required for Japanese)", variant="secondary")
                        unidic_output = gr.Textbox(label="UniDic Download Status", interactive=False)

                    # Set up the event handler for the UniDic button
                    unidic_btn.click(
                        fn=download_unidic,
                        inputs=[],
                        outputs=[unidic_output]
                    )

                    # Generate button for V2
                    generate_btn_v2 = gr.Button("Generate with OpenVoice V2", variant="primary")

            # Update speaker accent options based on language
            def update_accent_options(language):
                if language.startswith("en"):
                    return gr.Dropdown(
                        choices=[
                            ("Default", "default"),
                            ("American", "us"),
                            ("British", "uk"),
                            ("Australian", "au"),
                            ("Indian", "in"),
                        ],
                        value="default",
                        interactive=True,
                        label="Speaker Accent"
                    )
                else:
                    return gr.Dropdown(
                        choices=[("Default (Only option for non-English)", "default")],
                        value="default",
                        interactive=False,
                        label="Speaker Accent (Not applicable for non-English languages)"
                    )

            language_code.change(
                fn=update_accent_options,
                inputs=[language_code],
                outputs=[speaker_accent]
            )

            reference_audio = gr.Audio(
                label="Reference Audio",
                type="filepath",
                # INFO IS NOT AVAILABLE IN GRADIO 5.x!!! STOP INCLUDING IT
                # info="Upload your own target speaker audio"
            )

        with gr.Column():
            with gr.Accordion("Information", open=True):
                gr.Markdown(
                    """
                # OpenVoice

                OpenVoice is a versatile instant voice cloning approach that requires only a short audio clip from the reference speaker to replicate their voice and generate speech in multiple languages.

                ## Features
                - Instant voice cloning from a short audio clip
                - Multiple language support
                - Voice style control (emotion, accent, rhythm, etc.)
                - Cross-lingual voice cloning

                ## Usage
                1. Enter the text you want to generate
                2. Choose between OpenVoice V1 or V2 tabs
                3. Select language, accent, and style options
                4. Upload a reference audio file
                5. Click the appropriate "Generate" button for the version you want to use

                ## OpenVoice V1
                - Supports English (all styles)
                - Supports Chinese (only 'default' style)
                - Language is automatically detected from input text

                ## OpenVoice V2 with MeloTTS
                - Multi-accent and multi-lingual voice cloning
                - Supports English with multiple accents (American, British, Indian, Australian, Default)
                - Additional languages: Spanish, French, Chinese, Japanese, Korean (each with default accent only)
                - Adjustable speech speed

                **Note:** Speaker accents are only available for English languages. For other languages,
                only the default accent is available.

                **Japanese Support:** For Japanese text-to-speech, you need to install the UniDic dictionary.
                Use the "Download UniDic Dictionary" button in the OpenVoice V2 tab if you encounter errors
                with Japanese text.
                """
                )

            with gr.Column():
                unload_model_button("openvoice_v2")
                seed, randomize_seed_callback = randomize_seed_ui()

    with gr.Column():
        audio_out = gr.Audio(
            label="Generated Audio",
            type="numpy",
            autoplay=False,
        )

    # Set up event handlers for the V1 button
    generate_btn_v1.click(
        **randomize_seed_callback,
    ).then(
        **dictionarize(
            fn=tts_v1,
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
        api_name="openvoice_v1",
    )

    # Set up event handlers for the V2 button
    generate_btn_v2.click(
        **randomize_seed_callback,
    ).then(
        **dictionarize(
            fn=tts_v2,
            inputs={
                text: "text",
                reference_audio: "reference_audio",
                language_code: "language_code",
                speaker_accent: "speaker_accent",
                speed: "speed",
                seed: "seed",
            },
            outputs={
                "audio_out": audio_out,
                "metadata": gr.JSON(visible=False),
                "folder_root": gr.Textbox(visible=False),
            },
        ),
        api_name="openvoice_v2",
    )


if __name__ == "__main__":
    if "demo" in locals():
        locals()["demo"].close()
    with gr.Blocks() as demo:
        ui()
    demo.launch()
