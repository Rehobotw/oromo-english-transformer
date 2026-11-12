import gradio as gr
from pathlib import Path
from translator.config import TransformerConfig
from translator.inference import translate

# ---------------------------------------------------------------------------
# Config and checkpoint
# ---------------------------------------------------------------------------
config = TransformerConfig()

# Pick the latest checkpoint available, or fall back to epoch '00'
checkpoints = sorted(Path(config.model_folder).glob('tmodel_*.pt'))
latest_epoch = checkpoints[-1].stem.replace(config.model_basename, '') if checkpoints else '00'


# ---------------------------------------------------------------------------
# Translation function wired to the UI
# ---------------------------------------------------------------------------
def run_translation(text: str, direction: str) -> str:
    """Translate text in the chosen direction and return the result string."""
    if not text.strip():
        return ''

    cfg = TransformerConfig()

    if direction == 'English → Oromo':
        cfg.lang_src = 'eng'
        cfg.lang_tgt = 'orm'
    else:
        cfg.lang_src = 'orm'
        cfg.lang_tgt = 'eng'

    try:
        return translate(text, cfg, checkpoint_epoch=latest_epoch)
    except FileNotFoundError:
        return '⚠️  No trained checkpoint found. Run train.py first.'
    except ValueError as e:
        return f'⚠️  {e}'


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------
with gr.Blocks(title='Oromo ↔ English Translator') as demo:
    gr.Markdown('# Oromo ↔ English Translator')
    gr.Markdown('Translate between English and Afaan Oromo using a trained Transformer model.')

    direction = gr.Radio(
        choices=['English → Oromo', 'Oromo → English'],
        value='English → Oromo',
        label='Translation direction',
    )

    with gr.Row():
        source_box = gr.Textbox(label='Source text', lines=4, placeholder='Type a sentence…')
        output_box = gr.Textbox(label='Translation',  lines=4, interactive=False)

    translate_btn = gr.Button('Translate', variant='primary')

    translate_btn.click(fn=run_translation, inputs=[source_box, direction], outputs=output_box)
    source_box.submit(fn=run_translation, inputs=[source_box, direction], outputs=output_box)


if __name__ == '__main__':
    demo.launch()

