import bentoml
import gradio as gr


def extract_info(input_text, labels):
    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        output_dict = client.extract(input_text, labels)
    return output_dict


def process_input(input_text, labels_input):
    # Convert the comma-separated string to a list
    labels = [label.strip() for label in labels_input.split(",")]
    return extract_info(input_text, labels)


# Create the Gradio interface
iface = gr.Interface(
    fn=process_input,
    inputs=[gr.Textbox(lines=5, label="Input Text"), gr.Textbox(label="Labels (comma-separated)")],
    outputs=gr.JSON(label="Extracted Information"),
    title="Named Entity Extractor",
    description="Enter text and labels to extract entities.",
)

# Launch the app
iface.launch()
