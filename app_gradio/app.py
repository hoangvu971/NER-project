"""Gradio app for web deployment"""
import bentoml
import gradio as gr

README = "README.md"

def extract_info(input_text, labels):
    with bentoml.SyncHTTPClient("http://localhost:3000") as client:
        output_dict = client.extract(input_text, labels)
    return output_dict


def process_input(input_text, labels_input):
    # Convert the comma-separated string to a list
    labels = [label.strip() for label in labels_input.split(",")]
    return extract_info(input_text, labels)

def _load_readme():
    with open(README) as f:
        lines = f.readlines()
        readme = "".join(lines)
    return readme

readme = _load_readme()
examples = [["Libretto by Marius Petipa, based on the 1822 novella ``Trilby, ou Le Lutin d'Argail`` by Charles Nodier, first presented by the Ballet of the Moscow Imperial Bolshoi Theatre on January 25/February 6 (Julian/Gregorian calendar dates), 1870, in Moscow with Polina Karpakova as Trilby and Ludiia Geiten as Miranda and restaged by Petipa for the Imperial Ballet at the Imperial Bolshoi Kamenny Theatre on January 17–29, 1871 in St. Petersburg with Adèle Grantzow as Trilby and Lev Ivanov as Count Leopold." , "person, book, location, date, actor, character"]]

# Create the Gradio interface
iface = gr.Interface(
    fn=process_input,
    inputs=[gr.Textbox(lines=5, label="Input Text"), gr.Textbox(label="Labels (comma-separated)")],
    outputs=gr.JSON(label="Extracted Information"),
    title="Named Entity Extractor",
    description="Enter text and labels to extract entities.",
    article=readme,
    examples=examples
)

# Launch the app
iface.launch(share=True)
