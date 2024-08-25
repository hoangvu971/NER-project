"""Import Hugging-face model and convert to ONNX format"""
import argparse
import os
import subprocess

from gliner import GLiNER

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_name", type=str, default="urchade/gliner_medium-v2.1")
    parser.add_argument("--model_save_path", type=str, default="../models/gliner_medium-v2.1")
    args = parser.parse_args()

    model = GLiNER.from_pretrained(args.model_name)
    if not os.path.exists(args.model_save_path):
        os.mkdir(args.model_save_path)
    model.save_pretrained(args.model_save_path)

    # Convert the model to ONNX format
    script = "convert_to_onnx.py"
    subprocess.run(["python", script, "--model_path", args.model_save_path, "--save_path", args.model_save_path])

