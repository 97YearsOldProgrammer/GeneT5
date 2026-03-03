#!/usr/bin/env python3

import argparse

import torch

import lib.inference.engine as engine_lib
from web.app import create_app


parser = argparse.ArgumentParser(description='Launch GeneT5 web demo')
parser.add_argument('--model',     required=True, type=str, help='Model checkpoint path')
parser.add_argument('--tokenizer', default=None,  type=str, help='Tokenizer path (default: same as model)')
parser.add_argument('--device',    default=None,  type=str, help='Device (auto-detect if omitted)')
parser.add_argument('--port',      default=7860,  type=int, help='Server port')
parser.add_argument('--share',     action='store_true',     help='Create public Gradio link')
parser.add_argument('--host',      default='0.0.0.0', type=str, help='Server host')

args = parser.parse_args()

print("Loading model...")
device = torch.device(args.device) if args.device else None
inferencer = engine_lib.GeneT5Inference.from_pretrained(
    checkpoint_path = args.model,
    tokenizer_path  = args.tokenizer or args.model,
    device          = device,
)

print("Starting Gradio app...")
app = create_app(inferencer)
app.launch(
    server_name = args.host,
    server_port = args.port,
    share       = args.share,
)
