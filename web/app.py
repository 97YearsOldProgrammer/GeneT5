import time
import tempfile
import pathlib as pl

import gradio as gr


def create_app(inferencer):
    """Create Gradio Blocks app with shared inferencer"""

    import lib.inference.engine as engine_lib
    import lib.inference.output as output_lib

    def run_inference(fasta_file, paste_seq, diff_steps, temperature, device_choice):
        """Main inference callback"""

        # Parse input
        if fasta_file is not None:
            sequences, seqids = engine_lib.read_input(fasta_file)
        elif paste_seq and paste_seq.strip():
            text = paste_seq.strip()
            if text.startswith('>'):
                sequences, seqids = _parse_pasted_fasta(text)
            else:
                sequences = [text.replace('\n', '').replace(' ', '')]
                seqids    = ['seq_0']
        else:
            return (
                "No input provided",
                None, None, None, None, None,
                gr.Dataframe(value=[]),
                "0 sequences, 0 features, 0.0s",
            )

        gen_config = engine_lib.GenerationConfig(
            max_length  = 512,
            temperature = temperature,
        )

        t0      = time.time()
        results = inferencer.predict(
            sequences  = sequences,
            seqids     = seqids,
            gen_config = gen_config,
            batch_size = 1,
        )
        elapsed = time.time() - t0

        # Write all formats to temp dir
        tmp_dir = pl.Path(tempfile.mkdtemp(prefix="genet5_"))
        files   = {'gff3': [], 'fasta': [], 'protein': [], 'gtf': [], 'bed': []}

        all_gff_lines = []
        table_rows    = []
        total_feats   = 0

        for r in results:
            sid = r.metadata['seqid']
            paths = output_lib.write_all_formats(
                features       = r.parsed_features,
                input_sequence = r.input_sequence,
                seqid          = sid,
                output_dir     = tmp_dir,
                formats        = list(files.keys()),
                source         = "GeneT5",
            )

            for fmt, path in paths.items():
                files[fmt].append(str(path))

            # GFF3 text preview
            gff_lines = output_lib.features_to_gff3(
                r.parsed_features, r.input_sequence, seqid=sid)
            all_gff_lines.extend(gff_lines)

            # Table data
            n_exons = sum(1 for f in r.parsed_features if f.kind == "exon")
            n_utrs  = sum(1 for f in r.parsed_features if f.kind == "utr")
            total_feats += n_exons + n_utrs
            table_rows.append([sid, len(r.input_sequence), n_exons, n_utrs])

        gff_text = "##gff-version 3\n" + "\n".join(all_gff_lines) if all_gff_lines else ""

        # Return file paths (first of each format, or None)
        gff3_file    = files['gff3'][0]    if files['gff3']    else None
        fasta_out    = files['fasta'][0]   if files['fasta']   else None
        protein_out  = files['protein'][0] if files['protein'] else None
        gtf_out      = files['gtf'][0]     if files['gtf']     else None
        bed_out      = files['bed'][0]     if files['bed']     else None

        stats = f"{len(results)} sequences, {total_feats} features, {elapsed:.1f}s"

        return gff_text, gff3_file, fasta_out, protein_out, gtf_out, bed_out, table_rows, stats

    # Build UI
    with gr.Blocks(
        title="GeneT5 Gene Prediction",
        theme=gr.themes.Soft(),
    ) as app:

        gr.Markdown("# GeneT5 Gene Prediction\nPredict gene structures from DNA sequences using diffusion transformers")

        with gr.Row():
            # Left: Input
            with gr.Column(scale=1):
                gr.Markdown("### Input")
                fasta_upload = gr.File(label="Upload FASTA", file_types=[".fa", ".fasta", ".fna"])
                gr.Markdown("**OR** paste sequence below:")
                paste_input  = gr.Textbox(
                    label="Paste sequence",
                    placeholder=">seq1\nATGCATGC...",
                    lines=6,
                )

                gr.Markdown("### Parameters")
                diff_steps  = gr.Slider(minimum=4, maximum=128, value=32, step=4, label="Diffusion steps")
                temperature = gr.Slider(minimum=0.0, maximum=2.0, value=1.0, step=0.1, label="Temperature")
                device_sel  = gr.Dropdown(choices=["auto", "cuda", "cpu"], value="auto", label="Device")

                run_btn = gr.Button("Run Inference", variant="primary", size="lg")

            # Right: Output
            with gr.Column(scale=1):
                gr.Markdown("### Results")
                results_table = gr.Dataframe(
                    headers=["Sequence", "Length (bp)", "Exons", "UTRs"],
                    label="Feature summary",
                )
                stats_text = gr.Textbox(label="Stats", interactive=False)

                gr.Markdown("### GFF3 Preview")
                gff_preview = gr.Textbox(label="GFF3 output", lines=10, interactive=False)

                gr.Markdown("### Download")
                with gr.Row():
                    dl_gff3    = gr.File(label="GFF3")
                    dl_fasta   = gr.File(label="FASTA")
                    dl_protein = gr.File(label="Protein")
                with gr.Row():
                    dl_gtf = gr.File(label="GTF")
                    dl_bed = gr.File(label="BED")

        run_btn.click(
            fn=run_inference,
            inputs=[fasta_upload, paste_input, diff_steps, temperature, device_sel],
            outputs=[gff_preview, dl_gff3, dl_fasta, dl_protein, dl_gtf, dl_bed, results_table, stats_text],
        )

    return app


def _parse_pasted_fasta(text):
    """Parse pasted FASTA text into sequences and IDs"""

    sequences   = []
    seqids      = []
    current_seq = []
    current_id  = None

    for line in text.strip().split('\n'):
        line = line.strip()
        if line.startswith('>'):
            if current_id is not None:
                sequences.append(''.join(current_seq))
                seqids.append(current_id)
            current_id  = line[1:].split()[0]
            current_seq = []
        else:
            current_seq.append(line)

    if current_id is not None:
        sequences.append(''.join(current_seq))
        seqids.append(current_id)

    return sequences, seqids
