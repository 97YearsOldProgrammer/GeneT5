import os
import re
import subprocess
import tempfile
from pathlib import Path


def check_busco_installed():
    """
    Check if BUSCO is installed and available.

    Returns:
        True if BUSCO is installed, False otherwise
    """

    try:
        result = subprocess.run(
            ["busco", "--version"],
            capture_output = True,
            text           = True,
            timeout        = 10
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def parse_busco_output(short_summary_path):
    """
    Parse BUSCO short summary file.

    Args:
        short_summary_path: Path to BUSCO short_summary*.txt file

    Returns:
        Dict with complete, single_copy, duplicated, fragmented,
        missing (as fractions), total, and lineage
    """

    result = {
        "complete":    0.0,
        "single_copy": 0.0,
        "duplicated":  0.0,
        "fragmented":  0.0,
        "missing":     0.0,
        "total":       0,
        "lineage":     ""
    }

    with open(short_summary_path, 'r') as f:
        content = f.read()

    lineage_match = re.search(r'lineage dataset is:\s*(\S+)', content)
    if lineage_match:
        result["lineage"] = lineage_match.group(1)

    summary_match = re.search(
        r'C:([\d.]+)%\[S:([\d.]+)%,D:([\d.]+)%\],F:([\d.]+)%,M:([\d.]+)%,n:(\d+)',
        content
    )

    if summary_match:
        result["complete"]    = float(summary_match.group(1)) / 100
        result["single_copy"] = float(summary_match.group(2)) / 100
        result["duplicated"]  = float(summary_match.group(3)) / 100
        result["fragmented"]  = float(summary_match.group(4)) / 100
        result["missing"]     = float(summary_match.group(5)) / 100
        result["total"]       = int(summary_match.group(6))
    else:
        patterns = {
            "complete":    r'(\d+)\s+Complete BUSCOs',
            "single_copy": r'(\d+)\s+Complete and single-copy BUSCOs',
            "duplicated":  r'(\d+)\s+Complete and duplicated BUSCOs',
            "fragmented":  r'(\d+)\s+Fragmented BUSCOs',
            "missing":     r'(\d+)\s+Missing BUSCOs',
            "total":       r'(\d+)\s+Total BUSCO groups searched'
        }

        counts = {}
        for key, pattern in patterns.items():
            match = re.search(pattern, content)
            if match:
                counts[key] = int(match.group(1))

        if "total" in counts and counts["total"] > 0:
            total            = counts["total"]
            result["total"]  = total
            result["complete"]    = counts.get("complete", 0) / total
            result["single_copy"] = counts.get("single_copy", 0) / total
            result["duplicated"]  = counts.get("duplicated", 0) / total
            result["fragmented"]  = counts.get("fragmented", 0) / total
            result["missing"]     = counts.get("missing", 0) / total

    return result


def run_busco(input_file, lineage, mode="genome", output_dir=None, cpu=1, quiet=True):
    """
    Run BUSCO analysis on input file.

    Args:
        input_file: Path to input file (genome FASTA, proteins, or transcriptome)
        lineage:    BUSCO lineage dataset (e.g., "mammalia_odb10", "bacteria_odb10")
        mode:       Analysis mode - "genome", "proteins", or "transcriptome"
        output_dir: Output directory (default: temp directory)
        cpu:        Number of CPUs to use
        quiet:      Suppress BUSCO output

    Returns:
        Dict with complete, single_copy, duplicated, fragmented,
        missing (as fractions), total, and lineage

    Raises:
        RuntimeError:      If BUSCO is not installed or fails
        FileNotFoundError: If input file doesn't exist
    """

    if not check_busco_installed():
        raise RuntimeError(
            "BUSCO is not installed. Install with: conda install -c bioconda busco"
        )

    input_path = Path(input_file)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    valid_modes = ["genome", "proteins", "transcriptome"]
    if mode not in valid_modes:
        raise ValueError(f"Mode must be one of {valid_modes}, got: {mode}")

    use_temp = output_dir is None
    if use_temp:
        temp_dir   = tempfile.mkdtemp(prefix="busco_")
        output_dir = temp_dir

    output_name = f"busco_{input_path.stem}"

    cmd = [
        "busco",
        "-i", str(input_path),
        "-l", lineage,
        "-m", mode,
        "-o", output_name,
        "--out_path", output_dir,
        "-c", str(cpu),
        "-f"
    ]

    if quiet:
        cmd.append("--quiet")

    try:
        result = subprocess.run(
            cmd,
            capture_output = True,
            text           = True,
            timeout        = 3600
        )

        if result.returncode != 0:
            raise RuntimeError(f"BUSCO failed: {result.stderr}")

        output_path   = Path(output_dir) / output_name
        summary_files = list(output_path.glob("short_summary*.txt"))

        if not summary_files:
            raise RuntimeError(f"No BUSCO summary file found in {output_path}")

        busco_result = parse_busco_output(str(summary_files[0]))

        if not busco_result["lineage"]:
            busco_result["lineage"] = lineage

        return busco_result

    except subprocess.TimeoutExpired:
        raise RuntimeError("BUSCO timed out after 1 hour")
    finally:
        if use_temp:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
