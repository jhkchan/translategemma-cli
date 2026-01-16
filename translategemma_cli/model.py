"""Model management: download, convert, and load TranslateGemma."""

from __future__ import annotations

import os
import sys
import platform
import warnings
from pathlib import Path
from typing import Any, Literal

# Suppress tokenizer warnings before any transformers imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*tokenizer.*")
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress transformers logging
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from .config import (
    get_config,
    get_model_path,
    get_hf_model_id,
    MODEL_SIZES,
    MODEL_INFO,
    DEFAULT_MODEL_SIZE,
)

console = Console()

# Backend types
Backend = Literal["mlx", "pytorch"]


def get_backend() -> Backend:
    """
    Detect platform and return appropriate backend.
    
    Returns:
        "mlx" for macOS Apple Silicon, "pytorch" otherwise
    """
    system = platform.system()
    machine = platform.machine()
    
    if system == "Darwin" and machine == "arm64":
        # Check if MLX is available
        try:
            import mlx
            return "mlx"
        except ImportError:
            return "pytorch"
    
    return "pytorch"


def is_model_ready(
    model_size: str | None = None,
    quantization_bits: int | None = None,
) -> bool:
    """
    Check if the converted model exists.
    
    Args:
        model_size: Model size to check. If None, uses config default.
        quantization_bits: Quantization level (4 or 8). If None, uses config default.
    """
    config = get_config()
    size = model_size or config.model_size
    bits = config.quantization_bits if quantization_bits is None else quantization_bits
    model_path = get_model_path(size, bits)
    
    if not model_path.exists():
        return False
    
    # At minimum, check for config.json
    return (model_path / "config.json").exists()


def list_downloaded_models() -> list[dict]:
    """List all downloaded models with their info."""
    config = get_config()
    models = []
    
    for size in MODEL_SIZES:
        path = get_model_path(size, config.quantization_bits)
        info = MODEL_INFO[size].copy()
        info["size"] = size
        info["path"] = str(path)
        info["downloaded"] = is_model_ready(size)
        
        if info["downloaded"]:
            # Calculate actual size
            total_size = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
            info["actual_size_gb"] = round(total_size / (1024 ** 3), 2)
        
        models.append(info)
    
    return models


def download_and_convert_model(
    model_size: str | None = None,
    quantization_bits: int | None = None,
    force: bool = False,
) -> Path:
    """
    Download TranslateGemma from HuggingFace and convert to optimized format.
    
    Args:
        model_size: Model size (4b, 12b, 27b). If None, uses config default.
        quantization_bits: Quantization level (4 or 8). If None, uses config default.
        force: Force re-download and conversion even if model exists
        
    Returns:
        Path to the converted model
    """
    config = get_config()
    size = model_size or config.model_size
    bits = config.quantization_bits if quantization_bits is None else quantization_bits
    
    if size not in MODEL_SIZES:
        console.print(f"[red]Invalid model size: {size}[/red]")
        console.print(f"[dim]Available sizes: {', '.join(MODEL_SIZES)}[/dim]")
        raise SystemExit(1)
    
    if bits not in (4, 8):
        console.print(f"[red]Invalid quantization bits: {bits}[/red]")
        console.print("[dim]Available options: 4, 8[/dim]")
        raise SystemExit(1)
    
    model_path = get_model_path(size, bits)
    hf_model_id = get_hf_model_id(size)
    
    if is_model_ready(size, bits) and not force:
        console.print(f"[green]Model already available at {model_path}[/green]")
        return model_path
    
    if model_path.exists() and not is_model_ready(size, bits):
        # Remove incomplete conversions to avoid mlx failing on existing paths.
        if model_path.is_dir():
            import shutil
            shutil.rmtree(model_path)
        else:
            model_path.unlink()
    
    console.print("[bold]Model not found locally.[/bold]\n")
    
    backend = get_backend()
    
    if backend == "mlx":
        return _download_mlx(hf_model_id, model_path, bits)
    else:
        return _download_pytorch(hf_model_id, model_path, bits)


def _download_mlx(hf_model_id: str, model_path: Path, quantization_bits: int) -> Path:
    """Download and convert model using MLX backend."""
    try:
        from mlx_lm import convert
    except ImportError as e:
        console.print(f"[red]Error importing mlx-lm: {e}[/red]")
        console.print("\n[yellow]To fix, try:[/yellow]")
        console.print("  pip install --upgrade mlx mlx-lm torch")
        console.print("\n[dim]Note: mlx-lm requires PyTorch >= 2.2[/dim]")
        raise SystemExit(1)
    
    # Ensure parent directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract model size from hf_model_id (e.g., "google/translategemma-27b-it" -> "27b")
    model_name = hf_model_id.split("/")[-1]  # "translategemma-27b-it"
    size_key = model_name.replace("translategemma-", "").replace("-it", "")  # "27b"
    info = MODEL_INFO.get(size_key, {})
    size_hint = info.get("quantized_size_gb", "unknown")
    
    console.print(f"[cyan]Downloading {hf_model_id} from HuggingFace...[/cyan]")
    console.print(f"[dim]Quantized size will be ~{size_hint} GB[/dim]\n")
    
    console.print(f"[cyan]Converting to MLX format with {quantization_bits}-bit quantization...[/cyan]")
    console.print("[dim]This may take 10-20 minutes on first run.[/dim]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Converting model...", total=None)
        
        try:
            convert(
                hf_path=hf_model_id,
                mlx_path=str(model_path),
                quantize=True,
                q_bits=quantization_bits,
            )
            progress.update(task, completed=True)
        except Exception as e:
            console.print(f"\n[red]Error during conversion: {e}[/red]")
            console.print("\n[yellow]Troubleshooting:[/yellow]")
            console.print("1. Ensure you're logged in to HuggingFace: huggingface-cli login")
            console.print(f"2. Accept the model license at: https://huggingface.co/{hf_model_id}")
            console.print("3. Check available disk space")
            raise SystemExit(1)
    
    console.print(f"\n[green]✓ Model ready at {model_path}[/green]\n")
    return model_path


def _check_bitsandbytes() -> tuple[bool, str | None]:
    """
    Check if bitsandbytes is properly installed and functional.
    
    Returns:
        Tuple of (is_available, error_message)
    """
    import sys
    import io
    
    # First check if CUDA is available - no point checking bitsandbytes without it
    try:
        import torch
        if not torch.cuda.is_available():
            return False, "No NVIDIA GPU with CUDA detected (CPU-only mode)"
    except ImportError:
        return False, "PyTorch not installed"
    
    # Suppress all output during bitsandbytes check (it prints warnings on non-CUDA systems)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    try:
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()
        
        # Suppress warnings too
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            try:
                import bitsandbytes as bnb
            except ImportError as e:
                return False, f"bitsandbytes not installed (run: pip install bitsandbytes)"
            except Exception as e:
                error_msg = str(e)
                if "No package metadata" in error_msg or "metadata" in error_msg.lower():
                    return False, (
                        "bitsandbytes installation is corrupted. "
                        "Fix with: pip uninstall bitsandbytes && pip install bitsandbytes --no-cache-dir"
                    )
                return False, f"bitsandbytes import error: {e}"
            
            # Try to access package metadata to verify installation
            try:
                _ = bnb.__version__
            except AttributeError:
                return False, (
                    "bitsandbytes installation is incomplete. "
                    "Fix with: pip uninstall bitsandbytes && pip install bitsandbytes --no-cache-dir"
                )
            
            # Quick sanity check that bnb can work with CUDA
            try:
                _ = bnb.functional
                return True, None
            except Exception as e:
                return False, f"bitsandbytes CUDA initialization failed: {e}"
                
    except Exception as e:
        error_msg = str(e)
        if "No package metadata" in error_msg or "metadata" in error_msg.lower():
            return False, (
                "bitsandbytes installation is corrupted. "
                "Fix with: pip uninstall bitsandbytes && pip install bitsandbytes --no-cache-dir"
            )
        return False, f"bitsandbytes error: {e}"
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr


def _download_pytorch(hf_model_id: str, model_path: Path, quantization_bits: int) -> Path:
    """Download and convert model using PyTorch backend."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        console.print(f"[red]Error importing transformers/torch: {e}[/red]")
        console.print("\n[yellow]To fix, try:[/yellow]")
        console.print("  pip install transformers torch accelerate")
        raise SystemExit(1)
    
    # Ensure parent directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if bitsandbytes is available for quantization
    bnb_available, bnb_error = _check_bitsandbytes()
    use_quantization = bnb_available and torch.cuda.is_available()
    
    if not bnb_available:
        console.print(f"[yellow]⚠ Quantization unavailable: {bnb_error}[/yellow]")
        if torch.cuda.is_available():
            console.print("[yellow]Will download full-precision model instead (requires more VRAM).[/yellow]")
            console.print("[dim]To enable quantization, fix bitsandbytes:[/dim]")
            console.print("[dim]  pip uninstall bitsandbytes[/dim]")
            console.print("[dim]  pip install bitsandbytes --no-cache-dir[/dim]\n")
        else:
            console.print("[dim]Quantization requires NVIDIA GPU with CUDA.[/dim]\n")
    elif not torch.cuda.is_available():
        console.print("[yellow]⚠ No CUDA GPU detected. Using CPU mode (slower, no quantization).[/yellow]\n")
        use_quantization = False
    
    console.print(f"[cyan]Downloading {hf_model_id} from HuggingFace...[/cyan]")
    console.print("[dim]This may take a while depending on your connection.[/dim]\n")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Downloading model...", total=None)
        
        try:
            # Download tokenizer first
            progress.update(task, description="Downloading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
            tokenizer.save_pretrained(str(model_path))
            
            # Configure model loading based on availability
            if use_quantization:
                from transformers import BitsAndBytesConfig
                
                if quantization_bits == 4:
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_quant_type="nf4",
                    )
                else:
                    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
                
                progress.update(task, description=f"Downloading model ({quantization_bits}-bit quantized)...")
                model = AutoModelForCausalLM.from_pretrained(
                    hf_model_id,
                    quantization_config=bnb_config,
                    device_map="auto",
                    trust_remote_code=True,
                )
            else:
                # Fallback: load without quantization
                progress.update(task, description="Downloading model (full precision)...")
                device = "cuda" if torch.cuda.is_available() else "cpu"
                model = AutoModelForCausalLM.from_pretrained(
                    hf_model_id,
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    device_map="auto" if device == "cuda" else None,
                    trust_remote_code=True,
                    low_cpu_mem_usage=True,
                )
            
            progress.update(task, description="Saving model...")
            model.save_pretrained(str(model_path))
            
            progress.update(task, completed=True)
            
        except Exception as e:
            error_str = str(e)
            console.print(f"\n[red]Error during download: {e}[/red]")
            
            # Provide specific troubleshooting based on error
            console.print("\n[yellow]Troubleshooting:[/yellow]")
            
            if "bitsandbytes" in error_str.lower() or "bnb" in error_str.lower():
                console.print("1. bitsandbytes issue detected. Try reinstalling:")
                console.print("   pip uninstall bitsandbytes")
                console.print("   pip install bitsandbytes --no-cache-dir")
                console.print("2. Or install without quantization (requires more VRAM):")
                console.print("   pip install translategemma-cli[cpu]")
            elif "cuda" in error_str.lower() or "gpu" in error_str.lower():
                console.print("1. CUDA/GPU issue. Check your NVIDIA drivers:")
                console.print("   nvidia-smi")
                console.print("2. Ensure PyTorch is installed with CUDA support:")
                console.print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
            else:
                console.print("1. Ensure you're logged in to HuggingFace: huggingface-cli login")
                console.print(f"2. Accept the model license at: https://huggingface.co/{hf_model_id}")
                console.print("3. Check available disk space and GPU memory")
            
            raise SystemExit(1)
    
    if use_quantization:
        console.print(f"\n[green]✓ Quantized model ready at {model_path}[/green]\n")
    else:
        console.print(f"\n[green]✓ Model ready at {model_path}[/green]")
        console.print("[dim]Note: Running without quantization. Consider fixing bitsandbytes for lower memory usage.[/dim]\n")
    
    return model_path


def load_model(model_size: str | None = None) -> tuple[Any, Any, Backend]:
    """
    Load the TranslateGemma model and tokenizer.
    
    Args:
        model_size: Model size to load. If None, uses config default.
    
    Returns:
        Tuple of (model, tokenizer, backend)
    """
    config = get_config()
    size = model_size or config.model_size
    model_path = get_model_path(size, config.quantization_bits)
    
    if not is_model_ready(size):
        download_and_convert_model(size)
    
    backend = get_backend()
    
    if backend == "mlx":
        return _load_mlx(model_path)
    else:
        return _load_pytorch(model_path)


def _load_mlx(model_path: Path) -> tuple[Any, Any, Backend]:
    """Load model using MLX backend."""
    try:
        from mlx_lm import load, generate
        import mlx.core as mx
    except ImportError as e:
        console.print(f"[red]Error importing mlx-lm: {e}[/red]")
        console.print("\n[yellow]To fix, try:[/yellow]")
        console.print("  pip install --upgrade mlx mlx-lm torch")
        raise SystemExit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        
        # Use lazy=False to fully load model into memory at startup
        # This ensures consistent inference speed throughout the session
        model, tokenizer = load(
            str(model_path),
            lazy=False,
        )
        
        # Force model evaluation to ensure weights are fully loaded
        progress.update(task, description="Loading weights into memory...")
        mx.eval(model.parameters())
        
        # Warmup: Run a small inference to compile Metal shaders
        # This eliminates the cold start delay on first actual query
        progress.update(task, description="Warming up (compiling shaders)...")
        _ = generate(
            model,
            tokenizer,
            prompt="Hello",
            max_tokens=1,
            verbose=False,
        )
        
        progress.update(task, description="Model ready")
    
    console.print("[dim]Model loaded and ready.[/dim]\n")
    
    return model, tokenizer, "mlx"


def _load_pytorch(model_path: Path) -> tuple[Any, Any, Backend]:
    """Load model using PyTorch backend."""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch
    except ImportError as e:
        console.print(f"[red]Error importing transformers/torch: {e}[/red]")
        console.print("\n[yellow]To fix, try:[/yellow]")
        console.print("  pip install transformers torch accelerate")
        raise SystemExit(1)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Loading model...", total=None)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        
        # Try loading with bitsandbytes quantization first, fallback to standard loading
        model = None
        bnb_available, _ = _check_bitsandbytes()
        
        if device == "cuda" and bnb_available:
            try:
                # Try loading as quantized model
                model = AutoModelForCausalLM.from_pretrained(
                    str(model_path),
                    device_map="auto",
                    trust_remote_code=True,
                )
            except Exception:
                # Fallback to standard loading
                pass
        
        if model is None:
            # Standard loading without quantization
            model = AutoModelForCausalLM.from_pretrained(
                str(model_path),
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            if device == "cpu":
                model = model.to(device)
        
        # Warmup: Run a small inference to initialize CUDA kernels
        progress.update(task, description="Warming up...")
        inputs = tokenizer("Hello", return_tensors="pt")
        if device == "cuda":
            inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            _ = model.generate(**inputs, max_new_tokens=1)
        
        progress.update(task, description="Model ready")
    
    console.print("[dim]Model loaded and ready.[/dim]\n")
    
    return model, tokenizer, "pytorch"


def get_model_info(model_size: str | None = None) -> dict:
    """Get information about a model."""
    config = get_config()
    size = model_size or config.model_size
    model_path = get_model_path(size, config.quantization_bits)
    hf_model_id = get_hf_model_id(size)
    
    info = {
        "size": size,
        "hf_source": hf_model_id,
        "local_path": str(model_path),
        "ready": is_model_ready(size),
        "quantization_bits": config.quantization_bits,
        "backend": get_backend(),
        "params": MODEL_INFO[size]["params"],
    }
    
    if info["ready"]:
        # Calculate model size
        total_size = sum(f.stat().st_size for f in model_path.rglob("*") if f.is_file())
        info["size_gb"] = round(total_size / (1024 ** 3), 2)
    
    return info


def remove_model(model_size: str) -> bool:
    """
    Remove a downloaded model.
    
    Args:
        model_size: Model size to remove
        
    Returns:
        True if removed, False if not found
    """
    config = get_config()
    model_path = get_model_path(model_size, config.quantization_bits)
    
    if not model_path.exists():
        return False
    
    import shutil
    shutil.rmtree(model_path)
    return True
