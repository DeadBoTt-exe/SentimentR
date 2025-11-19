"""
Command-line interface for Enhanced SentimentR
"""
import asyncio
import typer
import json
import sys
import os
from pathlib import Path
from typing import Optional, List
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.markdown import Markdown
import time

# Handle both relative and absolute imports
try:
    from ..core.analyzer import HybridSentimentAnalyzer
    from ..core.models import AnalysisConfig, SentimentMethod
except ImportError:
    # Fallback for direct execution
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from enhanced_sentimentr.core.analyzer import HybridSentimentAnalyzer
    from enhanced_sentimentr.core.models import AnalysisConfig, SentimentMethod


app = typer.Typer(
    name="sentimentr",
    help="Enhanced SentimentR - Advanced sentiment analysis with Gemini integration",
    add_completion=False
)

console = Console()


def get_sentiment_emoji(polarity: float) -> str:
    """Get emoji based on sentiment polarity"""
    if polarity > 0.5:
        return "😊"
    elif polarity > 0.1:
        return "🙂"
    elif polarity < -0.5:
        return "😞"
    elif polarity < -0.1:
        return "😐"
    else:
        return "😶"


def get_sentiment_label(polarity: float) -> str:
    """Get text label for sentiment"""
    if polarity > 0.1:
        return "[green]Positive[/green]"
    elif polarity < -0.1:
        return "[red]Negative[/red]"
    else:
        return "[yellow]Neutral[/yellow]"


@app.command()
def analyze(
    text: str = typer.Argument(..., help="Text to analyze"),
    method: SentimentMethod = typer.Option(SentimentMethod.HYBRID, help="Analysis method"),
    emotions: bool = typer.Option(False, "--emotions", "-e", help="Include emotion analysis"),
    aspects: bool = typer.Option(False, "--aspects", "-a", help="Include aspect analysis"),
    explanation: bool = typer.Option(False, "--explanation", "-x", help="Include explanation"),
    gemini_key: Optional[str] = typer.Option(None, "--gemini-key", "-k", help="Gemini API key"),
    output_format: str = typer.Option("rich", "--format", "-f", help="Output format: rich, json, csv"),
    save_to: Optional[str] = typer.Option(None, "--save", "-s", help="Save results to file"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output")
):
    """
    Analyze sentiment of a single text
    """
    try:
        # Initialize analyzer
        analyzer = HybridSentimentAnalyzer(gemini_key)
        
        # Create config
        config = AnalysisConfig(
            method=method,
            include_emotions=emotions,
            include_aspects=aspects,
            include_explanation=explanation
        )
        
        # Show progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True
        ) as progress:
            task = progress.add_task("Analyzing sentiment...", total=None)
            
            # Run analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(analyzer.analyze(text, config))
            loop.close()
        
        # Output results
        if output_format == "json":
            output = json.dumps(result.to_dict(), indent=2)
        elif output_format == "csv":
            df = pd.DataFrame([{
                'text': text,
                'polarity': result.polarity,
                'confidence': result.confidence,
                'method': result.method.value,
                'sentiment': 'positive' if result.polarity > 0.1 else 'negative' if result.polarity < -0.1 else 'neutral'
            }])
            output = df.to_csv(index=False)
        else:  # rich format
            emoji = get_sentiment_emoji(result.polarity)
            sentiment_label = get_sentiment_label(result.polarity)
            
            # Main results panel
            results_content = f"""
**Text:** {text[:100]}{'...' if len(text) > 100 else ''}

**Results:**
• Sentiment: {emoji} {sentiment_label}
• Polarity: {result.polarity:.3f}
• Confidence: {result.confidence:.1%}
• Method: {result.method.value}
"""
            
            if result.processing_time:
                results_content += f"• Processing Time: {result.processing_time:.3f}s\n"
            
            if result.subjectivity is not None:
                results_content += f"• Subjectivity: {result.subjectivity:.1%}\n"
            
            if result.intensity is not None:
                results_content += f"• Intensity: {result.intensity:.1%}\n"
            
            console.print(Panel(
                Markdown(results_content),
                title="🎭 Sentiment Analysis Results",
                border_style="blue"
            ))
            
            # Emotions
            if result.emotions:
                emotions_table = Table(title="😊 Emotion Analysis", show_header=True)
                emotions_table.add_column("Emotion", style="cyan")
                emotions_table.add_column("Intensity", style="magenta")
                
                for emotion, score in result.emotions.items():
                    if score > 0.05:  # Only show significant emotions
                        emotions_table.add_row(emotion.value.title(), f"{score:.1%}")
                
                console.print(emotions_table)
            
            # Aspects
            if result.aspects:
                aspects_table = Table(title="🎯 Aspect-Based Sentiment", show_header=True)
                aspects_table.add_column("Aspect", style="cyan")
                aspects_table.add_column("Sentiment", style="magenta")
                
                for aspect, score in result.aspects.items():
                    sentiment = "positive" if score > 0.1 else "negative" if score < -0.1 else "neutral"
                    color = "green" if score > 0.1 else "red" if score < -0.1 else "yellow"
                    aspects_table.add_row(aspect, f"[{color}]{sentiment}[/{color}] ({score:.3f})")
                
                console.print(aspects_table)
            
            # Explanation
            if result.explanation:
                console.print(Panel(
                    result.explanation,
                    title="📋 Analysis Explanation",
                    border_style="green"
                ))
            
            # Verbose output
            if verbose:
                console.print(Panel(
                    json.dumps(result.to_dict(), indent=2),
                    title="🔍 Raw Data",
                    border_style="dim"
                ))
            
            output = ""  # Don't save rich output to file
        
        # Save to file if requested
        if save_to and output:
            Path(save_to).write_text(output)
            console.print(f"✅ Results saved to {save_to}")
        
        # Print output for non-rich formats
        if output_format != "rich":
            console.print(output)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def batch(
    input_file: str = typer.Argument(..., help="Input file (txt or csv)"),
    method: SentimentMethod = typer.Option(SentimentMethod.HYBRID, help="Analysis method"),
    emotions: bool = typer.Option(False, "--emotions", "-e", help="Include emotion analysis"),
    aspects: bool = typer.Option(False, "--aspects", "-a", help="Include aspect analysis"),
    explanation: bool = typer.Option(False, "--explanation", "-x", help="Include explanation"),
    gemini_key: Optional[str] = typer.Option(None, "--gemini-key", "-k", help="Gemini API key"),
    output_file: Optional[str] = typer.Option(None, "--output", "-o", help="Output file"),
    batch_size: int = typer.Option(10, "--batch-size", "-b", help="Batch processing size"),
    format: str = typer.Option("csv", "--format", "-f", help="Output format: csv, json")
):
    """
    Analyze sentiment of multiple texts from a file
    """
    try:
        # Read input file
        input_path = Path(input_file)
        if not input_path.exists():
            console.print(f"[red]Error:[/red] File {input_file} not found")
            raise typer.Exit(1)
        
        texts = []
        if input_path.suffix.lower() == '.csv':
            df = pd.read_csv(input_file)
            if 'text' in df.columns:
                texts = df['text'].astype(str).tolist()
            else:
                console.print("[red]Error:[/red] CSV file must have a 'text' column")
                raise typer.Exit(1)
        else:
            with open(input_file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
        
        if not texts:
            console.print("[red]Error:[/red] No texts found in input file")
            raise typer.Exit(1)
        
        console.print(f"📋 Found {len(texts)} texts to analyze")
        
        # Initialize analyzer
        analyzer = HybridSentimentAnalyzer(gemini_key)
        
        # Create config
        config = AnalysisConfig(
            method=method,
            include_emotions=emotions,
            include_aspects=aspects,
            include_explanation=explanation
        )
        
        # Run batch analysis with progress
        with Progress(console=console) as progress:
            task = progress.add_task("Analyzing texts...", total=len(texts))
            
            # Run analysis
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            batch_result = loop.run_until_complete(
                analyzer.batch_analyze(texts, config, batch_size)
            )
            loop.close()
            
            progress.update(task, completed=len(texts))
        
        # Show summary
        console.print(f"✅ Analysis complete!")
        console.print(f"• Total texts: {batch_result.total_texts}")
        console.print(f"• Successful: {batch_result.successful_analyses}")
        console.print(f"• Failed: {batch_result.failed_analyses}")
        console.print(f"• Average time: {batch_result.average_processing_time:.3f}s")
        console.print(f"• Total time: {batch_result.total_processing_time:.1f}s")
        
        # Prepare output data
        results_data = []
        for i, (text, result) in enumerate(zip(texts, batch_result.results)):
            row = {
                'index': i + 1,
                'text': text,
                'polarity': result.polarity,
                'confidence': result.confidence,
                'sentiment': 'positive' if result.polarity > 0.1 else 'negative' if result.polarity < -0.1 else 'neutral',
                'method': result.method.value,
                'processing_time': result.processing_time
            }
            
            if result.subjectivity is not None:
                row['subjectivity'] = result.subjectivity
            
            if result.intensity is not None:
                row['intensity'] = result.intensity
            
            if result.emotions:
                for emotion, score in result.emotions.items():
                    row[f'emotion_{emotion.value}'] = score
            
            if result.aspects:
                for aspect, score in result.aspects.items():
                    row[f'aspect_{aspect}'] = score
            
            if result.explanation:
                row['explanation'] = result.explanation
            
            results_data.append(row)
        
        # Output results
        if format == "json":
            output = json.dumps(results_data, indent=2)
        else:  # csv
            df = pd.DataFrame(results_data)
            output = df.to_csv(index=False)
        
        # Save or print results
        if output_file:
            Path(output_file).write_text(output)
            console.print(f"📁 Results saved to {output_file}")
        else:
            console.print(output)
        
        # Show basic statistics
        polarities = [r.polarity for r in batch_result.results]
        positive_count = sum(1 for p in polarities if p > 0.1)
        negative_count = sum(1 for p in polarities if p < -0.1)
        neutral_count = len(polarities) - positive_count - negative_count
        
        stats_table = Table(title="📊 Summary Statistics", show_header=True)
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Count", style="magenta")
        stats_table.add_column("Percentage", style="green")
        
        total = len(polarities)
        stats_table.add_row("Positive", str(positive_count), f"{positive_count/total:.1%}")
        stats_table.add_row("Negative", str(negative_count), f"{negative_count/total:.1%}")
        stats_table.add_row("Neutral", str(neutral_count), f"{neutral_count/total:.1%}")
        
        console.print(stats_table)
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def serve(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", "-r", help="Enable auto-reload")
):
    """
    Start the FastAPI web service
    """
    try:
        import uvicorn
        try:
            from ..api.main import app as fastapi_app
        except ImportError:
            from enhanced_sentimentr.api.main import app as fastapi_app
        
        console.print(f"🚀 Starting Enhanced SentimentR API server...")
        console.print(f"📍 URL: http://{host}:{port}")
        console.print(f"📚 Docs: http://{host}:{port}/docs")
        
        uvicorn.run(
            "enhanced_sentimentr.api.main:app",
            host=host,
            port=port,
            reload=reload
        )
        
    except ImportError:
        console.print("[red]Error:[/red] FastAPI dependencies not installed. Install with: pip install fastapi uvicorn")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def streamlit(
    port: int = typer.Option(8501, "--port", "-p", help="Port to bind to")
):
    """
    Start the Streamlit web interface
    """
    try:
        import subprocess
        import sys
        
        script_path = Path(__file__).parent.parent / "web" / "streamlit_app.py"
        
        console.print(f"🎭 Starting Enhanced SentimentR Streamlit app...")
        console.print(f"📍 URL: http://localhost:{port}")
        
        cmd = [sys.executable, "-m", "streamlit", "run", str(script_path), "--server.port", str(port)]
        subprocess.run(cmd)
        
    except ImportError:
        console.print("[red]Error:[/red] Streamlit not installed. Install with: pip install streamlit")
        raise typer.Exit(1)
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


@app.command()
def version():
    """
    Show version information
    """
    console.print("🎭 Enhanced SentimentR v2.0.0")
    console.print("Advanced sentiment analysis with Gemini integration")
    console.print("Built on the original SentimentR by Mohammad Darwich")


@app.command()
def config():
    """
    Show configuration information
    """
    try:
        analyzer = HybridSentimentAnalyzer()
        
        config_table = Table(title="⚙️ Configuration", show_header=True)
        config_table.add_column("Setting", style="cyan")
        config_table.add_column("Value", style="magenta")
        
        config_table.add_row("Gemini Available", "✅ Yes" if analyzer.is_gemini_available() else "❌ No")
        
        # Check for API key in environment
        gemini_key = os.getenv('GEMINI_API_KEY')
        config_table.add_row("Gemini API Key", "✅ Set" if gemini_key else "❌ Not set")
        
        # Available methods
        methods = ", ".join([method.value for method in SentimentMethod])
        config_table.add_row("Available Methods", methods)
        
        console.print(config_table)
        
        if not analyzer.is_gemini_available():
            console.print(Panel(
                "To enable Gemini analysis, set your API key:\n"
                "• Use --gemini-key option\n"
                "• Set GEMINI_API_KEY environment variable",
                title="💡 Tip",
                border_style="yellow"
            ))
        
    except Exception as e:
        console.print(f"[red]Error:[/red] {str(e)}")
        raise typer.Exit(1)


def main():
    """Main CLI entry point"""
    app()


if __name__ == "__main__":
    main()
