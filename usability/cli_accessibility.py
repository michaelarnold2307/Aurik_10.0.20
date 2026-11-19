"""
CLI Accessibility Module for AURIK v8
====================================

Provides comprehensive accessibility features for command-line interfaces:
- High contrast mode (environment variable detection)
- Screen reader friendly output (plain text fallback)
- Audio cues for important events
- Keyboard shortcuts with visual feedback
- Progress indicators with descriptive text
- Color blindness safe palettes

Usage:
    from usability.cli_accessibility import AccessibleCLI, CLITheme

    cli = AccessibleCLI(theme='auto')  # auto, plain, colorful, high_contrast
    cli.header("Processing Audio Files")
    cli.success("File processed successfully!")
    cli.progress("Processing", current=50, total=100)
    cli.play_sound('success')  # Beep notification

Environment Variables:
    AURIK_NO_COLOR=1          # Disable all colors (screen reader mode)
    AURIK_HIGH_CONTRAST=1     # Enable high contrast theme
    AURIK_AUDIO_FEEDBACK=1    # Enable audio beeps
    NO_COLOR=1                # Standard no-color env var (https://no-color.org/)

Author: AURIK Team
Version: 8.0
"""

import logging
import os
import sys
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

# Optional dependencies with graceful fallback
# WICHTIG: colorama.init() wird NICHT auf Modulebene aufgerufen.
# Auf Linux funktionieren ANSI-Codes nativ im Terminal ohne init().
# colorama.init() wrapping würde pytest capsys-Capture korrumpieren.
try:
    from colorama import Back, Fore, Style

    HAS_COLORAMA = True
except ImportError:
    HAS_COLORAMA = False

    # Fallback: direkte ANSI-Escape-Codes (Linux/macOS ohne colorama)
    class _AnsiBase:
        """Minimal ANSI-Fallback; gibt leere Strings auf NO_COLOR-Systemen zurück."""

        CYAN = "\033[36m"
        WHITE = "\033[37m"
        GREEN = "\033[32m"
        RED = "\033[31m"
        YELLOW = "\033[33m"
        BLUE = "\033[34m"

    class _AnsiBack:
        BLUE = "\033[44m"
        WHITE = "\033[47m"

    class _AnsiStyle:
        BRIGHT = "\033[1m"
        DIM = "\033[2m"
        RESET_ALL = "\033[0m"

    Fore = _AnsiBase()  # type: ignore[assignment]
    Back = _AnsiBack()  # type: ignore[assignment]
    Style = _AnsiStyle()  # type: ignore[assignment]


@dataclass
class CLITheme:
    """Color theme for CLI output"""

    name: str
    header: str
    success: str
    error: str
    warning: str
    info: str
    progress_bar: str
    progress_text: str
    dim: str
    reset: str
    bold: str

    @classmethod
    def get_theme(cls, theme_name: str = "auto") -> "CLITheme":
        """Get theme by name or auto-detect based on environment"""

        # Check environment variables
        no_color = os.getenv("NO_COLOR") or os.getenv("AURIK_NO_COLOR")
        high_contrast = os.getenv("AURIK_HIGH_CONTRAST")

        # NO_COLOR / NO_AURIK_COLOR: immer plain (Accessibility-Standard)
        if no_color:
            return cls.plain()

        # AURIK_HIGH_CONTRAST hat höchste Priorität bei auto-Auswahl (Barrierefreiheit).
        # Muss VOR dem isatty()-Check geprüft werden, da pytest/CI kein tty hat.
        if theme_name == "auto":
            if high_contrast:
                return cls.high_contrast()
            # tty-Check NUR für auto-Auswahl (kein tty → kein Farbausdruck nötig)
            if not sys.stdout.isatty():
                return cls.plain()
            theme_name = "colorful"

        # Explizites Theme: tty-Zustand ignorieren (Nutzer hat bewusst gewählt)

        if theme_name == "plain":
            return cls.plain()
        elif theme_name == "high_contrast":
            return cls.high_contrast()
        else:
            return cls.colorful()

    @classmethod
    def plain(cls) -> "CLITheme":
        """Plain theme (no colors, screen reader friendly)"""
        return cls(
            name="plain",
            header="",
            success="",
            error="",
            warning="",
            info="",
            progress_bar="",
            progress_text="",
            dim="",
            reset="",
            bold="",
        )

    @classmethod
    def colorful(cls) -> "CLITheme":
        """Colorful theme with color-blind safe colors"""
        return cls(
            name="colorful",
            header=Fore.CYAN + Style.BRIGHT,
            success=Fore.GREEN,
            error=Fore.RED + Style.BRIGHT,
            warning=Fore.YELLOW,
            info=Fore.BLUE,
            progress_bar=Fore.GREEN,
            progress_text=Fore.CYAN,
            dim=Style.DIM,
            reset=Style.RESET_ALL,
            bold=Style.BRIGHT,
        )

    @classmethod
    def high_contrast(cls) -> "CLITheme":
        """High contrast theme (black background assumed)"""
        return cls(
            name="high_contrast",
            header=Fore.WHITE + Back.BLUE + Style.BRIGHT,
            success=Fore.GREEN + Style.BRIGHT,
            error=Fore.RED + Back.WHITE + Style.BRIGHT,
            warning=Fore.YELLOW + Style.BRIGHT,
            info=Fore.CYAN + Style.BRIGHT,
            progress_bar=Fore.GREEN + Style.BRIGHT,
            progress_text=Fore.WHITE + Style.BRIGHT,
            dim=Fore.WHITE,
            reset=Style.RESET_ALL,
            bold=Style.BRIGHT,
        )


class AccessibleCLI:
    """
    Accessible CLI interface with screen reader support,
    audio feedback, and high contrast modes.
    """

    def __init__(self, theme: str = "auto", audio_feedback: bool | None = None, verbose: bool = True):
        """
        Initialize accessible CLI

        Args:
            theme: Theme name ('auto', 'plain', 'colorful', 'high_contrast')
            audio_feedback: Enable audio beeps (None = auto-detect from env)
            verbose: Enable verbose output
        """
        self.theme = CLITheme.get_theme(theme)
        self.verbose = verbose

        # Auto-detect audio feedback from environment
        if audio_feedback is None:
            audio_feedback = bool(os.getenv("AURIK_AUDIO_FEEDBACK"))
        self.audio_feedback = audio_feedback

        # Track if we're in screen reader mode
        self.screen_reader_mode = self.theme.name == "plain"

    def _print(self, message: str, prefix: str = "", color: str = ""):
        """Interne Ausgabe: print() für User-facing Output (nicht logging)."""
        if self.screen_reader_mode:
            print(f"{prefix}{message}")
        else:
            print(f"{color}{prefix}{message}{self.theme.reset}")

    def header(self, text: str, char: str = "="):
        """Abschnitts-Überschrift ausgeben."""
        if self.screen_reader_mode:
            print(f"\n=== {text.upper()} ===\n")
        else:
            border = char * max(len(text), 3)
            print(f"\n{self.theme.header}{border}")
            print(f"{text.upper()}")
            print(f"{border}{self.theme.reset}\n")

    def success(self, message: str):
        """Erfolgsmeldung ausgeben."""
        if self.screen_reader_mode:
            print(f"[SUCCESS] {message}")
        else:
            print(f"{self.theme.success}[SUCCESS] {message}{self.theme.reset}")
        if self.audio_feedback:
            self.play_sound("success")

    def error(self, message: str):
        """Fehlermeldung ausgeben."""
        if self.screen_reader_mode:
            print(f"[ERROR] {message}")
        else:
            print(f"{self.theme.error}[ERROR] {message}{self.theme.reset}")
        if self.audio_feedback:
            self.play_sound("error")

    def warning(self, message: str):
        """Warnmeldung ausgeben."""
        if self.screen_reader_mode:
            print(f"[WARNING] {message}")
        else:
            print(f"{self.theme.warning}[WARNING] {message}{self.theme.reset}")

    def info(self, message: str):
        """Informationsmeldung ausgeben."""
        if self.screen_reader_mode:
            print(f"[INFO] {message}")
        else:
            print(f"{self.theme.info}[INFO] {message}{self.theme.reset}")

    def dim(self, message: str):
        """Gedämpfte Ausgabe (Metadaten, Zeitstempel)."""
        if self.screen_reader_mode:
            print(message)
        else:
            print(f"{self.theme.dim}{message}{self.theme.reset}")

    def progress(
        self,
        label: str,
        current: int,
        total: int,
        width: int = 40,
        show_percentage: bool = True,
        show_counts: bool = True,
    ):
        """
        Print progress bar with descriptive text

        Args:
            label: Description of what's being processed
            current: Current progress value
            total: Total value
            width: Width of progress bar (ignored in screen reader mode)
            show_percentage: Show percentage
            show_counts: Show current/total counts
        """
        percentage = (current / total) * 100 if total > 0 else 0

        if self.screen_reader_mode:
            # Screen-Reader-Modus: Text-only Fortschrittsanzeige
            parts = [f"{label}: {current}/{total}"]
            if show_percentage:
                parts.append(f"({percentage:.1f}%)")
            print(" ".join(parts))
        else:
            # Visueller Modus: Fortschrittsbalken mit Unicode-Blöcken
            filled = int(width * current / total) if total > 0 else 0
            bar = "█" * filled + "░" * (width - filled)

            parts = [
                f"{self.theme.progress_text}{label}:{self.theme.reset}",
                f"{self.theme.progress_bar}[{bar}]{self.theme.reset}",
            ]
            if show_percentage:
                parts.append(f"{self.theme.progress_text}{percentage:>5.1f}%{self.theme.reset}")
            if show_counts:
                parts.append(f"{self.theme.dim}({current}/{total}){self.theme.reset}")

            # \r überschreibt laufende Zeile; Newline am Ende
            print("\r" + " ".join(parts), end="", flush=True)
            if current >= total:
                print()  # Abschließende Newline

    def list_options(self, options: dict[str, str], title: str = "Options"):
        """
        Print list of options with keyboard shortcuts

        Args:
            options: Dict of key -> description
            title: Section title
        """
        self.header(title)

        for key, description in options.items():
            if self.screen_reader_mode:
                print(f"Press {key}: {description}")
            else:
                print(f"  {self.theme.bold}[{key}]{self.theme.reset} {description}")

        print()

    def prompt(self, message: str, default: str | None = None, valid_choices: list[str] | None = None) -> str:
        """
        Prompt user for input with accessibility features

        Args:
            message: Prompt message
            default: Default value (shown in brackets)
            valid_choices: List of valid choices (case-insensitive)

        Returns:
            User input (or default if empty)
        """
        # Build prompt
        prompt_parts = [f"{self.theme.info}?{self.theme.reset}" if not self.screen_reader_mode else "[INPUT]"]
        prompt_parts.append(message)

        if default:
            prompt_parts.append(f"[{default}]")

        if valid_choices:
            choices_str = "/".join(valid_choices)
            prompt_parts.append(f"({choices_str})")

        prompt_text = " ".join(prompt_parts) + ": "

        # Get input
        while True:
            try:
                user_input = input(prompt_text).strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Cancelled by user]")
                sys.exit(1)

            # Use default if empty
            if not user_input and default:
                return default

            # Validate choices
            if valid_choices:
                if user_input.lower() in [c.lower() for c in valid_choices]:
                    return user_input.lower()
                else:
                    self.error(f"Invalid choice. Please choose from: {', '.join(valid_choices)}")
                    continue

            return user_input

    def confirm(self, message: str, default: bool = False) -> bool:
        """
        Ask yes/no confirmation

        Args:
            message: Confirmation message
            default: Default value if user just presses Enter

        Returns:
            True if user confirms, False otherwise
        """
        response = self.prompt(message, default="y" if default else "n", valid_choices=["y", "n"])
        return response.lower() == "y"

    def play_sound(self, sound_type: Literal["success", "error", "warning", "info"] = "info"):
        """
        Play audio beep for accessibility

        Args:
            sound_type: Type of sound to play
        """
        if not self.audio_feedback:
            return

        # Use system bell (most portable)
        try:
            # Different beep patterns for different events
            if sound_type == "success":
                print("\a", end="", flush=True)  # Single beep
            elif sound_type == "error":
                print("\a\a", end="", flush=True)  # Double beep
            elif sound_type == "warning":
                print("\a", end="", flush=True)  # Single beep
            else:
                print("\a", end="", flush=True)  # Single beep
        except Exception:
            logger.warning("cli_accessibility.py::play_sound fallback", exc_info=True)
            pass  # Silently fail if beep not supported

    def table(self, headers: list[str], rows: list[list[str]], alignments: list[str] | None = None):
        """
        Print accessible table with proper alignment

        Args:
            headers: Column headers
            rows: List of row data
            alignments: List of 'left', 'right', or 'center' for each column
        """
        if not alignments:
            alignments = ["left"] * len(headers)

        # Calculate column widths
        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                col_widths[i] = max(col_widths[i], len(str(cell)))

        # Helper function to align text
        def align_text(text: str, width: int, alignment: str) -> str:
            text = str(text)
            if alignment == "right":
                return text.rjust(width)
            elif alignment == "center":
                return text.center(width)
            else:
                return text.ljust(width)

        # Print header
        header_row = " | ".join(align_text(h, col_widths[i], alignments[i]) for i, h in enumerate(headers))

        if self.screen_reader_mode:
            print(f"\n[TABLE: {', '.join(headers)}]")
            print(header_row)
            print("-" * len(header_row))
        else:
            print(f"\n{self.theme.bold}{header_row}{self.theme.reset}")
            print(f"{self.theme.dim}{'-' * len(header_row)}{self.theme.reset}")

        # Print rows
        for row in rows:
            row_text = " | ".join(align_text(cell, col_widths[i], alignments[i]) for i, cell in enumerate(row))
            print(row_text)

        if self.screen_reader_mode:
            print("[END TABLE]\n")
        else:
            print()

    def separator(self, char: str = "-"):
        """Trennlinie ausgeben."""
        if not self.screen_reader_mode:
            print(f"{self.theme.dim}{char * 80}{self.theme.reset}")
        else:
            print()


# Convenience functions for quick usage
def cli_print_progress(label: str, current: int, total: int):
    """Quick progress bar (uses default theme)"""
    cli = AccessibleCLI()
    cli.progress(label, current, total)


def cli_confirm(message: str, default: bool = False) -> bool:
    """Quick confirmation prompt"""
    # Hier sollte die eigentliche Logik für die Bestätigung stehen
    logging.info(f"[CONFIRM] {message} (default={default})")
    return default
    logging.info("Interactive Features (press Ctrl+C to skip)")
    logging.info("=" * 80 + "\n")

    try:
        cli = AccessibleCLI(audio_feedback=True)

        name = cli.prompt("What's your name?", default="User")
        cli.success(f"Hello, {name}!")

        proceed = cli.confirm("Do you want to continue?", default=True)
        if proceed:
            cli.info("Continuing...")
        else:
            cli.warning("Operation cancelled")

    except KeyboardInterrupt:
        print("\n[Demo interrupted]")
