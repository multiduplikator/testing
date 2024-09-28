import os
import re
import csv
import json
import configparser
import subprocess
from pathlib import Path
from alive_progress import alive_bar

import nvidia.cudnn.lib


def load_config(config_path):
    """Load and return the configuration from the given path."""
    config = configparser.ConfigParser()
    config.read(config_path)
    return config


def get_paths_and_settings(config):
    """Retrieve relevant paths and settings from the config."""
    return {
        "ld_lib_path": Path(nvidia.cudnn.lib.__file__).parent,
        "video_directory": Path(
            config["transcriber"]["transcribe_directory"]
        ).expanduser(),
        "regex_file": Path(config["transcriber"]["transcribe_regexes"]).expanduser(),
        "language": config["transcriber"].get(
            "transcriber_language", "en"
        ),  # Default to "en" if not specified
    }


def load_regex_replacements(file_path):
    """Load regex patterns and replacements from a CSV file."""
    reg_repl_list = []
    with file_path.open(newline="", encoding="utf-8") as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header row if present
        for pattern, replacement, ignore_case_str in reader:
            if len([pattern, replacement, ignore_case_str]) == 3:
                ignore_case = ignore_case_str.strip().lower() == "true"
                flags = re.IGNORECASE if ignore_case else 0
                try:
                    reg_repl_list.append((re.compile(pattern, flags), replacement))
                except re.error as e:
                    print(f"Error compiling regex pattern '{pattern}': {e}")
    return reg_repl_list


def srt_to_json(srt_file, reg_repl_list):
    """Convert SRT file to JSON with regex replacements."""

    def offset_seconds(ts):
        """Convert timestamp (HH:MM:SS,mmm) to seconds."""
        return sum(
            h * s
            for h, s in zip(
                map(int, ts.replace(",", ":").split(":")), [3600, 60, 1, 0.001]
            )
        )

    def apply_replacements(text):
        """Apply regex replacements to text."""
        for reg, repl in reg_repl_list:
            text = re.sub(reg, repl, text)
        return text

    regex = r"(\d+:\d+:\d+,\d+) --> (\d+:\d+:\d+,\d+)\s+(.+?)(?:\n\n|$)"
    transcript = []
    with srt_file.open(encoding="utf-8") as f:
        for startTime, endTime, ref in re.findall(regex, f.read(), re.DOTALL):
            transcript.append(
                {
                    "start": offset_seconds(startTime),
                    "end": offset_seconds(endTime),
                    "text": apply_replacements(ref),
                }
            )

    json_path = srt_file.with_suffix(".json")
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(transcript, f, indent=4)


def execute_command(command_list, env):
    """Execute the command and handle output."""
    try:
        process = subprocess.Popen(
            command_list,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
        )
        stdout, stderr = process.communicate()
        if process.returncode != 0:
            print(f"Command failed with return code: {process.returncode}")
            print(stderr)
        print(stdout)
        return process.returncode
    except Exception as e:
        print(f"Error executing command: {e}")
        return -1


def main():
    config_path = "./config.ini"
    config = load_config(config_path)
    paths_and_settings = get_paths_and_settings(config)
    reg_repl_list = load_regex_replacements(paths_and_settings["regex_file"])

    command = "whisper-ctranslate2"
    parameters = [
        "--model",
        "medium",
        "--word_timestamps",
        "True",
        "--max_line_count",
        "10",
        "--max_line_width",
        "80",
        "--output_format",
        "srt",
        "--output_dir",
        str(paths_and_settings["video_directory"]),
        "--language",
        paths_and_settings["language"],  # Use the language from config
        "--local_files_only",
        "TRUE",
    ]

    video_files = list(paths_and_settings["video_directory"].glob("*.mp4"))

    with alive_bar(len(video_files)) as bar:
        for video_file in video_files:
            command_list = [command] + parameters + [str(video_file)]
            env = os.environ.copy()
            env["LD_LIBRARY_PATH"] = str(paths_and_settings["ld_lib_path"])

            if execute_command(command_list, env) == 0:
                srt_path = video_file.with_suffix(".srt")
                if srt_path.exists():
                    srt_to_json(srt_path, reg_repl_list)
                    bar.text(f"ok: {video_file}")
                else:
                    print(f"SRT file not found: {srt_path}")
                    bar.text(f"missing SRT: {video_file}")
            else:
                bar.text(f"failed: {video_file}")
            bar()


if __name__ == "__main__":
    main()
