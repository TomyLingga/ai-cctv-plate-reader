"""alpr-id: Indonesian licence plate reader for RTSP CCTV with a REST API."""

__version__ = "0.4.2"


def main() -> None:
    from alpr_id.cli import main as cli_main

    cli_main()
