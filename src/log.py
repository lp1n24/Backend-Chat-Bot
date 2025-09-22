import logging, os

# Set default level as INFO for normal monitoring. Set to DEBUG only when we need details
def setup():
    level = os.getenv("log_level", "info").upper()
    level = getattr(logging, level, logging.info)
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )